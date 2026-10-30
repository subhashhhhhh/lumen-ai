package com.subhashh.lumen.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Cutout: the subject of a photo, lifted off its background (long-press in the viewer). The model
 * ([CutoutModel]) marks every subject; only the one under the finger is kept, so one person can be
 * taken out of a group standing apart. Pressing the background keeps them all. Runs on the GPU
 * where it passed [GpuCheck] (in 16-bit floats), else the CPU. The model loads when first needed
 * and stays a little after the viewer closes ([release]).
 *
 * Photos with people get a light run around their outline as they settle in the viewer ([outline]),
 * so people find they can lift them out. Which photos have people comes from what's already indexed
 * ([hasPeople]: faces, or the search vector), so swiping through photos runs no model.
 */
object Cutout {
    /**
     * The subject with a transparent background, cropped to it; [bounds] is where that crop sits
     * in the photo, as fractions of its width and height (to lift it from its place on screen).
     */
    class Result(val bitmap: Bitmap, val bounds: RectF)

    private const val Tag = "LumenCutout"
    /** The photo is read at up to this size: the cutout is this sharp, and memory stays bounded. */
    private const val ReadSide = 2048
    private const val KeepMs = 30_000L
    /** A press this close to a subject (in mask pixels, of [CutoutModel.Side]) counts as on it. */
    private const val Reach = 12

    /**
     * Test: the 8-bit model compiled for this phone's NPU, put in the app's external files
     * (models/cutout/) by hand. When it's
     * there (and the phone has a supported NPU), cutouts run on the NPU instead of the GPU. Integer in
     * and out, with the scales int8.py chose: input 0.99291 (zero point -128), mask 1/256 (-128).
     * The 1024 one with a 640 backbone may take the photo twice, at 1024 and at [Small] (resizing
     * inside the model is slow on the NPU).
     */
    private val Int8Files = listOf(1024 to 0.9936917f, 512 to 0.9929091f)
    private val Small = 640 to 0.9951496f

    /** A mask's side: [CutoutModel.Side], or the 8-bit test model's. */
    private fun sideOf(mask: FloatArray) = kotlin.math.sqrt(mask.size.toDouble()).toInt()

    private class Loaded(
        val environment: Environment,
        val model: CompiledModel,
        val int8: Boolean = false,
        val side: Int = CutoutModel.Side,
        val inputScale: Float = 1f,
    ) : AutoCloseable {
        override fun close() {
            model.close()
            environment.close()
        }
    }

    private val lock = Mutex()
    private var loaded: Loaded? = null
    /** The last few photos' masks (every subject), so a hold after the outline doesn't run again. */
    private val masks = android.util.LruCache<String, FloatArray>(6)

    private fun key(item: MediaRef) = "${item.id}@${item.modified}"
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var releasing: Job? = null

    /** The model is on the phone (Settings → Smart Features → Cutout). */
    fun available(context: Context): Boolean = CutoutModel.installed(context)

    /** Loads the model ahead of the first long-press (the viewer opened). */
    fun warm(context: Context) {
        releasing?.cancel()
        if (!available(context)) return
        scope.launch { lock.withLock { runCatching { model(context) } } }
    }

    /**
     * The first load checks the GPU (~5 s) and builds its programs for the model (~10 s, kept on
     * disk: later loads take ~1.5 s). Done once the model has downloaded, so the first touch and
     * hold doesn't wait for it.
     */
    fun prepare(context: Context) {
        if (!available(context)) return
        scope.launch {
            lock.withLock { runCatching { model(context) } }
            release()
        }
    }

    /** Frees the model a little after the viewer closes, unless it opens again first. */
    fun release() {
        releasing?.cancel()
        releasing = scope.launch {
            delay(KeepMs)
            lock.withLock {
                loaded?.close()
                loaded = null
            }
        }
    }

    /**
     * Cuts out the subject at ([x], [y]), fractions of the photo's width and height. Null when
     * there's no model, the photo can't be read, or no subject was found.
     */
    suspend fun cut(context: Context, item: MediaRef, x: Float, y: Float): Result? = withContext(Dispatchers.Default) {
        if (item.isVideo || !available(context)) return@withContext null
        val photo = runCatching { MediaFrames.forReading(context, item, ReadSide) }.getOrNull() ?: return@withContext null
        try {
            val mask = (masks.get(key(item)) ?: masked(context, item, photo) ?: return@withContext null).copyOf()
            if (!keepUnder(mask, x, y)) return@withContext null
            compose(photo, mask)
        } finally {
            photo.recycle()
        }
    }

    private suspend fun masked(context: Context, item: MediaRef, photo: Bitmap): FloatArray? =
        lock.withLock { runCatching { mask(context, photo) }.onFailure { Log.w(Tag, "cutout failed", it) }.getOrNull() }
            ?.also { masks.put(key(item), it) }

    /**
     * The outline of each subject in the photo, as lines to draw around them: points (x, y, x, y…)
     * as fractions of the photo's width and height, in drawing order. Where a subject runs off the
     * photo (a person cut off at the bottom), the line starts and ends there, so it can be drawn
     * from one end up and around to the other. Null without the model, or with no subject.
     */
    suspend fun outline(context: Context, item: MediaRef): List<FloatArray>? = withContext(Dispatchers.Default) {
        if (item.isVideo || !available(context)) return@withContext null
        val mask = masks.get(key(item)) ?: run {
            // The model's input size is all it needs: a much smaller read than a cutout's.
            val photo = runCatching { MediaFrames.forReading(context, item, CutoutModel.Side * 2) }.getOrNull() ?: return@withContext null
            try { masked(context, item, photo) } finally { photo.recycle() }
        } ?: return@withContext null
        lines(mask).takeIf { it.isNotEmpty() }
    }

    /** Each sizeable subject's edge as one line, traced in order ([outline]). */
    private fun lines(mask: FloatArray): List<FloatArray> {
        val side = sideOf(mask)
        val label = IntArray(side * side)
        val sizes = ArrayList<Int>()
        val stack = IntArray(side * side)
        for (seed in label.indices) {
            if (label[seed] != 0 || mask[seed] <= 0.5f) continue
            val id = sizes.size + 1
            var top = 0
            var size = 0
            stack[top++] = seed
            label[seed] = id
            while (top > 0) {
                val i = stack[--top]
                size++
                val ix = i % side
                for (n in intArrayOf(if (ix > 0) i - 1 else -1, if (ix < side - 1) i + 1 else -1, i - side, i + side)) {
                    if (n >= 0 && n < label.size && label[n] == 0 && mask[n] > 0.5f) {
                        label[n] = id
                        stack[top++] = n
                    }
                }
            }
            sizes += size
        }
        // Subjects of at least 1% of the photo, the biggest four, left to right: drawn one after another.
        val kept = sizes.indices.filter { sizes[it] >= side * side / 100 }.sortedByDescending { sizes[it] }.take(4)
        return kept.flatMap { trace(label, it + 1, side) }.sortedBy { it[0] }
    }

    /**
     * The boundary of subject [id] (Moore neighbour tracing, from its lowest leftmost pixel),
     * as one smoothed line of fractions, open where the photo's edge cuts the subject off.
     */
    private fun trace(label: IntArray, id: Int, side: Int): List<FloatArray> {
        fun inside(x: Int, y: Int) = x in 0 until side && y in 0 until side && label[y * side + x] == id
        var sx = -1
        var sy = -1
        for (y in side - 1 downTo 0) {
            for (x in 0 until side) if (label[y * side + x] == id) { sx = x; sy = y; break }
            if (sx >= 0) break
        }
        if (sx < 0) return emptyList()
        // Clockwise on screen (y down), from west.
        val dx = intArrayOf(-1, -1, 0, 1, 1, 1, 0, -1)
        val dy = intArrayOf(0, -1, -1, -1, 0, 1, 1, 1)
        val xs = ArrayList<Int>()
        val ys = ArrayList<Int>()
        var x = sx
        var y = sy
        var from = 0 // the west neighbour is outside: it's the leftmost pixel of its row
        do {
            xs += x
            ys += y
            var found = false
            for (k in 1..8) {
                val d = (from + k) % 8
                if (inside(x + dx[d], y + dy[d])) {
                    x += dx[d]
                    y += dy[d]
                    // Next search starts just past the direction back to where we came from.
                    from = (d + 4) % 8
                    found = true
                    break
                }
            }
            if (!found) break
        } while ((x != sx || y != sy) && xs.size < side * 16)
        // One unbroken line. Where the subject is cut off by the photo's edge (its longest stretch
        // along the edge, usually the bottom), the line starts at one end of that cut and goes
        // all the way around to the other. Fully inside the photo: a loop from its lowest point.
        val n = xs.size
        fun border(i: Int) = xs[i] <= 1 || ys[i] <= 1 || xs[i] >= side - 2 || ys[i] >= side - 2
        if ((0 until n).all { border(it) }) return emptyList()
        var cutStart = -1
        var cutLength = 0
        run {
            var i = (0 until n).first { !border(it) }
            var k = 0
            while (k < n) {
                if (border(i)) {
                    var length = 0
                    while (length < n && border((i + length) % n)) length++
                    if (length > cutLength) { cutStart = i; cutLength = length }
                    i = (i + length) % n
                    k += length
                } else {
                    i = (i + 1) % n
                    k++
                }
            }
        }
        val order = if (cutStart < 0) IntArray(n + 1) { it % n } else IntArray(n - cutLength) { (cutStart + cutLength + it) % n }
        if (order.size < 24) return emptyList()
        return listOf(order).map { r ->
            // Every other pixel, averaged over a few neighbours: no staircase.
            val points = r.filterIndexed { i, _ -> i % 2 == 0 || i == r.size - 1 }
            FloatArray(points.size * 2).also { out ->
                for (j in points.indices) {
                    var ax = 0f
                    var ay = 0f
                    var c = 0
                    for (o in -3..3) {
                        val q = points[(j + o).coerceIn(0, points.size - 1)]
                        ax += xs[q]; ay += ys[q]; c++
                    }
                    out[2 * j] = (ax / c + 0.5f) / side
                    out[2 * j + 1] = (ay / c + 0.5f) / side
                }
            }
        }
    }

    /**
     * The photo likely shows people, from what's already indexed: faces found in it, or its
     * search vector closer to [PeoplePrompts] than to [OtherPrompts]. False when neither is
     * set up or it isn't indexed yet.
     */
    suspend fun hasPeople(context: Context, item: MediaRef): Boolean = withContext(Dispatchers.IO) {
        if (item.isVideo) return@withContext false
        if (FaceModel.installed(context) && runCatching { FaceStore.of(context).facesIn(item.id).isNotEmpty() }.getOrDefault(false)) {
            return@withContext true
        }
        val model = ModelDownloads.installed(context) ?: return@withContext false
        val vectors = runCatching { EmbeddingStore.of(context, model).vectorsOf(item.id) }.getOrNull()?.takeIf { it.isNotEmpty() }
            ?: return@withContext false
        val (people, other) = prompts(context, model) ?: return@withContext false
        fun best(prompts: List<FloatArray>) = vectors.maxOf { v -> prompts.maxOf { q -> var dot = 0f; for (d in v.indices) dot += v[d] * q[d]; dot } }
        best(people) > best(other)
    }

    private val PeoplePrompts = listOf("a photo of a person", "a photo of people", "a selfie", "a portrait of a man", "a portrait of a woman", "a photo of a child")
    private val OtherPrompts = listOf(
        "a photo of a landscape", "a photo of food", "a screenshot", "a photo of a document", "a photo of a building",
        "a photo of a car", "a photo of an animal", "a photo of a room", "a photo of an object", "a photo of the sky",
    )
    @Volatile private var cachedPrompts: Pair<String, Pair<List<FloatArray>, List<FloatArray>>>? = null

    /** The prompts' text vectors: embedded once per search model, then read from a small file. */
    private fun prompts(context: Context, model: AiModel): Pair<List<FloatArray>, List<FloatArray>>? {
        cachedPrompts?.takeIf { it.first == model.id }?.let { return it.second }
        val file = File(context.noBackupFilesDir, "search/cutout-prompts-${model.id}.bin")
        val all = PeoplePrompts + OtherPrompts
        val vectors = VocabularyFile.read(file)?.takeIf { it.first == all }?.second ?: runCatching {
            ModelRepair.loading(context) { OnnxEmbedder(model.dir(context), model.spec, threads = 2) }.use { e -> all.map { e.embedText(it) } }
                .also { VocabularyFile.write(file.apply { parentFile?.mkdirs() }, all, it) }
        }.getOrNull() ?: return null
        val split = vectors.take(PeoplePrompts.size) to vectors.drop(PeoplePrompts.size)
        cachedPrompts = model.id to split
        return split
    }

    private suspend fun model(context: Context): CompiledModel {
        loaded?.let { return it.model }
        // In the app's external files (adb can put it there on any build), not CutoutModel.dir.
        val found = Int8Files.firstNotNullOfOrNull { (side, scale) ->
            File(context.getExternalFilesDir(null), "models/cutout/cutout_${side}_int8_npu.tflite").takeIf { it.isFile }?.let { Triple(it, side, scale) }
        }
        if (found != null && Npu.chip(context) != null) {
            val (int8, side, scale) = found
            val environment = Npu.environment(context)
            val start = SystemClock.elapsedRealtime()
            val model = runCatching {
                CompiledModel.create(int8.path, CompiledModel.Options(Accelerator.NPU).apply { qualcommOptions = Npu.fast }, environment)
            }.onFailure { Log.w(Tag, "8-bit NPU model didn't load", it); environment.close() }.getOrNull()
            if (model != null) {
                Log.i(Tag, "loaded 8-bit ($side) on the NPU in ${SystemClock.elapsedRealtime() - start} ms")
                AiState(context).setProcessor(ModelPack.Cutout, "NPU")
                loaded = Loaded(environment, model, int8 = true, side = side, inputScale = scale)
                return model
            }
        }
        val path = File(CutoutModel.dir(context), CutoutModel.file.name).path
        val gpu = GpuCheck.useGpu(context, GpuCheck.Target(CutoutModel.file.name, File(path), 3 * CutoutModel.Side * CutoutModel.Side, mask = true))
        val accelerator = if (gpu) Accelerator.GPU else Accelerator.CPU
        val options = CompiledModel.Options(accelerator).apply {
            if (gpu) {
                // The GPU's programs for this model take ~9 s to build; kept on disk after the first.
                gpuOptions = CompiledModel.GpuOptions(
                    precision = CompiledModel.GpuOptions.Precision.FP16,
                    serializationDir = CutoutModel.gpuCache(context).apply { mkdirs() }.path,
                    modelCacheKey = CutoutModel.file.sha256.take(16),
                    serializeProgramCache = true,
                )
            } else {
                cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
            }
        }
        val environment = Environment.create()
        val start = SystemClock.elapsedRealtime()
        val model = CompiledModel.create(path, options, environment)
        Log.i(Tag, "loaded on $accelerator in ${SystemClock.elapsedRealtime() - start} ms")
        AiState(context).setProcessor(ModelPack.Cutout, if (gpu) "GPU" else "CPU")
        loaded = Loaded(environment, model)
        return model
    }

    /** The subject mask (0–1), [CutoutModel.Side] square, for [photo] stretched to that square. */
    private suspend fun mask(context: Context, photo: Bitmap): FloatArray {
        val model = model(context)
        val side = loaded?.side ?: CutoutModel.Side
        val scaled = Bitmap.createScaledBitmap(photo, side, side, true)
        val pixels = IntArray(side * side).also { scaled.getPixels(it, 0, side, 0, 0, side, side) }
        if (scaled !== photo) scaled.recycle()
        val input = FloatArray(3 * side * side)
        for (i in pixels.indices) {
            val p = pixels[i]
            input[3 * i] = (p shr 16 and 0xFF).toFloat()
            input[3 * i + 1] = (p shr 8 and 0xFF).toFloat()
            input[3 * i + 2] = (p and 0xFF).toFloat()
        }
        val int8 = loaded?.int8 == true
        val scale = loaded?.inputScale ?: 1f
        val inputs = model.createInputBuffers(0)
        val outputs = model.createOutputBuffers(0)
        try {
            val start = SystemClock.elapsedRealtime()
            if (int8) {
                inputs[0].writeInt8(int8Pixels(pixels, scale))
                if (inputs.size > 1) {
                    val (small, smallScale) = Small
                    val smaller = Bitmap.createScaledBitmap(photo, small, small, true)
                    val smallPixels = IntArray(small * small).also { smaller.getPixels(it, 0, small, 0, 0, small, small) }
                    if (smaller !== photo) smaller.recycle()
                    inputs[1].writeInt8(int8Pixels(smallPixels, smallScale))
                }
            } else {
                inputs[0].writeFloat(input)
            }
            val written = SystemClock.elapsedRealtime()
            model.run(inputs, outputs, 0)
            val ran = SystemClock.elapsedRealtime()
            val mask = if (int8) outputs[0].readInt8().let { b -> FloatArray(b.size) { (b[it] + 128) / 256f } } else outputs[0].readFloat()
            return mask.also {
                Log.i(Tag, "mask: write ${written - start} ms, run ${ran - written} ms, read ${SystemClock.elapsedRealtime() - ran} ms")
            }
        } finally {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
        }
    }

    /** RGB [pixels] as the 8-bit model's input: value / [scale] - 128. */
    private fun int8Pixels(pixels: IntArray, scale: Float) = ByteArray(3 * pixels.size) {
        val channel = pixels[it / 3] shr (16 - 8 * (it % 3)) and 0xFF
        (Math.round(channel / scale) - 128).coerceIn(-128, 127).toByte()
    }

    /**
     * Zeroes every subject but the one at ([x], [y]) (or the nearest within [Reach]), keeping a
     * few pixels around it for its soft edge. Pressed on the background: all are kept. False when
     * there's no subject at all.
     */
    private fun keepUnder(mask: FloatArray, x: Float, y: Float): Boolean {
        val side = sideOf(mask)
        val scale = side / CutoutModel.Side
        if (mask.none { it > 0.5f }) return false
        val px = (x * side).toInt().coerceIn(0, side - 1)
        val py = (y * side).toInt().coerceIn(0, side - 1)
        var seed = -1
        var best = Int.MAX_VALUE
        val reach = Reach * scale
        for (dy in -reach..reach) for (dx in -reach..reach) {
            val sx = px + dx
            val sy = py + dy
            if (sx !in 0 until side || sy !in 0 until side || mask[sy * side + sx] <= 0.5f) continue
            val d = dx * dx + dy * dy
            if (d < best) { best = d; seed = sy * side + sx }
        }
        if (seed < 0) return true
        // The connected part of the mask (above 0.5) the press is on.
        val keep = BooleanArray(side * side)
        val stack = IntArray(side * side)
        var top = 0
        stack[top++] = seed
        keep[seed] = true
        while (top > 0) {
            val i = stack[--top]
            val ix = i % side
            val iy = i / side
            for (n in intArrayOf(if (ix > 0) i - 1 else -1, if (ix < side - 1) i + 1 else -1, if (iy > 0) i - side else -1, if (iy < side - 1) i + side else -1)) {
                if (n >= 0 && !keep[n] && mask[n] > 0.5f) {
                    keep[n] = true
                    stack[top++] = n
                }
            }
        }
        // Grown by a few pixels, so the soft edge (hair, below 0.5) stays.
        val grown = grow(keep, side, 4 * scale)
        for (i in mask.indices) if (!grown[i]) mask[i] = 0f
        return true
    }

    private fun grow(set: BooleanArray, side: Int, by: Int): BooleanArray {
        var current = set
        repeat(by) {
            val next = current.copyOf()
            for (i in current.indices) {
                if (!current[i]) continue
                val ix = i % side
                if (ix > 0) next[i - 1] = true
                if (ix < side - 1) next[i + 1] = true
                if (i >= side) next[i - side] = true
                if (i < side * (side - 1)) next[i + side] = true
            }
            current = next
        }
        return current
    }

    /** [photo] with [mask] (stretched back over it) as its alpha, cropped to the subject. */
    private fun compose(photo: Bitmap, mask: FloatArray): Result? {
        val side = sideOf(mask)
        var left = side
        var top = side
        var right = -1
        var bottom = -1
        for (my in 0 until side) for (mx in 0 until side) {
            if (mask[my * side + mx] <= 0.02f) continue
            if (mx < left) left = mx
            if (mx > right) right = mx
            if (my < top) top = my
            if (my > bottom) bottom = my
        }
        if (right < 0) return null
        val w = photo.width
        val h = photo.height
        val x0 = (left * w / side - 2).coerceAtLeast(0)
        val y0 = (top * h / side - 2).coerceAtLeast(0)
        val x1 = ((right + 1) * w / side + 2).coerceAtMost(w)
        val y1 = ((bottom + 1) * h / side + 2).coerceAtMost(h)
        val cw = x1 - x0
        val ch = y1 - y0
        val pixels = IntArray(cw * ch).also { photo.getPixels(it, 0, cw, x0, y0, cw, ch) }
        val sx = side.toFloat() / w
        val sy = side.toFloat() / h
        for (y in 0 until ch) {
            // Bilinear: the mask is 512 wide, the photo up to 2048, and a blocky edge shows.
            val fy = ((y0 + y + 0.5f) * sy - 0.5f).coerceIn(0f, side - 1f)
            val my0 = fy.toInt()
            val my1 = minOf(my0 + 1, side - 1)
            val ty = fy - my0
            for (x in 0 until cw) {
                val fx = ((x0 + x + 0.5f) * sx - 0.5f).coerceIn(0f, side - 1f)
                val mx0 = fx.toInt()
                val mx1 = minOf(mx0 + 1, side - 1)
                val tx = fx - mx0
                val a = (mask[my0 * side + mx0] * (1 - tx) + mask[my0 * side + mx1] * tx) * (1 - ty) +
                    (mask[my1 * side + mx0] * (1 - tx) + mask[my1 * side + mx1] * tx) * ty
                val i = y * cw + x
                pixels[i] = ((a.coerceIn(0f, 1f) * 255 + 0.5f).toInt() shl 24) or (pixels[i] and 0xFFFFFF)
            }
        }
        val bitmap = Bitmap.createBitmap(pixels, cw, ch, Bitmap.Config.ARGB_8888)
        return Result(bitmap, RectF(x0.toFloat() / w, y0.toFloat() / h, x1.toFloat() / w, y1.toFloat() / h))
    }
}
