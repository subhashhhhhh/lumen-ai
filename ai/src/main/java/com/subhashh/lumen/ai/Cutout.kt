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
 * where it passed [GpuCheck] (in 16-bit floats), else the CPU. Like [LiveText], the model loads as
 * the viewer opens ([warm]) and stays a little after it closes ([release]).
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

    private class Loaded(val environment: Environment, val model: CompiledModel) : AutoCloseable {
        override fun close() {
            model.close()
            environment.close()
        }
    }

    private val lock = Mutex()
    private var loaded: Loaded? = null
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
            val mask = lock.withLock { runCatching { mask(context, photo) }.onFailure { Log.w(Tag, "cutout failed", it) }.getOrNull() }
                ?: return@withContext null
            if (!keepUnder(mask, x, y)) return@withContext null
            compose(photo, mask)
        } finally {
            photo.recycle()
        }
    }

    private suspend fun model(context: Context): CompiledModel {
        loaded?.let { return it.model }
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
        val side = CutoutModel.Side
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
        val model = model(context)
        val inputs = model.createInputBuffers(0)
        val outputs = model.createOutputBuffers(0)
        try {
            val start = SystemClock.elapsedRealtime()
            inputs[0].writeFloat(input)
            val written = SystemClock.elapsedRealtime()
            model.run(inputs, outputs, 0)
            val ran = SystemClock.elapsedRealtime()
            return outputs[0].readFloat().also {
                Log.i(Tag, "mask: write ${written - start} ms, run ${ran - written} ms, read ${SystemClock.elapsedRealtime() - ran} ms")
            }
        } finally {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
        }
    }

    /**
     * Zeroes every subject but the one at ([x], [y]) (or the nearest within [Reach]), keeping a
     * few pixels around it for its soft edge. Pressed on the background: all are kept. False when
     * there's no subject at all.
     */
    private fun keepUnder(mask: FloatArray, x: Float, y: Float): Boolean {
        val side = CutoutModel.Side
        if (mask.none { it > 0.5f }) return false
        val px = (x * side).toInt().coerceIn(0, side - 1)
        val py = (y * side).toInt().coerceIn(0, side - 1)
        var seed = -1
        var best = Int.MAX_VALUE
        for (dy in -Reach..Reach) for (dx in -Reach..Reach) {
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
        val grown = grow(keep, side, 4)
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
        val side = CutoutModel.Side
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
