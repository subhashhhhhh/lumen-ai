package com.subhashh.lumen.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.ensureActive
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * One faces pass: brings [FaceStore] in line with the library, then groups new faces into people
 * ([FaceGrouping]). Photos first, newest first; then videos, read like Smart Search reads them
 * (one frame a second up to two minutes, else 120 spread out, near-identical frames skipped) at
 * the Video Detail size, decoded a few frames ahead on another thread. A video keeps one face per
 * person in it, the biggest, with its moment and a saved crop; otherwise one reel would outweigh
 * a whole album. The first pass of the background job: it's the fastest.
 *
 * Decoding is most of a video's cost, so while a video is open its frames are also saved small
 * ([FrameCache]) for Smart Search ([model]), if that video still needs its vectors.
 */
@LumenAiEngine
class FaceIndexer(private val context: Context, private val model: AiModel?) {

    private sealed interface Queued {
        class Start(val item: MediaRef) : Queued
        class Frame(val atMs: Long, val bitmap: Bitmap) : Queued
        class End(val item: MediaRef, val started: Long, val frames: Int) : Queued
    }

    /** Looks at most [limit] photos and videos, only those in [only] when given (items that just arrived). */
    suspend fun run(limit: Int = Int.MAX_VALUE, only: Set<Long>? = null): Indexer.Outcome = coroutineScope {
        if (!FaceModel.ready(context)) return@coroutineScope Indexer.Outcome.Done
        val recognizer = FaceModel.recognizer(context)
        val dir = FaceModel.dir(context)
        // Accurate runs on the NPU only after matching its reference; a failure moves the phone to
        // Fast, whose file comes on the next start (Models.fetchNpuFiles).
        if (recognizer.npu) {
            val compiled = File(dir, recognizer.file.name)
            val reference = recognizer.reference?.let { File(dir, it.name) }
            if (!GpuCheck.useGpu(context, GpuCheck.Target(recognizer.id, compiled, FaceInputs, npu = compiled, expected = reference))) {
                return@coroutineScope Indexer.Outcome.Done
            }
        }
        AiState(context).setProcessor(ModelPack.Faces, if (recognizer.npu) "NPU" else "CPU")
        val store = FaceStore.of(context)
        // The model loads only when there are faces to move: loading it onto the NPU takes a while,
        // and most runs (the charger going in) find nothing to do.
        val moving = store.stale(recognizer.id).isNotEmpty()
        // Moving faces to another recognizer is backlog work: it follows the charger and the schedule.
        if (moving && only != null) return@coroutineScope Indexer.Outcome.Partial
        if (moving) {
            // Moved faces lose their people and aren't new: the next grouping has to be a full one.
            groups.edit().remove("${recognizer.id}:through").apply()
            ModelRepair.loading(context) { FaceReader(context, dir, Threads, recognizer) }.use { reader ->
                if (!moveOver(store, reader, limit)) return@coroutineScope Indexer.Outcome.Partial
            }
        } else {
            FaceModel.removeUnused(context)
        }
        val library = Indexer.eligible(context)
        val known = store.indexed()
        store.remove(known.keys - library.mapTo(HashSet()) { it.id })
        val todo = library.filter { known[it.id] != it.modified && (only == null || it.id in only) }.sortedByDescending { it.takenAt }
        val (videos, photos) = todo.partition { it.isVideo }
        var finished = library.size - todo.size
        SmartIndex.reportFaces(context, finished, library.size)
        if (todo.isEmpty()) {
            // Grouping takes a minute or more on a big library: only when faces came or went.
            if (moving || store.signature(recognizer.id) != groupedAt(recognizer.id)) group(store, recognizer)
            return@coroutineScope Indexer.Outcome.Done
        }
        val power = context.getSystemService(PowerManager::class.java)
        fun tooHot() = power.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE
        var budget = limit
        val streak = Indexer.FailureStreak()
        ModelRepair.loading(context) { FaceReader(context, dir, Threads, recognizer) }.use { reader ->
            for (item in photos) {
                if (budget-- <= 0) return@coroutineScope done(store, recognizer, Indexer.Outcome.Partial)
                ensureActive()
                if (tooHot()) return@coroutineScope done(store, recognizer, Indexer.Outcome.TooHot)
                val started = SystemClock.elapsedRealtime()
                val bitmap = Indexer.readOrNull(item) { MediaFrames.forReading(context, item, FaceReader.ReadSide) }
                val faces = if (bitmap == null) {
                    emptyList()
                } else {
                    try { Indexer.modelOrNull(item, streak) { reader.read(bitmap) } } finally { bitmap.recycle() }
                        ?: if (streak.tooMany) return@coroutineScope done(store, recognizer, Indexer.Outcome.Failing) else continue
                }
                store.put(item.id, item.modified, faces.map { NewFace(it.box, it.vector, it.quality) }, recognizer.id)
                SmartIndex.reportFaces(context, ++finished, library.size)
                Log.d(Tag, "${item.name.take(40)}: ${faces.size} faces, ${SystemClock.elapsedRealtime() - started} ms")
                if (finished % GroupEvery == 0) group(store, recognizer)
            }

            val batch = videos.take(budget.coerceAtLeast(0))
            if (batch.isEmpty()) return@coroutineScope done(store, recognizer, if (videos.isEmpty()) Indexer.Outcome.Done else Indexer.Outcome.Partial)
            val px = LumenAi.host.settings.videoDetailPx
            // Videos Smart Search hasn't done yet get their frames kept for it.
            val vectorsDone = model?.let { EmbeddingStore.of(context, it).indexed() }
            val queue = Channel<Queued>(capacity = QueueFrames)
            val decoder = launch(Dispatchers.IO) {
                try {
                    for (item in batch) {
                        val started = SystemClock.elapsedRealtime()
                        var frames = 0
                        queue.send(Queued.Start(item))
                        try {
                            MediaFrames.forEach(context, item, px) { at, frame ->
                                frames++
                                queue.send(Queued.Frame(at, frame))
                                true
                            }
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            Log.w(Tag, "Couldn't read ${item.name}", e)
                        }
                        queue.send(Queued.End(item, started, frames))
                    }
                } finally {
                    queue.close()
                }
            }
            try {
                var kept = ArrayList<Kept>()
                var cache: java.io.File? = null
                for (message in queue) {
                    when (message) {
                        is Queued.Start -> {
                            val item = message.item
                            // Keep its frames for Smart Search if that hasn't done this video yet.
                            cache = if (vectorsDone != null && vectorsDone[item.id] != item.modified) {
                                FrameCache.begin(context, item.id, item.modified)
                            } else null
                        }
                        is Queued.Frame -> {
                            if (tooHot()) {
                                message.bitmap.recycle()
                                return@coroutineScope done(store, recognizer, Indexer.Outcome.TooHot)
                            }
                            cache?.let { FrameCache.save(it, message.atMs, message.bitmap) }
                            for (face in reader.read(message.bitmap)) keep(kept, face, message.atMs, message.bitmap, recognizer.samePerson)
                            message.bitmap.recycle()
                        }
                        is Queued.End -> {
                            val item = message.item
                            cache?.let(FrameCache::finish)
                            cache = null
                            val ids = store.put(item.id, item.modified, kept.map { NewFace(it.face.box, it.face.vector, it.face.quality, it.atMs) }, recognizer.id)
                            for ((id, k) in ids.zip(kept)) {
                                runCatching { store.crop(id).outputStream().use { k.crop.compress(Bitmap.CompressFormat.JPEG, 88, it) } }
                                k.crop.recycle()
                            }
                            SmartIndex.reportFaces(context, ++finished, library.size)
                            Log.d(Tag, "${item.name.take(40)}: ${item.durationMs / 1000}s video, ${message.frames} frames, " +
                                "${kept.size} people, ${SystemClock.elapsedRealtime() - message.started} ms")
                            kept = ArrayList()
                            if (finished % GroupEvery == 0) group(store, recognizer)
                        }
                    }
                }
            } finally {
                decoder.cancel()
            }
            done(store, recognizer, if (batch.size < videos.size) Indexer.Outcome.Partial else Indexer.Outcome.Done)
        }
    }

    private fun done(store: FaceStore, recognizer: FaceRecognizer, outcome: Indexer.Outcome): Indexer.Outcome {
        group(store, recognizer)
        return outcome
    }

    /**
     * Groups faces into people: just the faces new since the last grouping (seconds), or all of
     * them (minutes on a big library) the first time, after faces moved to another recognizer,
     * and once every [FullEveryMs]. Notes which faces that covered ([groupedAt]).
     */
    private fun group(store: FaceStore, recognizer: FaceRecognizer) {
        val id = recognizer.id
        val through = groups.getLong("$id:through", -1L)
        val now = System.currentTimeMillis()
        val full = through < 0 || now - groups.getLong("$id:full", 0L) > FullEveryMs
        val newest = FaceGrouping.run(store, recognizer, since = if (full) null else through)
        groups.edit()
            .putString(id, store.signature(id))
            .putLong("$id:through", newest)
            .apply { if (full) putLong("$id:full", now) }
            .apply()
    }

    private val groups get() = context.getSharedPreferences("lumen-ai-face-groups", Context.MODE_PRIVATE)

    /** [FaceStore.signature] when faces were last grouped with this model. */
    private fun groupedAt(model: String): String? = groups.getString(model, null)

    /**
     * Moves faces another recognizer read (SFace before 0.3, or the other tier) to [reader]'s: each
     * is found again, in its saved crop (videos) or its photo, and read anew; faces not found go.
     * Once all have moved, people the user never named or hid are let go, to be regrouped with the
     * new numbers (until then People shows them as they were). At most [limit] photos and videos a
     * pass; false while some are left.
     */
    private suspend fun moveOver(store: FaceStore, reader: FaceReader, limit: Int): Boolean {
        val model = reader.recognizer.id
        val stale = store.stale(model)
        if (stale.isEmpty()) {
            FaceModel.removeUnused(context)
            return true
        }
        val started = SystemClock.elapsedRealtime()
        val items = Indexer.eligible(context).associateBy { it.id }
        val byMedia = stale.groupBy { it.mediaId }
        var moved = 0
        var gone = 0
        for ((n, entry) in byMedia.entries.withIndex()) {
            if (n >= limit) {
                Log.i(Tag, "Moved $moved faces to $model ($gone not found again), ${byMedia.size - n} photos and videos left")
                return false
            }
            coroutineContext.ensureActive()
            val (mediaId, faces) = entry
            val found = HashMap<Long, FaceReader.Found>()
            val lost = ArrayList<Long>()
            fun take(face: FaceStore.Stale, read: List<FaceReader.Found>, box: FloatArray) {
                val match = read.maxByOrNull { iou(it.box, box) }?.takeIf { iou(it.box, box) >= 0.3f }
                if (match != null) found[face.id] = match else lost += face.id
            }
            val inPhoto = faces.filter { it.atMs == null }
            if (inPhoto.isNotEmpty()) {
                val item = items[mediaId]
                val bitmap = item?.let { Indexer.readOrNull(it) { MediaFrames.forReading(context, it, FaceReader.ReadSide) } }
                val read = bitmap?.let { b -> try { runCatching { reader.read(b, FindAgainScore) }.getOrNull() } finally { b.recycle() } }.orEmpty()
                for (face in inPhoto) take(face, read, face.box)
            }
            for (face in faces.filter { it.atMs != null }) {
                val crop = BitmapFactory.decodeFile(store.crop(face.id).path)
                val read = crop?.let { b -> try { runCatching { reader.read(b, FindAgainScore) }.getOrNull() } finally { b.recycle() } }.orEmpty()
                take(face, read, CropFace)
            }
            store.revector(found, lost, model)
            moved += found.size
            gone += lost.size
        }
        Log.i(Tag, "Moved $moved faces to $model ($gone not found again) in ${SystemClock.elapsedRealtime() - started} ms")
        store.releaseUnnamed()
        FaceModel.removeUnused(context)
        return true
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val w = maxOf(0f, minOf(a[2], b[2]) - maxOf(a[0], b[0]))
        val h = maxOf(0f, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
        val inter = w * h
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9f)
    }

    private class Kept(var face: FaceReader.Found, var atMs: Long, var crop: Bitmap, var area: Float)

    /** A face joins whoever it matches among this video's faces (keeping the bigger one), or is someone new. */
    private fun keep(kept: MutableList<Kept>, face: FaceReader.Found, atMs: Long, frame: Bitmap, samePerson: Float) {
        val area = (face.box[2] - face.box[0]) * (face.box[3] - face.box[1])
        val same = kept.maxByOrNull { dot(it.face.vector, face.vector) }?.takeIf { dot(it.face.vector, face.vector) >= samePerson }
        when {
            same == null -> kept += Kept(face, atMs, crop(frame, face.box), area)
            area > same.area -> {
                same.crop.recycle()
                same.face = face
                same.atMs = atMs
                same.crop = crop(frame, face.box)
                same.area = area
            }
        }
    }

    /** The face with hair and chin, square, [CropPx] across: what People shows for a video face. */
    private fun crop(frame: Bitmap, box: FloatArray): Bitmap {
        val cx = (box[0] + box[2]) / 2 * frame.width
        val cy = (box[1] + box[3]) / 2 * frame.height
        val side = (maxOf((box[2] - box[0]) * frame.width, (box[3] - box[1]) * frame.height) * 1.7f)
            .coerceAtMost(minOf(frame.width, frame.height).toFloat())
        val s = side.toInt().coerceAtLeast(1)
        val left = (cx - side / 2).toInt().coerceIn(0, frame.width - s)
        val top = (cy - side / 2).toInt().coerceIn(0, frame.height - s)
        val square = Bitmap.createBitmap(frame, left, top, s, s)
        return Bitmap.createScaledBitmap(square, CropPx, CropPx, true).also { if (it !== square) square.recycle() }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }

    private companion object {
        const val Tag = "LumenFaces"
        val Threads = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(1, 4)
        const val GroupEvery = 200
        /** A full grouping (every waiting face against every face) at most this often. */
        const val FullEveryMs = 7 * 24 * 60 * 60_000L
        const val CropPx = 256
        /** A recognizer's test input for [GpuCheck]: one 112×112 RGB face. */
        const val FaceInputs = 112 * 112 * 3
        /** Finding a face again in its crop or photo: looser than finding new ones (0.8). */
        const val FindAgainScore = 0.5f
        /** Where the face sits in its saved crop, a square 1.7× the face around its centre. */
        val CropFace = floatArrayOf(0.5f - 0.5f / 1.7f, 0.5f - 0.5f / 1.7f, 0.5f + 0.5f / 1.7f, 0.5f + 0.5f / 1.7f)
        /** How far decoding may run ahead of the face models: a few frames, ~4 MB each at 1280. */
        const val QueueFrames = 6
    }
}
