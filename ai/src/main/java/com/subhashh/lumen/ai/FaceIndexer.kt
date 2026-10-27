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

    /** Looks at most [limit] photos and videos. */
    suspend fun run(limit: Int = Int.MAX_VALUE): Indexer.Outcome = coroutineScope {
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
        val store = FaceStore.of(context)
        ModelRepair.loading(context) { FaceReader(context, dir, Threads, recognizer) }.use { reader ->
            if (!moveOver(store, reader, limit)) return@coroutineScope Indexer.Outcome.Partial
        }
        val library = Indexer.eligible(context)
        val known = store.indexed()
        store.remove(known.keys - library.mapTo(HashSet()) { it.id })
        val todo = library.filter { known[it.id] != it.modified }.sortedByDescending { it.takenAt }
        val (videos, photos) = todo.partition { it.isVideo }
        var finished = library.size - todo.size
        SmartIndex.reportFaces(context, finished, library.size)
        if (todo.isEmpty()) {
            FaceGrouping.run(store, recognizer)
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
                if (finished % GroupEvery == 0) FaceGrouping.run(store, recognizer)
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
                            if (finished % GroupEvery == 0) FaceGrouping.run(store, recognizer)
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
        FaceGrouping.run(store, recognizer)
        return outcome
    }

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

/**
 * Groups faces into people, Immich-style and incremental: a face that has no person joins the
 * person of its most similar face that has one, if that's at least [FaceRecognizer.samePerson]
 * alike. Else, if [MinFaces] − 1 other unassigned faces are that alike, they start a new person
 * together; or if one waiting face in another photo or video is [FaceRecognizer.strongPair]
 * alike, the two do. Otherwise it waits for more faces of whoever it is. Faces with a person never
 * move: that's the user's to change (merge, split).
 *
 * One face at a time splits people whose faces vary a lot (actors across clips, lighting, makeup):
 * the first few faces start a person, faces from elsewhere start another, and nothing joined them.
 * So afterwards, people whose average faces are [FaceRecognizer.sameAverage] alike are joined
 * ([joinSplit]). Each recognizer has its own cutoffs ([FaceModel]); only faces it made are grouped.
 */
object FaceGrouping {
    private const val Tag = "LumenFaces"
    private const val MinFaces = 3

    @Synchronized
    fun run(store: FaceStore, recognizer: FaceRecognizer) {
        val samePerson = recognizer.samePerson
        val strongPair = recognizer.strongPair
        val started = SystemClock.elapsedRealtime()
        val faces = store.faces(recognizer.id)
        // Faces seen too unclearly to tell who they are stay out: blurry and dark ones otherwise
        // join whoever has the most faces (on the owner's library, a tennis player gathered
        // crowds, a visor and a man in a green cap).
        val clear = BooleanArray(faces.size) { (faces[it].quality ?: Float.MAX_VALUE) >= recognizer.minQuality }
        val person = arrayOfNulls<Long>(faces.size)
        faces.forEachIndexed { i, f -> person[i] = f.person }
        val joins = HashMap<Long, Long>()
        val newPeople = ArrayList<List<Long>>()
        // Placeholder ids for people made in this run (negative), real ones after saving.
        var nextNew = -1L
        val newMembers = HashMap<Long, MutableList<Long>>()
        // Rows go in newest photo first: so newest faces first, then a second try for the waiting.
        val order = faces.indices
        repeat(2) {
            for (i in order) {
                if (person[i] != null || !clear[i]) continue
                var best = -1
                var bestSim = samePerson
                var pair = -1
                var pairSim = strongPair
                val close = ArrayList<Int>()
                val v = faces[i].vector
                for (j in faces.indices) {
                    if (j == i || !clear[j]) continue
                    val s = dot(v, faces[j].vector)
                    if (s < samePerson) continue
                    close += j
                    if (person[j] == null && faces[j].mediaId != faces[i].mediaId && s >= pairSim) {
                        pairSim = s
                        pair = j
                    }
                    // Never back to a person the user took this face from.
                    if (person[j] != null && person[j] != faces[i].notPerson && s >= bestSim) {
                        bestSim = s
                        best = j
                    }
                }
                if (best >= 0) {
                    val p = person[best]!!
                    person[i] = p
                    if (p < 0) newMembers.getValue(p) += faces[i].id else joins[faces[i].id] = p
                } else if (close.size >= MinFaces - 1) {
                    val p = nextNew--
                    val members = mutableListOf(faces[i].id)
                    person[i] = p
                    for (j in close) if (person[j] == null) {
                        person[j] = p
                        members += faces[j].id
                    }
                    newMembers[p] = members
                } else if (pair >= 0) {
                    val p = nextNew--
                    person[i] = p
                    person[pair] = p
                    newMembers[p] = mutableListOf(faces[i].id, faces[pair].id)
                }
            }
        }
        newPeople += newMembers.values
        store.assign(joins, newPeople)
        joinSplit(store, recognizer)
        if (joins.isNotEmpty() || newPeople.isNotEmpty()) {
            Log.i(Tag, "Grouped ${faces.size} faces: ${joins.size} joined people, ${newPeople.size} new people, " +
                "${person.count { it == null }} waiting, ${SystemClock.elapsedRealtime() - started} ms")
        }
    }

    /**
     * Joins people whose average faces are [FaceRecognizer.sameAverage] alike, most alike first. Never two
     * people with different names, never a hidden person, and never a person into one the user
     * took any of its faces from (Not This Person). A named person keeps their name.
     */
    private fun joinSplit(store: FaceStore, recognizer: FaceRecognizer) {
        val labels = store.labels().toMutableMap()
        val sums = HashMap<Long, FloatArray>()
        val counts = HashMap<Long, Int>()
        val refused = HashMap<Long, MutableSet<Long>>()
        for (f in store.faces(recognizer.id)) {
            val p = f.person ?: continue
            f.notPerson?.let { refused.getOrPut(p) { HashSet() } += it }
            // Averages of clear faces only: blurry ones pull every average toward the same blur.
            if ((f.quality ?: Float.MAX_VALUE) < recognizer.minQuality) continue
            val sum = sums.getOrPut(p) { FloatArray(f.vector.size) }
            for (k in sum.indices) sum[k] += f.vector[k]
            counts[p] = (counts[p] ?: 0) + 1
        }
        fun average(p: Long): FloatArray {
            val sum = sums.getValue(p)
            val norm = kotlin.math.sqrt(sum.fold(0.0) { acc, x -> acc + x * x }).toFloat().coerceAtLeast(1e-6f)
            return FloatArray(sum.size) { sum[it] / norm }
        }
        fun allowed(a: Long, b: Long): Boolean {
            val (nameA, hiddenA) = labels[a] ?: (null to false)
            val (nameB, hiddenB) = labels[b] ?: (null to false)
            if (hiddenA || hiddenB) return false
            if (nameA != null && nameB != null && nameA != nameB) return false
            return refused[a]?.contains(b) != true && refused[b]?.contains(a) != true
        }
        // Every pair's likeness once; after a join only the kept person's row changes.
        val ids = sums.keys.toLongArray()
        val n = ids.size
        val averages = Array(n) { average(ids[it]) }
        val alive = BooleanArray(n) { true }
        val sim = FloatArray(n * n)
        for (i in 0 until n) for (j in i + 1 until n) sim[i * n + j] = dot(averages[i], averages[j])
        var joined = 0
        while (true) {
            var best = recognizer.sameAverage
            var bi = -1
            var bj = -1
            for (i in 0 until n) {
                if (!alive[i]) continue
                for (j in i + 1 until n) {
                    if (alive[j] && sim[i * n + j] >= best && allowed(ids[i], ids[j])) {
                        best = sim[i * n + j]
                        bi = i
                        bj = j
                    }
                }
            }
            if (bi < 0) break
            val a = ids[bi]
            val b = ids[bj]
            // The named one keeps its id (and name); else the one with more faces.
            val aFirst = labels[a]?.first != null || (labels[b]?.first == null && counts.getValue(a) >= counts.getValue(b))
            val keep = if (aFirst) a else b
            val gone = if (aFirst) b else a
            val ki = if (aFirst) bi else bj
            store.merge(keep, gone)
            val sum = sums.getValue(keep)
            val other = sums.remove(gone)!!
            for (k in sum.indices) sum[k] += other[k]
            counts[keep] = counts.getValue(keep) + (counts.remove(gone) ?: 0)
            refused.remove(gone)?.let { refused.getOrPut(keep) { HashSet() } += it }
            for (set in refused.values) if (set.remove(gone)) set += keep
            if (labels[keep]?.first == null) labels[gone]?.first?.let { labels[keep] = it to false }
            labels.remove(gone)
            alive[if (aFirst) bj else bi] = false
            averages[ki] = average(keep)
            for (o in 0 until n) if (alive[o] && o != ki) {
                val s = dot(averages[ki], averages[o])
                if (o < ki) sim[o * n + ki] = s else sim[ki * n + o] = s
            }
            joined++
        }
        if (joined > 0) Log.i(Tag, "Joined $joined people found twice")
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (k in a.indices) s += a[k] * b[k]
        return s
    }
}
