package com.subhashh.lumen.ai

import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Every vector of an index, packed for a fast scan and kept out of the Java heap.
 *
 * A large library has tens of thousands of vectors (a photo has one, a video one per kept frame).
 * As plain floats on the heap, 40,000 SigLIP vectors took 120 MB of an app's 256 MB, and Lumen
 * ran out of memory as soon as anything else (a video, a big photo) needed some. So each vector is
 * stored as 8-bit codes with one scale per row (a quarter of the size, and ranking is unaffected:
 * the error is far below the gaps between results), in shared memory mapped into the process
 * ([offHeap]), which doesn't count against the heap limit. (A "direct" ByteBuffer would: on
 * Android it is a Java byte array that the GC doesn't move.)
 *
 * Rows only ever get added: [append] while there is room, and [remove] marks rows dead
 * ([mediaId] = -1) instead of moving anything. When it fills up, or too much of it is dead, the
 * owner loads a fresh copy instead. Readers may scan while rows are added: they see the first
 * [size] rows, which never change once published. Only one thread may add or remove at a time.
 *
 * A media item's live rows are always next to each other: a full load is ordered by item, an
 * update removes an item's old rows before appending its new ones together.
 */
class Embeddings internal constructor(val dim: Int, val capacity: Int, buffer: (Int) -> ByteBuffer = ::offHeap) {
    private val codes: ByteBuffer = buffer(capacity * dim)
    private val scales = FloatArray(capacity)
    private val ids = LongArray(capacity)
    private val times = LongArray(capacity)

    /** Rows written so far, dead ones included. */
    @Volatile
    var size = 0
        private set

    /** How many of the first [size] rows are dead. */
    var dead = 0
        private set

    /** The item row [i] belongs to, or [Dead] once it was removed. */
    fun mediaId(i: Int): Long = ids[i]

    /** Where in a video row [i]'s frame is (0 for a photo). */
    fun atMs(i: Int): Long = times[i]

    /** Adds one vector; false when there's no room (load a fresh, bigger copy then). */
    internal fun append(mediaId: Long, atMs: Long, vector: FloatArray): Boolean {
        require(vector.size == dim) { "vector has ${vector.size} values, index has $dim" }
        val i = size
        if (i == capacity) return false
        var max = 0f
        for (x in vector) max = maxOf(max, abs(x))
        val scale = if (max > 0f) max / 127f else 0f
        val base = i * dim
        for (d in 0 until dim) {
            val code = if (scale > 0f) (vector[d] / scale).roundToInt().coerceIn(-127, 127) else 0
            codes.put(base + d, code.toByte())
        }
        scales[i] = scale
        ids[i] = mediaId
        times[i] = atMs
        size = i + 1 // published last: a reader that sees the new size sees the whole row
        return true
    }

    /** Marks every row of these items dead. */
    internal fun remove(mediaIds: Set<Long>) {
        if (mediaIds.isEmpty()) return
        for (i in 0 until size) {
            if (ids[i] != Dead && ids[i] in mediaIds) {
                ids[i] = Dead
                dead++
            }
        }
    }

    /**
     * Each of the first [rows] rows' dot product with [q], into [out]. Dead rows get a score
     * too; skip them by [mediaId]. Safe to call from several threads at once.
     */
    fun scores(q: FloatArray, out: FloatArray, rows: Int = size) {
        require(q.size == dim) { "query has ${q.size} values, index has $dim" }
        val view = codes.duplicate() // its own position, so scans don't disturb each other
        view.position(0)
        val row = ByteArray(dim)
        for (i in 0 until rows) {
            view.get(row, 0, dim)
            var dot = 0f
            for (d in 0 until dim) dot += q[d] * row[d]
            out[i] = dot * scales[i]
        }
    }

    companion object {
        const val Dead = -1L

        /**
         * [bytes] of anonymous shared memory, mapped read-write. Unmapped when the buffer is
         * garbage-collected; the descriptor isn't needed after mapping, so it's closed now.
         * Falls back to a direct buffer where shared memory isn't available (plain JVM tests).
         */
        internal fun offHeap(bytes: Int): ByteBuffer = try {
            val memory = android.os.SharedMemory.create("lumen-search", maxOf(bytes, 1))
            try { memory.mapReadWrite() } finally { memory.close() }
        } catch (e: Throwable) {
            ByteBuffer.allocateDirect(bytes)
        }

        /** Room for [rows] now and a good while of indexing after, so updates rarely need a reload. */
        internal fun withRoomFor(rows: Int, dim: Int) = Embeddings(dim, rows + rows / 4 + 256)
    }
}

/**
 * The items each recent write touched, numbered 1, 2, 3… (the index's version after it), so a
 * copy in memory can catch up on just those. Remembers the last [keep] writes; anything older
 * means "read everything again". Not thread-safe: the owner locks.
 */
internal class ChangeLog(private val keep: Int) {
    private val entries = ArrayDeque<Pair<Int, LongArray>>()

    /** The newest version: 0 before the first write. */
    var version = 0
        private set

    /** Records a write and returns its version. */
    fun add(mediaIds: LongArray): Int {
        version++
        entries.addLast(version to mediaIds)
        while (entries.size > keep) entries.removeFirst()
        return version
    }

    /** Every item touched after version [since]; null when the log doesn't reach back that far. */
    fun since(since: Int): Set<Long>? {
        if (since == version) return emptySet()
        if (since < 0 || since > version) return null
        val oldest = entries.firstOrNull()?.first ?: return null
        if (oldest > since + 1) return null
        val out = HashSet<Long>()
        for ((v, ids) in entries) if (v > since) ids.forEach { out += it }
        return out
    }
}
