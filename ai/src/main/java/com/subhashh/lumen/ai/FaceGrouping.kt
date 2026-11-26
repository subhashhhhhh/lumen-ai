package com.subhashh.lumen.ai

import android.os.SystemClock
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.sqrt

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
 * ([joinPeople]). Each recognizer has its own cutoffs ([FaceModel]); only faces it made are grouped.
 *
 * **Quick runs** ([run] with `since`): only faces added after that id, and waiting faces alike to
 * them, are tried; joining and releasing look only at the people that changed. A full run tried
 * every waiting face against every face (5,895 × 11,277, twice: 104 s on the owner's phone), so a
 * newly saved photo's face waited minutes to join its person. Nothing else can change a waiting
 * face's outcome, so quick runs miss little; a full run still comes now and then ([FaceIndexer]).
 *
 * **Memory**: the faces' numbers are read once into one block outside the Java heap ([FaceRows]),
 * and joining keeps each person's best match rather than every pair: the pair table took
 * people² floats (5,000 people: 100 MB, 8,000: the app's whole 256 MB), and big libraries ran out
 * of memory in the faces pass.
 */
object FaceGrouping {
    private const val Tag = "LumenFaces"
    internal const val MinFaces = 3
    /** Smaller people are left alone by [releaseChains]: too few faces to judge. */
    private const val ChainMinFaces = 8

    /**
     * Groups [recognizer]'s faces: all of them, or with [since] (a face id) only what's new after it.
     * Returns the newest face id it saw, the next quick run's [since].
     */
    @Synchronized
    fun run(store: FaceStore, recognizer: FaceRecognizer, since: Long? = null): Long {
        val started = SystemClock.elapsedRealtime()
        val set = store.faceSet(recognizer.id, recognizer.minQuality) ?: return since ?: 0L
        val n = set.size
        val newest = set.ids.maxOrNull() ?: 0L
        val wake = since?.let { s -> BooleanArray(n) { set.ids[it] > s } }
        if (wake != null && wake.none { it }) return newest
        val rules = Rules(recognizer.samePerson, recognizer.strongPair, recognizer.sameCentre)
        val result = assignFaces(set, rules, wake)
        val ids = store.assign(result.joins, result.newPeople)
        // Placeholder ids (negative) to the ones the store gave the new people.
        for (i in 0 until n) if (set.person[i] < 0) set.person[i] = ids[(-set.person[i] - 1).toInt()]
        val touched: Set<Long>? = if (wake == null) null else HashSet<Long>().apply {
            addAll(result.joins.values)
            addAll(ids)
        }
        if (touched?.isEmpty() != true) {
            val merged = joinSplit(store, recognizer, set, touched)
            releaseChains(store, recognizer, set, touched?.mapTo(HashSet()) { merged[it] ?: it })
        }
        if (result.joins.isNotEmpty() || result.newPeople.isNotEmpty()) {
            Log.i(Tag, "Grouped ${if (wake == null) "all" else "new"} of $n faces: ${result.joins.size} joined people, " +
                "${result.newPeople.size} new people, ${set.person.count { it == 0L }} waiting, ${SystemClock.elapsedRealtime() - started} ms")
        }
        return newest
    }

    /**
     * Joins people whose average faces are [FaceRecognizer.sameAverage] alike ([joinPeople]): of
     * [touched] with anyone, or everyone when null. Returns which person each joined one went into.
     */
    private fun joinSplit(store: FaceStore, recognizer: FaceRecognizer, set: FaceSet, touched: Set<Long>?): Map<Long, Long> {
        val labels = store.labels().toMutableMap()
        val dim = set.rows.dim
        val index = HashMap<Long, Int>()
        val people = ArrayList<Long>()
        val sums = ArrayList<FloatArray>()
        val counts = ArrayList<Int>()
        val refused = HashMap<Long, MutableSet<Long>>()
        val v = FloatArray(dim)
        for (i in 0 until set.size) {
            val p = set.person[i]
            if (p == 0L) continue
            if (set.notPerson[i] != 0L) refused.getOrPut(p) { HashSet() } += set.notPerson[i]
            // Averages of clear faces only: blurry ones pull every average toward the same blur.
            if (!set.clear[i]) continue
            val k = index.getOrPut(p) { people += p; sums += FloatArray(dim); counts += 0; people.lastIndex }
            set.rows.get(i, v)
            val sum = sums[k]
            for (d in 0 until dim) sum[d] += v[d]
            counts[k]++
        }
        val ids = people.toLongArray()
        val count = counts.toIntArray()
        val merged = HashMap<Long, Long>()
        val joined = joinPeople(
            sums = sums.toTypedArray(),
            counts = count,
            active = touched?.let { t -> BooleanArray(ids.size) { ids[it] in t } },
            threshold = recognizer.sameAverage,
            allowed = { a, b ->
                val (nameA, hiddenA) = labels[ids[a]] ?: (null to false)
                val (nameB, hiddenB) = labels[ids[b]] ?: (null to false)
                !hiddenA && !hiddenB && (nameA == null || nameB == null || nameA == nameB) &&
                    refused[ids[a]]?.contains(ids[b]) != true && refused[ids[b]]?.contains(ids[a]) != true
            },
            // The named one keeps its id (and name); else the one with more faces.
            keepFirst = { a, b ->
                labels[ids[a]]?.first != null || (labels[ids[b]]?.first == null && count[a] >= count[b])
            },
            onJoin = { k, g ->
                val keep = ids[k]
                val gone = ids[g]
                store.merge(keep, gone)
                refused.remove(gone)?.let { refused.getOrPut(keep) { HashSet() } += it }
                for (s in refused.values) if (s.remove(gone)) s += keep
                if (labels[keep]?.first == null) labels[gone]?.first?.let { labels[keep] = it to false }
                labels.remove(gone)
                for (e in merged.entries) if (e.value == gone) e.setValue(keep)
                merged[gone] = keep
                for (i in 0 until set.size) if (set.person[i] == gone) set.person[i] = keep
            },
        )
        if (joined > 0) Log.i(Tag, "Joined $joined people found twice")
        return merged
    }

    /**
     * Lets go of unnamed people whose faces don't look like one person ([FaceRecognizer.coherent]):
     * a chain of strangers, each a little like the next. Named and hidden people stay. Only
     * [touched] people, or all when null.
     */
    private fun releaseChains(store: FaceStore, recognizer: FaceRecognizer, set: FaceSet, touched: Set<Long>?) {
        if (recognizer.coherent <= 0f) return
        val labels = store.labels()
        val byPerson = HashMap<Long, MutableList<Int>>()
        for (i in 0 until set.size) {
            val p = set.person[i]
            if (p == 0L || !set.clear[i] || (touched != null && p !in touched)) continue
            byPerson.getOrPut(p) { ArrayList() } += i
        }
        val dim = set.rows.dim
        val v = FloatArray(dim)
        val chains = byPerson.filter { (p, rows) ->
            val (name, hidden) = labels[p] ?: (null to false)
            if (name != null || hidden || rows.size < ChainMinFaces) return@filter false
            val sum = FloatArray(dim)
            for (i in rows) {
                set.rows.get(i, v)
                for (d in 0 until dim) sum[d] += v[d]
            }
            val norm = sqrt(sum.fold(0.0) { acc, x -> acc + x * x }).toFloat().coerceAtLeast(1e-6f)
            val alike = rows.map { dot(set.rows.get(it, v), sum) / norm }.sorted()
            alike[alike.size / 2] < recognizer.coherent
        }.keys
        if (chains.isNotEmpty()) {
            store.release(chains)
            for (i in 0 until set.size) if (set.person[i] in chains) set.person[i] = 0L
            Log.i(Tag, "Let go of ${chains.size} people that were chains of strangers")
        }
    }
}

/** [FaceGrouping]'s cutoffs, from the recognizer. */
internal class Rules(val samePerson: Float, val strongPair: Float, val sameCentre: Float, val minFaces: Int = FaceGrouping.MinFaces)

/**
 * One recognizer's faces, compact: per face its id, photo or video, the person it's in (0: none;
 * negative: made in this run), the person the user said it isn't (0: none), and whether it's
 * clear enough to group; the numbers themselves in [rows].
 */
internal class FaceSet(
    val ids: LongArray,
    val mediaIds: LongArray,
    val person: LongArray,
    val notPerson: LongArray,
    val clear: BooleanArray,
    val rows: FaceRows,
) {
    val size get() = ids.size
}

/**
 * Faces' numbers, [count] rows of [dim] floats in one block outside the Java heap (shared memory,
 * like Smart Search's: a big library's faces took tens of MB of the app's 256). Not thread-safe:
 * grouping is one run at a time.
 */
internal class FaceRows(val count: Int, val dim: Int, buffer: (Int) -> ByteBuffer = { Embeddings.offHeap(it) }) {
    private val data: FloatBuffer = buffer(maxOf(1, count * dim * 4)).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val chunk = FloatArray(ChunkRows * dim)

    fun put(i: Int, v: FloatArray) {
        data.duplicate().apply { position(i * dim) }.put(v, 0, dim)
    }

    /** Row [i] into [out], which it returns. */
    fun get(i: Int, out: FloatArray): FloatArray {
        data.duplicate().apply { position(i * dim) }.get(out, 0, dim)
        return out
    }

    /** Every row's dot product with [q], into [out]. */
    fun scores(q: FloatArray, out: FloatArray) {
        val view = data.duplicate()
        view.position(0)
        var i = 0
        while (i < count) {
            val rows = minOf(ChunkRows, count - i)
            view.get(chunk, 0, rows * dim)
            for (r in 0 until rows) {
                val base = r * dim
                var s = 0f
                for (d in 0 until dim) s += q[d] * chunk[base + d]
                out[i + r] = s
            }
            i += rows
        }
    }

    private companion object {
        const val ChunkRows = 64
    }
}

/** What [assignFaces] decided: faces (by id) joining existing people, and new people's faces. */
internal class Assigned(val joins: Map<Long, Long>, val newPeople: List<List<Long>>)

/**
 * Gives waiting faces a person by [rules] (see [FaceGrouping]), in two rounds over the faces in
 * order. [wake] null tries every waiting face in both rounds. Otherwise only those it marks (new
 * faces), and then waiting faces alike to a face that's new or just got a person, which are the
 * only ones whose outcome can have changed; marks behind the current face wait for the second round.
 *
 * Updates [FaceSet.person]: new people get placeholder ids -1, -2, … in the order of
 * [Assigned.newPeople].
 */
internal fun assignFaces(set: FaceSet, rules: Rules, wake: BooleanArray?): Assigned {
    val n = set.size
    val rows = set.rows
    val dim = rows.dim
    val person = set.person
    val clear = set.clear
    val full = wake == null
    val isNew = wake?.copyOf()
    val marked = wake ?: BooleanArray(n)
    val joins = HashMap<Long, Long>()
    var nextNew = -1L
    val newMembers = LinkedHashMap<Long, MutableList<Long>>()
    // Each person's clear faces added up: their average face, for [Rules.sameCentre].
    val centres = HashMap<Long, FloatArray>()
    val v = FloatArray(dim)
    val other = FloatArray(dim)
    val sims = FloatArray(n)
    val aroundSims = FloatArray(n)
    fun addTo(p: Long, i: Int) {
        if (!clear[i]) return
        val sum = centres.getOrPut(p) { FloatArray(dim) }
        rows.get(i, other)
        for (d in 0 until dim) sum[d] += other[d]
    }
    fun nearCentre(p: Long, q: FloatArray): Boolean {
        if (rules.sameCentre <= 0f) return true
        val sum = centres[p] ?: return true
        val norm = sqrt(sum.fold(0.0) { acc, x -> acc + x * x }).toFloat()
        return norm == 0f || dot(q, sum) / norm >= rules.sameCentre
    }
    fun mark(j: Int) {
        if (person[j] == 0L && clear[j]) marked[j] = true
    }
    /** Marks the waiting faces alike to face [j], which just got a person. */
    fun markAround(j: Int) {
        rows.scores(rows.get(j, other), aroundSims)
        for (k in 0 until n) if (k != j && aroundSims[k] >= rules.samePerson) mark(k)
    }
    for (i in 0 until n) if (person[i] != 0L) addTo(person[i], i)
    repeat(2) {
        if (full) marked.fill(true)
        for (i in 0 until n) {
            if (person[i] != 0L || !clear[i] || !marked[i]) continue
            marked[i] = false
            // Each person's most alike face here; the most alike person near whose average it is wins.
            val byPerson = HashMap<Long, Float>()
            var pair = -1
            var pairSim = rules.strongPair
            val close = ArrayList<Int>()
            rows.get(i, v)
            rows.scores(v, sims)
            for (j in 0 until n) {
                if (j == i || !clear[j]) continue
                val s = sims[j]
                if (s < rules.samePerson) continue
                close += j
                if (person[j] == 0L && set.mediaIds[j] != set.mediaIds[i] && s >= pairSim) {
                    pairSim = s
                    pair = j
                }
                // Never back to a person the user took this face from.
                val pj = person[j]
                if (pj != 0L && pj != set.notPerson[i] && s > (byPerson[pj] ?: -1f)) byPerson[pj] = s
            }
            val joined = byPerson.entries.sortedByDescending { it.value }.firstOrNull { nearCentre(it.key, v) }?.key
            val grouped = ArrayList<Int>()
            if (joined != null) {
                person[i] = joined
                addTo(joined, i)
                if (joined < 0) newMembers.getValue(joined) += set.ids[i] else joins[set.ids[i]] = joined
            } else if (byPerson.isNotEmpty()) {
                // Alike to someone's face but not to them: it waits rather than starting a twin.
            } else if (close.size >= rules.minFaces - 1) {
                val p = nextNew--
                val members = mutableListOf(set.ids[i])
                person[i] = p
                addTo(p, i)
                for (j in close) if (person[j] == 0L) {
                    person[j] = p
                    members += set.ids[j]
                    addTo(p, j)
                    grouped += j
                }
                newMembers[p] = members
            } else if (pair >= 0) {
                val p = nextNew--
                person[i] = p
                person[pair] = p
                addTo(p, i)
                addTo(p, pair)
                newMembers[p] = mutableListOf(set.ids[i], set.ids[pair])
                grouped += pair
            }
            if (!full && (person[i] != 0L || isNew!![i])) {
                for (j in close) mark(j)
                for (j in grouped) markAround(j)
            }
        }
    }
    return Assigned(joins, newMembers.values.toList())
}

/**
 * Joins people, most alike pair first, while any allowed pair's average faces are at least
 * [threshold] alike: the pairs with an [active] person (all when null; a joined person becomes
 * active). Each person keeps only their best match ([best], [partner]), refreshed when it joins
 * or its match does, instead of a table of every pair. [sums] (each person's faces added up) and
 * [counts] are updated as people join; [onJoin] is told which kept which. Returns the joins made.
 */
internal fun joinPeople(
    sums: Array<FloatArray>,
    counts: IntArray,
    active: BooleanArray?,
    threshold: Float,
    allowed: (Int, Int) -> Boolean,
    keepFirst: (Int, Int) -> Boolean,
    onJoin: (keep: Int, gone: Int) -> Unit,
): Int {
    val n = sums.size
    if (n < 2) return 0
    fun unit(sum: FloatArray): FloatArray {
        val norm = sqrt(sum.fold(0.0) { acc, x -> acc + x * x }).toFloat().coerceAtLeast(1e-6f)
        return FloatArray(sum.size) { sum[it] / norm }
    }
    val averages = Array(n) { unit(sums[it]) }
    val alive = BooleanArray(n) { true }
    val act = active?.copyOf() ?: BooleanArray(n) { true }
    val best = FloatArray(n) { Float.NEGATIVE_INFINITY }
    val partner = IntArray(n) { -1 }
    fun offer(i: Int, j: Int, s: Float) {
        if (s > best[i]) {
            best[i] = s
            partner[i] = j
        }
    }
    fun pairs(i: Int, j: Int): Float {
        val s = dot(averages[i], averages[j])
        return if (s >= threshold && allowed(i, j)) s else Float.NEGATIVE_INFINITY
    }
    fun refresh(r: Int) {
        best[r] = Float.NEGATIVE_INFINITY
        partner[r] = -1
        for (o in 0 until n) if (o != r && alive[o] && (act[r] || act[o])) offer(r, o, pairs(r, o))
    }
    for (i in 0 until n) for (j in i + 1 until n) {
        if (!act[i] && !act[j]) continue
        val s = pairs(i, j)
        offer(i, j, s)
        offer(j, i, s)
    }
    var joined = 0
    while (true) {
        var i = -1
        for (r in 0 until n) if (alive[r] && partner[r] >= 0 && (i < 0 || best[r] > best[i])) i = r
        if (i < 0) break
        val j = partner[i]
        val (keep, gone) = if (keepFirst(i, j)) i to j else j to i
        onJoin(keep, gone)
        for (d in sums[keep].indices) sums[keep][d] += sums[gone][d]
        counts[keep] += counts[gone]
        alive[gone] = false
        act[keep] = true
        averages[keep] = unit(sums[keep])
        for (r in 0 until n) {
            if (!alive[r]) continue
            if (r == keep || partner[r] == keep || partner[r] == gone) refresh(r) else offer(r, keep, pairs(r, keep))
        }
        joined++
    }
    return joined
}

internal fun dot(a: FloatArray, b: FloatArray): Float {
    var s = 0f
    for (k in a.indices) s += a[k] * b[k]
    return s
}
