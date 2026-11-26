package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.sqrt

/**
 * Face grouping without a pair table and with quick runs: the same decisions as before. The old
 * algorithm is copied here ([oldAssign], [oldJoin]) as the reference.
 */
class FaceGroupingTest {
    private val dim = 32
    private val rules = Rules(samePerson = 0.6f, strongPair = 0.75f, sameCentre = 0.5f)

    private fun unit(v: FloatArray): FloatArray {
        val n = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
        return FloatArray(v.size) { v[it] / n }
    }

    /** Faces of [people] identities, each a noisy copy of its identity, in shuffled order. */
    private fun faces(r: Random, people: Int, perPerson: IntRange, noise: Float): List<Pair<Int, FloatArray>> {
        val ids = List(people) { unit(FloatArray(dim) { r.nextGaussian().toFloat() }) }
        val out = ArrayList<Pair<Int, FloatArray>>()
        for ((p, centre) in ids.withIndex()) {
            repeat(perPerson.first + r.nextInt(perPerson.last - perPerson.first + 1)) {
                out += p to unit(FloatArray(dim) { centre[it] + noise * r.nextGaussian().toFloat() })
            }
        }
        out.shuffle(r)
        return out
    }

    private fun set(vectors: List<FloatArray>, media: LongArray = LongArray(vectors.size) { it.toLong() }, clear: BooleanArray = BooleanArray(vectors.size) { true }): FaceSet {
        val rows = FaceRows(vectors.size, dim)
        vectors.forEachIndexed { i, v -> rows.put(i, v) }
        return FaceSet(LongArray(vectors.size) { it + 1L }, media, LongArray(vectors.size), LongArray(vectors.size), clear, rows)
    }

    @Test
    fun aFullRunDecidesExactlyAsBefore() {
        val r = Random(11)
        val data = faces(r, people = 30, perPerson = 1..9, noise = 0.12f)
        val vectors = data.map { it.second }
        val clear = BooleanArray(vectors.size) { r.nextFloat() > 0.1f }
        val media = LongArray(vectors.size) { (it / 2).toLong() } // some photos with two faces
        val s = set(vectors, media, clear)
        val expected = oldAssign(vectors, media, clear, LongArray(vectors.size))
        val got = assignFaces(s, rules, wake = null)
        assertEquals(expected.first, got.joins)
        assertEquals(expected.second, got.newPeople)
        assertTrue("some people were made", got.newPeople.isNotEmpty())
    }

    @Test
    fun aQuickRunPlacesNewFacesAsAFullRunWould() {
        val r = Random(12)
        val data = faces(r, people = 25, perPerson = 4..10, noise = 0.1f)
        val vectors = data.map { it.second }
        val old = vectors.size * 3 / 4
        // Grouped once with the older faces; then the rest arrive.
        val before = set(vectors.subList(0, old))
        assignFaces(before, rules, wake = null)
        val people = LongArray(vectors.size).also { before.person.copyInto(it) }
        // Placeholders as real ids: 1000 + n.
        for (i in people.indices) if (people[i] < 0) people[i] = 1000 - people[i]

        val quick = set(vectors).also { people.copyInto(it.person) }
        val full = set(vectors).also { people.copyInto(it.person) }
        val wake = BooleanArray(vectors.size) { it >= old }
        val q = assignFaces(quick, rules, wake)
        val f = assignFaces(full, rules, wake = null)
        assertEquals(f.joins, q.joins)
        assertEquals(f.newPeople.map { it.toSet() }.toSet(), q.newPeople.map { it.toSet() }.toSet())
        assertTrue("new faces joined their people", q.joins.isNotEmpty())
    }

    @Test
    fun aQuickRunWithNothingNewChangesNothing() {
        val r = Random(13)
        val vectors = faces(r, people = 10, perPerson = 1..2, noise = 0.6f).map { it.second }
        val s = set(vectors)
        val got = assignFaces(s, rules, wake = BooleanArray(vectors.size))
        assertTrue(got.joins.isEmpty() && got.newPeople.isEmpty())
    }

    @Test
    fun joiningPeopleFollowsTheOldMostAlikeFirstOrder() {
        val r = Random(14)
        val n = 120
        // Pairs of near-twins, so there's plenty to join, plus loners.
        val base = List(n / 2) { unit(FloatArray(dim) { r.nextGaussian().toFloat() }) }
        val sums = Array(n) { i -> unit(FloatArray(dim) { base[i / 2][it] + 0.08f * r.nextGaussian().toFloat() }).map { it * (1 + i % 3) }.toFloatArray() }
        val counts = IntArray(n) { 1 + it % 3 }
        val forbidden = setOf(4 to 5, 10 to 11)
        val allowed = { a: Int, b: Int -> (minOf(a, b) to maxOf(a, b)) !in forbidden }
        val keepFirst = { a: Int, b: Int -> a < b }
        val expected = oldJoin(Array(n) { sums[it].copyOf() }, counts.copyOf(), 0.75f, allowed, keepFirst)
        val got = ArrayList<Pair<Int, Int>>()
        joinPeople(sums, counts, active = null, threshold = 0.75f, allowed = allowed, keepFirst = keepFirst) { k, g -> got += k to g }
        assertEquals(expected, got)
        assertTrue("some people were joined", got.size >= 5)
    }

    @Test
    fun thousandsOfPeopleNeedNoPairTable() {
        // 6,000 people: the old table alone was 144 MB; this keeps two numbers per person.
        val r = Random(15)
        val n = 6000
        val small = 8
        val sums = Array(n) { FloatArray(small) { r.nextGaussian().toFloat() } }
        val joined = joinPeople(sums, IntArray(n) { 1 }, active = BooleanArray(n) { it < 50 }, threshold = 0.999f, allowed = { _, _ -> true }, keepFirst = { _, _ -> true }) { _, _ -> }
        assertTrue(joined >= 0)
    }

    /** The grouping loop as it was before (FaceIndexer.kt, 21 Nov), over plain arrays. */
    private fun oldAssign(faces: List<FloatArray>, media: LongArray, clear: BooleanArray, start: LongArray): Pair<Map<Long, Long>, List<List<Long>>> {
        val person = arrayOfNulls<Long>(faces.size)
        start.forEachIndexed { i, p -> if (p != 0L) person[i] = p }
        val joins = HashMap<Long, Long>()
        var nextNew = -1L
        val newMembers = HashMap<Long, MutableList<Long>>()
        val centres = HashMap<Long, FloatArray>()
        fun addTo(p: Long, i: Int) {
            if (!clear[i]) return
            val sum = centres.getOrPut(p) { FloatArray(dim) }
            for (k in sum.indices) sum[k] += faces[i][k]
        }
        fun nearCentre(p: Long, v: FloatArray): Boolean {
            val sum = centres[p] ?: return true
            val norm = sqrt(sum.fold(0.0) { acc, x -> acc + x * x }).toFloat()
            return norm == 0f || dot(v, sum) / norm >= rules.sameCentre
        }
        faces.indices.forEach { i -> person[i]?.let { addTo(it, i) } }
        repeat(2) {
            for (i in faces.indices) {
                if (person[i] != null || !clear[i]) continue
                val byPerson = HashMap<Long, Float>()
                var pair = -1
                var pairSim = rules.strongPair
                val close = ArrayList<Int>()
                val v = faces[i]
                for (j in faces.indices) {
                    if (j == i || !clear[j]) continue
                    val s = dot(v, faces[j])
                    if (s < rules.samePerson) continue
                    close += j
                    if (person[j] == null && media[j] != media[i] && s >= pairSim) {
                        pairSim = s
                        pair = j
                    }
                    val pj = person[j]
                    if (pj != null && s > (byPerson[pj] ?: -1f)) byPerson[pj] = s
                }
                val joined = byPerson.entries.sortedByDescending { it.value }.firstOrNull { nearCentre(it.key, v) }?.key
                if (joined != null) {
                    person[i] = joined
                    addTo(joined, i)
                    if (joined < 0) newMembers.getValue(joined) += i + 1L else joins[i + 1L] = joined
                } else if (byPerson.isNotEmpty()) {
                    continue
                } else if (close.size >= FaceGrouping.MinFaces - 1) {
                    val p = nextNew--
                    val members = mutableListOf(i + 1L)
                    person[i] = p
                    addTo(p, i)
                    for (j in close) if (person[j] == null) {
                        person[j] = p
                        members += j + 1L
                        addTo(p, j)
                    }
                    newMembers[p] = members
                } else if (pair >= 0) {
                    val p = nextNew--
                    person[i] = p
                    person[pair] = p
                    addTo(p, i)
                    addTo(p, pair)
                    newMembers[p] = mutableListOf(i + 1L, pair + 1L)
                }
            }
        }
        // New people in the order they were made (-1, -2, …).
        return joins to newMembers.entries.sortedByDescending { it.key }.map { it.value }
    }

    /** joinSplit's loop as it was before: every pair's likeness in a table, most alike first. */
    private fun oldJoin(sums: Array<FloatArray>, counts: IntArray, threshold: Float, allowed: (Int, Int) -> Boolean, keepFirst: (Int, Int) -> Boolean): List<Pair<Int, Int>> {
        val n = sums.size
        fun average(i: Int): FloatArray {
            val norm = sqrt(sums[i].fold(0.0) { acc, x -> acc + x * x }).toFloat().coerceAtLeast(1e-6f)
            return FloatArray(dim) { sums[i][it] / norm }
        }
        val averages = Array(n) { average(it) }
        val alive = BooleanArray(n) { true }
        val sim = FloatArray(n * n)
        for (i in 0 until n) for (j in i + 1 until n) sim[i * n + j] = dot(averages[i], averages[j])
        val out = ArrayList<Pair<Int, Int>>()
        while (true) {
            var best = threshold
            var bi = -1
            var bj = -1
            for (i in 0 until n) {
                if (!alive[i]) continue
                for (j in i + 1 until n) if (alive[j] && sim[i * n + j] >= best && allowed(i, j)) {
                    best = sim[i * n + j]; bi = i; bj = j
                }
            }
            if (bi < 0) break
            val aFirst = keepFirst(bi, bj)
            val keep = if (aFirst) bi else bj
            val gone = if (aFirst) bj else bi
            out += keep to gone
            for (k in 0 until dim) sums[keep][k] += sums[gone][k]
            counts[keep] += counts[gone]
            alive[gone] = false
            averages[keep] = average(keep)
            for (o in 0 until n) if (alive[o] && o != keep) {
                val s = dot(averages[keep], averages[o])
                if (o < keep) sim[o * n + keep] = s else sim[keep * n + o] = s
            }
        }
        return out
    }
}
