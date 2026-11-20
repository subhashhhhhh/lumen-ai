package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.sqrt

/** The search index in memory: 8-bit rows out of the heap, catching up, and ranking from it. */
class EmbeddingsTest {
    private val dim = 768

    private fun unit(r: Random) = FloatArray(dim) { r.nextGaussian().toFloat() }.also { v ->
        val n = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
        for (i in v.indices) v[i] /= n
    }

    private fun dot(a: FloatArray, b: FloatArray) = a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()

    @Test
    fun eightBitScoresStayCloseToTheFloatOnes() {
        val r = Random(1)
        val vectors = List(2000) { unit(r) }
        val all = Embeddings(dim, vectors.size)
        vectors.forEachIndexed { i, v -> assertTrue(all.append(i.toLong(), 0, v)) }
        val q = unit(r)
        val out = FloatArray(all.size)
        all.scores(q, out)
        // Unit vectors: dot products are in [-1, 1], and real result gaps are ~0.01 and up.
        val worst = vectors.indices.maxOf { abs(out[it] - dot(q, vectors[it])) }
        assertTrue("worst error $worst", worst < 0.004f)
    }

    @Test
    fun theBestMatchesAreTheSameAsWithFloats() {
        val r = Random(2)
        val vectors = List(5000) { unit(r) }
        val q = unit(r)
        // Make 20 items clearly relevant, as a real query has, at varied strengths.
        val relevant = (0 until 20).map { it * 250 }
        relevant.forEachIndexed { k, i -> for (d in 0 until dim) vectors[i][d] = 0.6f * vectors[i][d] + (0.3f + k * 0.02f) * q[d] }
        val all = Embeddings.withRoomFor(vectors.size, dim)
        vectors.forEachIndexed { i, v -> all.append(i.toLong(), 0, v) }
        val expected = vectors.indices.sortedByDescending { dot(q, vectors[it]) }.take(20).map { it.toLong() }
        val got = SmartSearch.scored(all, q).take(20).map { it.mediaId }
        // The same 20, and the same best one; neighbours closer than the rounding may swap.
        assertEquals(expected.toSet(), got.toSet())
        assertEquals(expected.first(), got.first())
    }

    @Test
    fun aVideoScoresAsItsBestFrame() {
        val r = Random(3)
        val q = unit(r)
        val all = Embeddings.withRoomFor(4, dim)
        all.append(1, 0, unit(r))
        all.append(2, 0, FloatArray(dim) { -q[it] }) // video 2: a bad frame,
        all.append(2, 4000, q.copyOf()) // then a perfect one at 4 s
        all.append(3, 0, unit(r))
        val hits = SmartSearch.scored(all, q)
        assertEquals(2L, hits.first().mediaId)
        assertEquals(4000L, hits.first().atMs)
        assertEquals(3, hits.size) // one hit per item, not per frame
        assertEquals(hits.sortedByDescending { it.score }, hits)
    }

    @Test
    fun negativeScoresSortBelowPositiveOnes() {
        val q = FloatArray(dim).also { it[0] = 1f }
        val all = Embeddings.withRoomFor(4, dim)
        for ((id, x) in listOf(1L to -0.9f, 2L to 0.2f, 3L to -0.1f, 4L to 0.7f)) {
            all.append(id, 0, FloatArray(dim).also { it[0] = x; it[1] = sqrt(1 - x * x) })
        }
        assertEquals(listOf(4L, 2L, 3L, 1L), SmartSearch.scored(all, q).map { it.mediaId })
    }

    @Test
    fun removedItemsDisappearAndTheirNewRowsCount() {
        val r = Random(4)
        val q = unit(r)
        val all = Embeddings.withRoomFor(3, dim)
        all.append(1, 0, q.copyOf())
        all.append(2, 0, unit(r))
        all.append(3, 0, unit(r))
        all.remove(setOf(1L))
        assertEquals(1, all.dead)
        assertEquals(Embeddings.Dead, all.mediaId(0))
        assertEquals(setOf(2L, 3L), SmartSearch.scored(all, q).map { it.mediaId }.toSet())
        // Re-indexed: its new row goes at the end and counts again.
        all.append(1, 0, q.copyOf())
        assertEquals(1L, SmartSearch.scored(all, q).first().mediaId)
        assertEquals(3, SmartSearch.scored(all, q).size)
    }

    @Test
    fun aFullIndexSaysSoInsteadOfGrowing() {
        val all = Embeddings(dim, 2)
        val v = FloatArray(dim).also { it[0] = 1f }
        assertTrue(all.append(1, 0, v))
        assertTrue(all.append(2, 0, v))
        assertFalse(all.append(3, 0, v))
        assertEquals(2, all.size)
    }

    @Test
    fun anAllZeroVectorScoresZero() {
        val all = Embeddings.withRoomFor(1, dim)
        all.append(1, 0, FloatArray(dim))
        val out = FloatArray(1)
        all.scores(FloatArray(dim) { 1f }, out)
        assertEquals(0f, out[0])
    }

    @Test
    fun theChangeLogNamesWhatChangedOrAdmitsItCantTell() {
        val log = ChangeLog(keep = 3)
        assertNull("before this process wrote anything, nothing can be caught up", log.since(-1))
        assertEquals(1, log.add(longArrayOf(10)))
        log.add(longArrayOf(11, 12))
        assertEquals(setOf(10L, 11L, 12L), log.since(0))
        assertEquals(setOf(11L, 12L), log.since(1))
        assertEquals(emptySet<Long>(), log.since(2))
        log.add(longArrayOf(13))
        log.add(longArrayOf(14)) // version 4: the write that made version 1 is forgotten
        assertNull(log.since(0))
        assertEquals(setOf(11L, 12L, 13L, 14L), log.since(1))
        assertNull("a version from the future means another process", log.since(9))
    }
}
