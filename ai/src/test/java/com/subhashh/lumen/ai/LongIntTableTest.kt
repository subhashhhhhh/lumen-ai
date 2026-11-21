package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Random

/** The tokenizer's merge-rank table: must answer exactly as the HashMap it replaced. */
class LongIntTableTest {
    private fun pair(a: Int, b: Int) = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)

    @Test
    fun answersLikeAHashMapForPackedIdPairs() {
        val r = Random(5)
        val n = 200_000
        val table = LongIntTable(n)
        val reference = HashMap<Long, Int>()
        for (rank in 0 until n) {
            // Real merges pair ids below 262,144, with many sharing a left or right half.
            val key = pair(r.nextInt(262_144), r.nextInt(4_000))
            table.putIfAbsent(key, rank)
            reference.putIfAbsent(key, rank)
        }
        for ((k, v) in reference) assertEquals(v, table.get(k))
        repeat(50_000) {
            val key = pair(r.nextInt(262_144) + 300_000, r.nextInt(4_000))
            assertEquals(-1, table.get(key))
        }
    }

    @Test
    fun theFirstRankForADuplicatedPairWins() {
        val table = LongIntTable(4)
        table.putIfAbsent(pair(1, 2), 7)
        table.putIfAbsent(pair(1, 2), 3)
        assertEquals(7, table.get(pair(1, 2)))
        assertEquals(-1, table.get(pair(2, 1)))
    }

    @Test
    fun negativeIdsAndZeroAreOrdinaryKeys() {
        val table = LongIntTable(4)
        table.putIfAbsent(0L, 0)
        table.putIfAbsent(pair(-1, -1), 1)
        assertEquals(0, table.get(0L))
        assertEquals(1, table.get(pair(-1, -1)))
    }
}
