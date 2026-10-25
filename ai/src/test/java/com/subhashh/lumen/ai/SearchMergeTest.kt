package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class SearchMergeTest {
    @Test
    fun exactMatchesComeFirst() {
        assertEquals(listOf(3L, 1L, 2L), Search.merge(listOf(3L, 1L), listOf(2L)))
    }

    @Test
    fun eachItemOnce_atItsBestPlace() {
        assertEquals(listOf(5L, 7L, 9L), Search.merge(listOf(5L, 7L), listOf(7L, 9L, 5L)))
    }

    @Test
    fun smartOnly() {
        assertEquals(listOf(4L, 2L), Search.merge(emptyList(), listOf(4L, 2L)))
    }
}
