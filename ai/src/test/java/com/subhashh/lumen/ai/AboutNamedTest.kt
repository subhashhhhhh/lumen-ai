package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Test

/** "messages with chiara": what a named person and other words give together. */
class AboutNamedTest {
    private fun hits(vararg ids: Long) = ids.mapIndexed { i, id -> SmartHit(id, 1f - i * 0.1f, 0) }
    private fun ids(hits: List<SmartHit>) = hits.map { it.mediaId }

    @Test
    fun whatIsAboutThemAndFitsComesFirst() {
        // 1 and 2 are chats with her name; 7 is a selfie of hers; 5 is another chat.
        val order = SmartSearch.aboutNamed(
            theirs = setOf(1, 2, 7),
            byRest = hits(5, 1, 2, 9, 7),
            fitting = hits(5, 1, 2),
            others = hits(5, 9),
        )
        assertEquals(listOf(1L, 2L, 5L, 9L, 7L), ids(order))
    }

    @Test
    fun plainMatchesStayWhenNothingOfTheirsFits() {
        // Her name was read nowhere and her face is in no chat: the chats still show, then her photos.
        val order = SmartSearch.aboutNamed(theirs = setOf(7, 8), byRest = hits(5, 6, 8, 7), fitting = hits(5, 6), others = hits(5, 6))
        assertEquals(listOf(5L, 6L, 8L, 7L), ids(order))
    }

    @Test
    fun eachItemOnce() {
        val order = SmartSearch.aboutNamed(theirs = setOf(1), byRest = hits(1, 2), fitting = hits(1), others = hits(1, 2))
        assertEquals(listOf(1L, 2L), ids(order))
    }
}
