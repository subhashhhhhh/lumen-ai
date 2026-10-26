package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class VideoTextRatesTest {

    @Test
    fun ratesFollowTheVideosLength() {
        val rates = VideoTextRates(short = 2, medium = 3, long = 4)
        assertEquals(60, rates.times(30_000, null).size) // 30 s at 2 a second
        assertEquals(270, rates.times(90_000, null).size) // 90 s at 3 a second
        assertEquals(1200, rates.times(300_000, null).size) // 5 min at 4 a second
    }

    @Test
    fun neverMoreThanTheVideoHas() {
        assertEquals(60, VideoTextRates(short = 60).times(2_000, 30f).size)
        assertEquals(120, VideoTextRates(short = 60).times(2_000, null).size)
    }

    @Test
    fun maxFramesSpreadsLongVideosOnly() {
        val capped = VideoTextRates(long = 1, longMaxFrames = 120)
        val hour = capped.times(3_600_000, null)
        assertEquals(120, hour.size)
        assertEquals(15_000L, hour.first()) // the middle of the first 30 s share
        assertEquals(3_600, VideoTextRates(long = 1).times(3_600_000, null).size) // 0: no cap
        assertEquals(100, capped.times(100_000, null).size) // medium videos aren't capped
        assertEquals(180, VideoTextRates(long = 1, longMaxFrames = 200).times(180_000, null).size) // under the cap: every second
    }

    @Test
    fun maxFramesIsAtMostAThousandTimesTheRate() {
        assertEquals(1_000, VideoTextRates(long = 1, longMaxFrames = 60_000).times(3_600_000, null).size)
    }
}
