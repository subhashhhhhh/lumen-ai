package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The catalogue of Cutout's NPU models: one per chip, where Hugging Face has them. */
class CutoutModelTest {
    @Test
    fun everyNpuChipHasACutoutModel() {
        for (chip in NpuModels.chips) {
            val file = CutoutModel.npu(chip)
            assertNotNull("no cutout model for $chip", file)
            assertEquals("npu/$chip/lumen_cutout_1024_int8_$chip.tflite", file!!.name)
            assertTrue("$chip: not a SHA-256", Regex("[0-9a-f]{64}").matches(file.sha256))
            assertTrue("$chip: an unlikely size", file.bytes in 40_000_000L..200_000_000L)
        }
    }

    @Test
    fun anUnknownChipHasNone() {
        assertNull(CutoutModel.npu("MT6989"))
    }

    @Test
    fun filesComeFromThePinnedCommit() {
        val url = CutoutModel.url(CutoutModel.npu("SM7450")!!)
        assertEquals("https://huggingface.co/subhashhhhhh/lumen-cutout/resolve/${CutoutModel.Commit}/npu/SM7450/lumen_cutout_1024_int8_SM7450.tflite", url)
    }
}
