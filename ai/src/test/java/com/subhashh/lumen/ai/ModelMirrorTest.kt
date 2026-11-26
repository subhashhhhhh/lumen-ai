package com.subhashh.lumen.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelMirrorTest {
    private val hf = "https://huggingface.co/subhashhhhhh/lumen-siglip2-b16/resolve/4b62b4b/visual.onnx"

    @Test
    fun aMirrorServesTheSamePath() {
        assertEquals(
            "https://hf-mirror.com/subhashhhhhh/lumen-siglip2-b16/resolve/4b62b4b/visual.onnx",
            ModelDownloads.mirrored(hf, "https://hf-mirror.com/"),
        )
    }

    @Test
    fun noMirrorLeavesHuggingFace() {
        assertEquals(hf, ModelDownloads.mirrored(hf, null))
    }

    @Test
    fun otherHostsAreLeftAlone() {
        val other = "https://example.org/model.onnx"
        assertEquals(other, ModelDownloads.mirrored(other, "https://hf-mirror.com"))
    }
}
