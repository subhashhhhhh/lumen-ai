package com.subhashh.lumen.ai

import java.io.DataInputStream
import java.io.File

/**
 * SigLIP 2's text tokenizer (Gemma's 256k BPE), read from the compact file that
 * tools/models/pack_gemma_tokenizer.py makes from tokenizer.json. Matches OpenCLIP's
 * "canonicalize" cleaning: no ASCII punctuation, lower case, single spaces. Then BPE over the
 * whole string (spaces as ▁), unknown characters as UTF-8 byte tokens, `<eos>`, zero padding.
 * Checked against the official tokenizer on English, Hindi, Telugu and emoji queries.
 */
@LumenAiEngine
class GemmaBpeTokenizer(file: File) {
    private val singles = HashMap<Int, Int>()
    private val byteIds = IntArray(256)
    /** (left id, right id) packed into a Long → merge rank. */
    private val ranks: LongIntTable
    private val merged: IntArray
    private val eos: Int

    init {
        DataInputStream(file.inputStream().buffered(1 shl 16)).use { input ->
            repeat(input.readInt()) { singles[input.readInt()] = input.readInt() }
            for (b in 0 until 256) byteIds[b] = input.readInt()
            val count = input.readInt()
            ranks = LongIntTable(count)
            merged = IntArray(count)
            for (rank in 0 until count) {
                val key = pair(input.readInt(), input.readInt())
                merged[rank] = input.readInt()
                ranks.putIfAbsent(key, rank)
            }
            eos = input.readInt()
        }
    }

    fun tokenize(text: String, contextLength: Int = 64): IntArray {
        val clean = text.replace('_', ' ').filterNot { it in Punctuation }.lowercase()
            .split(Whitespace).filter { it.isNotEmpty() }.joinToString("▁")
        val symbols = ArrayList<Int>()
        var i = 0
        while (i < clean.length) {
            val cp = clean.codePointAt(i)
            val id = singles[cp]
            if (id != null) symbols += id
            else String(Character.toChars(cp)).toByteArray(Charsets.UTF_8).forEach { symbols += byteIds[it.toInt() and 0xFF] }
            i += Character.charCount(cp)
        }
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var at = -1
            for (j in 0 until symbols.size - 1) {
                val rank = ranks.get(pair(symbols[j], symbols[j + 1]))
                if (rank < 0) continue
                if (rank < bestRank) {
                    bestRank = rank
                    at = j
                }
            }
            if (at < 0) break
            symbols[at] = merged[bestRank]
            symbols.removeAt(at + 1)
        }
        val ids = symbols.take(contextLength - 1) + eos
        return IntArray(contextLength) { ids.getOrElse(it) { 0 } }
    }

    private companion object {
        const val Punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        val Whitespace = Regex("\\s+")
        fun pair(a: Int, b: Int) = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
    }
}

/**
 * Long → non-negative Int, for the tokenizer's ~580,000 merge ranks: open addressing in two
 * plain arrays (about 13 MB) instead of a HashMap of boxed Longs and Integers (over 50 MB, which
 * helped run the app out of memory). Fixed size, no removal.
 */
internal class LongIntTable(expected: Int) {
    private val mask: Int
    private val keys: LongArray
    private val values: IntArray

    init {
        var capacity = 16
        while (capacity < expected + expected / 2) capacity = capacity shl 1 // at most 2/3 full
        mask = capacity - 1
        keys = LongArray(capacity)
        values = IntArray(capacity) { -1 } // -1: empty slot
    }

    /** Keeps the first value given for a key, as HashMap.putIfAbsent does. */
    fun putIfAbsent(key: Long, value: Int) {
        require(value >= 0)
        var i = slot(key)
        while (values[i] >= 0) {
            if (keys[i] == key) return
            i = (i + 1) and mask
        }
        keys[i] = key
        values[i] = value
    }

    /** The value for [key], or -1. */
    fun get(key: Long): Int {
        var i = slot(key)
        while (values[i] >= 0) {
            if (keys[i] == key) return values[i]
            i = (i + 1) and mask
        }
        return -1
    }

    private fun slot(key: Long): Int {
        // Spread the bits (packed id pairs are far from random) before taking the low ones.
        var h = key * -0x61c8864680b583ebL
        h = h xor (h ushr 29)
        return h.toInt() and mask
    }
}
