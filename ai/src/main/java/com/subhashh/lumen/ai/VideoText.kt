package com.subhashh.lumen.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

/**
 * A video's words, as the text pass reads them: as many frames a second as the user chose for
 * its length ([VideoTextRates]), never more than the video has, each line read only when it
 * changed ([TextReader.readChanged]).
 *
 * Two things keep it quick without missing captions:
 *  - the next frame decodes on another thread while the model reads this one (decoding is mostly
 *    the video hardware, the model the GPU and CPU);
 *  - after a frame with no text at all, the next look is at least a second later (at two or
 *    more frames a second). Text that appears is found at most a second late.
 */
@LumenAiEngine
object VideoText {

    class Result(
        /** Every distinct line, in the order first seen. */
        val lines: Set<String>,
        /** Frames decoded, and those the model looked at. */
        val frames: Int,
        val looked: Int,
        /** Lines actually read (the rest were unchanged from the frame before). */
        val read: Int,
    )

    /** Frames decoded ahead of the model: two keep it busy, a few MB at most. */
    private const val Ahead = 2
    /** After a frame with no text, the next one looked at is at least this much later. */
    private const val QuietGapMs = 1_000L

    /** Milliseconds between the frames read in [item]: the chosen rate, capped at the video's own. */
    fun stepMs(context: Context, item: MediaRef): Long {
        val (duration, fps) = MediaFrames.videoInfo(context, item)
        val wanted = LumenAi.host.settings.videoTextRates.forDuration(item.durationMs.takeIf { it > 0 } ?: duration ?: 0L)
        val rate = fps?.let { minOf(wanted, kotlin.math.ceil(it).toInt().coerceAtLeast(1)) } ?: wanted
        return (1000L / rate).coerceAtLeast(1)
    }

    /** [quick] false: the plain way (one frame at a time, every frame looked at, every changed-looking box read), for comparisons. */
    suspend fun read(context: Context, item: MediaRef, reader: TextReader, px: Int, quick: Boolean = true): Result {
        val lines = LinkedHashSet<String>()
        var frames = 0
        var looked = 0
        var read = 0
        var previous: TextReader.FrameText? = null
        var lastLook = Long.MIN_VALUE / 2
        fun look(at: Long, frame: Bitmap) {
            frames++
            if (quick && previous?.lines?.isEmpty() == true && at - lastLook < QuietGapMs) {
                frame.recycle()
                return
            }
            val text = try { reader.readChanged(frame, previous, smart = quick) } finally { frame.recycle() }
            looked++
            lastLook = at
            read += text.read
            text.lines.filterNotNullTo(lines)
            previous = text
        }
        if (!quick) {
            decode(context, item, px) { at, frame -> look(at, frame) }
            return Result(lines, frames, looked, read)
        }
        coroutineScope {
            val queue = Channel<Pair<Long, Bitmap>>(Ahead)
            val decoder = launch(Dispatchers.IO) {
                try {
                    decode(context, item, px) { at, frame -> queue.send(at to frame) }
                } finally {
                    queue.close()
                }
            }
            try {
                for ((at, frame) in queue) look(at, frame)
            } finally {
                decoder.cancel()
                // Frames decoded but never read (stopped early): free them.
                while (true) queue.tryReceive().getOrNull()?.second?.recycle() ?: break
            }
        }
        return Result(lines, frames, looked, read)
    }

    /** The video's frames; a decoding error partway ends them early, keeping what was read. */
    private suspend fun decode(context: Context, item: MediaRef, px: Int, onFrame: suspend (Long, Bitmap) -> Unit) {
        try {
            MediaFrames.forEach(context, item, px, skipSame = false, everyMs = stepMs(context, item)) { at, frame ->
                onFrame(at, frame)
                true
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w("LumenText", "Couldn't decode all of ${item.name}", e)
        }
    }
}
