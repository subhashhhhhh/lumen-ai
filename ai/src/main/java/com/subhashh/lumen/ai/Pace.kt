package com.subhashh.lumen.ai

import android.app.ActivityManager
import android.os.Process
import android.os.SystemClock
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import java.util.concurrent.Executors

/**
 * Keeps indexing out of the way of the app it runs in. It runs in the app's own process, so while
 * the app is on screen its grid and the indexer compete for the same cores (issue #39: the grid
 * stuttered during a first index).
 *
 * - [Background]: backlog passes run on threads at background priority, and so do the model
 *   runtimes' own threads, which inherit it when a pass creates them: the UI thread and
 *   RenderThread always go first.
 * - [rest], after each photo or video frame: while the app is on screen, waits three times as long
 *   as the work took, so indexing takes about a quarter of the phone's time; at full speed again
 *   once the app is in the background.
 */
internal class Pace {
    private var since = SystemClock.elapsedRealtime()

    suspend fun rest() {
        if (onScreen()) {
            val worked = SystemClock.elapsedRealtime() - since
            delay((worked * (Share - 1)).coerceIn(0L, MaxRestMs))
        }
        since = SystemClock.elapsedRealtime()
    }

    companion object {
        private const val Share = 4
        /** A slow first item (a model warming up) doesn't stall the next one for long. */
        private const val MaxRestMs = 1_500L
        private const val CheckEveryMs = 1_000L

        @Volatile private var checkedAt = 0L
        @Volatile private var visible = false

        /** Whether one of the app's screens is in front: a foreground service alone doesn't count. */
        fun onScreen(): Boolean {
            val now = SystemClock.elapsedRealtime()
            if (now - checkedAt > CheckEveryMs) {
                val info = ActivityManager.RunningAppProcessInfo()
                ActivityManager.getMyMemoryState(info)
                visible = info.importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND
                checkedAt = now
            }
            return visible
        }

        val Background: ExecutorCoroutineDispatcher by lazy {
            Executors.newFixedThreadPool(3) { work ->
                Thread({
                    Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                    work.run()
                }, "lumen-index")
            }.asCoroutineDispatcher()
        }
    }
}
