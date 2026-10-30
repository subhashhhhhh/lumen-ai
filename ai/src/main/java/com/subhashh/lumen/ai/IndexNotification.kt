package com.subhashh.lumen.ai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.work.ForegroundInfo

/**
 * A quiet progress notification while Smart Search works through a backlog: vectors first, then
 * text, then faces. Gone when the pass finishes; one that stops early (unplugged, too warm, Android's
 * 10-minute job limit) clears itself a few minutes after its last update unless the next run
 * picks it up. A backlog run also uses it as its foreground service's notification ([foreground]).
 */
object IndexNotification {
    private const val Id = 4202
    private const val Channel = "indexing"
    private const val ClearAfterMs = 3 * 60_000L
    private const val EveryMs = 1_000L
    private var lastShown = 0L

    enum class Pass(val title: String) {
        Vectors("Indexing photos and videos"),
        Text("Reading text in photos and videos"),
        Faces("Finding faces in photos"),
    }

    /**
     * For a backlog run as a foreground service: Android keeps it running past the 10-minute job limit
     * (and lists Lumen under Active apps). Media processing on Android 15+, data sync before. The
     * passes update the same notification.
     */
    fun foreground(context: Context): ForegroundInfo {
        val type = if (Build.VERSION.SDK_INT >= 35) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        return ForegroundInfo(Id, build(context, "Smart Features", "Getting ready", null), type)
    }

    fun update(context: Context, pass: Pass, done: Int, total: Int) {
        val manager = context.getSystemService(NotificationManager::class.java)
        if (done >= total) {
            manager.cancel(Id)
            return
        }
        // Once a second at most: a photo without text takes ~0.5 s.
        val now = SystemClock.elapsedRealtime()
        if (now - lastShown < EveryMs) return
        lastShown = now
        manager.notify(Id, build(context, pass.title, "%,d of %,d".format(done, total), done to total))
    }

    private fun build(context: Context, title: String, text: String, progress: Pair<Int, Int>?): Notification {
        val manager = context.getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(Channel) == null) {
            manager.createNotificationChannel(
                NotificationChannel(Channel, "Smart Features indexing", NotificationManager.IMPORTANCE_LOW).apply {
                    description = "Progress while Smart Features read your library"
                },
            )
        }
        val open = LumenAi.host.notifications.contentIntent(context)
        return NotificationCompat.Builder(context, Channel)
            .setSmallIcon(LumenAi.host.notifications.smallIcon)
            .setContentTitle(title)
            .setContentText(text)
            .apply { if (progress != null) setProgress(progress.second, progress.first, false) }
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setTimeoutAfter(ClearAfterMs)
            .setContentIntent(open)
            .build()
    }

    fun cancel(context: Context) = context.getSystemService(NotificationManager::class.java).cancel(Id)
}
