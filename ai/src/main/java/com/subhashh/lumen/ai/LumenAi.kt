package com.subhashh.lumen.ai

import android.app.PendingIntent
import android.content.Context
import android.media.MediaDataSource
import android.net.Uri
import androidx.annotation.DrawableRes
import androidx.core.content.edit

/**
 * The library's one link to the app using it. Install a [LumenAiHost] in `Application.onCreate`,
 * before anything else here runs: background indexing and model downloads start in their own
 * process launches and look the host up from there.
 */
object LumenAi {
    @Volatile private var installed: LumenAiHost? = null

    fun install(host: LumenAiHost) {
        installed = host
    }

    val host: LumenAiHost
        get() = installed ?: error("LumenAi.install(host) wasn't called; call it in Application.onCreate")

    /**
     * For an app that kept the download queue and video-text version itself before using the
     * library: hands them over once, so nothing is lost or redone. Later calls do nothing.
     */
    fun adoptState(context: Context, modelQueue: List<String>, textVideoVersion: Int) =
        AiState.adopt(context, modelQueue, textVideoVersion)
}

/** One photo or video, as the indexers and readers need it. */
data class MediaRef(
    /** Stable for the life of the file, e.g. its MediaStore id. Every index is keyed by it. */
    val id: Long,
    val uri: Uri,
    val mimeType: String,
    val isVideo: Boolean,
    /** Epoch millis: when it was taken. Newest are indexed first. */
    val takenAt: Long,
    /** 0 when unknown; read from the file then. */
    val durationMs: Long,
    /** Changes whenever the file does (DATE_MODIFIED, seconds): a changed file is indexed again. */
    val modified: Long,
    /** File name, for logs. */
    val name: String = "",
)

interface LumenAiHost {
    /** Everything the background indexers should cover, with the app's own exclusions applied. */
    suspend fun library(context: Context): List<MediaRef>

    /**
     * For media Android can't open by [MediaRef.uri], such as an app's own encrypted files: a data
     * source the library closes when it's done. Null (the default) reads the URI.
     */
    fun videoSource(context: Context, item: MediaRef): MediaDataSource? = null

    /** The user's settings; read each time they're needed, so changes apply at once. */
    val settings: AiSettings

    val notifications: AiNotifications
}

interface AiSettings {
    /** [ModelCatalog] id of the chosen search model; null for the default. */
    val searchModelId: String?
    /** Long side, in pixels, of the video frames read for faces and text. */
    val videoDetailPx: Int
    /** Index the backlog without the charger too. */
    val indexOnBattery: Boolean
    /** Index only inside these hours, plugged in or not; null for no schedule. */
    val indexSchedule: IndexSchedule?
}

/** Minutes after midnight; the window may cross midnight. */
data class IndexSchedule(val startMinute: Int, val endMinute: Int)

interface AiNotifications {
    /** Small icon for the download and indexing notifications: a monochrome drawable. */
    @get:DrawableRes val smallIcon: Int

    /** What tapping a notification opens. */
    fun contentIntent(context: Context): PendingIntent
}

/** The library's own bookkeeping, kept apart from the app's preferences. */
internal class AiState(context: Context) {
    private val prefs = context.getSharedPreferences("lumen_ai", Context.MODE_PRIVATE)

    /** Model packs waiting to download, in order. */
    var modelQueue: List<String>
        get() = prefs.getString(KeyModelQueue, null)?.split(',')?.filter { it.isNotEmpty() } ?: emptyList()
        set(value) = prefs.edit { putString(KeyModelQueue, value.joinToString(",")) }

    /** Version of the video text pass that last ran; a newer one re-reads videos once. */
    var textVideoVersion: Int
        get() = prefs.getInt(KeyTextVideoVersion, 1)
        set(value) = prefs.edit { putInt(KeyTextVideoVersion, value) }

    companion object {
        private const val KeyModelQueue = "model_queue"
        private const val KeyTextVideoVersion = "text_video_version"

        /**
         * One-time hand-over for an app that kept this state itself before using the library,
         * so a queued download or a finished video re-read isn't lost. Does nothing once set.
         */
        fun adopt(context: Context, modelQueue: List<String>, textVideoVersion: Int) {
            val prefs = context.getSharedPreferences("lumen_ai", Context.MODE_PRIVATE)
            if (prefs.getBoolean("adopted", false)) return
            prefs.edit {
                putString(KeyModelQueue, modelQueue.joinToString(","))
                putInt(KeyTextVideoVersion, textVideoVersion)
                putBoolean("adopted", true)
            }
        }
    }
}

/** Search words: lower case, split at anything that isn't a letter or digit, and between letters and digits. */
internal object Terms {
    private val NonWord = Regex("[^\\p{L}\\p{N}]+")
    private val LetterDigit = Regex("(?<=\\p{L})(?=\\p{N})|(?<=\\p{N})(?=\\p{L})")

    fun of(text: String): List<String> =
        text.lowercase(java.util.Locale.ROOT).split(NonWord).flatMap { it.split(LetterDigit) }.filter { it.isNotEmpty() }
}
