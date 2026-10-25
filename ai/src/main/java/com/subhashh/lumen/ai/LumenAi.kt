package com.subhashh.lumen.ai

import android.app.PendingIntent
import android.content.Context
import android.media.MediaDataSource
import android.net.Uri
import androidx.annotation.DrawableRes
import androidx.core.content.edit
import kotlinx.coroutines.flow.Flow

/**
 * Lumen AI's entry point.
 *
 * Install it once, in `Application.onCreate`, before anything else here runs: background
 * indexing and model downloads start in their own process launches and need it.
 *
 * ```
 * LumenAi.install(this)                       // every photo and video, default settings
 * LumenAi.models.download(context, ModelPack.Search)
 * val results = LumenAi.search(context, "cat sleeping on a chair")
 * ```
 */
object LumenAi {
    @Volatile private var installed: LumenAiHost? = null

    /**
     * Installs Lumen AI with defaults: it indexes every photo and video on the phone (from
     * MediaStore), shows its own notification icon, and opens your launcher activity from its
     * notifications. Change any of that in [configure].
     */
    fun install(context: Context, configure: LumenAiConfig.() -> Unit = {}) {
        install(LumenAiConfig().apply(configure).toHost(context.applicationContext))
    }

    /** Installs Lumen AI with a host of your own, for full control over every setting. */
    fun install(host: LumenAiHost) {
        installed = host
    }

    /** The installed host. */
    val host: LumenAiHost
        get() = installed ?: error("Lumen AI isn't installed: call LumenAi.install in Application.onCreate")

    /** Download, check and remove each feature's models. */
    val models: Models get() = Models

    /** Background indexing: when it runs, and how far it has got. */
    val indexing: Indexing get() = Indexing

    /** The people found in the library: list, name, merge, hide. */
    val people: People get() = People

    /**
     * Searches the library, the way Lumen does. Words in quotes find that exact phrase in the
     * text of photos and videos. Two or more names find the photos with all of those people.
     * Otherwise, photos whose text has the words come first, then Smart Search's matches.
     * Uses whichever features are downloaded; empty results when none are.
     */
    suspend fun search(context: Context, query: String): SearchResults = Search.run(context, query)

    /**
     * The lines of text in a photo, with their positions, for selecting and copying. Instant for
     * indexed photos; others are read now (about a second) and kept. Null when it has no text or
     * the Text feature isn't downloaded.
     */
    suspend fun liveText(context: Context, item: MediaRef): LiveText.Result? = Search.liveText(context, item)

    /** The text in one frame of a video, at [atMs]: for a paused video. */
    suspend fun liveText(context: Context, item: MediaRef, atMs: Long): LiveText.Result? =
        LiveText.readVideoFrame(context, item, atMs)

    /** Up to [max] words that describe a photo, e.g. "cat", "beach". Needs Smart Search. */
    suspend fun labels(context: Context, mediaId: Long, max: Int = 3): List<String> = Search.labels(context, mediaId, max)

    /** Search suggestions drawn from the library: things there are many photos of. */
    suspend fun suggestions(context: Context): List<Suggestion> = Search.suggestions(context)

    /**
     * Makes subtitles for a video, starting at [fromMs], and emits all cues so far as each part
     * is done. Made parts are kept, so calling it again continues. Needs Subtitles; ARM only.
     */
    fun subtitles(context: Context, item: MediaRef, fromMs: Long = 0): Flow<List<Cue>> = Search.subtitles(context, item, fromMs)

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
