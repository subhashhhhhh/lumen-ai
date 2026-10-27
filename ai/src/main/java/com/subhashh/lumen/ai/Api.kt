package com.subhashh.lumen.ai

import android.app.PendingIntent
import android.content.ContentUris
import android.content.Context
import android.media.MediaDataSource
import android.provider.MediaStore
import androidx.annotation.DrawableRes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.withContext
import java.io.File

// ---------- Setup ----------

/** Lumen AI's search models. */
enum class SearchModel(internal val id: String) {
    /** SigLIP 2: the better results. About 670 MB; runs on the GPU where it can. */
    Accurate("siglip2-b16"),

    /** CLIP ViT-B/32: smaller and faster, for phones with less memory or storage. About 240 MB. */
    Fast("clip-vit-b32"),
}

/** Settings for [LumenAi.install]. Every one has a default. */
class LumenAiConfig internal constructor() {
    /** What to index. Default: every photo and video in MediaStore (needs the media permissions). */
    var library: (suspend (Context) -> List<MediaRef>)? = null

    var searchModel: SearchModel = SearchModel.Accurate

    /** Long side, in pixels, of the video frames read for faces and text. Lower is faster. */
    var videoDetailPx: Int = 1280

    /** Frames a second read for text in short, medium and long videos ([VideoTextRates]). */
    var videoTextRates: VideoTextRates = VideoTextRates()

    /** Index without the charger too. Default: only while charging. */
    var indexOnBattery: Boolean = false

    /** Index only inside these hours, plugged in or not. Default: none. */
    var indexSchedule: IndexSchedule? = null

    /** Small icon for the download and indexing notifications. Default: Lumen AI's own. */
    @DrawableRes var notificationIcon: Int = R.drawable.lumen_ai_notification

    /** What tapping a notification opens. Default: your launcher activity. */
    var openOnTap: ((Context) -> PendingIntent)? = null

    /** For videos Android can't open by URI, such as encrypted ones. Default: none. */
    var videoSource: ((Context, MediaRef) -> MediaDataSource?)? = null

    internal fun toHost(app: Context): LumenAiHost {
        val config = this
        return object : LumenAiHost {
            override suspend fun library(context: Context) = config.library?.invoke(context) ?: MediaStoreLibrary.load(context)
            override fun videoSource(context: Context, item: MediaRef) = config.videoSource?.invoke(context, item)
            override val settings = object : AiSettings {
                override val searchModelId = config.searchModel.id
                override val videoDetailPx = config.videoDetailPx
                override val indexOnBattery = config.indexOnBattery
                override val indexSchedule = config.indexSchedule
                override val videoTextRates = config.videoTextRates
            }
            override val notifications = object : AiNotifications {
                override val smallIcon = config.notificationIcon
                override fun contentIntent(context: Context): PendingIntent = config.openOnTap?.invoke(context)
                    ?: PendingIntent.getActivity(
                        context, 0, app.packageManager.getLaunchIntentForPackage(app.packageName), PendingIntent.FLAG_IMMUTABLE,
                    )
            }
        }
    }
}

/** Every photo and video MediaStore knows, newest first: the default library. */
object MediaStoreLibrary {
    suspend fun load(context: Context): List<MediaRef> = withContext(Dispatchers.IO) {
        val projection = arrayOf(
            MediaStore.Files.FileColumns._ID,
            MediaStore.Files.FileColumns.MEDIA_TYPE,
            MediaStore.Files.FileColumns.MIME_TYPE,
            MediaStore.Files.FileColumns.DATE_TAKEN,
            MediaStore.Files.FileColumns.DATE_MODIFIED,
            MediaStore.Files.FileColumns.DURATION,
            MediaStore.Files.FileColumns.DISPLAY_NAME,
        )
        val selection = "${MediaStore.Files.FileColumns.MEDIA_TYPE} IN (?, ?)"
        val args = arrayOf(
            MediaStore.Files.FileColumns.MEDIA_TYPE_IMAGE.toString(),
            MediaStore.Files.FileColumns.MEDIA_TYPE_VIDEO.toString(),
        )
        val items = ArrayList<MediaRef>()
        try {
            context.contentResolver.query(
                MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL), projection, selection, args,
                "${MediaStore.Files.FileColumns.DATE_MODIFIED} DESC",
            )?.use { c ->
                while (c.moveToNext()) {
                    val id = c.getLong(0)
                    val video = c.getInt(1) == MediaStore.Files.FileColumns.MEDIA_TYPE_VIDEO
                    val modified = c.getLong(4)
                    val base = if (video) MediaStore.Video.Media.EXTERNAL_CONTENT_URI else MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                    items += MediaRef(
                        id = id,
                        uri = ContentUris.withAppendedId(base, id),
                        mimeType = c.getString(2) ?: "",
                        isVideo = video,
                        takenAt = c.getLong(3).takeIf { it > 0 } ?: (modified * 1000),
                        durationMs = c.getLong(5),
                        modified = modified,
                        name = c.getString(6) ?: "",
                    )
                }
            }
        } catch (e: SecurityException) {
            // No media permission yet: nothing to index until it's granted.
        }
        items
    }
}

// ---------- Models ----------

/** Where a feature's models are. */
sealed interface FeatureStatus {
    data object NotDownloaded : FeatureStatus
    /** Asked for; waits for a network, or for the feature before it to finish. */
    data object Queued : FeatureStatus
    data class Downloading(val done: Long, val total: Long) : FeatureStatus
    /** Works, but a newer version of its models can be downloaded. */
    data object UpdateAvailable : FeatureStatus
    data object Ready : FeatureStatus
}

/** Download, check and remove each feature's models ([ModelPack]). */
object Models {
    /** The running download, if any. */
    val progress: StateFlow<DownloadState> get() = ModelDownloads.state

    fun status(context: Context, feature: ModelPack): FeatureStatus {
        val running = ModelDownloads.state.value
        return when {
            running is DownloadState.Running && running.pack == feature -> FeatureStatus.Downloading(running.done, running.total)
            feature in ModelDownloads.queue(context) -> FeatureStatus.Queued
            !ModelDownloads.usable(context, feature) -> FeatureStatus.NotDownloaded
            !ModelDownloads.complete(context, feature) -> FeatureStatus.UpdateAvailable
            else -> FeatureStatus.Ready
        }
    }

    /** Bytes to download for [feature] (all of it; less if part of it is already there). */
    fun size(context: Context, feature: ModelPack): Long = ModelDownloads.remainingBytes(context, feature)

    /**
     * Downloads [feature]'s models in the background, with a progress notification, and
     * indexes the library with it afterwards. Several requests queue up.
     */
    fun download(context: Context, feature: ModelPack, wifiOnly: Boolean = true) = ModelDownloads.start(context, feature, wifiOnly)

    fun cancel(context: Context, feature: ModelPack, wifiOnly: Boolean = true) = ModelDownloads.cancel(context, feature, wifiOnly)

    /** Deletes [feature]'s models and what it found (removing Faces deletes people and their names). */
    fun remove(context: Context, feature: ModelPack, wifiOnly: Boolean = true) = ModelDownloads.delete(context, feature, wifiOnly)

    /**
     * On a phone whose NPU Lumen AI can use: fetches the NPU's files for features already
     * downloaded, on Wi-Fi, if they're missing (installed before NPU support, or a model
     * changed). Call while the app is on screen, like [download].
     */
    fun fetchNpuFiles(context: Context) = ModelDownloads.fetchNpuFiles(context)

    /** Accurate faces can run here: a supported NPU that passed its check (or hasn't run it yet). */
    fun accurateFacesAvailable(context: Context): Boolean = FaceModel.accurate(context) != null

    /** See [FaceModel.upgradeToAsk]: then ask, set [AiSettings.faceModelId] and call [fetchNpuFiles]. */
    fun faceUpgradeToAsk(context: Context): Boolean = FaceModel.upgradeToAsk(context)

    /** This phone's chip when its NPU can be used ("SM7450"), else null. */
    fun npuChip(context: Context): String? = Npu.chip(context)

    /** What ran [feature]'s main model last time: "NPU", "GPU" or "CPU"; null before it first ran. */
    fun processor(context: Context, feature: ModelPack): String? = AiState(context).processor(feature)
}

// ---------- Indexing ----------

/** How far each pass has got; null for a pass that isn't running. */
data class IndexStatus(val search: IndexProgress?, val text: IndexProgress?, val faces: IndexProgress?)

/**
 * Background indexing. It runs while the phone charges (or as set in [LumenAiConfig]), in short
 * jobs, and stops when the phone gets warm or the battery is low.
 */
object Indexing {
    /** Makes sure indexing is scheduled. Cheap; call it at app start. */
    fun start(context: Context) = SmartIndex.schedule(context)

    /** Indexes the newest items now, charging or not (a model download does this itself). */
    fun runNow(context: Context) = SmartIndex.startNow(context)

    /** Stops indexing until [start] is called again. */
    fun stop(context: Context) = SmartIndex.cancel(context)

    val progress: Flow<IndexStatus> =
        combine(SmartIndex.progress, SmartIndex.textProgress, SmartIndex.faceProgress) { s, t, f -> IndexStatus(s, t, f) }
}

// ---------- People ----------

/** The people found in the library. Needs Faces. */
object People {
    suspend fun all(context: Context, includeHidden: Boolean = false): List<Person> = withContext(Dispatchers.IO) {
        if (!FaceModel.installed(context)) return@withContext emptyList()
        FaceStore.of(context).people().filter { includeHidden || !it.hidden }
    }

    /** The people in one photo or video. */
    suspend fun inMedia(context: Context, mediaId: Long): List<Person> = all(context, includeHidden = true).filter { mediaId in it.mediaIds }

    /** A square crop of [person]'s face, for avatars. */
    fun faceImage(context: Context, person: Person): File = FaceStore.cropFile(context, person.coverFaceId)

    /** Names [person]; blank clears the name. */
    suspend fun name(context: Context, person: Person, name: String) = withContext(Dispatchers.IO) { FaceStore.of(context).rename(person.id, name) }

    /** Makes [gone] part of [keep]: for two groups that are the same person. */
    suspend fun merge(context: Context, keep: Person, gone: Person) = withContext(Dispatchers.IO) { FaceStore.of(context).merge(keep.id, gone.id) }

    suspend fun setHidden(context: Context, person: Person, hidden: Boolean) =
        withContext(Dispatchers.IO) { FaceStore.of(context).setHidden(person.id, hidden) }

    /** Uses [person]'s face in this photo or video as their picture. */
    suspend fun setCover(context: Context, person: Person, mediaId: Long) =
        withContext(Dispatchers.IO) { FaceStore.of(context).setCover(person.id, mediaId) }
}

// ---------- Search ----------

data class SearchResults(
    /** Every match, best first, each once: the list to show. */
    val items: List<Long>,
    /** People whose names match the query: show them as a row above the results. */
    val people: List<Person> = emptyList(),
    /** Items whose text (read in them) matches. */
    val textMatches: List<Long> = emptyList(),
    /** Smart Search's matches, with scores and, for videos, the moment that matched. */
    val smartMatches: List<SmartHit> = emptyList(),
    /** A person not named yet who is in most of the best matches: offer to name them. */
    val unnamedSubject: Person? = null,
) {
    companion object {
        val Empty = SearchResults(emptyList())
    }
}

internal object Search {
    suspend fun run(context: Context, query: String): SearchResults = withContext(Dispatchers.Default) {
        val q = query.trim()
        if (q.isEmpty()) return@withContext SearchResults.Empty
        val text = if (OcrModel.installed(context)) TextStore.of(context) else null

        QueryParts.phrases(q)?.let { phrases ->
            val ids = text?.searchPhrases(phrases).orEmpty()
            return@withContext SearchResults(items = ids, textMatches = ids)
        }

        val people = People.all(context)
        QueryParts.people(q, people)?.let { group ->
            val ids = group.map { it.mediaIds }.reduce { a, b -> a intersect b }.sortedDescending()
            return@withContext SearchResults(items = ids, people = group)
        }

        val terms = Terms.of(q).filter { it.length >= 2 }
        val named = people.filter { p -> p.name != null && Terms.of(p.name).any { w -> terms.any { w.startsWith(it) } } }
        val textIds = text?.search(q).orEmpty()
        val smart = SmartSearch.search(context, q)
        SearchResults(
            items = merge(textIds, smart?.hits.orEmpty().map { it.mediaId }),
            people = named,
            textMatches = textIds,
            smartMatches = smart?.hits.orEmpty(),
            unnamedSubject = smart?.unnamedSubject,
        )
    }

    /** Exact matches first, then Smart Search's, each item once. */
    fun merge(exact: List<Long>, smart: List<Long>): List<Long> = LinkedHashSet<Long>(exact.size + smart.size).apply {
        addAll(exact)
        addAll(smart)
    }.toList()

    suspend fun liveText(context: Context, item: MediaRef): LiveText.Result? {
        LiveText.cached(item)?.let { return it }
        if (!OcrModel.installed(context)) return null
        val (stored, known) = LiveText.stored(context, item)
        if (stored != null) return stored
        if (known == TextStore.Known.NoText) return null
        return LiveText.read(context, item)
    }

    suspend fun labels(context: Context, mediaId: Long, max: Int): List<String> = withContext(Dispatchers.IO) {
        val model = ModelDownloads.installed(context) ?: return@withContext emptyList()
        SearchSuggestions.looksLike(context, model, mediaId, max)
    }

    suspend fun suggestions(context: Context): List<Suggestion> = withContext(Dispatchers.IO) {
        val model = ModelDownloads.installed(context) ?: return@withContext emptyList()
        SearchSuggestions.load(context, model)
    }

    fun subtitles(context: Context, item: MediaRef, fromMs: Long): Flow<List<Cue>> = Subtitles.generate(context, item, fromMs)
}

/** These cues as an SRT subtitle file. */
fun List<Cue>.toSrt(): String = Subtitles.srt(this)

/** These cues as a WebVTT subtitle file. */
fun List<Cue>.toVtt(): String = Subtitles.vtt(this)
