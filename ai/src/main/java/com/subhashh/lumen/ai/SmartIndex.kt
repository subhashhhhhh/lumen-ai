package com.subhashh.lumen.ai

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.BatteryManager
import android.provider.MediaStore
import android.util.Log
import androidx.work.OutOfQuotaPolicy
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.WorkInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.time.Duration

data class IndexProgress(val indexed: Int, val total: Int)

/**
 * Keeps Smart Search's index current, in the background, with nothing for the user to do:
 * - **Backlog** (everything not indexed yet) runs only while charging and the battery isn't low,
 *   so a first index of thousands of photos and videos never costs battery; unless the user
 *   turns on Index on Battery (Settings → Smart Search), which drops the charger rule.
 * - **New items**: MediaStore wakes Lumen shortly after a photo or video is added; up to
 *   [NewLimit] of the items added since the last such run ([arrivals]) get indexed right away,
 *   charging or not. More than that, and everything older, waits for the backlog. (It used to take
 *   the newest unindexed items of the whole library, so any change to MediaStore, a screenshot or
 *   a deletion, indexed part of the backlog on battery, outside Scheduled Indexing: issue #36.)
 * - **Scheduled Indexing** (Settings): the backlog runs only in the user's hours (overnight,
 *   say), plugged in or not. An alarm that fires in Doze ([WindowAlarm]) wakes Lumen at the window's
 *   start and end (a delayed job alone waits for Doze's next maintenance window, hours at night), and
 *   the backlog then starts as an expedited job, which Doze lets run.
 * - A backlog run makes itself a foreground service ([IndexNotification.foreground]) so Android
 *   doesn't stop it after 10 minutes. Android only allows that from the background when Lumen is
 *   exempt from battery optimization ("Unrestricted"); otherwise it runs as a plain job, as before.
 *   Only once a run has real work ([ForegroundAfterMs]): a run that finds nothing new used to
 *   flash "Getting ready" every time the charger went in. And Android 15+ allows a media
 *   processing service 6 hours a day, until Lumen is opened again; after that every start is
 *   stopped at once, so runs go on as plain jobs ([foregroundSpent]) instead of looping.
 * - **Pause** ([pause]): nothing indexes until the time given or [resume]; a run in progress
 *   stops. Items that arrive meanwhile are indexed by the first new-items run after it.
 * - Steps can be turned off one by one ([AiSettings.indexFaces] and the rest).
 * Both resume where they stopped. Scheduled at app start and when a model finishes downloading.
 */
@LumenAiEngine
object SmartIndex {
    private val mutableProgress = MutableStateFlow<IndexProgress?>(null)
    /** Latest counts from a pass (null until one runs in this process). */
    val progress: StateFlow<IndexProgress?> = mutableProgress.asStateFlow()

    private const val Backlog = "index-backlog"
    private const val New = "index-new"
    private const val Window = "index-window"
    private const val Resume = "index-resume"
    private const val KeyPausedUntil = "paused-until"
    private const val NewLimit = 30
    private const val KeyMode = "mode"
    /** A new-items run that takes the newest unindexed items, arrived or not (after a model download). */
    private const val KeyAny = "any"
    private const val KeyWindow = "window"
    private const val Tag = "SmartIndex"

    private val mutableTextProgress = MutableStateFlow<IndexProgress?>(null)
    /** The same for reading text in photos ([TextIndexer]). */
    val textProgress: StateFlow<IndexProgress?> = mutableTextProgress.asStateFlow()

    fun report(context: Context, indexed: Int, total: Int) {
        mutableProgress.value = IndexProgress(indexed, total)
        IndexNotification.update(context, IndexNotification.Pass.Vectors, indexed, total)
    }

    private val mutableFaceProgress = MutableStateFlow<IndexProgress?>(null)
    /** The same for finding faces ([FaceIndexer]). */
    val faceProgress: StateFlow<IndexProgress?> = mutableFaceProgress.asStateFlow()

    fun reportFaces(context: Context, done: Int, total: Int) {
        mutableFaceProgress.value = IndexProgress(done, total)
        IndexNotification.update(context, IndexNotification.Pass.Faces, done, total)
    }

    fun reportText(context: Context, read: Int, total: Int) {
        mutableTextProgress.value = IndexProgress(read, total)
        IndexNotification.update(context, IndexNotification.Pass.Text, read, total)
    }

    /**
     * Makes sure both jobs are queued, if a search, text or face model is installed. Safe to call often (app start). A
     * backlog job waiting out a back-off is replaced by a fresh one, so opening Lumen restarts a
     * stalled index; a running one is left alone.
     */
    fun schedule(context: Context, replace: Boolean = false) {
        val app = context.applicationContext
        scope.launch { scheduleNow(app, replace) }
    }

    private suspend fun scheduleNow(app: Context, replace: Boolean) {
        run {
            if (!anythingToIndex(app)) return
            val work = WorkManager.getInstance(app)
            if (pausedUntil(app) != 0L) {
                // Paused: no backlog. The watch for new items stays; its runs end at once and leave
                // arrivals for the first run after the pause.
                work.cancelUniqueWork(Backlog)
                watchForNew(app, ExistingWorkPolicy.KEEP)
                return
            }
            val settings = LumenAi.host.settings
            var charging = !settings.indexOnBattery
            val hours = settings.indexSchedule
            if (hours != null) {
                val inside = inWindow(hours.startMinute, hours.endMinute)
                // Wake at the window's next edge: its end while inside it, else its start.
                val edge = if (inside) hours.endMinute else hours.startMinute
                work.enqueueUniqueWork(
                    Window,
                    ExistingWorkPolicy.REPLACE,
                    OneTimeWorkRequestBuilder<WindowWorker>().setInitialDelay(untilMinute(edge)).build(),
                )
                WindowAlarm.set(app, untilMinute(edge))
                if (!inside) {
                    // Outside the hours: no backlog; new items still come in.
                    work.cancelUniqueWork(Backlog)
                    watchForNew(app, ExistingWorkPolicy.KEEP)
                    return
                }
                // Inside them: plugged in or not, as an expedited job so it runs in Doze too. Expedited
                // jobs can't require a battery level: the worker checks that itself.
                work.enqueueUniqueWork(
                    Backlog,
                    ExistingWorkPolicy.REPLACE.takeIf { replace } ?: ExistingWorkPolicy.KEEP,
                    OneTimeWorkRequestBuilder<IndexWorker>()
                        .setInputData(workDataOf(KeyMode to Backlog, KeyWindow to true))
                        .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
                        .setBackoffCriteria(BackoffPolicy.LINEAR, Duration.ofMinutes(1))
                        .build(),
                )
                watchForNew(app, ExistingWorkPolicy.KEEP)
                return
            } else {
                work.cancelUniqueWork(Window)
                WindowAlarm.cancel(app)
            }
            val waiting = work.getWorkInfosForUniqueWorkFlow(Backlog).first()
                .any { it.state == WorkInfo.State.ENQUEUED && it.runAttemptCount > 0 }
            work.enqueueUniqueWork(
                Backlog,
                if (waiting || replace) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP,
                OneTimeWorkRequestBuilder<IndexWorker>()
                    .setInputData(workDataOf(KeyMode to Backlog))
                    .setConstraints(Constraints.Builder().setRequiresCharging(charging).setRequiresBatteryNotLow(true).build())
                    // Android stops a job after ~10 minutes; a long first index carries on a minute
                    // later each time, rather than WorkManager's default doubling wait.
                    .setBackoffCriteria(BackoffPolicy.LINEAR, Duration.ofMinutes(1))
                    .build(),
            )
            watchForNew(app, ExistingWorkPolicy.KEEP)
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Each pack downloads on its own: faces or text can be on the phone without search. */
    private fun anythingToIndex(context: Context) =
        ModelDownloads.installed(context) != null || FaceModel.installed(context) || OcrModel.installed(context)

    /**
     * After a model download: index the newest [NewLimit] items right away, charging or not, so
     * search works at once. Not with Scheduled Indexing on: the user chose when indexing runs.
     */
    fun startNow(context: Context) {
        if (LumenAi.host.settings.indexSchedule == null) {
            WorkManager.getInstance(context).enqueueUniqueWork(
                New,
                ExistingWorkPolicy.REPLACE,
                OneTimeWorkRequestBuilder<IndexWorker>().setInputData(workDataOf(KeyMode to New, KeyAny to true)).build(),
            )
        }
        schedule(context)
    }

    private val mutablePaused = MutableStateFlow(0L)
    private var pauseLoaded = false

    /**
     * Until when indexing is paused, epoch milliseconds: 0 when it isn't, [Long.MAX_VALUE] until
     * [resume]. A pause that ran out reads as 0.
     */
    fun paused(context: Context): StateFlow<Long> {
        pausedUntil(context)
        return mutablePaused.asStateFlow()
    }

    private fun pausedUntil(context: Context): Long {
        val prefs = state(context)
        var until = prefs.getLong(KeyPausedUntil, 0L)
        if (until != 0L && until <= System.currentTimeMillis()) {
            prefs.edit().remove(KeyPausedUntil).apply()
            until = 0L
        }
        if (!pauseLoaded || mutablePaused.value != until) {
            mutablePaused.value = until
            pauseLoaded = true
        }
        return until
    }

    /**
     * Pauses all indexing until [untilMs] (epoch milliseconds; [Long.MAX_VALUE]: until [resume]).
     * A run in progress stops at once; what it did is kept.
     */
    fun pause(context: Context, untilMs: Long) {
        val app = context.applicationContext
        state(app).edit().putLong(KeyPausedUntil, untilMs).apply()
        mutablePaused.value = untilMs
        pauseLoaded = true
        val work = WorkManager.getInstance(app)
        work.cancelUniqueWork(Backlog)
        // Stops a new-items run in progress; the watch is queued again (its runs wait out the pause).
        work.cancelUniqueWork(New)
        watchForNew(app, ExistingWorkPolicy.REPLACE)
        if (untilMs == Long.MAX_VALUE) {
            work.cancelUniqueWork(Resume)
        } else {
            work.enqueueUniqueWork(
                Resume,
                ExistingWorkPolicy.REPLACE,
                OneTimeWorkRequestBuilder<WindowWorker>()
                    .setInitialDelay(Duration.ofMillis((untilMs - System.currentTimeMillis()).coerceAtLeast(0L)))
                    .build(),
            )
        }
        IndexNotification.cancel(app)
    }

    /** Ends a pause: indexing carries on where it stopped. */
    fun resume(context: Context) {
        val app = context.applicationContext
        state(app).edit().remove(KeyPausedUntil).apply()
        mutablePaused.value = 0L
        WorkManager.getInstance(app).cancelUniqueWork(Resume)
        schedule(app, replace = true)
    }

    /**
     * Items added since the last call: the library's ids are kept in a file, and anything not in
     * it is new. The first call only records them. Every arrival counts once, also those beyond
     * [NewLimit]: those are left to the backlog.
     */
    private suspend fun arrivals(context: Context): Set<Long> {
        val ids = Indexer.eligible(context).mapTo(HashSet()) { it.id }
        val file = File(context.noBackupFilesDir, "index-seen.bin")
        val seen = runCatching {
            DataInputStream(file.inputStream().buffered()).use { input -> HashSet<Long>().apply { repeat(input.readInt()) { add(input.readLong()) } } }
        }.getOrNull()
        val tmp = File(file.path + ".tmp")
        DataOutputStream(tmp.outputStream().buffered()).use { out ->
            out.writeInt(ids.size)
            ids.forEach { out.writeLong(it) }
        }
        tmp.renameTo(file)
        return if (seen == null) emptySet() else ids - seen
    }

    fun cancel(context: Context) {
        WorkManager.getInstance(context).apply {
            cancelUniqueWork(Backlog)
            cancelUniqueWork(New)
        }
        mutableProgress.value = null
        mutableTextProgress.value = null
        mutableFaceProgress.value = null
        IndexNotification.cancel(context)
    }

    /** A content-triggered job fires once; each run queues the next one behind itself. */
    private fun watchForNew(context: Context, policy: ExistingWorkPolicy) {
        val triggers = Constraints.Builder()
            .addContentUriTrigger(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, true)
            .addContentUriTrigger(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, true)
            // Camera bursts and downloads arrive in clumps: wait for them to settle.
            .setTriggerContentUpdateDelay(Duration.ofSeconds(15))
            .setTriggerContentMaxDelay(Duration.ofMinutes(2))
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            New,
            policy,
            OneTimeWorkRequestBuilder<IndexWorker>().setInputData(workDataOf(KeyMode to New)).setConstraints(triggers).build(),
        )
    }

    /** How long a backlog run works before it asks for a foreground service. */
    private const val ForegroundAfterMs = 30_000L
    private const val AllowanceMs = 24 * 60 * 60_000L
    private const val KeySpentAt = "foreground-spent-at"

    private fun state(context: Context) = context.getSharedPreferences("lumen-ai-index", Context.MODE_PRIVATE)

    /** Whether Android's daily foreground-service time is used up: since the last timeout, Lumen wasn't opened and a day hasn't passed. */
    private fun foregroundSpent(context: Context): Boolean {
        val at = state(context).getLong(KeySpentAt, 0L)
        return at != 0L && System.currentTimeMillis() - at < AllowanceMs
    }

    /**
     * Lumen is on screen: Android resets its foreground-service time, so the next backlog run may
     * use one again. Call from the app's activity when it starts.
     */
    fun appOpened(context: Context) {
        val prefs = state(context)
        if (prefs.contains(KeySpentAt)) prefs.edit().remove(KeySpentAt).apply()
    }

    class IndexWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
        /** An expedited run (Scheduled Indexing) needs this before Android 12. */
        override suspend fun getForegroundInfo() = IndexNotification.foreground(applicationContext)

        override suspend fun doWork(): Result {
            if (!anythingToIndex(applicationContext)) return Result.success()
            if (pausedUntil(applicationContext) != 0L) {
                if (inputData.getString(KeyMode) != Backlog) watchForNew(applicationContext, ExistingWorkPolicy.APPEND_OR_REPLACE)
                return Result.success()
            }
            val model = ModelDownloads.installed(applicationContext)
            val backlog = inputData.getString(KeyMode) == Backlog
            if (backlog && inputData.getBoolean(KeyWindow, false) && batteryLow(applicationContext)) return Result.retry()
            val limit = if (backlog) Int.MAX_VALUE else NewLimit
            // A new-items run looks only at what just arrived; with nothing new it's done.
            val only = if (backlog || inputData.getBoolean(KeyAny, false)) null else arrivals(applicationContext)
            if (only != null && only.isEmpty()) {
                watchForNew(applicationContext, ExistingWorkPolicy.APPEND_OR_REPLACE)
                return Result.success()
            }
            // One pass at a time: both jobs would otherwise work through the same items side by
            // side, each at half speed. The backlog waits its turn; a new-items job finding the
            // backlog running leaves new items to it.
            val outcome = try {
                coroutineScope {
                    // Past Android's 10-minute job limit as a foreground service, where Android
                    // allows it (Lumen exempt from battery optimization, time left today); else a
                    // plain job that resumes next run. Only once it's clearly busy: a run that
                    // finds nothing to do is over in seconds and shows nothing.
                    val promote = if (backlog && !foregroundSpent(applicationContext)) launch {
                        delay(ForegroundAfterMs)
                        runCatching { setForeground(IndexNotification.foreground(applicationContext)) }
                            .onFailure { Log.i(Tag, "Indexing as a plain job: ${it.javaClass.simpleName}") }
                    } else null
                    try {
                        when {
                            backlog -> passes.withLock {
                                // Items that arrive during a pass miss the steps already done
                                // (faces go first and list the library once): go round again.
                                var outcome: Indexer.Outcome
                                do {
                                    arrived = false
                                    outcome = pass(model, limit, null)
                                } while (arrived && outcome == Indexer.Outcome.Done)
                                outcome
                            }
                            passes.tryLock() -> try { pass(model, limit, only) } finally { passes.unlock() }
                            else -> {
                                arrived = true // the backlog picks them up when its pass ends
                                Indexer.Outcome.Partial
                            }
                        }
                    } finally {
                        promote?.cancel()
                    }
                }
            } catch (e: CancellationException) {
                if (stopReason == WorkInfo.STOP_REASON_FOREGROUND_SERVICE_TIMEOUT) {
                    // Today's foreground time is used up: carry on as plain jobs until Lumen is opened.
                    Log.i(Tag, "Foreground service time used up; indexing as plain jobs until Lumen is opened")
                    state(applicationContext).edit().putLong(KeySpentAt, System.currentTimeMillis()).apply()
                }
                throw e
            }
            if (backlog) {
                // Too warm, or the model kept failing: try again later (WorkManager backs off).
                return if (outcome == Indexer.Outcome.TooHot || outcome == Indexer.Outcome.Failing) Result.retry() else Result.success()
            }
            // New-items job: queue the next watch, and leave anything beyond the limit to the charger.
            watchForNew(applicationContext, ExistingWorkPolicy.APPEND_OR_REPLACE)
            if (outcome != Indexer.Outcome.Done) schedule(applicationContext)
            return Result.success()
        }

        /**
         * Faces first (fastest; while videos are decoded for faces, their frames are kept for
         * the next step), then Smart Search's vectors, then the words in photos. Steps whose
         * models aren't downloaded are skipped (each indexer checks its own).
         */
        private suspend fun pass(model: AiModel?, limit: Int, only: Set<Long>?): Indexer.Outcome {
            // Arrivals go through every step even if one leaves some for the backlog (faces moving
            // to another model, say); a backlog pass stops at the first step that isn't done.
            fun stop(outcome: Indexer.Outcome) = outcome == Indexer.Outcome.TooHot || outcome == Indexer.Outcome.Failing ||
                (only == null && outcome != Indexer.Outcome.Done)
            val settings = LumenAi.host.settings
            var result = Indexer.Outcome.Done
            if (settings.indexFaces) {
                val faces = FaceIndexer(applicationContext, model).run(limit, only)
                if (stop(faces)) return faces
                if (faces != Indexer.Outcome.Done) result = faces
            }
            if (model != null && settings.indexSearch) {
                val vectors = Indexer(applicationContext, model).run(limit, only)
                if (stop(vectors)) return vectors
                if (vectors != Indexer.Outcome.Done) result = vectors
                // Suggested searches follow what's in the library; refreshed once it's all indexed.
                if (only == null) SearchSuggestions.refreshIfStale(applicationContext, model)
            }
            if (!settings.indexText) return result
            val text = TextIndexer(applicationContext).run(limit, only)
            return if (text != Indexer.Outcome.Done) text else result
        }
    }

    private val passes = Mutex()

    /** New items came in while the backlog was in a pass. */
    @Volatile private var arrived = false

    /** Whether now is between [start] and [end], minutes after midnight (a window may cross it). */
    private fun inWindow(start: Int, end: Int): Boolean {
        val now = java.time.LocalTime.now().let { it.hour * 60 + it.minute }
        return if (start <= end) now in start until end else now >= start || now < end
    }

    /** From now until the next [minute] after midnight (today or tomorrow). */
    private fun untilMinute(minute: Int): Duration {
        val now = java.time.LocalDateTime.now()
        var at = now.toLocalDate().atStartOfDay().plusMinutes(minute.toLong())
        if (!at.isAfter(now)) at = at.plusDays(1)
        return Duration.between(now, at)
    }

    /** Wakes at a Scheduled Indexing edge: starts or stops the backlog, and arms the next edge. */
    /** What [Constraints.Builder.setRequiresBatteryNotLow] means, for jobs that can't have it: under 15%, not charging. */
    private fun batteryLow(context: Context): Boolean {
        val battery = context.getSystemService(BatteryManager::class.java)
        return !battery.isCharging && battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) in 0..14
    }

    /**
     * Wakes Lumen at a Scheduled Indexing window's start or end, also in Doze (inexact: within a few
     * minutes). No exact-alarm permission needed.
     */
    class WindowAlarm : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val pending = goAsync()
            scope.launch {
                try { scheduleNow(context.applicationContext, replace = true) } finally { pending.finish() }
            }
        }

        companion object {
            private fun intent(context: Context) =
                PendingIntent.getBroadcast(context, 0, Intent(context, WindowAlarm::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)

            fun set(context: Context, after: Duration) {
                context.getSystemService(AlarmManager::class.java)
                    .setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, System.currentTimeMillis() + after.toMillis(), intent(context))
            }

            fun cancel(context: Context) = context.getSystemService(AlarmManager::class.java).cancel(intent(context))
        }
    }

    class WindowWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
        override suspend fun doWork(): Result {
            schedule(applicationContext, replace = true)
            return Result.success()
        }
    }
}
