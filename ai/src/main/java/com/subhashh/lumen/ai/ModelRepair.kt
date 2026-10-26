package com.subhashh.lumen.ai

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Models are checked against their SHA-256 when they download, and trusted by size after that.
 * If one fails to load, a file may have been damaged on disk since: every downloaded pack's files
 * are then hashed again, and any that don't match are deleted. The feature then shows as needing
 * a download in Settings, instead of failing each time it's used. Hashing all of them takes a few
 * seconds, so it runs in the background, at most once per process.
 */
internal object ModelRepair {
    private const val Tag = "LumenModels"
    private val checked = AtomicBoolean(false)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Runs [load]; if it throws, checks the model files in the background and rethrows. */
    inline fun <T> loading(context: Context, load: () -> T): T = try {
        load()
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        failed(context, e)
        throw e
    }

    /** A model didn't load: check every downloaded file once, in the background. */
    fun failed(context: Context, cause: Throwable? = null) {
        Log.w(Tag, "A model didn't load; checking the downloaded files", cause)
        if (!checked.compareAndSet(false, true)) return
        val app = context.applicationContext
        scope.launch {
            for ((pack, files, dir) in ModelDownloads.downloadedFiles(app)) {
                for (file in files) {
                    val path = File(dir, file.name)
                    if (path.length() != file.bytes) continue
                    if (ModelDownloader.sha256(path) != file.sha256 && path.delete()) {
                        Log.w(Tag, "${pack.id}: ${file.name} was damaged; deleted, download it again")
                    }
                }
            }
        }
    }
}
