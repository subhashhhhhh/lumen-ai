package com.subhashh.lumen.ai

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileNotFoundException

/** Installs locally selected model files using the same catalog as model downloads. */
@LumenAiEngine
object ModelImports {
    enum class Error { UnknownFile, SizeMismatch, ChecksumMismatch, Archive, DuplicateFile, ReadOrWrite, RollbackFailed }

    data class Rejection(val name: String, val error: Error)

    /** Number of committed files, the first rejection, and the packs confirmed ready. */
    data class Result(
        val verified: Int,
        val rejection: Rejection?,
        val ready: Set<ModelPack>,
        /** The committed files remain installed, but readiness or automatic activation failed. */
        val activationFailed: Boolean = false,
        val incomplete: Set<ModelPack> = emptySet(),
    )

    /**
     * Imports one selection of original files and ZIPs. A selection may supply part of a model
     * pack. Returns null while another operation owns the model files. The caller holds read grants.
     */
    suspend fun importFiles(context: Context, documents: List<Uri>): Result? = withContext(Dispatchers.IO) {
        if (!ModelDownloads.fileLock.tryLock()) return@withContext null
        try {
            val app = context.applicationContext
            val resolver = app.contentResolver
            val inputs = mutableListOf<ModelImporter.Document>()
            for (uri in documents.distinct()) {
                ensureActive()
                val name = documentName(resolver, uri)
                    ?: return@withContext Result(0, Rejection("?", Error.ReadOrWrite), emptySet())
                inputs += ModelImporter.Document(name) {
                    resolver.openInputStream(uri) ?: throw FileNotFoundException()
                }
            }
            val targets = ModelDownloads.downloadedFiles(app).flatMap { (pack, files, dir) ->
                files.map { ModelImporter.Target(pack, it, dir) }
            }
            val staging = File(app.noBackupFilesDir, "model-import-staging")
            val outcome = ModelImporter(targets, staging).import(inputs)
            finishImport(
                outcome,
                isReady = { target ->
                    when (target.pack) {
                        ModelPack.Search -> ModelCatalog.all.any { it.dir(app) == target.directory && it.installed(app) }
                        ModelPack.Faces -> FaceModel.ready(app)
                        else -> ModelDownloads.usable(app, target.pack)
                    }
                },
                notify = ModelDownloads::filesChanged,
                activate = { ModelDownloads.activate(app, it) },
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            Result(0, Rejection("?", Error.ReadOrWrite), emptySet())
        } finally {
            ModelDownloads.fileLock.unlock()
        }
    }

    private fun documentName(resolver: ContentResolver, uri: Uri): String? {
        if (uri.scheme != ContentResolver.SCHEME_CONTENT) return null
        return resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        }
    }

    internal fun finishImport(
        outcome: ModelImporter.Outcome,
        isReady: (ModelImporter.Target) -> Boolean,
        notify: () -> Unit,
        activate: (Set<ModelPack>) -> Unit,
    ): Result {
        val ready = mutableSetOf<ModelPack>()
        val incomplete = mutableSetOf<ModelPack>()
        val activationFailed = try {
            if (outcome.targets.isNotEmpty() || outcome.rejection?.error == Error.RollbackFailed) notify()
            for (target in outcome.targets) {
                if (isReady(target)) ready += target.pack else incomplete += target.pack
            }
            if (ready.isNotEmpty()) activate(ready)
            false
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            true
        }
        return Result(outcome.targets.size, outcome.rejection, ready, activationFailed, incomplete)
    }
}
