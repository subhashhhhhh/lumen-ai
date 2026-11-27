package com.subhashh.lumen.ai

import com.subhashh.lumen.ai.ModelImports.Error
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption.ATOMIC_MOVE
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.security.MessageDigest
import java.util.zip.CRC32
import java.util.zip.ZipException
import java.util.zip.ZipFile

/**
 * Validates the whole selection before publishing it. Only catalog paths are used for writes.
 */
internal class ModelImporter(
    private val models: List<Target>,
    private val temporaryDirectory: File,
    private val move: (File, File) -> Unit = { from, to ->
        Files.move(from.toPath(), to.toPath(), ATOMIC_MOVE, REPLACE_EXISTING)
    },
) {
    data class Target(val pack: ModelPack, val file: ModelFile, val directory: File) {
        val destination: File get() = File(directory, file.name)
    }

    data class Document(val name: String, val open: () -> InputStream)
    data class Outcome(val targets: List<Target>, val rejection: ModelImports.Rejection? = null)

    private data class Staged(val target: Target, val file: File)
    private data class Change(val staged: Staged, val backup: File?)
    private class Invalid(val problem: Error) : IOException()

    private val byName = models.groupBy { it.file.name.substringAfterLast('/') }
    // Allow ZIP headers, including a conservative allowance for incompressible data.
    private val modelBytes = models.distinctBy { it.destination }.sumOf { it.file.bytes }
    private val archiveLimit = modelBytes + modelBytes / 100 + (1L shl 20)

    suspend fun import(documents: List<Document>): Outcome {
        if (documents.isEmpty()) return Outcome(emptyList())
        val staged = mutableListOf<Staged>()
        val destinations = mutableSetOf<File>()
        var name = documents.first().name
        try {
            for (document in documents) {
                name = document.name
                if (name.endsWith(".zip", ignoreCase = true)) {
                    document.open().use { stageArchive(it, staged, destinations) }
                } else {
                    val candidates = byName[name] ?: throw Invalid(Error.UnknownFile)
                    document.open().use { record(stage(it, candidates), staged, destinations) }
                }
            }
            publish(staged)
            return Outcome(staged.map { it.target })
        } catch (e: IOException) {
            return rejected(name, when (e) {
                is RollbackFailed -> Error.RollbackFailed
                is Invalid -> e.problem
                is ZipException -> Error.Archive
                else -> Error.ReadOrWrite
            })
        } catch (_: SecurityException) {
            return rejected(name, Error.ReadOrWrite)
        } catch (_: IllegalArgumentException) {
            return rejected(name, Error.ReadOrWrite)
        } finally {
            staged.forEach { it.file.delete() }
        }
    }

    private fun rejected(name: String, error: Error) =
        Outcome(emptyList(), ModelImports.Rejection(name.take(120), error))

    private fun record(model: Staged, staged: MutableList<Staged>, destinations: MutableSet<File>) {
        staged += model
        if (!destinations.add(model.target.destination)) throw Invalid(Error.DuplicateFile)
    }

    private suspend fun publish(staged: List<Staged>) {
        val prepared = mutableListOf<Change>()
        val published = mutableListOf<Change>()
        val recovery = mutableSetOf<File>()
        try {
            for (model in staged) {
                currentCoroutineContext().ensureActive()
                val destination = model.target.destination
                val directory = checkNotNull(destination.parentFile)
                Files.createDirectories(directory.toPath())
                val backup = if (destination.exists()) {
                    if (!destination.isFile) throw IOException("Model destination must be a file")
                    // Keep the old bytes on the same filesystem without copying the model.
                    Files.createLink(
                        File(temporaryDirectory, "${model.file.name}.backup").toPath(),
                        destination.toPath(),
                    ).toFile()
                } else null
                prepared += Change(model, backup)
            }

            currentCoroutineContext().ensureActive()
            // Finish publication or rollback even if the caller cancels during this short phase.
            for (change in prepared) {
                move(change.staged.file, change.staged.target.destination)
                published += change
            }
        } catch (failure: Exception) {
            var rollbackFailed = false
            for ((model, backup) in published.asReversed()) {
                try {
                    if (backup == null) Files.delete(model.target.destination.toPath())
                    else move(backup, model.target.destination)
                } catch (rollback: Exception) {
                    rollbackFailed = true
                    if (backup != null) recovery += backup
                    failure.addSuppressed(rollback)
                }
            }
            if (rollbackFailed) throw RollbackFailed(failure)
            throw failure
        } finally {
            prepared.mapNotNull { it.backup }.filterNot { it in recovery }.forEach(File::delete)
        }
    }

    private suspend fun stageArchive(input: InputStream, staged: MutableList<Staged>, destinations: MutableSet<File>) {
        temporaryDirectory.mkdirs()
        val archive = File.createTempFile("model-archive-", ".tmp", temporaryDirectory)
        try {
            copy(input, archive, archiveLimit, Error.Archive)
            ZipFile(archive).use { zip ->
                if (zip.size() !in 1..MaxEntries) throw Invalid(Error.Archive)
                val before = staged.size
                val paths = mutableSetOf<String>()
                val entries = zip.entries()
                while (entries.hasMoreElements()) {
                    currentCoroutineContext().ensureActive()
                    val entry = entries.nextElement()
                    val name = entry.name
                    if (!safePath(name) || !paths.add(name)) throw Invalid(Error.Archive)
                    if (entry.isDirectory) {
                        if (entry.size != 0L || zip.getInputStream(entry).use { it.read() } != -1) {
                            throw Invalid(Error.Archive)
                        }
                        continue
                    }
                    val candidates = byName[name.substringAfterLast('/')]
                        ?: throw Invalid(Error.UnknownFile)
                    val sized = candidates.filter { it.file.bytes == entry.size }
                    if (sized.isEmpty()) throw Invalid(Error.SizeMismatch)
                    zip.getInputStream(entry).use { record(stage(it, sized, entry.crc), staged, destinations) }
                }
                if (staged.size == before) throw Invalid(Error.Archive)
            }
        } finally {
            archive.delete()
        }
    }

    private suspend fun stage(input: InputStream, candidates: List<Target>, crc: Long? = null): Staged {
        val digest = MessageDigest.getInstance("SHA-256")
        val crc32 = CRC32()
        // Stage under noBackupFilesDir with the model directories so a rename preserves the
        // normal app-data ownership as well as staying on the same filesystem.
        temporaryDirectory.mkdirs()
        val file = File.createTempFile("model-", ".tmp", temporaryDirectory)
        try {
            val bytes = copy(input, file, candidates.maxOf { it.file.bytes }, Error.SizeMismatch) { buffer, length ->
                digest.update(buffer, 0, length)
                if (crc != null) crc32.update(buffer, 0, length)
            }
            if (crc != null && crc32.value != crc) throw Invalid(Error.Archive)
            val sizes = candidates.filter { it.file.bytes == bytes }
            if (sizes.isEmpty()) throw Invalid(Error.SizeMismatch)
            val hash = digest.digest().joinToString("") { "%02x".format(it) }
            val target = sizes.firstOrNull { it.file.sha256 == hash } ?: throw Invalid(Error.ChecksumMismatch)
            return Staged(target, file)
        } catch (e: Exception) {
            file.delete()
            throw e
        }
    }

    private suspend fun copy(
        input: InputStream,
        file: File,
        limit: Long,
        problem: Error,
        observe: (ByteArray, Int) -> Unit = { _, _ -> },
    ): Long {
        var bytes = 0L
        file.outputStream().use { output ->
            val buffer = ByteArray(1 shl 18)
            while (true) {
                currentCoroutineContext().ensureActive()
                val length = input.read(buffer)
                if (length < 0) break
                bytes += length
                if (bytes > limit) throw Invalid(problem)
                output.write(buffer, 0, length)
                observe(buffer, length)
            }
        }
        return bytes
    }

    companion object {
        private const val MaxEntries = 128

        internal fun safePath(name: String): Boolean =
            name.length in 1..512 && name.none { it == '\\' || it == ':' || it.isISOControl() } &&
                name.removeSuffix("/").split('/').all { it.isNotEmpty() && it != "." && it != ".." }
    }

    private class RollbackFailed(cause: Exception) : IOException(cause)
}
