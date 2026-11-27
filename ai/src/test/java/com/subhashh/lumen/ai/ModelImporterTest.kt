package com.subhashh.lumen.ai

import com.subhashh.lumen.ai.ModelImports.Error
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption.ATOMIC_MOVE
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

@OptIn(LumenAiEngine::class)
class ModelImporterTest {
    @get:Rule val folder = TemporaryFolder()

    private val detector = byteArrayOf(1, 2, 3, 4)
    private val recognizer = byteArrayOf(9, 8, 7)

    private fun target(
        name: String,
        bytes: ByteArray,
        pack: ModelPack = ModelPack.Faces,
        directory: String = "faces",
    ): ModelImporter.Target {
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        return ModelImporter.Target(pack, ModelFile(name, bytes.size.toLong(), hash), File(folder.root, directory))
    }

    private fun importer(vararg targets: ModelImporter.Target) =
        ModelImporter(targets.toList(), File(folder.root, "staging"))

    private fun document(name: String, bytes: ByteArray) = ModelImporter.Document(name) { bytes.inputStream() }

    private suspend fun ModelImporter.import(name: String, bytes: ByteArray) = import(listOf(document(name, bytes)))

    @Test fun installsOneModelAndKeepsAnIncompletePackPartial() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val recognize = target("edgeface_s.onnx", recognizer)

        val outcome = importer(detect, recognize).import("yunet.onnx", detector)
        assertEquals(listOf(detect), outcome.targets)
        assertNull(outcome.rejection)
        assertArrayEquals(detector, detect.destination.readBytes())
        assertFalse(recognize.destination.exists())
        assertTrue(File(folder.root, "staging").listFiles().orEmpty().isEmpty())
    }

    @Test fun corruptFileWithCorrectLengthDoesNotReplaceExistingModel() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val importer = importer(detect)
        assertNull(importer.import("yunet.onnx", detector).rejection)

        val failed = importer.import("yunet.onnx", byteArrayOf(0, 2, 3, 4))
        assertEquals(Error.ChecksumMismatch, failed.rejection?.error)
        assertArrayEquals(detector, detect.destination.readBytes())
    }

    @Test fun verifiedFileRepairsAnExistingSameLengthCorruption() = runBlocking {
        val detect = target("yunet.onnx", detector)
        detect.destination.parentFile!!.mkdirs()
        detect.destination.writeBytes(byteArrayOf(0, 2, 3, 4))

        assertNull(importer(detect).import("yunet.onnx", detector).rejection)
        assertArrayEquals(detector, detect.destination.readBytes())
    }

    @Test fun zipWithOriginalModelNamesIsAccepted() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val recognize = target("SM8650/adaface_ir101.tflite", recognizer)
        val zip = archive("faces/yunet.onnx" to detector, "SM8650/adaface_ir101.tflite" to recognizer)

        val outcome = importer(detect, recognize).import("models.ZIP", zip)
        assertNull(outcome.rejection)
        assertEquals(listOf(detect, recognize), outcome.targets)
        assertArrayEquals(recognizer, recognize.destination.readBytes())
    }

    @Test fun rejectsEntireZipWhenALaterEntryHasAnUnsafePath() = runBlocking {
        val detect = target("yunet.onnx", detector)
        for (path in listOf("../yunet.onnx", "/yunet.onnx", "a/../../yunet.onnx", "a\\..\\yunet.onnx", "C:/yunet.onnx")) {
            val zip = archive("faces/yunet.onnx" to detector, path to detector)
            assertEquals(path, Error.Archive, importer(detect).import("faces.zip", zip).rejection?.error)
        }
        assertFalse(detect.destination.exists())
        assertFalse(File(folder.root, "yunet.onnx").exists())
    }

    @Test fun rejectsZipWithDuplicateLogicalDestinationOrMissingCentralDirectory() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val importer = importer(detect)
        val valid = archive("yunet.onnx" to detector)
        val duplicate = archive("a/yunet.onnx" to detector, "b/yunet.onnx" to detector)
        assertEquals(Error.DuplicateFile, importer.import("models.zip", duplicate).rejection?.error)
        assertEquals(Error.Archive, importer.import("models.zip", valid.copyOf(valid.size - 22)).rejection?.error)
        assertFalse(detect.destination.exists())
    }

    @Test fun rejectsFileExceedingCatalogSizeAndUnrecognizedFileName() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val importer = importer(detect)
        for (bad in listOf(byteArrayOf(1), detector + detector)) {
            assertEquals(Error.SizeMismatch, importer.import("yunet.onnx", bad).rejection?.error)
        }
        val unrecognized = importer.import(listOf(ModelImporter.Document("notes.txt") { error("Unexpected stream read") }))
        assertEquals(Error.UnknownFile, unrecognized.rejection?.error)
        assertFalse(detect.destination.exists())
        assertTrue(File(folder.root, "staging").listFiles().orEmpty().isEmpty())
    }

    @Test fun identifiesSameNameCatalogEntriesByHash() = runBlocking {
        val first = target("visual.onnx", byteArrayOf(4, 3, 2, 1), directory = "fast")
        val second = target("chip/visual.onnx", detector, directory = "accurate")
        val outcome = importer(first, second).import("visual.onnx", detector)
        assertEquals(listOf(second), outcome.targets)
        assertFalse(first.destination.exists())
        assertArrayEquals(detector, second.destination.readBytes())
    }

    @Test fun installsMixedSelectionAndRejectsDuplicateDestinationsAcrossDocuments() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val recognize = target("recognizer.onnx", recognizer)
        val first = document("yunet.onnx", detector)
        val zipped = document("model.zip", archive("recognizer.onnx" to recognizer))
        val importer = importer(detect, recognize)

        val failure = importer.import(listOf(first, zipped, document("recognizer.onnx", recognizer)))
        assertEquals(Error.DuplicateFile, failure.rejection?.error)
        assertFalse(detect.destination.exists())
        assertFalse(recognize.destination.exists())
        val success = importer.import(listOf(first, zipped))
        assertEquals(listOf(detect, recognize), success.targets)
        assertArrayEquals(detector, detect.destination.readBytes())
        assertArrayEquals(recognizer, recognize.destination.readBytes())
    }

    @Test fun laterZipChecksumFailureLeavesEarlierSelectionUnpublished() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val recognize = target("recognizer.onnx", recognizer)
        val result = importer(detect, recognize).import(listOf(
            document("yunet.onnx", detector),
            document("models.zip", archive("recognizer.onnx" to byteArrayOf(0, 8, 7))),
        ))
        assertEquals(Error.ChecksumMismatch, result.rejection?.error)
        assertTrue(result.targets.isEmpty())
        assertFalse(detect.destination.exists())
        assertFalse(recognize.destination.exists())
    }

    @Test fun writeFailureRestoresReplacedFileAndRemovesNewFile() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val recognize = target("recognizer.onnx", recognizer)
        val blocked = target("other.onnx", detector)
        val old = byteArrayOf(5, 5, 5, 5)
        detect.destination.parentFile!!.mkdirs()
        detect.destination.writeBytes(old)
        val staging = folder.newFolder()
        val importer = ModelImporter(listOf(detect, recognize, blocked), staging) { source, destination ->
            if (destination == blocked.destination) throw IOException("Simulated publish failure")
            Files.move(source.toPath(), destination.toPath(), ATOMIC_MOVE, REPLACE_EXISTING)
        }
        val outcome = importer.import("models.zip", archive(
            "yunet.onnx" to detector, "recognizer.onnx" to recognizer, "other.onnx" to detector,
        ))
        assertEquals(Error.ReadOrWrite, outcome.rejection?.error)
        assertTrue(outcome.targets.isEmpty())
        assertArrayEquals(old, detect.destination.readBytes())
        assertFalse(recognize.destination.exists())
        assertFalse(blocked.destination.exists())
        assertTrue(staging.listFiles().orEmpty().isEmpty())
    }

    @Test fun refreshesPartialPacksAndStartsIndexingOnceTheSelectionCompletesThem() = runBlocking {
        val detect = target("yunet.onnx", detector)
        val recognize = target("recognizer.onnx", recognizer)
        val text = target("ocr.onnx", detector, ModelPack.Text, "text")
        val cutout = target("cutout.tflite", detector, ModelPack.Cutout, "cutout")
        val subtitles = target("speech.bin", detector, ModelPack.Subtitles, "speech")
        val importer = importer(detect, recognize, text, cutout, subtitles)
        var refreshes = 0
        var indexing = 0
        var warmups = 0
        fun finish(outcome: ModelImporter.Outcome) = ModelImports.finishImport(
            outcome,
            isReady = { target -> if (target.pack == ModelPack.Faces) recognize.destination.isFile else target.destination.isFile },
            notify = { refreshes++ },
            activate = { activateModelPacks(it, { warmups++ }, { indexing++ }) },
        )

        assertTrue(finish(importer.import("yunet.onnx", detector)).ready.isEmpty())
        assertEquals(1, refreshes)
        assertEquals(0, indexing)
        assertTrue(finish(importer.import("recognizer.onnx", detector)).ready.isEmpty())
        assertEquals(1, refreshes)
        assertEquals(0, indexing)
        assertEquals(setOf(ModelPack.Cutout, ModelPack.Subtitles), finish(importer.import(listOf(
            document("cutout.tflite", detector), document("speech.bin", detector),
        ))).ready)
        assertEquals(1, warmups)
        assertEquals(0, indexing)
        val ready = finish(importer.import(listOf(document("recognizer.onnx", recognizer), document("ocr.onnx", detector))))
        assertEquals(setOf(ModelPack.Faces, ModelPack.Text), ready.ready)
        assertEquals(3, refreshes)
        assertEquals(1, indexing)
    }

    private fun archive(vararg entries: Pair<String, ByteArray>): ByteArray {
        val result = ByteArrayOutputStream()
        ZipOutputStream(result).use { output ->
            for ((path, content) in entries) {
                output.putNextEntry(ZipEntry(path))
                output.write(content)
                output.closeEntry()
            }
        }
        return result.toByteArray()
    }
}
