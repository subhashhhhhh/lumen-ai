package com.subhashh.lumen.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Finds the faces in a photo and gives each a fingerprint ([FaceModel]), in three steps:
 * 1. **YuNet** marks faces: a box, a confidence and 5 landmarks (eyes, nose tip, mouth corners).
 * 2. Each face is **straightened** onto a 112×112 template from its landmarks (the rotation, scale
 *    and shift that fit them best), the way the recognizer was trained.
 * 3. The **recognizer** ([FaceModel.recognizer]) turns it into 512 numbers; the same person's
 *    faces point the same way.
 *
 * Same steps as tools/faces/face_crops.py and face_embed.py, which check them on the Mac.
 */
@LumenAiEngine
class FaceReader(context: Context, dir: File, threads: Int, val recognizer: FaceRecognizer) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    private val det = env.createSession(File(dir, "yunet.onnx").path, options)
    private var argb = IntArray(0)
    private var detInput = FloatArray(0)
    private var frameW = 0
    private var frameH = 0
    private val rec = if (recognizer.npu) null else env.createSession(File(dir, recognizer.file.name).path, options)
    private val npu = if (recognizer.npu) NpuRecognizer(context, File(dir, recognizer.file.name)) else null

    /**
     * Time spent per step, in ns, when set (Search Lab's profile); null in normal use. Keys:
     * prepare, detect, decodeOutput, straighten, embedPrepare, embed.
     */
    var timings: MutableMap<String, Long>? = null

    private inline fun <T> timed(step: String, block: () -> T): T {
        val t = timings ?: return block()
        val start = System.nanoTime()
        return block().also { t[step] = (t[step] ?: 0L) + (System.nanoTime() - start) }
    }

    /** A face: its box as fractions of the photo ([left, top, right, bottom]) and fingerprint. */
    class Found(val box: FloatArray, val vector: FloatArray, val quality: Float)

    /** Every face at least [minScore] sure (lower to find one again that was found before). */
    fun read(original: Bitmap, minScore: Float = MinScore): List<Found> {
        val image = if (original.config == Bitmap.Config.HARDWARE) original.copy(Bitmap.Config.ARGB_8888, false) else original
        try {
            return detect(image, minScore).map { (box, landmarks) ->
                val face = timed("straighten") { straighten(image, landmarks) }
                val raw = embed(face)
                face.recycle()
                val length = sqrt(raw.fold(0f) { acc, x -> acc + x * x })
                Found(
                    floatArrayOf(box[0] / image.width, box[1] / image.height, box[2] / image.width, box[3] / image.height),
                    normalize(raw),
                    length,
                )
            }
        } finally {
            if (image !== original) image.recycle()
        }
    }

    /** Boxes in pixels and their 5 landmarks (x0, y0, … x4, y4). */
    private fun detect(image: Bitmap, minScore: Float): List<Pair<FloatArray, FloatArray>> {
        val w = image.width
        val h = image.height
        val pw = (w + 31) / 32 * 32
        val ph = (h + 31) / 32 * 32
        val plane = pw * ph
        val input = timed("prepare") {
            // Reused between frames of one size: a 1280 frame is ~11 MB of floats.
            if (argb.size != w * h) argb = IntArray(w * h)
            if (detInput.size != 3 * plane || frameW != w || frameH != h) {
                detInput = FloatArray(3 * plane) // The padding stays black (0) from here on.
                frameW = w
                frameH = h
            }
            val input = detInput
            image.getPixels(argb, 0, w, 0, 0, w, h)
            // BGR, raw 0–255, padded right and bottom with black.
            var src = 0
            for (y in 0 until h) {
                var i = y * pw
                for (x in 0 until w) {
                    val p = argb[src++]
                    input[i] = (p and 0xFF).toFloat()
                    input[plane + i] = (p shr 8 and 0xFF).toFloat()
                    input[2 * plane + i] = (p shr 16 and 0xFF).toFloat()
                    i++
                }
            }
            input
        }
        val candidates = ArrayList<Candidate>()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, ph.toLong(), pw.toLong())).use { x ->
            timed("detect") { det.run(mapOf("input" to x)) }.use { result -> timed("decodeOutput") {
                fun out(name: String): FloatBuffer = (result.get(name).get() as OnnxTensor).floatBuffer
                for (stride in intArrayOf(8, 16, 32)) {
                    val cols = pw / stride
                    val cells = cols * (ph / stride)
                    val cls = out("cls_$stride")
                    val obj = out("obj_$stride")
                    val bbox = out("bbox_$stride")
                    val kps = out("kps_$stride")
                    for (i in 0 until cells) {
                        val score = sqrt(cls[i].coerceIn(0f, 1f) * obj[i].coerceIn(0f, 1f))
                        if (score < minScore) continue
                        val r = i / cols
                        val c = i % cols
                        val cx = (c + bbox[4 * i]) * stride
                        val cy = (r + bbox[4 * i + 1]) * stride
                        val bw = exp(bbox[4 * i + 2]) * stride
                        val bh = exp(bbox[4 * i + 3]) * stride
                        val landmarks = FloatArray(10) { k ->
                            if (k % 2 == 0) (kps[10 * i + k] + c) * stride else (kps[10 * i + k] + r) * stride
                        }
                        candidates += Candidate(floatArrayOf(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), score, landmarks)
                    }
                }
            } }
        }
        // Overlapping boxes for one face: keep the most confident.
        val kept = ArrayList<Candidate>()
        for (c in candidates.sortedByDescending { it.score }) {
            if (kept.none { iou(it.box, c.box) > MaxOverlap }) kept += c
        }
        return kept.filter { min(it.box[2] - it.box[0], it.box[3] - it.box[1]) >= MinFacePx }.map { it.box to it.landmarks }
    }

    private class Candidate(val box: FloatArray, val score: Float, val landmarks: FloatArray)

    /**
     * The rotation, uniform scale and shift that best map the landmarks onto [Template] (least
     * squares), applied with a canvas: a 112×112 face, eyes level.
     */
    private fun straighten(image: Bitmap, landmarks: FloatArray): Bitmap {
        var mx = 0f; var my = 0f; var tx = 0f; var ty = 0f
        for (k in 0 until 5) {
            mx += landmarks[2 * k]; my += landmarks[2 * k + 1]
            tx += Template[2 * k]; ty += Template[2 * k + 1]
        }
        mx /= 5; my /= 5; tx /= 5; ty /= 5
        var dot = 0f; var cross = 0f; var norm = 0f
        for (k in 0 until 5) {
            val px = landmarks[2 * k] - mx
            val py = landmarks[2 * k + 1] - my
            val qx = Template[2 * k] - tx
            val qy = Template[2 * k + 1] - ty
            dot += px * qx + py * qy
            cross += px * qy - py * qx
            norm += px * px + py * py
        }
        // [a −b; b a] is rotation × scale.
        val a = dot / norm
        val b = cross / norm
        val matrix = Matrix().apply {
            setValues(floatArrayOf(a, -b, tx - (a * mx - b * my), b, a, ty - (b * mx + a * my), 0f, 0f, 1f))
        }
        return Bitmap.createBitmap(FaceSize, FaceSize, Bitmap.Config.ARGB_8888).also {
            Canvas(it).drawBitmap(image, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
        }
    }

    /** RGB, raw 0–255 (channels first for ONNX, last for the NPU); the recognizer's raw numbers out. */
    private fun embed(face: Bitmap): FloatArray {
        val plane = FaceSize * FaceSize
        val nhwc = recognizer.npu
        val input = timed("embedPrepare") {
            val argb = IntArray(plane)
            face.getPixels(argb, 0, FaceSize, 0, 0, FaceSize, FaceSize)
            val input = FloatArray(3 * plane)
            for (i in 0 until plane) {
                val p = argb[i]
                val r = (p shr 16 and 0xFF).toFloat()
                val g = (p shr 8 and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                if (nhwc) {
                    input[3 * i] = r; input[3 * i + 1] = g; input[3 * i + 2] = b
                } else {
                    input[i] = r; input[plane + i] = g; input[2 * plane + i] = b
                }
            }
            input
        }
        npu?.let { return timed("embed") { it.run(input) } }
        val rec = checkNotNull(rec)
        val shape = longArrayOf(1, 3, FaceSize.toLong(), FaceSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { x ->
            timed("embed") { rec.run(mapOf(rec.inputNames.first() to x)) }.use { result ->
                val out = (result[0] as OnnxTensor).floatBuffer
                FloatArray(out.remaining()).also { out.get(it) }
            }
        }
    }

    override fun close() {
        det.close()
        rec?.close()
        npu?.close()
        options.close()
    }

    /** The recognizer compiled for this phone's NPU, in its fastest mode. */
    private class NpuRecognizer(context: Context, file: File) : AutoCloseable {
        private val env = Npu.environment(context)
        private val model = CompiledModel.create(file.path, CompiledModel.Options(Accelerator.NPU).apply { qualcommOptions = Npu.fast }, env)
        private val inputs = model.createInputBuffers(0)
        private val outputs = model.createOutputBuffers(0)

        fun run(input: FloatArray): FloatArray {
            inputs[0].writeFloat(input)
            model.run(inputs, outputs, 0)
            return outputs[0].readFloat()
        }

        override fun close() {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()
            env.close()
        }
    }

    companion object {
        /** Long side photos are looked at: faces of [MinFacePx] still show up in group shots. */
        const val ReadSide = 1280
        private const val MinScore = 0.8f
        private const val MaxOverlap = 0.3f
        /** Smaller faces are too blurry to tell apart reliably. */
        private const val MinFacePx = 40f
        private const val FaceSize = 112
        /** Where eyes, nose tip and mouth corners sit on a 112×112 face (ArcFace's template). */
        private val Template = floatArrayOf(
            38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
        )

        private fun iou(a: FloatArray, b: FloatArray): Float {
            val w = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
            val h = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
            val inter = w * h
            return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9f)
        }
    }
}
