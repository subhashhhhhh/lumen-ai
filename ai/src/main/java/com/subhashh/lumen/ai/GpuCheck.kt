package com.subhashh.lumen.ai

import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlinx.coroutines.delay
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Can this phone's GPU (or NPU, [Target.npu]) run a model, and run it right? Asked once per model
 * and system build.
 *
 * The check runs in its own process ([GpuCheckService], `:gpucheck`) because a GPU driver that
 * can't handle the model doesn't throw: LiteRT aborts the whole process (seen on the edge 50 with
 * an earlier export). So the check writes "running" first; if its process dies, that's all the
 * app ever finds, and it uses the CPU. A GPU that does compile must also give the CPU's answer on
 * a fixed test image (fp16 on this model compiled fine and returned wrong vectors).
 */
@LumenAiEngine
object GpuCheck {
    private const val Tag = "LumenGpu"
    private const val Timeout = 120_000L

    /**
     * A LiteRT model to check: [id] names its result file; [inputs] floats of test input. With
     * [npu], that file (compiled for this phone's NPU) is checked against [expected], the output
     * computed ahead of time, if it's there, else against [file] on the CPU.
     */
    class Target(
        val id: String,
        val file: File,
        val inputs: Int,
        val npu: File? = null,
        val expected: File? = null,
        /**
         * An image model taking raw 0–255 pixels and giving a 0–1 mask, run on the GPU in 16-bit
         * floats as it will be used: checked by how far the mask strays from the CPU's, since
         * masks are all positive and would look alike to [cosine] even when wrong.
         */
        val mask: Boolean = false,
    )

    private fun target(context: Context, model: AiModel) =
        Target(model.id, File(model.dir(context), model.spec.visualFile), 3 * model.spec.size * model.spec.size)

    private fun file(context: Context, target: Target) =
        File(context.noBackupFilesDir, if (target.npu != null) "npu-check-${target.id}.txt" else "gpu-check-${target.id}.txt")

    /** Results only hold for the system build they were taken on (drivers change with updates). */
    private val build get() = Build.FINGERPRINT

    /** True when [model] should run on the GPU. Checks on first call (up to two minutes). */
    suspend fun useGpu(context: Context, model: AiModel): Boolean =
        model.spec.visualFile.endsWith(".tflite") && useGpu(context, target(context, model))

    /** True when [model] should run on the NPU, from [npu] (its file compiled for this chip). */
    suspend fun useNpu(context: Context, model: AiModel, npu: File, expected: File): Boolean =
        useGpu(context, target(context, model).let { Target(it.id, it.file, it.inputs, npu, expected) })

    /** [id]'s NPU check ran on this system build and failed. */
    fun npuFailed(context: Context, id: String): Boolean = read(File(context.noBackupFilesDir, "npu-check-$id.txt")) == false

    /** True when [target] should run on the GPU (the NPU, with [Target.npu]). Checks on first call (up to two minutes). */
    suspend fun useGpu(context: Context, target: Target): Boolean {
        val canCompare = target.file.isFile || target.expected?.isFile == true
        if (!canCompare || target.npu?.isFile == false) return false
        val file = file(context, target)
        read(file)?.let { return it }
        file.delete()
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) = Unit
            override fun onServiceDisconnected(name: ComponentName?) = Unit
        }
        val intent = Intent(context, GpuCheckService::class.java)
            .putExtra(GpuCheckService.ExtraId, target.id)
            .putExtra(GpuCheckService.ExtraPath, target.file.path)
            .putExtra(GpuCheckService.ExtraInputs, target.inputs)
            .putExtra(GpuCheckService.ExtraNpu, target.npu?.path)
            .putExtra(GpuCheckService.ExtraExpected, target.expected?.path)
            .putExtra(GpuCheckService.ExtraMask, target.mask)
        if (!context.bindService(intent, connection, Context.BIND_AUTO_CREATE)) return false
        try {
            val start = SystemClock.elapsedRealtime()
            while (SystemClock.elapsedRealtime() - start < Timeout) {
                delay(500)
                read(file)?.let { return it }
            }
            // Never answered: the check crashed or hung. Remember, so it isn't retried every run.
            file.writeText("fail\n$build\n")
            Log.w(Tag, "${if (target.npu != null) "NPU" else "GPU"} check for ${target.id} didn't finish")
            return false
        } finally {
            runCatching { context.unbindService(connection) }
        }
    }

    /** "ok"/"fail" for this build; null if unknown or left at "running" (crashed: treated as fail). */
    private fun read(file: File): Boolean? {
        val lines = runCatching { file.readLines() }.getOrNull() ?: return null
        if (lines.getOrNull(1) != build) return null
        return when (lines.firstOrNull()) {
            "ok" -> true
            "fail" -> false
            else -> null
        }
    }

    /** Runs in the `:gpucheck` process. */
    internal fun run(context: Context, target: Target) {
        val file = file(context, target)
        file.writeText("running\n$build\n")
        val what = if (target.npu != null) "NPU" else "GPU"
        val ok = try {
            val path = target.file.path
            val input = FloatArray(target.inputs) { i -> ((i * 7919) % 2000) / 1000f - 1f }
            if (target.mask) for (i in input.indices) input[i] = (input[i] + 1f) * 127.5f
            val fast = if (target.npu != null) {
                infer(context, target.npu.path, Accelerator.NPU, input)
            } else {
                infer(context, path, Accelerator.GPU, input, fp16 = target.mask)
            }
            val cpu = target.expected?.takeIf { it.isFile }?.let(::readFloats) ?: infer(context, path, Accelerator.CPU, input)
            if (target.mask) {
                val stray = meanDifference(fast, cpu)
                Log.i(Tag, "$what check ${target.id}: mean mask difference $stray")
                stray <= 0.03f
            } else {
                val agreement = cosine(fast, cpu)
                Log.i(Tag, "$what check ${target.id}: agreement $agreement")
                agreement >= 0.99f
            }
        } catch (e: Throwable) {
            Log.w(Tag, "$what check ${target.id} failed", e)
            false
        }
        file.writeText("${if (ok) "ok" else "fail"}\n$build\n")
    }

    private fun infer(context: Context, path: String, accelerator: Accelerator, input: FloatArray, fp16: Boolean = false): FloatArray {
        val environment = if (accelerator == Accelerator.NPU) Npu.environment(context) else Environment.create()
        val options = CompiledModel.Options(accelerator).apply {
            if (accelerator == Accelerator.GPU) {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = if (fp16) CompiledModel.GpuOptions.Precision.FP16 else CompiledModel.GpuOptions.Precision.FP32,
                )
            }
        }
        val model = CompiledModel.create(path, options, environment)
        val inputs = model.createInputBuffers(0)
        val outputs = model.createOutputBuffers(0)
        try {
            inputs[0].writeFloat(input)
            model.run(inputs, outputs, 0)
            // Every output, end to end: the split line reader's second half can be the wrong one.
            val parts = outputs.map { it.readFloat() }
            return FloatArray(parts.sumOf { it.size }).also { all ->
                var at = 0
                for (part in parts) { part.copyInto(all, at); at += part.size }
            }
        } finally {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()
            environment.close()
        }
    }

    /** Little-endian floats, as numpy's tofile writes them. */
    private fun readFloats(file: File): FloatArray {
        val buffer = java.nio.ByteBuffer.wrap(file.readBytes()).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    /** Mean absolute difference; NaN (a broken driver's output) counts as everything wrong. */
    private fun meanDifference(a: FloatArray, b: FloatArray): Float {
        var sum = 0.0
        for (i in a.indices) sum += abs(a[i] - b[i]).let { if (it.isNaN()) 1f else it }
        return (sum / a.size).toFloat()
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return (dot / (sqrt(na) * sqrt(nb))).toFloat()
    }
}

/** Hosts [GpuCheck.run] in a separate process, so a GPU or NPU driver crash can't take Lumen down. */
class GpuCheckService : Service() {
    override fun onBind(intent: Intent): IBinder {
        val target = GpuCheck.Target(
            intent.getStringExtra(ExtraId)!!,
            File(intent.getStringExtra(ExtraPath)!!),
            intent.getIntExtra(ExtraInputs, 0),
            intent.getStringExtra(ExtraNpu)?.let(::File),
            intent.getStringExtra(ExtraExpected)?.let(::File),
            intent.getBooleanExtra(ExtraMask, false),
        )
        Thread { GpuCheck.run(applicationContext, target) }.start()
        return Binder()
    }

    companion object {
        const val ExtraId = "id"
        const val ExtraPath = "path"
        const val ExtraInputs = "inputs"
        const val ExtraNpu = "npu"
        const val ExtraExpected = "expected"
        const val ExtraMask = "mask"
    }
}
