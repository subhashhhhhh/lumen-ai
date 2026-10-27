package com.subhashh.lumen.ai

import android.content.Context
import android.os.Build
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Qualcomm's NPU (Hexagon), through LiteRT. Models are compiled ahead of time for each chip
 * ([NpuModels]), so nothing compiles on the phone. Needs Android 12 (LiteRT's NPU path and
 * [Build.SOC_MODEL]) and Qualcomm's runtime libraries in the app, which only the GitHub build
 * ships: elsewhere [chip] is null and everything runs as before.
 */
@LumenAiEngine
object Npu {
    /** This phone's chip ("SM7450") when there are models for it and the runtime is in the app. */
    fun chip(context: Context): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S || Build.SOC_MANUFACTURER != "QTI") return null
        val chip = Build.SOC_MODEL.takeIf { it in NpuModels.chips } ?: return null
        return chip.takeIf { File(context.applicationInfo.nativeLibraryDir, DispatchLibrary).isFile }
    }

    /**
     * LiteRT with the NPU. The DSP loads its half of Qualcomm's runtime (the "skel" library) from
     * a search path, which must include the app's own library folder.
     */
    fun environment(context: Context): Environment {
        val libs = context.applicationInfo.nativeLibraryDir
        android.system.Os.setenv("ADSP_LIBRARY_PATH", "$libs;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp", true)
        return Environment.create(context, BuiltinNpuAcceleratorProvider(context))
    }

    /**
     * [model]'s image model compiled for this phone's NPU, and the reference it's checked
     * against; null where the NPU can't run it or failed its check (then the standard one is used).
     */
    fun imageFiles(context: Context, model: AiModel): List<ModelFile>? {
        if (model != ModelCatalog.Accurate || GpuCheck.npuFailed(context, model.id)) return null
        val compiled = chip(context)?.let(NpuModels::siglip) ?: return null
        return listOf(compiled, NpuModels.Reference)
    }

    /**
     * The search model's NPU file, when it's downloaded and matched the reference in [GpuCheck].
     * Then the standard image model isn't needed any more, and goes (369 MB for Accurate).
     */
    suspend fun searchModel(context: Context, model: AiModel): NpuFile? {
        val wanted = imageFiles(context, model)?.first() ?: return null
        val file = File(model.dir(context), wanted.name)
        val reference = File(model.dir(context), NpuModels.Reference.name)
        if (file.length() != wanted.bytes || !GpuCheck.useNpu(context, model, file, reference)) return null
        File(model.dir(context), model.spec.visualFile).delete()
        return NpuFile(context, file)
    }

    /** Text detection's NPU file, likewise. */
    suspend fun textDetector(context: Context): NpuFile? {
        val wanted = chip(context)?.let(NpuModels::textDetector) ?: return null
        val file = File(OcrModel.dir(context), wanted.name)
        if (file.length() != wanted.bytes) return null
        val target = GpuCheck.Target(OcrModel.DetId, File(OcrModel.dir(context), OcrModel.DetGpu), OcrModel.DetInputs, npu = file)
        return if (GpuCheck.useGpu(context, target)) NpuFile(context, file) else null
    }

    /** Fastest clocks while running: about twice the default's speed (98 → 46 ms for SigLIP 2). */
    val fast get() = CompiledModel.QualcommOptions(htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE)

    private const val DispatchLibrary = "libLiteRtDispatch_Qualcomm.so"
}

/** A model [file] compiled for this phone's NPU, with the [context] LiteRT's NPU path needs. */
class NpuFile(val context: Context, val file: File)
