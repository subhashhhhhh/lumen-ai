package com.subhashh.lumen.ai

import android.content.Context
import java.io.File

/** [optional]: added in a later version; the model works without it until it's downloaded. */
data class ModelFile(val name: String, val bytes: Long, val sha256: String, val optional: Boolean = false)

/**
 * A downloadable model, pinned to one commit of its Hugging Face repo so the files can't change
 * under us; each file's SHA-256 is checked before it's used.
 */
data class AiModel(
    val id: String,
    /** What users see: the tier, never the model's name. */
    val label: String,
    val repo: String,
    val commit: String,
    val files: List<ModelFile>,
    /** How to run it once downloaded. */
    val spec: OnnxSpec,
) {
    val bytes: Long get() = files.sumOf { it.bytes }

    fun url(file: ModelFile) = "https://huggingface.co/$repo/resolve/$commit/${file.name}"

    /** App-private, never backed up: the files are large and re-downloadable. */
    fun dir(context: Context) = File(context.noBackupFilesDir, "models/$id")

    /** It works: every required file is in place. */
    fun installed(context: Context): Boolean =
        files.filterNot { it.optional }.all { File(dir(context), it.name).length() == it.bytes }

    /** Every file of this version is in place; otherwise Settings offers the rest as an update. */
    fun complete(context: Context): Boolean = files.all { File(dir(context), it.name).length() == it.bytes }
}

/**
 * Smart Search's quality tiers. Chosen in the Search Lab on the owner's photos; EmbeddingGemma 2,
 * MobileCLIP2 and heavier models (PE-Core, SigLIP 2 So400m, Qwen3-VL-Embedding) were tested or
 * researched and left out. Conversions: tools/models/. Hosting: huggingface.co/subhashhhhhh.
 */
@LumenAiEngine
object ModelCatalog {
    /** Google's SigLIP 2 B/16 (Apache 2.0): fp32 image encoder (LiteRT, GPU-ready), int8 text, packed tokenizer, Looks like vocabulary. */
    val Accurate = AiModel(
        id = "siglip2-b16",
        label = "Accurate",
        repo = "subhashhhhhh/lumen-siglip2-b16",
        commit = "a04c26dcc208923ac1c0c4e7126be041ff65ce49",
        files = listOf(
            ModelFile("visual.tflite", 369_330_540, "b358184561275f176135257853e9927dd5fa0ca6e9cd3f52c63572d5a9740951"),
            ModelFile("textual_int8.onnx", 283_187_112, "6fdbedc85c54262bcbf380cfb048f03e10cf07237333fe4ca1e204a8f954d608"),
            ModelFile("gemma_bpe.bin", 7_134_060, "e9163029a76e66dadb07afd9373531a24687e25d7a027f65d13a2977331f53c2"),
            ModelFile("vocabulary.bin", 6_536_435, "96070e6bdc36e01255b0501d6259c0482e9e7d85faafb4c0d949f871b8def1bf", optional = true),
        ),
        spec = OnnxEmbedder.Siglip,
    )

    /** OpenCLIP ViT-B/32 LAION-2B (MIT), int8: about 8x faster to index, English only. */
    val Fast = AiModel(
        id = "clip-vit-b32",
        label = "Fast",
        repo = "subhashhhhhh/lumen-clip-vit-b32",
        commit = "5eca4f3a8eb045c138d7dbadc9a478735a279fbd",
        files = listOf(
            ModelFile("visual_int8.onnx", 95_777_877, "17755a1228aee44f4274eadac14a600a5e45ab9f6325b6ad070b5d0d75e05073"),
            ModelFile("textual_int8.onnx", 140_296_927, "e1631f9b00b2a14e5b268a9c73000b1eec3c79bd7d4876ce10fe4f0de1a03b1e"),
            ModelFile("vocab.json", 862_328, "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"),
            ModelFile("merges.txt", 524_619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
            ModelFile("vocabulary.bin", 4_372_211, "0f0ab53ade495ed6dbaa0d238bd1a4f7c3b2e6970309f174ad0c02add9abbcac", optional = true),
        ),
        spec = OnnxEmbedder.Clip,
    )

    val all = listOf(Accurate, Fast)

    fun byId(id: String?) = all.firstOrNull { it.id == id } ?: Accurate
}

/**
 * Reading text in photos: Baidu's PP-OCRv6 small (Apache 2.0), text detection and recognition,
 * 50 languages (Chinese, English, Japanese, Latin scripts), plus our GPU conversion of the detector
 * (tools/ocr/; the repo also has GPU line readers, tried and left out: no faster on real lines). Downloaded with either Smart Search tier. `rec.yml` carries the recognizer's
 * character list.
 */
@LumenAiEngine
object OcrModel {
    const val Repo = "subhashhhhhh/lumen-ppocrv6-small"
    const val Commit = "891518718e7870b86c9e4859404a149c5028e4de"

    /** Detection for LiteRT's GPU, 960×960; used where [GpuCheck] passes. */
    const val DetGpu = "det_gpu.tflite"
    /** [GpuCheck]'s name for the detector, and its input: a 960×960 RGB square. */
    internal const val DetId = "ppocrv6-det"
    internal const val DetInputs = 960 * 960 * 3

    private val det = ModelFile("det.onnx", 9_880_512, "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e")
    /** The line reader with int8 convolutions (tools/ocr/quantize_rec.py): ~1.45× faster, same words. */
    private val rec = ModelFile("rec_int8.onnx", 13_022_934, "67c0300e647071eac9a0b56de2a1a17dcf04928cab6e3c861049c677d267e686")
    /** The original line reader, from earlier downloads: read with until [rec] arrives. */
    private val oldRec = ModelFile("rec.onnx", 21_159_378, "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634")
    private val chars = ModelFile("rec.yml", 150_579, "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1")

    val files = listOf(
        det,
        rec,
        chars,
        ModelFile(DetGpu, 9_869_976, "8a7676cbc5a01ca47544bf95af2196c8b1b6c7b75beec9ad9b17857dffb71d1e"),
    )
    val bytes: Long get() = files.sumOf { it.bytes }

    fun url(file: ModelFile) = "https://huggingface.co/$Repo/resolve/$Commit/${file.name}"

    fun dir(context: Context) = File(context.noBackupFilesDir, "models/ppocrv6-small")

    private fun inPlace(context: Context, file: ModelFile) = File(dir(context), file.name).length() == file.bytes

    /** The line reader to use: the int8 one once downloaded, else an earlier download's. */
    fun recFile(context: Context): File = File(dir(context), if (inPlace(context, rec) || !inPlace(context, oldRec)) rec.name else oldRec.name)

    /** Text can be read; an earlier download, missing this version's newer files, still reads. */
    fun installed(context: Context): Boolean =
        inPlace(context, det) && inPlace(context, chars) && (inPlace(context, rec) || inPlace(context, oldRec))

    /** Every file of this version is in place; otherwise Settings offers the rest as an update. */
    fun complete(context: Context): Boolean = files.all { inPlace(context, it) }

    /** Deletes files this version doesn't use: GPU line readers, and the old reader once replaced. */
    fun removeUnused(context: Context) {
        val wanted = files.flatMapTo(HashSet()) { listOf(it.name, it.name + ".part") }
        if (!inPlace(context, rec)) wanted += oldRec.name
        // The NPU's detector, in a folder named for the chip.
        Npu.chip(context)?.let { wanted += it }
        dir(context).listFiles()?.filter { it.name !in wanted }?.forEach { it.deleteRecursively() }
    }
}

/**
 * Smart Search's image model and text detection compiled ahead of time for Qualcomm NPUs, one
 * folder per chip (huggingface.co/subhashhhhhh/lumen-npu, built by lumen-npu-build). Optional
 * files of the Search and Text packs, used once [GpuCheck] has seen them match the CPU.
 */
@LumenAiEngine
object NpuModels {
    const val Repo = "subhashhhhhh/lumen-npu"
    const val Commit = "e696eeba06e0cc4bc0454b5848e5185979f3b6b2"

    /** chip to (SigLIP 2 B/16 image model, PP-OCRv6 detection), as (bytes, SHA-256). */
    private val table: Map<String, Pair<Pair<Long, String>, Pair<Long, String>>> = mapOf(
        "SM4250" to ((375_764_088L to "f012f6223ce9f63c70ff0d5a3b8b5f67ddbb259a58694c727580210c1779a66e") to (11_054_088L to "cfc84a821f6c9f3d5ca9fa301dc4c7238b0eb953b3547eb3199de88478df2cb8")),
        "SM4350" to ((375_764_088L to "e3143680ce3719770a40f2ff6fb414ac6b533dd75d865415881f40e864d57c9d") to (11_054_088L to "435eca7d0a1de259cf51c24cd223386779cd7cb0db5b89b68fbaffe47ae83a9b")),
        "SM4375" to ((375_764_088L to "99362a36787b2d003d10a27aeaf78af7bac487ffc8cf7f864e01e040adc95e0d") to (11_054_088L to "fb6258013da08199e4cf2dfbc193266914e4a58f16b1d2ac235908525a84b3f0")),
        "SM4450" to ((375_762_808L to "76ccac07a64d2a3c82a779f4a27cb5910bf0edf52c19a81cba35a2c2863e18af") to (11_053_880L to "ac5c242bcf083ca61afadc65f6de8f24b522a2a1440bcebc7af57ea11697ecff")),
        "SM4635" to ((375_762_808L to "5898a34fd17e9209ea26e89242ef75943c21b90f0dc57aaa4bae120938fb7f7a") to (11_053_880L to "d8335c4b1bb222efab81e7de19cc0f4ee351115e95a4952ffa8b4734dde5406a")),
        "SM6115" to ((375_764_088L to "91a856e3ae95ca7c6c67c7a6f56ed394970d37eba503b6bbfe916341ee955336") to (11_054_088L to "84c52a0a7c647a42d62d09f56c639af328ed210f71fbf8c014be8f1541a1addf")),
        "SM6125" to ((375_764_088L to "9adb2fdbc2c473ddbb2d085c701ba5cbded5f16be196facd7c1729a9042b5a2b") to (11_054_088L to "21f763f1c34a150a91251ced5eb712a817c33c2a4b54e2d8be4e86a3f4c805f2")),
        "SM6150" to ((375_764_088L to "3fff35a833dd1a3878eb60eea481484fe89d29339a8e73be7dcefa0970a77161") to (11_054_088L to "425d891d6e8d0bb6d6024da77aa3046b9c9fa56c03d425cfd0cd6781a1b32581")),
        "SM6225" to ((375_764_088L to "122fc2968bafd4299ba2a2670afc12b94f0b36361bee8a922a6753aa3d51125c") to (11_054_088L to "c7a97c6d1ca604341da14d378798d34acd6766e6b75714742020ed0c221a143d")),
        "SM6250" to ((375_764_088L to "d9b7c6a845222da8f1af3393cdd0897b007e3b6570e74eb94e0347032cf09551") to (11_054_088L to "305b2d0e198da4c509f7943a00307972f7acd2b7a0dab3392bbbd2bc25ef0cdc")),
        "SM6350" to ((375_764_088L to "c0f1a126a62b2f0646370b493fed9eeccadb070d532b3b494abbccf7812b82d3") to (11_054_088L to "113de91a3b8d001728937026e215e3314be23613e698ece4089a55a578adda45")),
        "SM6375" to ((375_764_088L to "2ded56a0f6adf8a8f2d20de35478f261fc44036658b83e1bfe191603226d05e4") to (11_054_088L to "77ebfe95a0064419b9a2e45b26509cdc3ffb924e3c23e2c5837ba146c086fac2")),
        "SM6450" to ((375_814_296L to "8803f865cc79171a26c9fcc26cbb17087788752879955f664ad6fe417879d5e9") to (11_055_024L to "2321db79bea7eb3286e05a94a5b62144e2373819b9af9361345e57b492e6b654")),
        "SM6450Q" to ((375_762_808L to "a77be36b00679c72071950bceb8e701406c666be9fc1f74532471d61c1ad082c") to (11_053_880L to "bd43352ec4e9e56672d496141e23268171decd534759202a0571d8db3c522d7c")),
        "SM6475" to ((375_814_296L to "750422f87c0c19d6fc4816303010020b307d8d7d785cef321465bc485e751eba") to (11_055_024L to "1b9c1282543308e62f0207e115b6b9277412bac3278df4f4aeb4e24039dbe1e7")),
        "SM6650" to ((375_818_392L to "98e3d0a7c951c90b215197bf2dad64f6d220cdedd88bb9d87ab2d5d52b07b01f") to (11_055_024L to "adfa6ec3ad35981c1a263d5452e021518d1ed7149601e05d55c252220fc1b106")),
        "SM7150" to ((375_764_088L to "d0d10432d28acf9c74b2bdface2b48d7510d0db5d1e2181837386e922f5cf30c") to (11_054_088L to "98ff797ab8a027772cddb7b6f07c4674ace43471805729732f55ce6b6899059d")),
        "SM7225" to ((375_764_088L to "1a4843f19345a83b970befb4ff0c7ca856f9bf8f11346ede9d260abbda018ec8") to (11_054_088L to "cd0b03d3c52e1e9504c4ef19ac62585f5d410118ba61f348473376a7c26bce2b")),
        "SM7250" to ((375_764_088L to "dcc92d3b92ac3193f0cf6b6b6b77e50e0328cd4758728248ecf1ccb0c5acb056") to (11_054_088L to "7af63d5bf16616b01a28b302432fddf6db73c8aae583468ad82c619014b9e7fd")),
        "SM7315" to ((375_814_296L to "18367904c130bc5ca39c24c0f6aa5320c1ef63b2bf2f4997b583497601d93f1e") to (11_055_024L to "e16c9007269c18727d9af8d123c073303f1dcfd013efb2f6b5fdb770c111889f")),
        "SM7325" to ((375_814_296L to "1ad6ae2d8554eb8c8d1526caa4ade9cc0ee76ccb2ed4e321cb1d1eece8a3fb74") to (11_055_024L to "c03e64a472d34e7d0c049123e404f8b9d2ff062a48c7fd639e830f139e4aae32")),
        "SM7350" to ((375_814_296L to "91213d2ee58898360569041375d1a8c2f216803299540a086466dcfddb5e3540") to (11_055_024L to "a77b270d77aec6b168affb7ed7b3c84112c9ad77c7a45ba100db10ec20bb8ea2")),
        "SM7435" to ((375_814_296L to "b0eaa3aae78272d613a855a18d722cba18bc150e1e967846864bbf0e97755c22") to (11_055_024L to "82d4fe7c92da11355d401133a72aac17af2da8ec78e70f049eb0303bc59165ac")),
        "SM7450" to ((185_727_720L to "e0fec347f4290ae3c30ee7b34e2a450233aa2f4e0309fa37adc929d105c66321") to (9_200_000L to "cabfb386835026e3f9f0902043342ab74082c6e75e568371a88a739e923ceb0f")),
        "SM7475" to ((375_814_296L to "a96028181a0f7cdfd9d4a460c4323c9e62c3270af1129987889aa81861151708") to (11_055_024L to "c1b10e362904567dca7344cac86039056c6147db844fd0cea387629e758f2640")),
        "SM7550" to ((375_818_392L to "4a3f5f081fc281da8e898ee4f3eba390b12506691be8dc357efa8301a7db311f") to (11_055_024L to "c998e7ed548e33369598362ce3b7654d1e21fbf69c5cc9307b8fa05daf1956e7")),
        "SM7635" to ((375_818_392L to "26d36e6e419c9018f76275210ecccef522d18c81c54eb7401a659bfe1af6234a") to (11_055_024L to "4602b91e206fde72740f17763783906f10f298d16888d015b091eebc93f190fb")),
        "SM7675" to ((375_818_392L to "a216b41d5745f2cc479dbde838785ab3e7f4ff0f8e1d5e495fe5dc2ab49aeb66") to (11_055_024L to "0dec44a0add15b7e8a03bd883f8bd19514f71557056887e242f3f919862a51e9")),
        "SM7750" to ((375_818_392L to "c9769ed6324cdb93e37d0f262cae3a2d91a31a07b0fd6b09f16c41ac68060d75") to (11_055_024L to "8b2f664de3a4f7d64a281800381e65aff4566c6bd303bab13b5e292a8c251e76")),
        "SM8325" to ((375_814_296L to "7fce5017045327ad23e0b4dff844e40ae11d80e6da95f54c5f73cb37770da63b") to (11_055_024L to "6c252cab8507f5b4ed3d12ae1b7df46d42e3a4e286a7621cb4f80109e89eacfe")),
        "SM8450" to ((185_592_552L to "9508632c92243f5f8d73401b5ed2c94ac47ea4c9677ecf09c688e41601fab61a") to (8_606_080L to "b7ad3d2dd4e07a8e68caa793bb23ed33d19de88170fcbdbe8e48ebf81e310ceb")),
        "SM8475" to ((185_596_648L to "142fe87e8d0b2bd6840bc398816b07da320565da106aeb9237e27eb5febb1944") to (8_606_080L to "f811cda0b3820c39c4196071761fb690ac73d4025ce3498365ba2860e23fd2de")),
        "SM8550" to ((187_345_640L to "d84344a2b0f5a9ece861ca23f7a8a06fa1a10c8660ae3b11f3627b4540ebb71c") to (6_537_600L to "340d1ab378dc971be60a63865b182f8ec2d18847357700e65576e160f7a5d572")),
        "SM8635" to ((375_818_392L to "35ec49604f5f710343bb9795bc9afc1950ad787760e61fb21b03f4cee309cf12") to (11_055_024L to "ad1d0f161b7fc58f16fa419a987b007405843f0b656235e04c8f758a9c3dc1cd")),
        "SM8650" to ((187_341_544L to "b3a1b6db98a25013d4a76c588f64e80288499b0909e96616a6ce9ee02eb0a277") to (6_537_600L to "b3524e70551d63ff371311f3b027e7f17e0a29cc5f37430f03985cfde9f922c2")),
        "SM8735" to ((375_769_240L to "1773d694052eea8613b923597cb9dc905a49a068460ad02afcbb85e4aa818695") to (11_055_024L to "3b815ad67dcef034e2a045d86083bc83eb64cfa228e3cb3972f32dd487589f6d")),
        "SM8750" to ((187_349_736L to "1e3e74c13f07380203a1628badb8bea2cdc9c1f8362dea4b445fe4b539f8f549") to (6_549_888L to "e8da9c161ff9912fbef899cd119b1f93e315d00700a5ef0dddb75dfbd5b5cbb2")),
        "SM8845" to ((187_353_832L to "24eb73dde01f59fed2c6394d90668d45f6e30f1480f841a0cd86aec9a313b0ac") to (6_553_984L to "653441793cc5395d1c36c16ad25cc8ff9d6a5a124b39712ffd06e9d5d191478d")),
        "SM8845P" to ((187_353_832L to "fdedae390cf770e017c8eee697d20b767b116c840c3548af293bb418fd4d192a") to (6_553_984L to "220e03196636a64bb54593ef975b617fae4c7f2532581264ac0705ede7b4923d")),
        "SM8850" to ((187_411_176L to "95ab3ef08bb9c8a098c045e70e01471dba13f08af27c244f5c4a454424ad387e") to (6_668_672L to "0e5430ca14cbca80749db1cc943974d42f0a1a61babd6bc795eac6cb8d34508f")),
    )

    val chips: Set<String> get() = table.keys

    /** The image model for [chip] (Accurate only: Fast stays on ONNX). */
    fun siglip(chip: String): ModelFile? = table[chip]?.first?.let { (bytes, sha) -> ModelFile("$chip/siglip_visual.tflite", bytes, sha, optional = true) }

    fun textDetector(chip: String): ModelFile? = table[chip]?.second?.let { (bytes, sha) -> ModelFile("$chip/ocr_det.tflite", bytes, sha, optional = true) }

    fun url(file: ModelFile) = "https://huggingface.co/$Repo/resolve/$Commit/${file.name}"
}

/**
 * Grouping photos by person: YuNet finds faces and their landmarks (MIT), SFace turns each
 * straightened face into a 128-number fingerprint (Apache 2.0). From OpenCV Zoo, unchanged;
 * chosen on the owner's photos over AuraFace and InsightFace's (tools/faces/). Downloaded with
 * either Smart Search tier.
 */
@LumenAiEngine
object FaceModel {
    const val Repo = "subhashhhhhh/lumen-faces"
    const val Commit = "009b05388d32e9d7d487fa2a78fb9e4a3c30b6f3"

    val files = listOf(
        ModelFile("yunet.onnx", 229_738, "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"),
        ModelFile("sface.onnx", 38_696_353, "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    )
    val bytes: Long get() = files.sumOf { it.bytes }

    fun url(file: ModelFile) = "https://huggingface.co/$Repo/resolve/$Commit/${file.name}"

    fun dir(context: Context) = File(context.noBackupFilesDir, "models/faces")

    fun installed(context: Context): Boolean = files.all { File(dir(context), it.name).length() == it.bytes }
}

/**
 * Subtitles: Cactus Compute's Whistle (speech to text, Apache 2.0) and Silero VAD v6.2.3 (where
 * people speak, MIT), so music isn't "transcribed" into invented lines. Unchanged copies, hosted
 * together; downloaded with either Smart Search tier.
 */
@LumenAiEngine
object SpeechModel {
    const val Repo = "subhashhhhhh/lumen-speech"
    const val Commit = "7ead30428d4c748df73fe9c2d659cd1eb8ab997f"

    /** Settings' choices, by quality: Whistle for now; Whisper or Parakeet if they earn a place. */
    class Tier(val id: String, val label: String)
    val tiers = listOf(Tier("whistle", "Fast"))

    val files = listOf(
        ModelFile("whistle.cact", 16_919_407, "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"),
        ModelFile("vad.onnx", 1_289_603, "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"),
    )
    val bytes: Long get() = files.sumOf { it.bytes }

    fun url(file: ModelFile) = "https://huggingface.co/$Repo/resolve/$Commit/${file.name}"

    fun dir(context: Context) = File(context.noBackupFilesDir, "models/speech")

    fun installed(context: Context): Boolean = files.all { File(dir(context), it.name).length() == it.bytes }
}
