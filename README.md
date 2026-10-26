# Lumen AI

Smart photo features for Android apps, on the phone.
Nothing leaves the phone.

[![Maven Central](https://img.shields.io/maven-central/v/app.lumen-photos/lumen-ai.svg)](https://central.sonatype.com/artifact/app.lumen-photos/lumen-ai)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Android 11+](https://img.shields.io/badge/Android-11%2B-3ddc84.svg)

With Lumen AI, a gallery app can find a photo from a description and read the text in photos
and videos. It can also group faces into people and make subtitles. All of the work happens on
the phone. The library
does not send photos, faces or text to a server.

Lumen AI is the engine of [Lumen](https://github.com/subhashhhhhh/Lumen), a gallery app for
Android. It is a separate library so that other apps can use it too.

> [!IMPORTANT]
> Lumen AI is at an early stage. The code works in Lumen every day, but the API (application
> programming interface) can change before version 1.0.

## Features

- **Smart Search.** Find photos and videos from a description, for example "cat sleeping on a
  chair". Put words in quotes to find an exact phrase in the text of a photo. Write two names to
  find photos with both people.
- **Text in photos and videos.** Read the text in photos and in video frames. The library keeps
  the position of each line, so an app can let users select and copy words.
- **People.** Find the faces in photos and videos and group them into people. Users can name,
  merge and hide people. Search uses the people to give better results for names.
- **Subtitles.** Make subtitles for a video, and save them as SRT or WebVTT files.
- **Labels.** Show words that describe a photo, for example "cat" or "beach", and make search
  suggestions from the user's own photos.
- **Background indexing.** Index the library while the phone charges, or in hours that the
  user selects. Indexing stops when the phone gets warm or the battery is low.
- **Model downloads.** Download each model only when the user turns on its feature, and check
  each file before use.

## Privacy

- The library does all of its work on the phone.
- It goes online only to download models from Hugging Face, and only when the app asks it to.
- It does not use Google Play services.
- It removes the telemetry provider of ONNX Runtime from the app.
- It does not collect data, and it has no analytics.

## Requirements

- Android 11 (API level 30) or later.
- Subtitles work only on ARM phones. The other features work on all phones.
- Storage for the models that the user turns on:

| Feature | Models | Download |
|---|---|---|
| Smart Search, Accurate | SigLIP 2 | 670 MB |
| Smart Search, Fast | CLIP ViT-B/32 | 240 MB |
| Text | PP-OCRv6 small | 23 MB |
| People | YuNet and SFace | 39 MB |
| Subtitles | Whistle and Silero VAD | 18 MB |

## How it works

The library does not include the models. When the user turns on a feature, the library
downloads its files from a fixed commit of a Hugging Face repository. Then it checks each file
with a SHA-256 checksum. A file that does not match is not used.

The Accurate search model and the text finder run on the GPU (graphics processing unit) when
they can. The library tests the GPU one time, in a separate process. If the test fails, these
models run on the CPU (central processing unit) instead. A broken GPU driver cannot stop the
app. The other models always run on the CPU.

The library keeps its indexes in the private storage of the app. Other apps cannot read them.

## Add Lumen AI to an app

Lumen AI is on Maven Central. Add it to the build file of your app:

```kotlin
dependencies {
    implementation("app.lumen-photos:lumen-ai:0.1.5")
}
```

To build it from source instead, add this repository next to your project and include its
build in `settings.gradle.kts`:

```kotlin
includeBuild("../lumen-ai") {
    dependencySubstitution {
        substitute(module("app.lumen-photos:lumen-ai")).using(project(":ai"))
    }
}
```

## Use it

Install Lumen AI in `Application.onCreate`. Background work starts in its own process, so
install it before any other code runs. With no settings, it indexes every photo and video on
the phone.

```kotlin
class App : Application() {
    override fun onCreate() {
        super.onCreate()
        LumenAi.install(this)
        LumenAi.indexing.start(this)
    }
}
```

Then download the features that you want, and use them:

```kotlin
LumenAi.models.download(context, ModelPack.Search)

val results = LumenAi.search(context, "cat sleeping on a chair")   // media ids, best first
val text = LumenAi.liveText(context, item)                          // lines, with positions
val people = LumenAi.people.all(context)
val labels = LumenAi.labels(context, mediaId)                       // ["cat", "sofa"]
LumenAi.subtitles(context, video).collect { cues -> show(cues) }
```

Your app asks for the media permissions itself (`READ_MEDIA_IMAGES`, `READ_MEDIA_VIDEO`, and
`POST_NOTIFICATIONS` for the progress notifications).

To change a default, use the install block:

```kotlin
LumenAi.install(this) {
    searchModel = SearchModel.Fast          // 240 MB instead of 670 MB
    indexSchedule = IndexSchedule(60, 420)  // 1:00 to 7:00, plugged in or not
    library = { context -> myOwnMediaList(context) }
}
```

For full control, implement `LumenAiHost` and install it instead. Lumen does this in
[`AiHost.kt`](https://github.com/subhashhhhhh/Lumen/blob/main/app/src/main/java/com/subhashh/lumen/data/AiHost.kt).

The stores, indexers and readers behind these calls are public, but marked `@LumenAiEngine`.
To use them, opt in. Their API can change in any release.

## Documentation

- [Guide](https://lumen-photos.app/ai): setup and every feature, with examples.
- [API reference](https://lumen-photos.app/ai/reference/index.html), made from the code with Dokka.
- [Sample app](sample): a small gallery that uses only the calls above.

## Permissions

The library adds these permissions to your app, for model downloads:

| Permission | Why |
|---|---|
| `INTERNET` | To download models |
| `ACCESS_NETWORK_STATE` | To wait for Wi-Fi when the user asks for it |
| `RUN_USER_INITIATED_JOBS` | To download on Android 14 and later |
| `FOREGROUND_SERVICE` and `FOREGROUND_SERVICE_DATA_SYNC` | To download on Android 11 to 13 |

An app that imports models by hand, with no internet access, can remove them with
`tools:node="remove"` in its manifest.

## Build

You need these tools:

- JDK (Java Development Kit) 17 or later
- The Android SDK (software development kit), with NDK (native development kit) 30

Then run these commands:

```sh
git clone https://github.com/subhashhhhhh/lumen-ai.git
cd lumen-ai
./gradlew :ai:assembleRelease      # the library
./gradlew :ai:testDebugUnitTest    # the tests
./gradlew :sample:assembleDebug    # the sample app
./gradlew :ai:dokkaGenerate        # the API reference, in ai/build/dokka/html
```

## Plans

- Separate modules for each feature, so an app takes only what it needs.
- Version 1.0, with a stable API.

## Credits

| Model or library | Use | License |
|---|---|---|
| [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), Google | Smart Search, Accurate | Apache 2.0 |
| [OpenCLIP ViT-B/32](https://github.com/mlfoundations/open_clip), LAION and OpenAI | Smart Search, Fast | MIT |
| [PP-OCRv6](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx), PaddlePaddle | Text | Apache 2.0 |
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), OpenCV Zoo | Finding faces | MIT |
| [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), OpenCV Zoo | Telling faces apart | Apache 2.0 |
| [Whistle and needle](https://huggingface.co/Cactus-Compute/whistle), Cactus Compute | Subtitles | Apache 2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad), Silero | Finding speech | MIT |
| [Recognize Anything](https://github.com/xinyu1205/recognize-anything) | Labels and suggestions | Apache 2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), Microsoft | Runs models on the CPU | MIT |
| [LiteRT](https://github.com/google-ai-edge/LiteRT), Google | Runs models on the GPU | Apache 2.0 |

## License

Lumen AI is free and open source under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
