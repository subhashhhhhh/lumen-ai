# Lumen AI

On-device smart features for Android photo apps: search photos by describing them, read the
text in photos and videos, group faces into people, and make subtitles. Everything runs on the
phone. Nothing is uploaded.

This is the engine of [Lumen](https://github.com/subhashhhhhh/Lumen), split out so that other
gallery apps can use it too.

> **Status: early.** The code is being moved out of Lumen. The API will change before the first
> release. It is not on Maven Central yet.

## What it does

| Feature | Models | Runs on |
|---|---|---|
| Smart Search (photos and videos by description) | SigLIP 2 (Accurate) or CLIP (Fast) | GPU when it works, else CPU |
| Text in photos and videos, with the position of each line | PP-OCRv6 | GPU and CPU |
| Faces, grouped into people | YuNet and SFace | CPU |
| Subtitles | Whistle and Silero VAD | CPU (ARM only) |

The models are not in the library. The app downloads them when the user turns a feature on, from
pinned commits on Hugging Face, and checks each file with SHA-256.

## Use it in an app

1. Add the `:ai` module to your build. (A Maven artifact comes later.)
2. Implement `LumenAiHost` and install it in `Application.onCreate`:

```kotlin
class App : Application() {
    override fun onCreate() {
        super.onCreate()
        LumenAi.install(object : LumenAiHost {
            override suspend fun library(context: Context) = myMediaList().map { it.toMediaRef() }
            override val settings = object : AiSettings {
                override val searchModelId: String? = null
                override val videoDetailPx = 1280
                override val indexOnBattery = false
                override val indexSchedule: IndexSchedule? = null
            }
            override val notifications = object : AiNotifications {
                override val smallIcon = R.drawable.ic_notification
                override fun contentIntent(context: Context) = PendingIntent.getActivity(
                    context, 0, Intent(context, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
                )
            }
        })
    }
}
```

## Network and permissions

The library goes online only to download models. It declares the permissions for that
(`INTERNET`, `ACCESS_NETWORK_STATE`, and those for a download job). An app that imports models
by hand can remove them with `tools:node="remove"`.

The library does not use Google Play services, and it removes ONNX Runtime's telemetry provider.

## Build

You need JDK 17 or later, the Android SDK and NDK. Then:

```sh
./gradlew :ai:assembleRelease
./gradlew :ai:testDebugUnitTest
```

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
