plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.subhashh.lumen.ai"
    compileSdk = 37

    defaultConfig {
        minSdk = 30
        consumerProguardFiles("consumer-rules.pro")
        // The speech engine (src/main/cpp/needle) ships for ARM only; elsewhere subtitles are off.
        externalNativeBuild {
            cmake {
                abiFilters += listOf("arm64-v8a", "armeabi-v7a")
                arguments += "-DANDROID_STL=c++_static"
            }
        }
    }

    ndkVersion = "30.0.16248370"
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
    // Model downloads on Android 11–13 (14+ uses a user-initiated job instead).
    implementation(libs.androidx.work.runtime)
    // ONNX Runtime for the text models and Fast's image model (CPU); LiteRT for Accurate's image
    // model and text detection, on the GPU where it works. Excluded from LiteRT: Google Play's
    // AI-pack model delivery (proprietary; only used for models shipped through Play).
    api(libs.onnxruntime.android)
    implementation(libs.litert) { exclude(group = "com.google.android.play") }
    implementation(libs.litert.gpu) { exclude(group = "com.google.android.play") }

    testImplementation(libs.junit)
    // Android's org.json is a stub in local tests; the real one stands in.
    testImplementation(libs.org.json)
}
