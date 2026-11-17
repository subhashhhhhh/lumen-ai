import com.vanniktech.maven.publish.AndroidSingleVariantLibrary
import com.vanniktech.maven.publish.JavadocJar
import com.vanniktech.maven.publish.SourcesJar
import java.security.MessageDigest

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.dokka)
    alias(libs.plugins.maven.publish)
}

group = "app.lumen-photos"
version = "0.1.7"

/**
 * Where there's no NDK (Google ships none for Linux on ARM, such as a build server), the speech
 * library isn't compiled but taken ready-made: `-PlumenAi.nativeLibs=<folder>`, holding
 * `<abi>/liblumen_speech.so` and `SOURCE_HASH` as scripts/package-native.sh writes them on a
 * machine that has the NDK. The hash is of src/main/cpp, so a build whose native sources have
 * changed since stops, instead of quietly shipping the old library.
 */
val prebuiltNative: File? = providers.gradleProperty("lumenAi.nativeLibs").orNull?.let(::File)
if (prebuiltNative != null) {
    val sources = file("src/main/cpp")
    val sha = { bytes: ByteArray -> MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) } }
    // The same lines `shasum -a 256` prints for the sorted files, hashed again (see the script).
    val listing = sources.walkTopDown().filter { it.isFile }
        .map { "./" + it.relativeTo(sources).invariantSeparatorsPath to it }
        .sortedBy { it.first }
        .joinToString("") { (path, source) -> "${sha(source.readBytes())}  $path\n" }
    val built = prebuiltNative.resolve("SOURCE_HASH").takeIf { it.isFile }?.readText()?.trim()
    if (built != sha(listing.toByteArray())) {
        throw GradleException(
            "The ready-made speech library in $prebuiltNative is missing or was built from other sources than src/main/cpp. " +
                "Run scripts/package-native.sh on a machine with the NDK and copy its output there.",
        )
    }
}

android {
    namespace = "com.subhashh.lumen.ai"
    compileSdk = 37

    defaultConfig {
        minSdk = 30
        consumerProguardFiles("consumer-rules.pro")
        // The speech engine (src/main/cpp/needle) ships for ARM only; elsewhere subtitles are off.
        if (prebuiltNative == null) {
            externalNativeBuild {
                cmake {
                    abiFilters += listOf("arm64-v8a", "armeabi-v7a")
                    arguments += "-DANDROID_STL=c++_static"
                }
            }
        }
    }

    if (prebuiltNative == null) {
        ndkVersion = "30.0.16248370"
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "4.1.2"
            }
        }
    } else {
        sourceSets.getByName("main").jniLibs.directories.add(prebuiltNative.path)
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        // The library's own code uses its engine freely.
        optIn.add("com.subhashh.lumen.ai.LumenAiEngine")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    // Flow and StateFlow are in the public API.
    api(libs.kotlinx.coroutines.android)
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

// API reference: ./gradlew :ai:dokkaGenerate (HTML in build/dokka/html), published with the guides
// at lumen-photos.app/ai.
dokka {
    moduleName.set("lumen-ai")
    moduleVersion.set(version.toString())
    pluginsConfiguration.html {
        footerMessage.set("Lumen AI · Apache License 2.0 · <a href=\"https://lumen-photos.app/ai\">Guide</a> · <a href=\"https://github.com/subhashhhhhh/lumen-ai\">GitHub</a>")
    }
    dokkaSourceSets.configureEach {
        includes.from("Module.md")
        sourceLink {
            localDirectory.set(file("src/main/java"))
            remoteUrl("https://github.com/subhashhhhhh/lumen-ai/tree/main/ai/src/main/java")
        }
    }
}

// Maven Central: app.lumen-photos:lumen-ai. Publish with scripts/publish.sh, which reads the
// Central Portal token and the signing key's passphrase from the macOS Keychain.
mavenPublishing {
    coordinates(group.toString(), "lumen-ai", version.toString())
    configure(
        AndroidSingleVariantLibrary(
            javadocJar = JavadocJar.Dokka("dokkaGeneratePublicationHtml"),
            sourcesJar = SourcesJar.Sources(),
            variant = "release",
        ),
    )
    publishToMavenCentral()
    signAllPublications()
    pom {
        name.set("Lumen AI")
        description.set(
            "On-device smart photo features for Android apps: search by description, text in photos and " +
                "videos, people, labels and subtitles. Nothing leaves the phone.",
        )
        inceptionYear.set("2026")
        url.set("https://lumen-photos.app/ai")
        licenses {
            license {
                name.set("The Apache License, Version 2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                distribution.set("repo")
            }
        }
        developers {
            developer {
                id.set("subhashhhhhh")
                name.set("Subhash Gottumukkala")
                url.set("https://github.com/subhashhhhhh")
            }
        }
        scm {
            url.set("https://github.com/subhashhhhhh/lumen-ai")
            connection.set("scm:git:https://github.com/subhashhhhhh/lumen-ai.git")
            developerConnection.set("scm:git:ssh://git@github.com/subhashhhhhh/lumen-ai.git")
        }
    }
}
