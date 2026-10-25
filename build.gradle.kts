plugins {
    alias(libs.plugins.android.library) apply false
    // Only puts this Kotlin version on the build classpath for AGP's built-in Kotlin to use.
    alias(libs.plugins.kotlin.android) apply false
}
