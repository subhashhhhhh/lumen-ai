plugins {
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
    // Only puts this Kotlin version on the build classpath for AGP's built-in Kotlin to use.
    alias(libs.plugins.kotlin.android) apply false
}
