# Applied to every app that uses Lumen AI.

# The runtimes call back into their Java/Kotlin classes from native code (JNI), by name.
-keep class ai.onnxruntime.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# Only LiteRT's AiPackModelProvider (Play's AI-pack model delivery) uses these; that library is
# excluded, and Lumen AI never takes models from Play.
-dontwarn com.google.android.play.core.aipacks.**
-dontwarn com.google.android.gms.tasks.**
