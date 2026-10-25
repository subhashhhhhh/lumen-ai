package com.subhashh.lumen.ai

/**
 * Marks Lumen AI's lower-level engine: the stores, indexers, readers and runtimes behind
 * [LumenAi]'s calls. They work, and Lumen uses them directly, but their API can change in any
 * release. Prefer [LumenAi]; opt in only when you need finer control.
 */
@RequiresOptIn(
    level = RequiresOptIn.Level.ERROR,
    message = "Lumen AI's lower-level engine: its API can change in any release. Prefer LumenAi's calls.",
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
annotation class LumenAiEngine
