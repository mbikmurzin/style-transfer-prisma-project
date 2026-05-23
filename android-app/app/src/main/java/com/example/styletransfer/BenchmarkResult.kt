package com.example.styletransfer

data class BenchmarkResult(
    val inputImageSize: String,
    val gpuEnabled: Boolean,
    val numberOfThreads: Int,
    val preProcessTimeMs: Long,
    val predictingStyleTimeMs: Long,
    val transferringStyleTimeMs: Long,
    val postProcessTimeMs: Long,
    val fullExecutionTimeMs: Long
) {
    fun asDisplayText(): String = """
        Input Image Size: $inputImageSize
        GPU enabled: $gpuEnabled
        Number of threads: $numberOfThreads
        Pre-process execution time: ${preProcessTimeMs} ms
        Predicting style execution time: ${predictingStyleTimeMs} ms
        Transferring style execution time: ${transferringStyleTimeMs} ms
        Post-process execution time: ${postProcessTimeMs} ms
        Full execution time: ${fullExecutionTimeMs} ms
    """.trimIndent()
}

data class StyleTransferResult(
    val bitmap: android.graphics.Bitmap,
    val benchmark: BenchmarkResult
)
