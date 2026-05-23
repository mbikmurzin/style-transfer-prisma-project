package com.example.styletransfer

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelsNotFoundException : Exception(
    "Models not found. Put style_predict.tflite and style_transform.tflite into assets/models."
)

class InvalidModelContractException(message: String) : Exception(message)

class StyleTransferModelRunner(
    private val context: Context,
    private val useGpu: Boolean,
    private val numberOfThreads: Int
) : AutoCloseable {
    private val gpuDelegates = mutableListOf<GpuDelegate>()
    private lateinit var predictInterpreter: Interpreter
    private lateinit var transformInterpreter: Interpreter
    private var gpuEnabled = false

    init {
        checkModelsAvailable()
        createInterpreters()
    }

    fun run(contentImage: Bitmap, styleImage: Bitmap): StyleTransferResult {
        return try {
            runInternal(contentImage, styleImage)
        } catch (gpuError: Exception) {
            if (!gpuEnabled) throw gpuError
            closeInterpreters()
            createCpuInterpreters()
            runInternal(contentImage, styleImage)
        }
    }

    private fun runInternal(contentImage: Bitmap, styleImage: Bitmap): StyleTransferResult {
        val fullStart = SystemClock.elapsedRealtimeNanos()
        val predictInput = predictInterpreter.getInputTensor(0)
        val transformContentIndex = findContentInputIndex()
        val transformStyleIndex = if (transformContentIndex == 0) 1 else 0
        val transformContentInput = transformInterpreter.getInputTensor(transformContentIndex)
        val transformStyleInput = transformInterpreter.getInputTensor(transformStyleIndex)
        val styleOutputTensor = predictInterpreter.getOutputTensor(0)
        validateBottleneckContract(styleOutputTensor, transformStyleInput)

        val preprocessingStart = SystemClock.elapsedRealtimeNanos()
        val styleBuffer = ImageUtils.bitmapToTensorBuffer(styleImage, predictInput)
        val contentBuffer = ImageUtils.bitmapToTensorBuffer(contentImage, transformContentInput)
        val preprocessingMs = elapsedMs(preprocessingStart)

        val styleBottleneck = ByteBuffer.allocateDirect(styleOutputTensor.numBytes())
            .order(ByteOrder.nativeOrder())
        val predictingStart = SystemClock.elapsedRealtimeNanos()
        predictInterpreter.run(styleBuffer, styleBottleneck)
        val predictingMs = elapsedMs(predictingStart)
        styleBottleneck.rewind()

        val resultTensor = transformInterpreter.getOutputTensor(0)
        val resultBuffer = ByteBuffer.allocateDirect(resultTensor.numBytes())
            .order(ByteOrder.nativeOrder())
        val transformInputs = arrayOfNulls<Any>(2)
        transformInputs[transformContentIndex] = contentBuffer
        transformInputs[transformStyleIndex] = styleBottleneck
        val transferringStart = SystemClock.elapsedRealtimeNanos()
        transformInterpreter.runForMultipleInputsOutputs(
            transformInputs.requireNoNulls(),
            mapOf(0 to resultBuffer)
        )
        val transferringMs = elapsedMs(transferringStart)

        val postProcessingStart = SystemClock.elapsedRealtimeNanos()
        val resultBitmap = ImageUtils.outputBufferToBitmap(resultBuffer, resultTensor)
        val postProcessingMs = elapsedMs(postProcessingStart)
        val contentShape = transformContentInput.shape()
        val benchmark = BenchmarkResult(
            inputImageSize = "${contentShape[2]} x ${contentShape[1]}",
            gpuEnabled = gpuEnabled,
            numberOfThreads = numberOfThreads,
            preProcessTimeMs = preprocessingMs,
            predictingStyleTimeMs = predictingMs,
            transferringStyleTimeMs = transferringMs,
            postProcessTimeMs = postProcessingMs,
            fullExecutionTimeMs = elapsedMs(fullStart)
        )
        return StyleTransferResult(resultBitmap, benchmark)
    }

    private fun createInterpreters() {
        if (useGpu) {
            try {
                val compatibilityList = CompatibilityList()
                if (compatibilityList.isDelegateSupportedOnThisDevice) {
                    val predictDelegate = GpuDelegate(compatibilityList.bestOptionsForThisDevice)
                    val transformDelegate = GpuDelegate(compatibilityList.bestOptionsForThisDevice)
                    gpuDelegates += predictDelegate
                    gpuDelegates += transformDelegate
                    val predictOptions = Interpreter.Options()
                        .setNumThreads(numberOfThreads)
                        .addDelegate(predictDelegate)
                    val transformOptions = Interpreter.Options()
                        .setNumThreads(numberOfThreads)
                        .addDelegate(transformDelegate)
                    predictInterpreter = Interpreter(loadModel(PREDICT_MODEL), predictOptions)
                    transformInterpreter = Interpreter(loadModel(TRANSFORM_MODEL), transformOptions)
                    gpuEnabled = true
                    return
                }
            } catch (_: Exception) {
                closeInterpreters()
            }
        }
        createCpuInterpreters()
    }

    private fun createCpuInterpreters() {
        val cpuOptions = Interpreter.Options().setNumThreads(numberOfThreads)
        predictInterpreter = Interpreter(loadModel(PREDICT_MODEL), cpuOptions)
        transformInterpreter = Interpreter(loadModel(TRANSFORM_MODEL), cpuOptions)
        gpuEnabled = false
    }

    private fun findContentInputIndex(): Int {
        if (transformInterpreter.inputTensorCount != 2) {
            throw InvalidModelContractException(
                "style_transform model must expose two inputs: content image and style bottleneck."
            )
        }
        for (index in 0 until transformInterpreter.inputTensorCount) {
            val tensor = transformInterpreter.getInputTensor(index)
            val name = tensor.name().lowercase()
            val shape = tensor.shape()
            if ("content" in name || (shape.size == 4 && shape[3] == 3)) {
                return index
            }
        }
        return 0
    }

    private fun validateBottleneckContract(styleOutput: org.tensorflow.lite.Tensor, transformStyleInput: org.tensorflow.lite.Tensor) {
        if (!styleOutput.shape().contentEquals(transformStyleInput.shape()) ||
            styleOutput.dataType() != transformStyleInput.dataType()
        ) {
            throw InvalidModelContractException(
                "Incompatible models: style_predict output does not match style_transform bottleneck input."
            )
        }
    }

    private fun checkModelsAvailable() {
        val available = context.assets.list(MODELS_FOLDER)?.toSet().orEmpty()
        if (PREDICT_MODEL.substringAfterLast('/') !in available ||
            TRANSFORM_MODEL.substringAfterLast('/') !in available
        ) {
            throw ModelsNotFoundException()
        }
    }

    private fun loadModel(path: String): MappedByteBuffer {
        context.assets.openFd(path).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.startOffset,
                    descriptor.declaredLength
                )
            }
        }
    }

    private fun elapsedMs(startNanos: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000

    private fun closeInterpreters() {
        if (::predictInterpreter.isInitialized) predictInterpreter.close()
        if (::transformInterpreter.isInitialized) transformInterpreter.close()
        gpuDelegates.forEach { it.close() }
        gpuDelegates.clear()
    }

    override fun close() {
        closeInterpreters()
    }

    private companion object {
        const val MODELS_FOLDER = "models"
        const val PREDICT_MODEL = "$MODELS_FOLDER/style_predict.tflite"
        const val TRANSFORM_MODEL = "$MODELS_FOLDER/style_transform.tflite"
    }
}
