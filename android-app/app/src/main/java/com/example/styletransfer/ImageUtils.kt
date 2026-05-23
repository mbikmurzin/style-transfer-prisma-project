package com.example.styletransfer

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

object ImageUtils {
    fun decodeUri(contentResolver: ContentResolver, uri: Uri): Bitmap =
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        }

    fun loadAssetBitmap(context: Context, path: String): Bitmap =
        context.assets.open(path).use { input ->
            android.graphics.BitmapFactory.decodeStream(input)
                ?: error("Unable to decode built-in style: $path")
        }

    fun bitmapToTensorBuffer(bitmap: Bitmap, tensor: Tensor): ByteBuffer {
        val shape = tensor.shape()
        require(shape.size == 4 && shape[0] == 1 && shape[3] == 3) {
            "Expected image tensor [1, height, width, 3], got ${shape.contentToString()}."
        }
        val height = shape[1]
        val width = shape[2]
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            putNormalized(buffer, tensor, ((pixel shr 16) and 0xff) / 255f)
            putNormalized(buffer, tensor, ((pixel shr 8) and 0xff) / 255f)
            putNormalized(buffer, tensor, (pixel and 0xff) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    fun outputBufferToBitmap(buffer: ByteBuffer, tensor: Tensor): Bitmap {
        val shape = tensor.shape()
        require(shape.size == 4 && shape[0] == 1 && shape[3] == 3) {
            "Expected output tensor [1, height, width, 3], got ${shape.contentToString()}."
        }
        val height = shape[1]
        val width = shape[2]
        buffer.rewind()
        val pixels = IntArray(width * height)
        for (index in pixels.indices) {
            val red = (readNormalized(buffer, tensor) * 255).roundToInt().coerceIn(0, 255)
            val green = (readNormalized(buffer, tensor) * 255).roundToInt().coerceIn(0, 255)
            val blue = (readNormalized(buffer, tensor) * 255).roundToInt().coerceIn(0, 255)
            pixels[index] = (0xff shl 24) or (red shl 16) or (green shl 8) or blue
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun putNormalized(buffer: ByteBuffer, tensor: Tensor, value: Float) {
        when (tensor.dataType()) {
            DataType.FLOAT32 -> buffer.putFloat(value)
            DataType.UINT8 -> buffer.put(quantize(value, tensor).coerceIn(0, 255).toByte())
            DataType.INT8 -> buffer.put(quantize(value, tensor).coerceIn(-128, 127).toByte())
            else -> error("Unsupported model input type: ${tensor.dataType()}")
        }
    }

    private fun readNormalized(buffer: ByteBuffer, tensor: Tensor): Float =
        when (tensor.dataType()) {
            DataType.FLOAT32 -> buffer.float.coerceIn(0f, 1f)
            DataType.UINT8 -> dequantize(buffer.get().toInt() and 0xff, tensor).coerceIn(0f, 1f)
            DataType.INT8 -> dequantize(buffer.get().toInt(), tensor).coerceIn(0f, 1f)
            else -> error("Unsupported model output type: ${tensor.dataType()}")
        }

    private fun quantize(value: Float, tensor: Tensor): Int {
        val parameters = tensor.quantizationParams()
        return if (parameters.scale == 0f) {
            (value * 255f).roundToInt()
        } else {
            (value / parameters.scale + parameters.zeroPoint).roundToInt()
        }
    }

    private fun dequantize(value: Int, tensor: Tensor): Float {
        val parameters = tensor.quantizationParams()
        return if (parameters.scale == 0f) {
            value / 255f
        } else {
            (value - parameters.zeroPoint) * parameters.scale
        }
    }
}
