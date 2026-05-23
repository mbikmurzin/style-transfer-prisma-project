package com.example.styletransfer

import android.content.ContentValues
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.styletransfer.databinding.ActivityMainBinding
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private var contentBitmap: Bitmap? = null
    private var styleBitmap: Bitmap? = null
    private var resultBitmap: Bitmap? = null
    private var pendingCameraUri: Uri? = null

    private val contentPicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { loadBitmapFromUri(it, ::setContentBitmap) }
    }

    private val stylePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { loadBitmapFromUri(it, ::setStyleBitmap) }
    }

    private val cameraCapture = registerForActivityResult(ActivityResultContracts.TakePicture()) { succeeded ->
        if (succeeded) {
            pendingCameraUri?.let { loadBitmapFromUri(it, ::setContentBitmap) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupThreadSelection()
        loadFirstBuiltInStyle()

        binding.selectContentButton.setOnClickListener { contentPicker.launch("image/*") }
        binding.cameraButton.setOnClickListener { captureContentImage() }
        binding.selectStyleButton.setOnClickListener { stylePicker.launch("image/*") }
        binding.builtInStyleButton.setOnClickListener { showBuiltInStyles() }
        binding.stylizeButton.setOnClickListener { runStyleTransfer() }
        binding.saveButton.setOnClickListener { resultBitmap?.let(::saveResultToGallery) }
    }

    private fun setupThreadSelection() {
        val threadOptions = listOf("1", "2", "4")
        binding.threadsSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            threadOptions
        )
        binding.threadsSpinner.setSelection(threadOptions.indexOf("4"))
    }

    private fun loadFirstBuiltInStyle() {
        val styles = builtInStylePaths()
        if (styles.isNotEmpty()) {
            loadBuiltInStyle(styles.first())
        }
    }

    private fun captureContentImage() {
        runCatching {
            val target = File.createTempFile("content_camera_", ".jpg", cacheDir)
            FileProvider.getUriForFile(this, "$packageName.fileprovider", target)
        }.onSuccess { uri ->
            pendingCameraUri = uri
            cameraCapture.launch(uri)
        }.onFailure {
            toast(getString(R.string.camera_open_failed))
        }
    }

    private fun showBuiltInStyles() {
        val styles = builtInStylePaths()
        if (styles.isEmpty()) {
            toast(getString(R.string.no_built_in_styles))
            return
        }
        val labels = styles.map { it.substringAfterLast('/').substringBeforeLast('.') }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle(R.string.select_built_in_style)
            .setItems(labels) { _, index ->
                loadBuiltInStyle(styles[index])
            }
            .show()
    }

    private fun builtInStylePaths(): List<String> =
        assets.list("styles")
            ?.filter { file -> file.endsWith(".jpg", true) || file.endsWith(".png", true) }
            ?.sorted()
            ?.map { "styles/$it" }
            .orEmpty()

    private fun setContentBitmap(bitmap: Bitmap) {
        contentBitmap = bitmap
        binding.contentImage.setImageBitmap(bitmap)
    }

    private fun setStyleBitmap(bitmap: Bitmap) {
        styleBitmap = bitmap
        binding.styleImage.setImageBitmap(bitmap)
    }

    private fun loadBitmapFromUri(uri: Uri, onLoaded: (Bitmap) -> Unit) {
        runCatching { ImageUtils.decodeUri(contentResolver, uri) }
            .onSuccess(onLoaded)
            .onFailure { toast(getString(R.string.image_load_failed)) }
    }

    private fun loadBuiltInStyle(path: String) {
        runCatching { ImageUtils.loadAssetBitmap(this, path) }
            .onSuccess(::setStyleBitmap)
            .onFailure { toast(getString(R.string.image_load_failed)) }
    }

    private fun runStyleTransfer() {
        val content = contentBitmap
        val style = styleBitmap
        if (content == null || style == null) {
            toast(getString(R.string.choose_images_first))
            return
        }

        val requestedGpu = binding.gpuSwitch.isChecked
        val threads = binding.threadsSpinner.selectedItem.toString().toInt()
        showBusy(true)
        inferenceExecutor.execute {
            try {
                StyleTransferModelRunner(applicationContext, requestedGpu, threads).use { runner ->
                    val result = runner.run(content, style)
                    runOnUiThread {
                        resultBitmap = result.bitmap
                        binding.resultImage.setImageBitmap(result.bitmap)
                        binding.metricsText.text = result.benchmark.asDisplayText()
                        binding.saveButton.isEnabled = true
                        showBusy(false)
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    showBusy(false)
                    binding.metricsText.text = error.message ?: getString(R.string.inference_failed)
                    toast(error.message ?: getString(R.string.inference_failed))
                }
            }
        }
    }

    private fun saveResultToGallery(bitmap: Bitmap) {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "stylized_$timestamp.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/NeuralCanvas")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            toast(getString(R.string.save_failed))
            return
        }

        runCatching {
            contentResolver.openOutputStream(uri)?.use { stream ->
                check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                    "Bitmap compression failed."
                }
            } ?: error("Unable to open gallery output stream.")
            contentResolver.update(uri, ContentValues().apply {
                put(MediaStore.Images.Media.IS_PENDING, 0)
            }, null, null)
        }.onSuccess {
            toast(getString(R.string.saved_result))
        }.onFailure {
            contentResolver.delete(uri, null, null)
            toast(getString(R.string.save_failed))
        }
    }

    private fun showBusy(isBusy: Boolean) {
        binding.progressBar.visibility = if (isBusy) View.VISIBLE else View.GONE
        binding.stylizeButton.isEnabled = !isBusy
    }

    private fun toast(message: String) =
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        inferenceExecutor.shutdown()
        super.onDestroy()
    }
}
