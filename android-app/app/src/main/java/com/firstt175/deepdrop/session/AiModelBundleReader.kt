package com.firstt175.deepdrop.session

import android.content.Context
import android.net.Uri
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.zip.Inflater

sealed class ModelBundleResult {
    data class Success(val precisionUsed: String, val graphs: List<String>, val modelDir: String) : ModelBundleResult()
    data class Failure(val message: String) : ModelBundleResult()
}

private data class BundleSection(
    val offset: Int,
    val compressedSize: Int,
    val uncompressedSize: Int,
    val sha256: String,
)

/**
 * Reads the "FIBD" model bundle produced by `build_model_bundle.py` in the training notebook
 * (best-fg.ipynb, cell "build_model_bundle.py"). Layout:
 *
 *   [4B magic "FIBD"] [2B version LE] [4B header_len LE] [header JSON utf-8] [zlib blobs...]
 *
 * The header JSON lists every packed section (e.g. `ncnn_flownet_lite_v2_fp16_param`,
 * `ncnn_flownet_lite_v2_fp16_bin`, ...) with its byte offset/size relative to the start of the
 * blob area plus a sha256 of the *uncompressed* bytes. We only ever need the two ncnn graphs
 * (FlowNetLite, RefineNetLite) at one precision, so we seek straight to those sections, inflate
 * them, verify the checksum, and write plain `.ncnn.param` / `.ncnn.bin` files to app-private
 * storage — the same layout NcnnInterpolator expects to load from disk.
 */
object AiModelBundleReader {

    private const val TAG = "AiModelBundleReader"
    private const val MAGIC = "FIBD"
    private val GRAPHS = listOf("flownet_lite_v2", "refinenet_lite_v2")

    fun extract(ctx: Context, bundleUri: Uri, preferFp16: Boolean): ModelBundleResult {
        val rawDir = File(ctx.filesDir, "ai_bundle_raw").apply { mkdirs() }
        val outDir = File(ctx.filesDir, "ai_model").apply { mkdirs() }
        val rawFile = File(rawDir, "bundle.fibd")

        try {
            ctx.contentResolver.openInputStream(bundleUri)?.use { input ->
                rawFile.outputStream().use { output -> input.copyTo(output) }
            } ?: return ModelBundleResult.Failure("Couldn't open the selected file (revoked SAF permission?)")
        } catch (t: Throwable) {
            Log.e(TAG, "Copy failed", t)
            return ModelBundleResult.Failure("Couldn't copy the model bundle: ${t.message}")
        }

        return try {
            RandomAccessFile(rawFile, "r").use { raf -> parseAndExtract(raf, outDir, preferFp16) }
        } catch (t: Throwable) {
            Log.e(TAG, "Bundle parse failed", t)
            ModelBundleResult.Failure("Failed to read the bundle: ${t.message}")
        } finally {
            rawFile.delete()
        }
    }

    private fun parseAndExtract(raf: RandomAccessFile, outDir: File, preferFp16: Boolean): ModelBundleResult {
        val magic = ByteArray(4)
        raf.readFully(magic)
        if (String(magic, Charsets.US_ASCII) != MAGIC) {
            return ModelBundleResult.Failure(
                "Not a model bundle (bad magic). Export one with build_model_bundle.py from the training notebook first."
            )
        }

        readU16LE(raf) // version — unused for now, format has been stable since v1
        val headerLen = readU32LE(raf)
        if (headerLen <= 0 || headerLen > 8 * 1024 * 1024) {
            return ModelBundleResult.Failure("Corrupt bundle header.")
        }
        val headerBytes = ByteArray(headerLen)
        raf.readFully(headerBytes)
        val header = JSONObject(String(headerBytes, Charsets.UTF_8))
        if (header.optString("format") != "frame-interpolator-model-bundle") {
            return ModelBundleResult.Failure("Unrecognized bundle format.")
        }

        val sectionsJson = header.optJSONObject("sections")
            ?: return ModelBundleResult.Failure("Bundle header has no sections.")
        val dataStart = raf.filePointer

        val sections = HashMap<String, BundleSection>()
        sectionsJson.keys().forEach { name ->
            val s = sectionsJson.getJSONObject(name)
            sections[name] = BundleSection(
                offset = s.getInt("offset"),
                compressedSize = s.getInt("compressed_size"),
                uncompressedSize = s.getInt("uncompressed_size"),
                sha256 = s.getString("sha256"),
            )
        }

        outDir.listFiles()?.forEach { it.delete() }

        val preferred = if (preferFp16) "fp16" else "fp32"
        val fallback = if (preferred == "fp16") "fp32" else "fp16"
        var precisionUsed: String? = null
        val extractedGraphs = mutableListOf<String>()

        for (graph in GRAPHS) {
            var tag = preferred
            var paramSec = sections["ncnn_${graph}_${tag}_param"]
            var binSec = sections["ncnn_${graph}_${tag}_bin"]
            if (paramSec == null || binSec == null) {
                tag = fallback
                paramSec = sections["ncnn_${graph}_${tag}_param"]
                binSec = sections["ncnn_${graph}_${tag}_bin"]
            }
            if (paramSec == null || binSec == null) {
                return ModelBundleResult.Failure(
                    "Bundle is missing the \"$graph\" graph (no fp16 or fp32 sections). Re-export from the notebook."
                )
            }

            val paramBytes = inflateSection(raf, dataStart, paramSec)
                ?: return ModelBundleResult.Failure("Checksum mismatch on ncnn_${graph}_${tag}_param — bundle is corrupt.")
            val binBytes = inflateSection(raf, dataStart, binSec)
                ?: return ModelBundleResult.Failure("Checksum mismatch on ncnn_${graph}_${tag}_bin — bundle is corrupt.")

            File(outDir, "$graph.ncnn.param").writeBytes(paramBytes)
            File(outDir, "$graph.ncnn.bin").writeBytes(binBytes)
            extractedGraphs += "$graph ($tag)"

            // Both graphs should agree on precision; if the bundle mixes them (one fp16, one
            // fp32 fallback) report the lower-common-denominator so the UI doesn't overclaim.
            precisionUsed = when {
                precisionUsed == null -> tag
                precisionUsed != tag -> "mixed"
                else -> precisionUsed
            }
        }

        return ModelBundleResult.Success(
            precisionUsed = precisionUsed ?: preferred,
            graphs = extractedGraphs,
            modelDir = outDir.absolutePath,
        )
    }

    private fun readU16LE(raf: RandomAccessFile): Int {
        val b0 = raf.read(); val b1 = raf.read()
        return (b0 and 0xFF) or ((b1 and 0xFF) shl 8)
    }

    private fun readU32LE(raf: RandomAccessFile): Int {
        val b0 = raf.read(); val b1 = raf.read(); val b2 = raf.read(); val b3 = raf.read()
        return (b0 and 0xFF) or ((b1 and 0xFF) shl 8) or ((b2 and 0xFF) shl 16) or ((b3 and 0xFF) shl 24)
    }

    private fun inflateSection(raf: RandomAccessFile, dataStart: Long, sec: BundleSection): ByteArray? {
        raf.seek(dataStart + sec.offset)
        val comp = ByteArray(sec.compressedSize)
        raf.readFully(comp)

        val inflater = Inflater()
        inflater.setInput(comp)
        val out = ByteArray(sec.uncompressedSize)
        var total = 0
        try {
            while (!inflater.finished() && total < out.size) {
                val n = inflater.inflate(out, total, out.size - total)
                if (n == 0 && inflater.needsInput()) break
                total += n
            }
        } finally {
            inflater.end()
        }
        if (total != sec.uncompressedSize) return null

        val sha = MessageDigest.getInstance("SHA-256").digest(out).joinToString("") { "%02x".format(it) }
        if (sha != sec.sha256) return null
        return out
    }
}
