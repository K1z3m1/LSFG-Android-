package com.firstt175.deepdrop.session

import android.util.Log
import com.firstt175.deepdrop.shizuku.IShizukuCaptureService
import com.firstt175.deepdrop.shizuku.IShizukuFrameCallback
import java.util.concurrent.atomic.AtomicBoolean

class ShizukuCaptureUserService : IShizukuCaptureService.Stub() {

    private val running = AtomicBoolean(false)
    private var worker: Thread? = null

    override fun startCapture(
        targetUid: Int,
        width: Int,
        height: Int,
        callback: IShizukuFrameCallback,
    ) {
        stopCapture()
        running.set(true)
        worker = Thread({
            runCaptureLoop(targetUid, width, height, callback)
        }, "lsfg-shizuku-capture").also { it.start() }
    }

    override fun stopCapture() {
        running.set(false)
        worker?.interrupt()
        worker = null
    }

    override fun describeBackend(): String {
        return "uid=${android.os.Process.myUid()} sdk=${android.os.Build.VERSION.SDK_INT}"
    }

    override fun destroy() {
        stopCapture()
        System.exit(0)
    }

    private fun runCaptureLoop(
        targetUid: Int,
        width: Int,
        height: Int,
        callback: IShizukuFrameCallback,
    ) {
        val capture = runCatching { PrivilegedScreenCapture(width, height, targetUid) }
            .getOrElse { e ->
                Log.w(TAG, "Unable to initialize privileged capture", e)
                callback.onError("Shizuku capture unavailable: ${e.message ?: e.javaClass.simpleName}")
                running.set(false)
                return
            }

        var lastFrameNs = 0L
        var frameLogCount = 0
        while (running.get()) {
            val hb = runCatching { capture.captureHardwareBuffer() }
                .onFailure {
                    Log.w(TAG, "captureHardwareBuffer failed", it)
                    callback.onError("Shizuku capture failed: ${it.message ?: it.javaClass.simpleName}")
                }
                .getOrNull()

            if (hb != null) {
                if (frameLogCount < 8) {
                    frameLogCount++
                    Log.i(TAG, "captured frame #$frameLogCount uid=$targetUid ${hb.width}x${hb.height} fmt=${hb.format}")
                }
                val timestampNs = System.nanoTime()
                val frameTimeNs = if (lastFrameNs > 0L) timestampNs - lastFrameNs else 0L
                lastFrameNs = timestampNs
                try {
                    callback.onFrameMetrics(timestampNs, frameTimeNs, 0L)
                    callback.onFrame(hb, timestampNs)
                } catch (t: Throwable) {
                    Log.w(TAG, "frame callback failed", t)
                    running.set(false)
                } finally {
                    runCatching { hb.close() }
                }
            }
        }
    }

    companion object {
        private const val TAG = "ShizukuUserCapture"
    }
}
