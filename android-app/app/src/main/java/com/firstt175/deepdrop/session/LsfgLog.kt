package com.firstt175.deepdrop.session

import android.content.Context
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Logging wrapper that mirrors every call to both logcat and the shared
 * `filesDir/lsfg.log` file used by the crash reporter. Use instead of
 * android.util.Log for any message that should survive a remote test session.
 *
 * Initialised lazily from [init] (called by LsfgApplication). If init was
 * never reached, we silently fall through to logcat-only.
 */
object LsfgLog {

    private var file: File? = null
    private val timestampFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    // One BufferedWriter kept open for the process lifetime instead of a
    // fresh FileWriter per call. The previous version opened+wrote+closed a
    // new file handle (two syscalls plus object allocation) on every single
    // i()/w()/e() call across ~85 call sites in service/capture code — this
    // mirrors the persistent-fd approach the native ring logger already uses
    // for the same lsfg.log file. writeLock also confines timestampFormat
    // (a SimpleDateFormat, which is documented as not thread-safe) to one
    // critical section: LsfgLog is called from several different
    // HandlerThreads concurrently (capture, fps, service), so the shared
    // formatter was previously a latent race that could corrupt timestamps
    // or throw.
    private val writeLock = Any()
    private var writer: BufferedWriter? = null

    fun init(ctx: Context) {
        file = CrashReporter.logFile(ctx)
    }

    fun i(tag: String, msg: String) {
        Log.i(tag, msg)
        append('I', tag, msg, null)
    }

    fun w(tag: String, msg: String, tr: Throwable? = null) {
        if (tr != null) Log.w(tag, msg, tr) else Log.w(tag, msg)
        append('W', tag, msg, tr)
    }

    fun e(tag: String, msg: String, tr: Throwable? = null) {
        if (tr != null) Log.e(tag, msg, tr) else Log.e(tag, msg)
        append('E', tag, msg, tr)
    }

    private fun append(level: Char, tag: String, msg: String, tr: Throwable?) {
        val f = file ?: return
        synchronized(writeLock) {
            try {
                val w = writer ?: BufferedWriter(FileWriter(f, true)).also { writer = it }
                val ts = timestampFormat.format(Date())
                w.write(ts); w.write(" "); w.write(level.toString()); w.write("/")
                w.write(tag); w.write(": "); w.write(msg); w.newLine()
                if (tr != null) {
                    val sw = StringWriter()
                    tr.printStackTrace(PrintWriter(sw))
                    w.write(sw.toString().trimEnd()); w.newLine()
                }
                // Flush (cheap) rather than close (expensive): keeps the
                // on-disk copy current for crash forensics without paying
                // open/close syscalls on every line.
                w.flush()
            } catch (_: Throwable) {
                // Best-effort: never let logging crash the app. Drop the
                // writer so the next call retries a fresh open rather than
                // reusing a handle that may be in a bad state.
                writer = null
            }
        }
    }
}
