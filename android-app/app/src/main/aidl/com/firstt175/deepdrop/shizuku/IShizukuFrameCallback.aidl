package com.firstt175.deepdrop.shizuku;

import android.hardware.HardwareBuffer;

interface IShizukuFrameCallback {
    void onFrame(in HardwareBuffer buffer, long timestampNs);
    void onFrameMetrics(long timestampNs, long frameTimeNs, long pacingJitterNs);
    void onError(String message);
}
