package com.firstt175.deepdrop.shizuku;

import com.firstt175.deepdrop.shizuku.IShizukuFrameCallback;

interface IShizukuCaptureService {
    void startCapture(int targetUid, int width, int height, IShizukuFrameCallback callback) = 1;
    void stopCapture() = 2;
    String describeBackend() = 3;
    void destroy() = 16777114;
}
