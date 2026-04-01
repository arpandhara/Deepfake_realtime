/**
 * ScreenShareCapture.jsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PHASE 1 — Core Screen Capture Component
 *
 * Handles `getDisplayMedia()` screen sharing, exposes a `captureFrame()` helper
 * that returns a base64 JPEG of the current video frame, and passes the full
 * MediaStream (video + audio) back to the parent via callbacks.
 *
 * IMPORTANT:
 *  - Requires HTTPS or localhost (browser security requirement).
 *  - Audio is captured with `audio: true` but is NOT sent to /predict.
 *    The full stream is forwarded via `onStreamStarted(stream)` so the parent
 *    can hold the audio track for future audio-deepfake detection.
 *  - This component owns stream lifecycle — always stops tracks on unmount.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import React, { useRef, useState, useEffect, useCallback, useImperativeHandle, forwardRef } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Props:
 *   showPreview       {boolean}  — Whether to render the live screen preview <video>
 *   onFrameCaptured   {(base64: string) => void}  — Called each time captureFrame() fires
 *   onStreamStarted   {(stream: MediaStream) => void}  — Called when share begins (full stream)
 *   onStreamStopped   {() => void}  — Called when share ends (user stop OR track ended by OS)
 *   onError           {(err: string) => void}  — Called on getDisplayMedia or capture error
 *
 * Ref methods (via forwardRef):
 *   captureFrame()    {() => string | null}  — Returns base64 JPEG of the current frame
 *   stopStream()      {() => void}  — Programmatically stop the stream
 */
const ScreenShareCapture = forwardRef(function ScreenShareCapture(
  { showPreview = true, onFrameCaptured, onStreamStarted, onStreamStopped, onError },
  ref
) {
  const videoRef = useRef(null);   // <video> playing the screen stream
  const canvasRef = useRef(null);   // hidden <canvas> for frame extraction
  const streamRef = useRef(null);   // holds the live MediaStream

  const [isSharing, setIsSharing] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [httpsWarning, setHttpsWarning] = useState(false);

  // ── Check browser support on mount ────────────────────────────────────────
  useEffect(() => {
    const supported = !!navigator.mediaDevices?.getDisplayMedia;
    if (!supported) {
      setIsSupported(false);
    }
    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
      setHttpsWarning(true);
    }
    // onError intentionally excluded — it's a stable prop callback, not state
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Stop all tracks and notify parent ─────────────────────────────────────
  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsSharing(false);
    onStreamStopped?.();
  }, [onStreamStopped]);

  // ── Cleanup on unmount ─────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // ── captureFrame — draws current video frame to canvas, returns base64 ────
  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < 2) return null;  // not ready
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;  // no frame yet

    // Match canvas to current video resolution
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Return as base64 JPEG (strip the data:... prefix for the AI server)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    const base64 = dataUrl.split(',')[1];

    onFrameCaptured?.(base64);
    return base64;
  }, [onFrameCaptured]);

  // ── Start screen share ─────────────────────────────────────────────────────
  const startShare = useCallback(async () => {
    if (!isSupported) return;

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: { ideal: 30 },  // smooth preview
          displaySurface: 'window',    // hint to prefer window picker (not tab/screen)
        },
        audio: true,   // audio captured — forwarded to parent for future audio deepfake detection
      });

      streamRef.current = stream;

      // Attach video tracks to the preview element
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;  // mute preview locally to avoid echo
      }

      setIsSharing(true);
      onStreamStarted?.(stream);  // parent receives the full stream (incl. audio)

      // Listen for OS-level stop (user clicks "Stop Sharing" in browser toolbar)
      stream.getVideoTracks()[0].addEventListener('ended', () => {
        stopStream();
      });

    } catch (err) {
      if (err.name === 'NotAllowedError') {
        onError?.('Screen share permission denied. Please allow screen sharing and try again.');
      } else if (err.name === 'NotSupportedError') {
        onError?.('Screen sharing is not supported in this browser.');
      } else {
        onError?.(`Failed to start screen share: ${err.message}`);
      }
    }
  }, [isSupported, onStreamStarted, onError, stopStream]);

  // ── Expose captureFrame + stopStream to parent via ref ────────────────────
  useImperativeHandle(ref, () => ({
    captureFrame,
    stopStream,
  }), [captureFrame, stopStream]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-4 w-full">

      {/* ── HTTPS Warning ── */}
      {httpsWarning && (
        <div
          id="screen-capture-https-warning"
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-medium"
        >
          <span>⚠️</span>
          <span>Screen sharing requires HTTPS. It will work on localhost during development.</span>
        </div>
      )}

      {/* ── Browser Not Supported ── */}
      {!isSupported && (
        <div
          id="screen-capture-not-supported"
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs font-medium"
        >
          <span>❌</span>
          <span>Screen capture is not supported in this browser. Please use Chrome or Edge.</span>
        </div>
      )}

      {/* ── Screen Share Preview Video ── */}
      {showPreview && (
        <div
          id="screen-share-preview-container"
          className="relative w-full rounded-2xl overflow-hidden border-2 border-white/10 shadow-2xl shadow-violet-500/10 bg-black aspect-video"
        >
          <video
            id="screen-share-preview-video"
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-contain"
          />

          {/* Empty state overlay when not sharing */}
          {!isSharing && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/80">
              <div
                id="screen-share-idle-icon"
                className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-3xl"
              >
                🖥️
              </div>
              <p className="text-slate-500 text-sm font-medium">
                Share your call window to begin analysis
              </p>
            </div>
          )}

          {/* "LIVE" indicator badge when sharing */}
          {isSharing && (
            <div
              id="screen-share-live-badge"
              className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/90 backdrop-blur-sm text-white text-[11px] font-bold tracking-wide shadow-lg"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
              LIVE
            </div>
          )}

          {/* Scanning animation overlay when sharing */}
          {isSharing && (
            <div
              id="screen-share-scan-overlay"
              className="absolute inset-0 pointer-events-none overflow-hidden"
            >
              <div
                className="w-full h-[2px] bg-violet-500 shadow-[0_0_12px_3px_rgba(139,92,246,0.5)]"
                style={{ animation: 'liveScan 3s linear infinite' }}
              />
            </div>
          )}

          {/* Inline keyframes for the scan line */}
          <style>{`
            @keyframes liveScan {
              0%   { transform: translateY(0); }
              50%  { transform: translateY(calc(var(--preview-h, 400px) - 2px)); }
              100% { transform: translateY(0); }
            }
          `}</style>
        </div>
      )}

      {/* ── Hidden Canvas for frame extraction (never rendered visually) ── */}
      <canvas
        id="screen-capture-offscreen-canvas"
        ref={canvasRef}
        className="hidden"
        aria-hidden="true"
      />

      {/* ── Action Buttons ── */}
      <div id="screen-capture-controls" className="flex gap-3">
        {!isSharing ? (
          <button
            id="screen-share-start-btn"
            onClick={startShare}
            disabled={!isSupported}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-sm tracking-tight transition-all duration-300
              bg-violet-600 text-white hover:bg-violet-500
              shadow-[0_0_20px_rgba(139,92,246,0.3)] hover:shadow-[0_0_28px_rgba(139,92,246,0.5)]
              disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span>🖥️</span>
            Share Interview Window
          </button>
        ) : (
          <button
            id="screen-share-stop-btn"
            onClick={stopStream}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-sm tracking-tight transition-all duration-300
              bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20"
          >
            <span>⏹</span>
            Stop Sharing
          </button>
        )}
      </div>
    </div>
  );
});

export default ScreenShareCapture;
