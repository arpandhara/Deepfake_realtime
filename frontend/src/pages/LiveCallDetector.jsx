/**
 * LiveCallDetector.jsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PHASE 3 — Main "Live Interview Detector" page.
 *
 * Orchestrates:
 *   - ScreenShareCapture  →  grabs frames every `scanInterval` ms
 *   - POST /predict       →  FastAPI AI server (port 8000)
 *   - DetectionBadge      →  shows REAL / FAKE / UNCERTAIN
 *   - ConfidenceMeter     →  animated confidence bar
 *   - ScanHistoryLog      →  rolling 20-entry result log
 *   - AlertBanner         →  fires after 3 consecutive FAKEs
 *
 * Environment:
 *   VITE_AI_SERVER_URL  — FastAPI base URL (default: http://localhost:8000)
 *   (VITE_BACKEND_URL is the Node backend — NOT used here; /predict is called directly)
 *
 * Protected files guarantee: This file does NOT touch any .env, backend, or
 * model files. All server communication is via existing /predict endpoint only.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { UserButton } from '@clerk/clerk-react';

import ScreenShareCapture from '../components/live-call/ScreenShareCapture';
import DetectionBadge from '../components/live-call/DetectionBadge';
import ConfidenceMeter from '../components/live-call/ConfidenceMeter';
import ScanHistoryLog from '../components/live-call/ScanHistoryLog';
import AlertBanner from '../components/live-call/AlertBanner';

// ─── Constants ───────────────────────────────────────────────────────────────
const AI_SERVER_URL = import.meta.env.VITE_AI_SERVER_URL || 'http://localhost:8000';
const MAX_HISTORY = 20;    // rolling log limit
const CONSECUTIVE_THRESHOLD = 3; // # of FAKEs before alert fires

// Scan interval options (ms)
const INTERVAL_OPTIONS = [
  { label: '1s', value: 1000 },
  { label: '2s', value: 2000 },
  { label: '3s', value: 3000 },
  { label: '5s', value: 5000 },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────
function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

// ─── Main Component ──────────────────────────────────────────────────────────
export default function LiveCallDetector() {
  // Screen share capture ref (access captureFrame / stopStream imperatively)
  const captureRef = useRef(null);

  // Polling interval ref
  const intervalRef = useRef(null);

  // ── State ──────────────────────────────────────────────────────────────────
  const [isCapturing, setIsCapturing] = useState(false);
  const [prediction, setPrediction] = useState(null);    // latest AI result
  const [history, setHistory] = useState([]);       // last MAX_HISTORY results
  const [consecutiveFakes, setConsecutiveFakes] = useState(0);
  const [scanInterval, setScanInterval] = useState(2000);
  const [error, setError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);   // request in-flight
  const [totalScans, setTotalScans] = useState(0);
  const [fakeCount, setFakeCount] = useState(0);

  // ── Core: send one frame to /predict ──────────────────────────────────────
  const runPrediction = useCallback(async () => {
    const capture = captureRef.current;
    if (!capture) return;

    const base64 = capture.captureFrame();
    if (!base64) return;  // video not ready or no frame

    setIsAnalyzing(true);

    try {
      const response = await fetch(`${AI_SERVER_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_b64: base64 }),
      });

      if (!response.ok) {
        throw new Error(`Server error ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      // data: { label, confidence, uncertain, face_detected, latency_ms }

      // Resolve the display label
      let displayLabel = data.label;
      if (!data.face_detected) displayLabel = 'UNKNOWN';
      else if (data.uncertain) displayLabel = 'UNCERTAIN';

      const entry = {
        id: generateId(),
        label: displayLabel,
        confidence: data.confidence ?? 0,
        latency_ms: data.latency_ms ?? null,
        timestamp: Date.now(),
      };

      // Update latest prediction
      setPrediction(entry);

      // Update rolling history (max MAX_HISTORY)
      setHistory(prev => {
        const next = [entry, ...prev];
        return next.slice(0, MAX_HISTORY);
      });

      // Track total scans and FAKE count
      setTotalScans(n => n + 1);
      if (displayLabel === 'FAKE') setFakeCount(n => n + 1);

      // Update consecutive FAKE counter
      setConsecutiveFakes(prev => {
        if (displayLabel === 'FAKE') return prev + 1;
        return 0;  // any non-FAKE resets the streak
      });

      // Clear any connection error
      setError(null);

    } catch (err) {
      console.error('[LiveCallDetector] Prediction error:', err);
      setError(err.message || 'Failed to reach the AI server. Is it running on port 8000?');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  // ── Start / stop polling loop ──────────────────────────────────────────────
  useEffect(() => {
    if (isCapturing) {
      // Immediate first scan, then interval
      runPrediction();
      intervalRef.current = setInterval(runPrediction, scanInterval);
    } else {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => clearInterval(intervalRef.current);
  }, [isCapturing, scanInterval, runPrediction]);

  // ── Stream lifecycle callbacks (from ScreenShareCapture) ──────────────────
  const handleStreamStarted = useCallback(() => {
    setIsCapturing(true);
    setError(null);
    setConsecutiveFakes(0);
  }, []);

  const handleStreamStopped = useCallback(() => {
    setIsCapturing(false);
    setIsAnalyzing(false);
    setPrediction(null);
    setConsecutiveFakes(0);
  }, []);

  const handleCaptureError = useCallback((msg) => {
    setError(msg);
    setIsCapturing(false);
  }, []);

  // ── Derived stats ─────────────────────────────────────────────────────────
  const fakeRate = totalScans > 0 ? ((fakeCount / totalScans) * 100).toFixed(0) : '—';

  // ─── Render ─────────────────────────────────────────────────────────────
  return (
    <div
      className="min-h-screen flex flex-col bg-[#050505] text-white"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >

      {/* ── Header ── */}
      <header className="fixed top-0 w-full px-8 py-4 flex justify-between items-center z-50 bg-[#050505]/80 backdrop-blur-xl border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-[10px]">
            🛡️
          </div>
          <span className="text-[13px] font-semibold tracking-tight">
            DeepSheild.ai
          </span>
          <span className="text-white/20 text-xs">•</span>
          <span className="text-[12px] text-slate-400 font-medium">
            Live Interview Detector
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Live indicator */}
          {isCapturing && (
            <div
              id="live-indicator"
              className="flex items-center gap-1.5 text-[11px] font-semibold text-red-400"
            >
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              ANALYZING
            </div>
          )}
          <UserButton
            afterSignOutUrl="/"
            appearance={{ elements: { userButtonAvatarBox: 'w-8 h-8 border border-white/10' } }}
          />
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="flex-1 flex flex-col lg:flex-row gap-6 p-6 mt-[73px]">

        {/* ══ LEFT COLUMN — Screen Share Preview + Controls ══ */}
        <div className="flex flex-col gap-5 flex-1 min-w-0">

          {/* Section title */}
          <div>
            <h1 className="text-base font-bold tracking-tight">Interview Window</h1>
            <p className="text-slate-500 text-xs mt-0.5">
              Share the window showing your call to begin real-time analysis
            </p>
          </div>

          {/* Screen Share Capture component */}
          <ScreenShareCapture
            ref={captureRef}
            showPreview={true}
            onStreamStarted={handleStreamStarted}
            onStreamStopped={handleStreamStopped}
            onError={handleCaptureError}
          />

          {/* Scan interval selector */}
          <div
            id="scan-interval-selector"
            className="flex flex-col gap-2 p-4 rounded-2xl bg-white/5 border border-white/10"
          >
            <span className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold">
              Scan Frequency
            </span>
            <div className="flex gap-2 flex-wrap">
              {INTERVAL_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  id={`interval-btn-${opt.label}`}
                  onClick={() => setScanInterval(opt.value)}
                  disabled={isCapturing}
                  className={`
                    px-4 py-1.5 rounded-lg text-xs font-semibold tracking-tight
                    transition-all duration-200
                    ${scanInterval === opt.value
                      ? 'bg-violet-600 text-white shadow-[0_0_12px_rgba(139,92,246,0.3)]'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10 border border-white/10'
                    }
                    disabled:opacity-40 disabled:cursor-not-allowed
                  `}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-600 mt-1">
              {isCapturing
                ? 'Stop sharing to change scan frequency'
                : `One scan every ${INTERVAL_OPTIONS.find(o => o.value === scanInterval)?.label}`
              }
            </p>
          </div>

          {/* Error message */}
          {error && (
            <div
              id="live-call-error"
              className="flex items-start gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs"
            >
              <span className="shrink-0">❌</span>
              <span>{error}</span>
            </div>
          )}

          {/* Session stats row */}
          {totalScans > 0 && (
            <div
              id="session-stats"
              className="grid grid-cols-3 gap-3"
            >
              {[
                { label: 'Total Scans', value: totalScans },
                { label: 'FAKE Detected', value: fakeCount },
                { label: 'FAKE Rate', value: `${fakeRate}%` },
              ].map(stat => (
                <div
                  key={stat.label}
                  className="flex flex-col items-center gap-1 p-3 rounded-xl bg-white/5 border border-white/10"
                >
                  <span className="text-lg font-black text-white tabular-nums">
                    {stat.value}
                  </span>
                  <span className="text-[10px] text-slate-500 font-medium text-center">
                    {stat.label}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ══ RIGHT COLUMN — Detection Results ══ */}
        <div className="flex flex-col gap-4 w-full lg:w-[360px] shrink-0">

          {/* Section title */}
          <div>
            <h2 className="text-base font-bold tracking-tight">Detection Result</h2>
            <p className="text-slate-500 text-xs mt-0.5">
              Real-time analysis of the interviewee's face
            </p>
          </div>

          {/* Alert banner */}
          <AlertBanner
            consecutiveFakes={consecutiveFakes}
            threshold={CONSECUTIVE_THRESHOLD}
          />

          {/* Detection badge */}
          <DetectionBadge
            label={prediction?.label ?? null}
            confidence={prediction?.confidence ?? 0}
            latency_ms={prediction?.latency_ms ?? null}
          />

          {/* Confidence meter */}
          <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
            <ConfidenceMeter
              confidence={prediction?.confidence ?? 0}
              label={prediction?.label ?? null}
            />
          </div>

          {/* In-flight indicator */}
          {isAnalyzing && (
            <div
              id="analyzing-indicator"
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400 text-xs font-medium"
            >
              <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
              <span>Analyzing frame…</span>
            </div>
          )}

          {/* Scan history log */}
          <div className="flex-1 p-4 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex justify-between items-center mb-3">
              <span className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold">
                Scan History
              </span>
              {history.length > 0 && (
                <span className="text-[10px] text-slate-600">
                  {history.length} / {MAX_HISTORY}
                </span>
              )}
            </div>
            <ScanHistoryLog history={history} />
          </div>

        </div>
      </main>
    </div>
  );
}
