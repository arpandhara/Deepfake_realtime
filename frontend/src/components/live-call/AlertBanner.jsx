/**
 * AlertBanner.jsx
 * ─────────────────────────────────────────────
 * PHASE 2 — Dismissible alert shown when N consecutive
 * FAKE detections are recorded.
 * Auto-reappears on the next threshold hit after dismiss.
 * ─────────────────────────────────────────────
 * Props:
 *   consecutiveFakes {number}   current run of FAKE results
 *   threshold        {number}   default 3 — how many FAKEs before alerting
 * ─────────────────────────────────────────────
 */

import React, { useState } from 'react';

/**
 * Inner banner — receives a `resetKey` so that when the parent
 * mounts a fresh instance (key changes), dismissed state resets to false.
 */
function AlertBannerInner({ consecutiveFakes }) {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <div
      id="fake-alert-banner"
      role="alert"
      aria-live="assertive"
      className="relative flex items-start gap-3 px-5 py-4 rounded-2xl
        bg-red-500/10 border border-red-500/30
        shadow-xl shadow-red-500/10
        animate-[fadeInDown_0.3s_ease_forwards]"
    >
      {/* Animated pulse ring */}
      <span className="absolute inset-0 rounded-2xl border border-red-500/30 animate-ping opacity-20 pointer-events-none" />

      {/* Icon */}
      <span className="text-2xl shrink-0" role="img" aria-label="Warning">⚠️</span>

      {/* Text content */}
      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
        <p className="text-red-400 font-bold text-sm tracking-tight leading-snug">
          Deepfake Alert
        </p>
        <p className="text-red-300/70 text-xs leading-relaxed">
          {consecutiveFakes} consecutive FAKE detections recorded. The person on screen may be AI-generated. Proceed with caution.
        </p>
      </div>

      {/* Dismiss button */}
      <button
        id="fake-alert-dismiss-btn"
        onClick={() => setDismissed(true)}
        aria-label="Dismiss alert"
        className="shrink-0 text-red-400/60 hover:text-red-300 text-lg transition-colors duration-200 mt-0.5"
      >
        ✕
      </button>

      {/* Fade-in animation */}
      <style>{`
        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

/**
 * Public component — tracks which threshold-crossing episode we're on
 * and uses that as a React key so AlertBannerInner resets dismissed state
 * each time a new streak crosses the threshold.
 */
export default function AlertBanner({ consecutiveFakes = 0, threshold = 3 }) {
  if (consecutiveFakes < threshold) return null;

  // Each new streak episode gets its own key → fresh dismissed=false state
  const episodeKey = Math.floor(consecutiveFakes / threshold);

  return (
    <AlertBannerInner
      key={episodeKey}
      consecutiveFakes={consecutiveFakes}
    />
  );
}
