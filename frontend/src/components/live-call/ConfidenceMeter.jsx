/**
 * ConfidenceMeter.jsx
 * ─────────────────────────────────────────────
 * PHASE 2 — Animated confidence progress bar.
 * Color zones: green (>80%), amber (60–80%), red (<60%).
 * ─────────────────────────────────────────────
 * Props:
 *   confidence {number}  0.0 – 1.0
 *   label      {string}  "REAL" | "FAKE" | "UNCERTAIN" | "UNKNOWN" | null
 * ─────────────────────────────────────────────
 */

import React from 'react';

function getBarColor(label, confidence) {
  if (label === 'FAKE') return 'bg-red-500';
  if (label === 'REAL' && confidence > 0.8) return 'bg-emerald-500';
  if (label === 'REAL') return 'bg-emerald-400';
  if (label === 'UNCERTAIN') return 'bg-amber-400';
  return 'bg-slate-500';
}

function getGlowColor(label) {
  if (label === 'FAKE') return 'shadow-[0_0_12px_rgba(239,68,68,0.4)]';
  if (label === 'REAL') return 'shadow-[0_0_12px_rgba(52,211,153,0.4)]';
  if (label === 'UNCERTAIN') return 'shadow-[0_0_12px_rgba(251,191,36,0.4)]';
  return '';
}

export default function ConfidenceMeter({ confidence = 0, label = null }) {
  const pct = Math.round((confidence ?? 0) * 100);
  const barColor = getBarColor(label, confidence);
  const glow = getGlowColor(label);

  return (
    <div id="confidence-meter" className="w-full flex flex-col gap-2">
      {/* Header row */}
      <div className="flex justify-between items-center">
        <span className="text-[11px] uppercase tracking-widest text-slate-500 font-semibold">
          Confidence
        </span>
        <span
          id="confidence-meter-pct"
          className="text-sm font-bold text-white tabular-nums"
        >
          {label ? `${pct}%` : '—'}
        </span>
      </div>

      {/* Track */}
      <div className="relative w-full h-3 rounded-full bg-white/5 overflow-hidden border border-white/10">
        {/* Fill bar — CSS transition handles animation */}
        <div
          id="confidence-meter-fill"
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor} ${glow}`}
          style={{ width: label ? `${pct}%` : '0%' }}
        />

        {/* Shimmer overlay */}
        {label && pct > 0 && (
          <div
            className="absolute inset-0 rounded-full opacity-30"
            style={{
              background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)',
              animation: 'shimmer 2.5s infinite',
              backgroundSize: '200% 100%',
            }}
          />
        )}
      </div>

      {/* Zone labels */}
      <div className="flex justify-between text-[10px] text-slate-600 font-medium">
        <span>0%</span>
        <span className="text-amber-600">60%</span>
        <span className="text-emerald-600">80%</span>
        <span>100%</span>
      </div>

      {/* Shimmer keyframe */}
      <style>{`
        @keyframes shimmer {
          0%   { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>
    </div>
  );
}
