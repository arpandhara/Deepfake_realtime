/**
 * DetectionBadge.jsx
 * ─────────────────────────────────────────────
 * PHASE 2 — Stateless result badge component.
 * Shows REAL / FAKE / UNCERTAIN with icon + color.
 * ─────────────────────────────────────────────
 * Props:
 *   label      {string|null}  "REAL" | "FAKE" | "UNCERTAIN" | "UNKNOWN" | null
 *   confidence {number}       0.0 – 1.0
 *   latency_ms {number}       server response time in ms
 * ─────────────────────────────────────────────
 */

import React from 'react';

const CONFIG = {
  REAL: {
    icon: '🛡️',
    label: 'REAL',
    textColor: 'text-emerald-400',
    borderColor: 'border-emerald-500/30',
    bgColor: 'bg-emerald-500/10',
    glowColor: 'shadow-emerald-500/20',
    pulse: false,
    description: 'Person appears to be genuine',
  },
  FAKE: {
    icon: '⚠️',
    label: 'FAKE',
    textColor: 'text-red-400',
    borderColor: 'border-red-500/30',
    bgColor: 'bg-red-500/10',
    glowColor: 'shadow-red-500/20',
    pulse: true,
    description: 'Possible AI-generated face detected',
  },
  UNCERTAIN: {
    icon: '❓',
    label: 'UNCERTAIN',
    textColor: 'text-amber-400',
    borderColor: 'border-amber-500/30',
    bgColor: 'bg-amber-500/10',
    glowColor: 'shadow-amber-500/20',
    pulse: false,
    description: 'Low confidence — try adjusting angle',
  },
  UNKNOWN: {
    icon: '👤',
    label: 'NO FACE',
    textColor: 'text-slate-400',
    borderColor: 'border-white/10',
    bgColor: 'bg-white/5',
    glowColor: 'shadow-transparent',
    pulse: false,
    description: 'No face detected in frame',
  },
};

export default function DetectionBadge({ label = null, confidence = 0, latency_ms = null }) {
  if (!label) {
    // Idle state
    return (
      <div
        id="detection-badge-idle"
        className="flex flex-col items-center justify-center gap-3 p-8 rounded-2xl bg-white/5 border border-white/10 text-center"
      >
        <span className="text-4xl opacity-30">🔍</span>
        <p className="text-slate-500 text-sm font-medium">Awaiting first scan…</p>
      </div>
    );
  }

  const cfg = CONFIG[label] ?? CONFIG.UNKNOWN;

  return (
    <div
      id={`detection-badge-${label.toLowerCase()}`}
      className={`
        relative flex flex-col items-center gap-4 p-8 rounded-2xl
        border ${cfg.borderColor} ${cfg.bgColor}
        shadow-xl ${cfg.glowColor}
        transition-all duration-500
      `}
    >
      {/* Pulse ring for FAKE detections */}
      {cfg.pulse && (
        <span className="absolute inset-0 rounded-2xl border border-red-500/40 animate-ping opacity-30 pointer-events-none" />
      )}

      {/* Icon */}
      <span className="text-5xl" role="img" aria-label={cfg.label}>
        {cfg.icon}
      </span>

      {/* Label */}
      <span
        className={`text-4xl font-black tracking-tight ${cfg.textColor}`}
        style={{ fontFamily: "'Inter', sans-serif" }}
      >
        {cfg.label}
      </span>

      {/* Confidence */}
      {confidence > 0 && (
        <span className="text-sm text-slate-300 font-medium">
          Confidence:{' '}
          <span className={`font-bold ${cfg.textColor}`}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </span>
      )}

      {/* Description */}
      <p className="text-xs text-slate-500 text-center leading-relaxed max-w-[180px]">
        {cfg.description}
      </p>

      {/* Latency chip */}
      {latency_ms !== null && (
        <span
          id="detection-badge-latency"
          className="absolute top-3 right-3 text-[10px] text-slate-600 font-mono bg-white/5 px-2 py-0.5 rounded-full"
        >
          {Math.round(latency_ms)}ms
        </span>
      )}
    </div>
  );
}
