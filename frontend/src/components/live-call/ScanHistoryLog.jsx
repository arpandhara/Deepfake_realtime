/**
 * ScanHistoryLog.jsx
 * ─────────────────────────────────────────────
 * PHASE 2 — Scrollable history of last 20 scan results.
 * Most recent entry at top.
 * ─────────────────────────────────────────────
 * Props:
 *   history  {Array<{
 *     id:         string,
 *     label:      "REAL"|"FAKE"|"UNCERTAIN"|"UNKNOWN",
 *     confidence: number,
 *     timestamp:  number,   (Date.now())
 *     latency_ms: number,
 *   }>}
 * ─────────────────────────────────────────────
 */

import React from 'react';

const LABEL_STYLE = {
  REAL: { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
  FAKE: { text: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/20' },
  UNCERTAIN: { text: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
  UNKNOWN: { text: 'text-slate-500', bg: 'bg-white/5', border: 'border-white/10' },
};

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export default function ScanHistoryLog({ history = [] }) {
  if (history.length === 0) {
    return (
      <div
        id="scan-history-empty"
        className="flex flex-col items-center justify-center gap-2 py-8 text-center"
      >
        <span className="text-2xl opacity-20">📋</span>
        <p className="text-slate-600 text-xs font-medium">No scans yet — results will appear here</p>
      </div>
    );
  }

  return (
    <div id="scan-history-log" className="flex flex-col gap-1.5 max-h-64 overflow-y-auto pr-1 custom-scroll">
      {/* Column headers */}
      <div className="grid grid-cols-4 text-[10px] uppercase tracking-widest text-slate-600 font-semibold px-3 pb-1 border-b border-white/5">
        <span>Time</span>
        <span>Result</span>
        <span>Confidence</span>
        <span>Latency</span>
      </div>

      {/* Rows — newest first */}
      {[...history].reverse().map((entry, idx) => {
        const style = LABEL_STYLE[entry.label] ?? LABEL_STYLE.UNKNOWN;
        const isFirst = idx === 0;

        return (
          <div
            key={entry.id}
            id={`scan-history-entry-${entry.id}`}
            className={`
              grid grid-cols-4 items-center gap-1 px-3 py-2 rounded-xl
              border transition-all duration-300
              ${isFirst ? 'border-white/15 bg-white/5' : 'border-transparent bg-transparent'}
            `}
          >
            {/* Timestamp */}
            <span className="text-[11px] text-slate-500 font-mono tabular-nums">
              {formatTime(entry.timestamp)}
            </span>

            {/* Label pill */}
            <span
              className={`
                inline-flex w-fit items-center px-2 py-0.5 rounded-full
                text-[10px] font-bold tracking-wide border
                ${style.text} ${style.bg} ${style.border}
              `}
            >
              {entry.label === 'UNKNOWN' ? 'NO FACE' : entry.label}
            </span>

            {/* Confidence */}
            <span className={`text-[11px] font-semibold tabular-nums ${style.text}`}>
              {entry.label === 'UNKNOWN' ? '—' : `${(entry.confidence * 100).toFixed(1)}%`}
            </span>

            {/* Latency */}
            <span className="text-[11px] text-slate-600 font-mono tabular-nums">
              {entry.latency_ms != null ? `${Math.round(entry.latency_ms)}ms` : '—'}
            </span>
          </div>
        );
      })}

      {/* Scroll hint */}
      <style>{`
        .custom-scroll::-webkit-scrollbar        { width: 4px; }
        .custom-scroll::-webkit-scrollbar-track  { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb  { background: rgba(255,255,255,0.1); border-radius: 99px; }
      `}</style>
    </div>
  );
}
