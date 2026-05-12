"use client";

import { useState } from "react";
import { WebSocketProvider, useWs } from "@/components/WebSocketProvider";
import EchoGhost3D from "@/components/EchoGhost3D";
import ControlPanel from "@/components/ControlPanel";
import StatusHUD from "@/components/StatusHUD";

type TabId = "3d" | "spectrum" | "breathing" | "heatmap";

function TabBar({ active, onChange }: { active: TabId; onChange: (t: TabId) => void }) {
  const tabs: { id: TabId; label: string }[] = [
    { id: "3d", label: "3D Room" },
    { id: "spectrum", label: "Spectrum" },
    { id: "breathing", label: "Breathing" },
    { id: "heatmap", label: "Heatmap" },
  ];

  return (
    <div className="flex gap-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`text-xs font-medium px-3 py-1.5 rounded-md transition-colors ${
            active === t.id
              ? "bg-emerald-700 text-emerald-100"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

function SpectrumView() {
  const { frame } = useWs();
  if (!frame?.spectrum?.length) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
        No spectrum data yet
      </div>
    );
  }

  const values = frame.spectrum;
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;
  const barWidth = 100 / values.length;

  return (
    <div className="flex items-end h-full gap-[1px] px-1">
      {values.map((v, i) => {
        const pct = ((v - minVal) / range) * 100;
        const hue = 120 - ((v - minVal) / range) * 120; // green to red
        return (
          <div
            key={i}
            className="flex-1 rounded-t"
            style={{
              height: `${Math.max(pct, 1)}%`,
              backgroundColor: `hsl(${hue}, 80%, 50%)`,
              width: `${barWidth}%`,
            }}
          />
        );
      })}
    </div>
  );
}

function BreathingView() {
  const { frame } = useWs();
  const bpm = frame?.breathing?.bpm;
  const conf = frame?.breathing?.confidence;

  return (
    <div className="flex flex-col items-center justify-center h-full text-center">
      {bpm ? (
        <>
          <p className="text-6xl font-bold text-emerald-400 font-mono">
            {bpm.toFixed(1)}
          </p>
          <p className="text-sm text-zinc-500 mt-2">BPM</p>
          <p className="text-xs text-zinc-600 mt-1">
            confidence: {(conf ?? 0).toFixed(3)}
          </p>
        </>
      ) : (
        <p className="text-zinc-600 text-sm">No breathing data yet</p>
      )}
      <div className="mt-4 text-xs text-zinc-500">
        <p>Motion: {frame?.motion?.label ?? "—"}</p>
        <p>Score: {(frame?.motion?.score ?? 0).toFixed(6)}</p>
        <p>Confidence: {(frame?.motion?.confidence ?? 0).toFixed(3)}</p>
      </div>
    </div>
  );
}

function HeatmapView() {
  const { frame } = useWs();
  const rows = frame?.heatmap_rows ?? 0;
  const cols = frame?.heatmap_cols ?? 0;
  const data = frame?.heatmap_data;

  if (!data?.length || rows === 0 || cols === 0) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
        No heatmap data yet
      </div>
    );
  }

  const minVal = Math.min(...data);
  const maxVal = Math.max(...data);
  const range = maxVal - minVal || 1;

  const cellH = 100 / rows;
  const cellW = 100 / cols;

  return (
    <div className="flex flex-col h-full p-1">
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex" style={{ height: `${cellH}%` }}>
          {Array.from({ length: cols }).map((_, c) => {
            const idx = r * cols + c;
            const v = data[idx] ?? minVal;
            const pct = (v - minVal) / range;
            const hue = 240 - pct * 240;
            return (
              <div
                key={c}
                className="flex-1"
                style={{
                  backgroundColor: `hsl(${hue}, 80%, ${20 + pct * 40}%)`,
                  width: `${cellW}%`,
                }}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

function DashboardContent() {
  const [activeTab, setActiveTab] = useState<TabId>("3d");
  const { frame } = useWs();

  const motionLabel = frame?.motion?.label ?? "—";
  const motionScore = frame?.motion?.score ?? 0;
  const bpm = frame?.breathing?.bpm;

  return (
    <div className="h-screen w-screen bg-[#050a18] text-zinc-200 flex flex-col overflow-hidden">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-[#0a0f1e] shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
          <h1 className="text-sm font-bold tracking-widest text-emerald-400 uppercase">
            EchoGhost Hub Ultra
          </h1>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs font-mono text-zinc-500">
            {frame?.mode?.toUpperCase() ?? "—"} | {frame?.waveform?.toUpperCase() ?? "—"}
          </span>
          <span className={`text-xs font-mono ${motionScore > 0.01 ? "text-orange-400" : "text-zinc-600"}`}>
            {motionLabel}
          </span>
          {bpm && (
            <span className="text-xs font-mono text-cyan-400">
              {bpm.toFixed(1)} BPM
            </span>
          )}
        </div>
      </header>

      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">
        {/* 3D / Visualization area */}
        <main className="flex-1 relative">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="absolute inset-0 top-10">
            {activeTab === "3d" && <EchoGhost3D />}
            {activeTab === "spectrum" && (
              <div className="h-full p-4">
                <SpectrumView />
              </div>
            )}
            {activeTab === "breathing" && (
              <div className="h-full p-4">
                <BreathingView />
              </div>
            )}
            {activeTab === "heatmap" && (
              <div className="h-full p-4">
                <HeatmapView />
              </div>
            )}
          </div>
        </main>

        {/* Sidebar */}
        <aside className="w-64 bg-[#0a0f1e] border-l border-zinc-800 p-4 overflow-y-auto shrink-0">
          <ControlPanel />

          <hr className="border-zinc-800 my-4" />

          <div className="text-xs text-zinc-500">
            <h3 className="font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
              Status
            </h3>
            <StatusHUD />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <WebSocketProvider>
      <DashboardContent />
    </WebSocketProvider>
  );
}
