"use client";

import { useState } from "react";
import { useWs } from "./WebSocketProvider";
import type { SessionConfig, OperatingMode, WaveformKind, BackendKind } from "@/lib/types";

const MODES: OperatingMode[] = ["simulation", "active", "passive"];
const WAVEFORMS: WaveformKind[] = [
  "tone",
  "chirp",
  "chaotic",
  "chaotic_henon",
  "chaotic_lorenz",
  "chaotic_ks",
  "prn",
];
const BACKENDS: BackendKind[] = ["simulation", "hackrf", "soapy", "multi_hackrf"];
const FRAME_SIZES = [1024, 2048, 4096, 8192];

export default function ControlPanel() {
  const { connected, startSession, stopSession, updateConfig } = useWs();

  const [mode, setMode] = useState<OperatingMode>("simulation");
  const [waveform, setWaveform] = useState<WaveformKind>("tone");
  const [backend, setBackend] = useState<BackendKind>("simulation");
  const [freqMhz, setFreqMhz] = useState(915);
  const [sampleRateMhz, setSampleRateMhz] = useState(2);
  const [frameSize, setFrameSize] = useState(4096);
  const [txGain, setTxGain] = useState(18);
  const [rxGain, setRxGain] = useState(24);
  const [adaptive, setAdaptive] = useState(false);
  const [running, setRunning] = useState(false);

  const buildConfig = (): SessionConfig => ({
    mode,
    waveform,
    backend,
    center_frequency_hz: freqMhz * 1e6,
    sample_rate_sps: sampleRateMhz * 1e6,
    frame_size: frameSize,
    tx_gain_db: txGain,
    rx_gain_db: rxGain,
    adaptive,
  });

  const handleStart = () => {
    startSession(buildConfig());
    setRunning(true);
  };

  const handleStop = () => {
    stopSession();
    setRunning(false);
  };

  const handleApply = () => {
    updateConfig(buildConfig());
  };

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">
        Controls
      </h2>

      <div>
        <label className="text-xs text-zinc-500 block mb-1">Mode</label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as OperatingMode)}
          className="w-full bg-zinc-800 text-zinc-200 text-xs rounded px-2 py-1.5 border border-zinc-700"
        >
          {MODES.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="text-xs text-zinc-500 block mb-1">Waveform</label>
        <select
          value={waveform}
          onChange={(e) => setWaveform(e.target.value as WaveformKind)}
          className="w-full bg-zinc-800 text-zinc-200 text-xs rounded px-2 py-1.5 border border-zinc-700"
        >
          {WAVEFORMS.map((w) => (
            <option key={w} value={w}>{w}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="text-xs text-zinc-500 block mb-1">Backend</label>
        <select
          value={backend}
          onChange={(e) => setBackend(e.target.value as BackendKind)}
          className="w-full bg-zinc-800 text-zinc-200 text-xs rounded px-2 py-1.5 border border-zinc-700"
        >
          {BACKENDS.map((b) => (
            <option key={b} value={b}>{b}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="text-xs text-zinc-500 block mb-1">
          Center Frequency: {freqMhz} MHz
        </label>
        <input
          type="range"
          min={100}
          max={6000}
          step={1}
          value={freqMhz}
          onChange={(e) => setFreqMhz(Number(e.target.value))}
          className="w-full accent-emerald-500"
        />
      </div>

      <div>
        <label className="text-xs text-zinc-500 block mb-1">
          Sample Rate: {sampleRateMhz} MSPS
        </label>
        <input
          type="range"
          min={0.5}
          max={20}
          step={0.5}
          value={sampleRateMhz}
          onChange={(e) => setSampleRateMhz(Number(e.target.value))}
          className="w-full accent-emerald-500"
        />
      </div>

      <div>
        <label className="text-xs text-zinc-500 block mb-1">Frame Size</label>
        <div className="flex gap-1">
          {FRAME_SIZES.map((fs) => (
            <button
              key={fs}
              onClick={() => setFrameSize(fs)}
              className={`text-xs px-2 py-1 rounded ${
                frameSize === fs
                  ? "bg-emerald-700 text-emerald-200"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              {fs}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-2 items-center">
        <label className="text-xs text-zinc-500">Adaptive</label>
        <input
          type="checkbox"
          checked={adaptive}
          onChange={(e) => setAdaptive(e.target.checked)}
          className="accent-emerald-500"
        />
      </div>

      <div className="pt-2 space-y-2">
        {!running ? (
          <button
            onClick={handleStart}
            disabled={!connected}
            className="w-full text-xs font-semibold py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-black disabled:bg-zinc-800 disabled:text-zinc-600 transition-colors"
          >
            START SESSION
          </button>
        ) : (
          <>
            <button
              onClick={handleApply}
              className="w-full text-xs font-semibold py-2 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
            >
              APPLY CONFIG
            </button>
            <button
              onClick={handleStop}
              className="w-full text-xs font-semibold py-2 rounded bg-red-700 hover:bg-red-600 text-white transition-colors"
            >
              STOP
            </button>
          </>
        )}
      </div>
    </div>
  );
}
