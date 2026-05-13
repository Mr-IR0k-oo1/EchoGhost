"use client";

/**
 * ControlPanel — RF and mode configuration controls.
 *
 * Provides sliders, buttons, and toggles for:
 *   - Mode selection (active chaotic, passive WiFi, motion, etc.)
 *   - Frequency, sample rate, gain settings
 *   - Waveform type selection
 *   - TX amplitude with safety limits
 *   - TCXO toggle
 *   - Start/Stop streaming
 *   - Safety status indicator
 *
 * All config changes are sent to the backend via WebSocket.
 */

import { useState, useCallback, useEffect } from "react";
import type { RadarMode, WaveformType, ConnectionStatus } from "../lib/socket";
import { socket } from "../lib/socket";

// ── Mode Definitions ──────────────────────────────────────────────────────

interface ModeInfo {
  id: RadarMode;
  label: string;
  description: string;
  color: string;
}

const MODES: ModeInfo[] = [
  {
    id: "active_chaotic",
    label: "Chaotic",
    description: "LPI noise-like spread-spectrum",
    color: "#00f5ff",
  },
  {
    id: "passive_wifi",
    label: "Passive WiFi",
    description: "Use ambient WiFi signals",
    color: "#00ff88",
  },
  {
    id: "motion",
    label: "Motion",
    description: "Movement detection & tracking",
    color: "#ffaa00",
  },
  {
    id: "breathing",
    label: "Breathing",
    description: "Chest micro-movement vital signs",
    color: "#00ff88",
  },
  {
    id: "micro_doppler",
    label: "Micro-Doppler",
    description: "Activity recognition",
    color: "#ff00e5",
  },
  {
    id: "range_doppler",
    label: "Range-Doppler",
    description: "Range-velocity heatmap",
    color: "#00f5ff",
  },
  {
    id: "through_wall",
    label: "Through-Wall",
    description: "915 MHz through-obstacle",
    color: "#ff3355",
  },
  {
    id: "gesture",
    label: "Gesture",
    description: "Hand gesture control",
    color: "#ff00e5",
  },
  {
    id: "artistic",
    label: "Generative",
    description: "Motion → audio/visual art",
    color: "#aa00ff",
  },
  {
    id: "security",
    label: "Security",
    description: "Anomaly / intrusion detection",
    color: "#ff3355",
  },
];

const WAVEFORMS: { id: WaveformType; label: string }[] = [
  { id: "tone", label: "Tone (CW)" },
  { id: "fmcw", label: "FMCW (Chirp)" },
  { id: "chaotic", label: "Chaotic (Lorenz)" },
  { id: "noise", label: "Noise (Spread)" },
];

// ── Props ─────────────────────────────────────────────────────────────────

interface ControlPanelProps {
  connectionStatus: ConnectionStatus;
  onConfigChange?: () => void;
}

// ── Component ─────────────────────────────────────────────────────────────

export default function ControlPanel({
  connectionStatus,
  onConfigChange,
}: ControlPanelProps) {
  // Mode
  const [mode, setMode] = useState<RadarMode>("active_chaotic");
  const [waveform, setWaveform] = useState<WaveformType>("chaotic");

  // RF params
  const [frequency, setFrequency] = useState(915); // MHz
  const [sampleRate, setSampleRate] = useState(20); // Msps
  const [lnaGain, setLnaGain] = useState(16);
  const [vgaGain, setVgaGain] = useState(16);
  const [txGain, setTxGain] = useState(10);
  const [txAmplitude, setTxAmplitude] = useState(0.5);
  const [ampEnable, setAmpEnable] = useState(false);
  const [useTCXO, setUseTCXO] = useState(true);

  // Streaming
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [safetyOk, setSafetyOk] = useState(true);

  // Subscribe to WebSocket events
  useEffect(() => {
    const unsubStatus = socket.onStatusMessage((msg) => {
      setStatusMessage(msg);
    });

    const unsubError = socket.onErrorMessage((msg) => {
      setStatusMessage(`⚠ ${msg}`);
      setSafetyOk(false);
    });

    return () => {
      unsubStatus();
      unsubError();
    };
  }, []);

  // ── Handlers ────────────────────────────────────────────────────────────

  const sendConfig = useCallback((config: Record<string, unknown>) => {
    socket.updateConfig(config);
    onConfigChange?.();
  }, [onConfigChange]);

  const handleModeChange = useCallback((newMode: RadarMode) => {
    setMode(newMode);
    sendConfig({ radar: { mode: newMode } });
  }, [sendConfig]);

  const handleWaveformChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const wf = e.target.value as WaveformType;
    setWaveform(wf);
    sendConfig({ radar: { waveform_type: wf } });
  }, [sendConfig]);

  const handleFrequencyChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setFrequency(val);
    sendConfig({ hackrf: { frequency_hz: val * 1e6 } });
  }, [sendConfig]);

  const handleSampleRateChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setSampleRate(val);
    sendConfig({ hackrf: { sample_rate_hz: val * 1e6 } });
  }, [sendConfig]);

  const handleLnaGain = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setLnaGain(val);
    sendConfig({ hackrf: { lna_gain_db: val } });
  }, [sendConfig]);

  const handleVgaGain = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setVgaGain(val);
    sendConfig({ hackrf: { vga_gain_db: val } });
  }, [sendConfig]);

  const handleTxGain = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setTxGain(val);
    sendConfig({ hackrf: { txvga_gain_db: val } });
  }, [sendConfig]);

  const handleTxAmplitude = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setTxAmplitude(val);
    sendConfig({ radar: { tx_amplitude: val } });
  }, [sendConfig]);

  const handleAmpToggle = useCallback(() => {
    const val = !ampEnable;
    setAmpEnable(val);
    sendConfig({ hackrf: { amp_enable: val } });
  }, [ampEnable, sendConfig]);

  const handleTcxoToggle = useCallback(() => {
    const val = !useTCXO;
    setUseTCXO(val);
    sendConfig({ radar: { use_tcxo: val } });
  }, [useTCXO, sendConfig]);

  const handleStartStop = useCallback(() => {
    if (isStreaming) {
      socket.stopStreaming();
      setIsStreaming(false);
    } else {
      socket.startStreaming();
      setIsStreaming(true);
    }
  }, [isStreaming]);

  const handleSafetyCheck = useCallback(() => {
    socket.runSafetyCheck();
    setSafetyOk(true);
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────

  const isConnected = connectionStatus === "connected" || connectionStatus === "streaming";
  const freqWarning = frequency === 915 || frequency === 2450 ? "" : "outside ISM band";

  return (
    <div className="panel h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="panel-header flex items-center justify-between">
        <span className="flex items-center gap-2">
          <span className="text-gradient">⚡</span>
          Control Panel
        </span>
        <span className="flex items-center gap-2">
          <span className={`status-dot ${connectionStatus}`} />
          <span className="text-[0.65rem] text-text-dim uppercase tracking-wider">
            {connectionStatus}
          </span>
        </span>
      </div>

      {/* Content (scrollable) */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* ── Mode Selector ── */}
        <div>
          <div className="value-label mb-2">Mode</div>
          <div className="flex flex-wrap gap-1.5">
            {MODES.map((m) => (
              <button
                key={m.id}
                onClick={() => handleModeChange(m.id)}
                className={`mode-tab text-xs ${mode === m.id ? "active" : ""}`}
                title={m.description}
                style={mode === m.id ? {
                  borderColor: m.color,
                  color: m.color,
                  background: `${m.color}15`,
                } : undefined}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Waveform ── */}
        <div>
          <div className="value-label mb-1.5">Waveform</div>
          <select
            value={waveform}
            onChange={handleWaveformChange}
            className="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-primary cursor-pointer"
          >
            {WAVEFORMS.map((wf) => (
              <option key={wf.id} value={wf.id}>
                {wf.label}
              </option>
            ))}
          </select>
        </div>

        {/* ── Frequency ── */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="value-label">Frequency</span>
            <span className="value-display text-sm">{frequency} MHz</span>
          </div>
          <input
            type="range"
            min={902}
            max={2484}
            step={1}
            value={frequency}
            onChange={handleFrequencyChange}
            className="slider-track w-full"
          />
          <div className="flex justify-between text-[0.6rem] text-text-muted mt-0.5">
            <span>902 MHz</span>
            <span>2484 MHz</span>
          </div>
          {freqWarning && (
            <div className="safety-badge warning mt-1 text-[0.6rem]">
              {freqWarning}
            </div>
          )}
        </div>

        {/* ── Sample Rate ── */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="value-label">Sample Rate</span>
            <span className="value-display text-sm">{sampleRate} Msps</span>
          </div>
          <input
            type="range"
            min={2}
            max={20}
            step={0.5}
            value={sampleRate}
            onChange={handleSampleRateChange}
            className="slider-track w-full"
          />
        </div>

        {/* ── Gains ── */}
        <div className="grid grid-cols-3 gap-3">
          <div>
            <div className="value-label mb-1">LNA</div>
            <input
              type="range"
              min={0}
              max={40}
              step={8}
              value={lnaGain}
              onChange={handleLnaGain}
              className="slider-track w-full"
            />
            <div className="text-[0.65rem] text-text-dim text-center mt-0.5">
              {lnaGain} dB
            </div>
          </div>
          <div>
            <div className="value-label mb-1">VGA</div>
            <input
              type="range"
              min={0}
              max={62}
              step={2}
              value={vgaGain}
              onChange={handleVgaGain}
              className="slider-track w-full"
            />
            <div className="text-[0.65rem] text-text-dim text-center mt-0.5">
              {vgaGain} dB
            </div>
          </div>
          <div>
            <div className="value-label mb-1">TX</div>
            <input
              type="range"
              min={0}
              max={47}
              step={1}
              value={txGain}
              onChange={handleTxGain}
              className={`slider-track w-full ${txGain > 20 ? "opacity-50" : ""}`}
            />
            <div className="text-[0.65rem] text-text-dim text-center mt-0.5">
              {txGain} dB
            </div>
          </div>
        </div>

        {/* ── TX Amplitude ── */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="value-label">TX Amplitude</span>
            <span className="value-display text-sm">{txAmplitude.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={txAmplitude}
            onChange={handleTxAmplitude}
            className={`slider-track w-full ${txAmplitude > 0.7 ? "opacity-50" : ""}`}
          />
          {txAmplitude > 0.7 && (
            <div className="safety-badge warning mt-1 text-[0.6rem]">
              High TX amplitude — reduce if possible
            </div>
          )}
        </div>

        {/* ── Toggles ── */}
        <div className="space-y-2">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="value-label">
              Amp {ampEnable ? "ON" : "OFF"}
            </span>
            <button
              onClick={handleAmpToggle}
              className={`relative w-10 h-5 rounded-full transition-colors ${
                ampEnable ? "bg-danger" : "bg-border"
              }`}
            >
              <span
                className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                  ampEnable ? "translate-x-5" : "translate-x-0.5"
                }`}
              />
            </button>
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <span className="value-label">10 MHz TCXO</span>
            <button
              onClick={handleTcxoToggle}
              className={`relative w-10 h-5 rounded-full transition-colors ${
                useTCXO ? "bg-primary" : "bg-border"
              }`}
            >
              <span
                className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                  useTCXO ? "translate-x-5" : "translate-x-0.5"
                }`}
              />
            </button>
          </label>
        </div>

        {/* ── Safety Status ── */}
        <div className="flex items-center gap-2">
          <span className={`safety-badge ${safetyOk ? "safe" : "danger"}`}>
            {safetyOk ? "SAFE" : "⚠ SAFETY"}
          </span>
          {!isConnected && (
            <span className="text-[0.65rem] text-text-muted">
              Connect to backend first
            </span>
          )}
        </div>

        {statusMessage && (
          <div className="text-[0.65rem] text-text-dim bg-surface-2 rounded-lg px-3 py-2">
            {statusMessage}
          </div>
        )}

        {/* ── Action Buttons ── */}
        <div className="flex gap-2 pt-2">
          <button
            onClick={handleSafetyCheck}
            disabled={!isConnected}
            className="flex-1 px-3 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider
              border border-border text-text-dim hover:bg-surface-2 hover:text-text
              disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            Safety Check
          </button>

          <button
            onClick={handleStartStop}
            disabled={!isConnected}
            className={`flex-1 px-3 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider
              transition-all disabled:opacity-30 disabled:cursor-not-allowed
              ${isStreaming
                ? "bg-danger/20 text-danger border border-danger/40 hover:bg-danger/30"
                : "bg-primary/10 text-primary border border-primary/30 hover:bg-primary/20"
              }`}
          >
            {isStreaming ? "Stop" : "Start"}
          </button>
        </div>

        {/* ── Safety Reminder ── */}
        <div className="text-[0.55rem] text-text-muted leading-relaxed border-t border-border pt-3 mt-2">
          <strong className="text-warning">⚠ SAFETY:</strong> Always use a
          dummy load or appropriate antenna. Observe local regulations. TX
          power should be kept low for indoor experimental use.
        </div>
      </div>
    </div>
  );
}
