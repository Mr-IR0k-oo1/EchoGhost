"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import EchoGhost3D from "../../components/EchoGhost3D";
import ControlPanel from "../../components/ControlPanel";
import { socket, type RadarData, type ConnectionStatus } from "../../lib/socket";

interface Metrics {
  breathingRate: number;
  breathingConfidence: number;
  motionDetected: boolean;
  motionMagnitude: number;
  targetRange: number;
  targetVelocity: number;
  activityLabel: string;
  activityConfidence: number;
  cpiIndex: number;
}

const INITIAL_METRICS: Metrics = {
  breathingRate: 0,
  breathingConfidence: 0,
  motionDetected: false,
  motionMagnitude: 0,
  targetRange: 0,
  targetVelocity: 0,
  activityLabel: "idle",
  activityConfidence: 0,
  cpiIndex: 0,
};

const ACTIVITY_COLORS: Record<string, string> = {
  idle: "#555577",
  walking: "#00f5ff",
  sitting: "#00ff88",
  gesturing: "#ff00e5",
  falling: "#ff3355",
};

function MiniHeatmap({ data, label }: { data?: number[][]; label: string }) {
  if (!data || data.length === 0) {
    return (
      <div className="panel flex flex-col">
        <div className="panel-header">{label}</div>
        <div className="flex-1 flex items-center justify-center text-text-muted text-xs">
          Awaiting data...
        </div>
      </div>
    );
  }

  const rows = data.length;
  const cols = data[0]?.length ?? rows;

  return (
    <div className="panel flex flex-col">
      <div className="panel-header">{label}</div>
      <div className="flex-1 p-2">
        <svg viewBox={`0 0 ${cols} ${rows}`} className="w-full h-full">
          {data.map((row, r) =>
            row.map((val, c) => (
              <rect
                key={`${r}-${c}`}
                x={c}
                y={r}
                width={1}
                height={1}
                fill={val > 0.8 ? "#ff00e5" : val > 0.5 ? "#00f5ff" : val > 0.2 ? "#1a1a2e" : "#0a0a0f"}
                opacity={0.3 + val * 0.7}
              />
            ))
          )}
        </svg>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  unit,
  color = "#00f5ff",
  confidence,
}: {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
  confidence?: number;
}) {
  return (
    <div className="panel p-3 flex flex-col">
      <span className="value-label mb-1">{label}</span>
      <div className="flex items-baseline gap-1.5">
        <span className="value-display" style={{ color }}>{value}</span>
        {unit && <span className="text-[0.65rem] text-text-dim">{unit}</span>}
      </div>
      {confidence !== undefined && (
        <div className="mt-1.5 flex items-center gap-1.5">
          <div className="flex-1 h-1 rounded-full bg-surface-2 overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${confidence * 100}%`, background: color }}
            />
          </div>
          <span className="text-[0.55rem] text-text-muted">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
}

function ConnectionBanner({ status }: { status: ConnectionStatus }) {
  if (status === "connected" || status === "streaming") return null;

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 panel px-6 py-3 flex items-center gap-3 animate-pulse-glow">
      <span className="status-dot disconnected" />
      <span className="text-sm text-text-dim">
        {status === "connecting"
          ? "Connecting to EchoGhost backend..."
          : 'Disconnected -- start the backend with: python main.py'}
      </span>
    </div>
  );
}

function StreamStatusBar({ isStreaming, cpiIndex }: { isStreaming: boolean; cpiIndex: number }) {
  if (!isStreaming) return null;
  return (
    <div className="absolute top-4 right-4 z-50 flex items-center gap-2 text-[0.65rem] text-text-dim">
      <span className="status-dot streaming" />
      <span>CPI #{cpiIndex}</span>
    </div>
  );
}

function ModeIndicator({ mode }: { mode: string }) {
  const modeLabels: Record<string, string> = {
    active_chaotic: "Active Chaotic Sensing",
    passive_wifi: "Passive WiFi Sensing",
    motion: "Motion Detection",
    breathing: "Breathing / Vital Signs",
    micro_doppler: "Micro-Doppler Analysis",
    range_doppler: "Range-Doppler Imaging",
    through_wall: "Through-Wall Mode",
    gesture: "Gesture Control",
    artistic: "Generative Art Mode",
    security: "Security / Anomaly Detection",
  };

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-2 border border-border">
      <span className="text-[0.6rem] uppercase tracking-widest text-text-muted">Mode:</span>
      <span className="text-xs font-medium text-primary">{modeLabels[mode] ?? mode}</span>
    </div>
  );
}

export default function DashboardPage() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [isStreaming, setIsStreaming] = useState(false);
  const [heatmap, setHeatmap] = useState<number[][] | undefined>();
  const [rangeDoppler, setRangeDoppler] = useState<number[][] | undefined>();
  const [microDoppler, setMicroDoppler] = useState<number[][] | undefined>();
  const [metrics, setMetrics] = useState<Metrics>(INITIAL_METRICS);
  const [currentMode, setCurrentMode] = useState("active_chaotic");
  const [rangeProfile, setRangeProfile] = useState<number[] | undefined>();

  useEffect(() => {
    const unsubConnection = socket.onConnectionChange((status) => {
      setConnectionStatus(status);
      if (status === "streaming") setIsStreaming(true);
      else if (status === "connected") setIsStreaming(false);
    });

    const unsubRadar = socket.onRadar((data: RadarData) => {
      setHeatmap(data.heatmap);
      setRangeDoppler(prev => {
        const rd = data.range_profile;
        if (!rd || rd.length === 0) return prev;
        const size = Math.floor(Math.sqrt(rd.length));
        if (size < 2) return prev;
        const arr: number[][] = [];
        for (let i = 0; i < size; i++) {
          arr.push(rd.slice(i * (rd.length / size), (i + 1) * (rd.length / size)));
        }
        return arr;
      });
      setMicroDoppler(data.micro_doppler);
      setRangeProfile(data.range_profile);
      setMetrics({
        breathingRate: data.breathing.rate_bpm,
        breathingConfidence: data.breathing.confidence,
        motionDetected: data.motion.detected,
        motionMagnitude: data.motion.magnitude,
        targetRange: data.motion.target_range_m,
        targetVelocity: data.motion.target_velocity_mps,
        activityLabel: data.activity.label,
        activityConfidence: data.activity.confidence,
        cpiIndex: data.cpi_index,
      });
    });

    const unsubError = socket.onErrorMessage((msg) => {
      console.error("[Dashboard] Backend error:", msg);
    });

    socket.connect();

    return () => {
      unsubConnection();
      unsubRadar();
      unsubError();
      socket.disconnect();
    };
  }, []);

  const handleConfigChange = useCallback(() => {
    setCurrentMode(socket.status === "streaming" ? currentMode : currentMode);
  }, [currentMode]);

  return (
    <div className="h-screen w-screen bg-bg text-text flex flex-col relative overflow-hidden">
      <ConnectionBanner status={connectionStatus} />
      <StreamStatusBar isStreaming={isStreaming} cpiIndex={metrics.cpiIndex} />

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border bg-surface/80 backdrop-blur-md z-40 shrink-0">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold">
            <span className="text-gradient">EchoGhost</span>
            <span className="text-text-dim font-light"> Hub Ultra</span>
          </h1>
          <ModeIndicator mode={currentMode} />
        </div>
        <div className="flex items-center gap-3">
          <span className={`safety-badge safe text-[0.6rem]`}>EXPERIMENTAL</span>
          <span className="text-[0.6rem] text-text-muted">
            v1.0.0
          </span>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex-1 flex gap-2 p-2 overflow-hidden">
        {/* Left: Control Panel */}
        <div className="w-72 shrink-0">
          <ControlPanel
            connectionStatus={connectionStatus}
            onConfigChange={handleConfigChange}
          />
        </div>

        {/* Center: 3D Visualization */}
        <div className="flex-1 panel overflow-hidden relative">
          <EchoGhost3D
            heatmap={heatmap}
            motionMagnitude={metrics.motionMagnitude}
            targetRange={metrics.targetRange}
            breathingRate={metrics.breathingRate}
            activityLabel={metrics.activityLabel}
            mode={currentMode}
          />
        </div>

        {/* Right: Metrics & Mini Panels */}
        <div className="w-80 shrink-0 flex flex-col gap-2 overflow-y-auto">
          {/* Activity */}
          <div className="panel p-3 flex items-center justify-between">
            <span className="value-label">Activity</span>
            <div className="flex items-center gap-2">
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: ACTIVITY_COLORS[metrics.activityLabel] ?? "#555577" }}
              />
              <span
                className="text-sm font-bold uppercase"
                style={{ color: ACTIVITY_COLORS[metrics.activityLabel] ?? "#555577" }}
              >
                {metrics.activityLabel}
              </span>
              <span className="text-[0.55rem] text-text-muted">
                {(metrics.activityConfidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Breathing"
              value={metrics.breathingRate > 0 ? metrics.breathingRate.toFixed(1) : "---"}
              unit="BPM"
              color="#00ff88"
              confidence={metrics.breathingConfidence}
            />
            <MetricCard
              label="Motion"
              value={metrics.motionDetected ? "YES" : "no"}
              color={metrics.motionDetected ? "#ffaa00" : "#555577"}
              confidence={metrics.motionMagnitude}
            />
            <MetricCard
              label="Range"
              value={metrics.targetRange > 0 ? metrics.targetRange.toFixed(1) : "---"}
              unit="m"
              color="#00f5ff"
            />
            <MetricCard
              label="Velocity"
              value={metrics.targetVelocity !== 0 ? metrics.targetVelocity.toFixed(2) : "---"}
              unit="m/s"
              color="#ff00e5"
            />
          </div>

          {/* Heatmap */}
          <div className="flex-1 min-h-[120px]">
            <MiniHeatmap data={heatmap} label="Range-Doppler Heatmap" />
          </div>

          {/* Micro-Doppler */}
          <div className="flex-1 min-h-[100px]">
            <MiniHeatmap data={microDoppler} label="Micro-Doppler" />
          </div>

          {/* Range Profile */}
          {rangeProfile && rangeProfile.length > 0 && (
            <div className="panel p-3">
              <div className="value-label mb-2">Range Profile</div>
              <svg viewBox={`0 0 ${rangeProfile.length} 40`} className="w-full h-10">
                {rangeProfile.map((val, i) => (
                  <rect
                    key={i}
                    x={i}
                    y={40 - val * 40}
                    width={1}
                    height={val * 40}
                    fill="#00f5ff"
                    opacity={0.5 + val * 0.5}
                  />
                ))}
              </svg>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
