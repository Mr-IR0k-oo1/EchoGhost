export interface MotionData {
  score: number;
  label: string;
  confidence: number;
}

export interface BreathingData {
  bpm: number | null;
  confidence: number;
}

export interface PositionData {
  x: number;
  y: number;
  z: number;
  intensity: number;
  label: string;
}

export interface SensingFrame {
  t: number;
  mode: string;
  waveform: string;
  backend: string;
  status: string;
  motion: MotionData;
  breathing: BreathingData;
  ambient_energy_db: number;
  spectrum: number[];
  heatmap_rows: number;
  heatmap_cols: number;
  heatmap_data: number[];
  positions: PositionData[];
}

export type OperatingMode = "simulation" | "active" | "passive";
export type WaveformKind =
  | "tone"
  | "chirp"
  | "chaotic"
  | "chaotic_henon"
  | "chaotic_lorenz"
  | "chaotic_ks"
  | "prn";
export type BackendKind = "simulation" | "hackrf" | "soapy" | "multi_hackrf";

export interface SessionConfig {
  mode: OperatingMode;
  waveform: WaveformKind;
  backend: BackendKind;
  center_frequency_hz: number;
  sample_rate_sps: number;
  frame_size: number;
  tx_gain_db: number;
  rx_gain_db: number;
  adaptive: boolean;
}

export interface ServerStatus {
  running: boolean;
  mode: string;
  waveform: string;
  backend: string;
  uptime_s: number;
  frame_count: number;
}
