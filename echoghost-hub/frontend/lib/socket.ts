/**
 * WebSocket client for EchoGhost Hub Ultra.
 *
 * Manages the real-time connection to the FastAPI backend,
 * handles reconnection, and provides typed message interfaces.
 */

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "streaming";

// ── Message Types (Server -> Client) ──────────────────────────────────────

export interface RadarData {
  type: "radar_data";
  timestamp: number;
  cpi_index: number;
  heatmap: number[][];
  range_profile: number[];
  doppler_profile: number[];
  micro_doppler: number[][];
  micro_doppler_freqs: number[];
  breathing: {
    rate_bpm: number;
    confidence: number;
    phase: number[];
  };
  motion: {
    detected: boolean;
    magnitude: number;
    target_range_m: number;
    target_velocity_mps: number;
  };
  activity: {
    label: ActivityLabel;
    confidence: number;
  };
}

export interface StatusMessage {
  type: "status";
  message: string;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = RadarData | StatusMessage | ErrorMessage;

// ── Message Types (Client -> Server) ──────────────────────────────────────

export interface ConfigMessage {
  type: "config";
  config: Partial<{
    hackrf: Partial<HackRFConfig>;
    radar: Partial<RadarModeConfig>;
  }>;
}

export interface StartMessage {
  type: "start";
}

export interface StopMessage {
  type: "stop";
}

export interface SafetyCheckMessage {
  type: "safety_check";
}

export type ClientMessage = ConfigMessage | StartMessage | StopMessage | SafetyCheckMessage;

// ── Config Interfaces ─────────────────────────────────────────────────────

export interface HackRFConfig {
  frequency_hz: number;
  sample_rate_hz: number;
  lna_gain_db: number;
  vga_gain_db: number;
  txvga_gain_db: number;
  amp_enable: boolean;
  bandwidth_hz: number;
}

export interface RadarModeConfig {
  mode: RadarMode;
  waveform_type: WaveformType;
  pulse_repetition_hz: number;
  chirp_bw_hz: number;
  chirp_duration_s: number;
  max_range_m: number;
  tx_amplitude: number;
  use_tcxo: boolean;
}

export type RadarMode =
  | "active_chaotic"
  | "passive_wifi"
  | "motion"
  | "breathing"
  | "micro_doppler"
  | "range_doppler"
  | "through_wall"
  | "gesture"
  | "artistic"
  | "security";

export type WaveformType = "tone" | "fmcw" | "chaotic" | "noise";

export type ActivityLabel =
  | "idle"
  | "walking"
  | "sitting"
  | "gesturing"
  | "falling";

// ── Event Handlers ────────────────────────────────────────────────────────

export type RadarDataHandler = (data: RadarData) => void;
export type StatusHandler = (message: string) => void;
export type ErrorHandler = (message: string) => void;
export type ConnectionChangeHandler = (status: ConnectionStatus) => void;

// ── WebSocket Client ──────────────────────────────────────────────────────

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 10;

export class EchoGhostSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _status: ConnectionStatus = "disconnected";

  private onRadarData: RadarDataHandler[] = [];
  private onStatus: StatusHandler[] = [];
  private onError: ErrorHandler[] = [];
  private onConnectionChange: ConnectionChangeHandler[] = [];

  constructor(url: string = `ws://${window.location.hostname}:8000/ws`) {
    this.url = url;
  }

  // ── Public API ──────────────────────────────────────────────────────────

  get status(): ConnectionStatus {
    return this._status;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.setStatus("connecting");

    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      console.error("[EchoGhostSocket] Failed to create WebSocket:", err);
      this.handleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log("[EchoGhostSocket] Connected");
      this.reconnectAttempts = 0;
      this.setStatus("connected");
    };

    this.ws.onclose = (event) => {
      console.log(`[EchoGhostSocket] Disconnected (code: ${event.code})`);
      this.setStatus("disconnected");
      this.handleReconnect();
    };

    this.ws.onerror = (event) => {
      console.error("[EchoGhostSocket] Error:", event);
    };

    this.ws.onmessage = (event) => {
      try {
        const message: ServerMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (err) {
        console.error("[EchoGhostSocket] Failed to parse message:", err);
      }
    };
  }

  disconnect(): void {
    this.reconnectAttempts = MAX_RECONNECT_ATTEMPTS; // Prevent reconnect
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus("disconnected");
  }

  send(message: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn("[EchoGhostSocket] Cannot send: not connected");
    }
  }

  startStreaming(): void {
    this.send({ type: "start" });
    this.setStatus("streaming");
  }

  stopStreaming(): void {
    this.send({ type: "stop" });
    this.setStatus("connected");
  }

  updateConfig(config: ConfigMessage["config"]): void {
    this.send({ type: "config", config });
  }

  runSafetyCheck(): void {
    this.send({ type: "safety_check" });
  }

  // ── Event Registration ──────────────────────────────────────────────────

  onRadar(handler: RadarDataHandler): () => void {
    this.onRadarData.push(handler);
    return () => {
      this.onRadarData = this.onRadarData.filter((h) => h !== handler);
    };
  }

  onStatusMessage(handler: StatusHandler): () => void {
    this.onStatus.push(handler);
    return () => {
      this.onStatus = this.onStatus.filter((h) => h !== handler);
    };
  }

  onErrorMessage(handler: ErrorHandler): () => void {
    this.onError.push(handler);
    return () => {
      this.onError = this.onError.filter((h) => h !== handler);
    };
  }

  onConnectionChange(handler: ConnectionChangeHandler): () => void {
    this.onConnectionChange.push(handler);
    return () => {
      this.onConnectionChange = this.onConnectionChange.filter(
        (h) => h !== handler
      );
    };
  }

  // ── Private ─────────────────────────────────────────────────────────────

  private setStatus(status: ConnectionStatus): void {
    if (this._status !== status) {
      this._status = status;
      this.onConnectionChange.forEach((h) => h(status));
    }
  }

  private handleMessage(message: ServerMessage): void {
    switch (message.type) {
      case "radar_data":
        this.onRadarData.forEach((h) => h(message));
        break;
      case "status":
        this.onStatusMessage.forEach((h) => h(message.message));
        break;
      case "error":
        this.onErrorMessage.forEach((h) => h(message.message));
        break;
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error("[EchoGhostSocket] Max reconnect attempts reached");
      return;
    }

    this.reconnectAttempts++;
    const delay = RECONNECT_DELAY_MS * Math.min(this.reconnectAttempts, 5);

    console.log(
      `[EchoGhostSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`
    );

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }
}

// Singleton instance
export const socket = new EchoGhostSocket();
