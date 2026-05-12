import type { SensingFrame, SessionConfig } from "./types";

export type WsMessageHandler = (frame: SensingFrame) => void;
export type WsStatusHandler = (message: string) => void;

export class SensingWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private onFrame: WsMessageHandler;
  private onStatus: WsStatusHandler;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private shouldReconnect = false;

  constructor(
    url: string,
    onFrame: WsMessageHandler,
    onStatus: WsStatusHandler
  ) {
    this.url = url;
    this.onFrame = onFrame;
    this.onStatus = onStatus;
  }

  connect() {
    this.shouldReconnect = true;
    this._connect();
  }

  private _connect() {
    if (this.ws) {
      this.ws.close();
    }

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log("[ws] connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
          this.onFrame(data as SensingFrame);
        } else if (data.type === "status") {
          this.onStatus(data.message);
        }
      } catch {
        console.warn("[ws] failed to parse message");
      }
    };

    this.ws.onclose = () => {
      console.log("[ws] disconnected");
      if (this.shouldReconnect) {
        this.reconnectTimer = setTimeout(() => this._connect(), 2000);
      }
    };

    this.ws.onerror = () => {
      console.warn("[ws] error");
    };
  }

  sendStart(config: SessionConfig) {
    this._send({ action: "start", config });
  }

  sendStop() {
    this._send({ action: "stop" });
  }

  sendConfig(config: SessionConfig) {
    this._send({ action: "config", config });
  }

  sendSubscribe() {
    this._send({ action: "subscribe" });
  }

  sendUnsubscribe() {
    this._send({ action: "unsubscribe" });
  }

  private _send(data: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
