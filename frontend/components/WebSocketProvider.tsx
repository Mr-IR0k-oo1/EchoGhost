"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type { SensingFrame, SessionConfig } from "@/lib/types";
import { SensingWebSocket } from "@/lib/websocket";

interface WsContextValue {
  frame: SensingFrame | null;
  connected: boolean;
  statusMessage: string;
  startSession: (config: SessionConfig) => void;
  stopSession: () => void;
  updateConfig: (config: SessionConfig) => void;
}

const WsContext = createContext<WsContextValue>({
  frame: null,
  connected: false,
  statusMessage: "",
  startSession: () => {},
  stopSession: () => {},
  updateConfig: () => {},
});

export function useWs() {
  return useContext(WsContext);
}

const WS_URL = "ws://localhost:8000/ws/sensing";

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const [frame, setFrame] = useState<SensingFrame | null>(null);
  const [connected, setConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const wsRef = useRef<SensingWebSocket | null>(null);

  useEffect(() => {
    const ws = new SensingWebSocket(
      WS_URL,
      (f) => setFrame(f),
      (msg) => setStatusMessage(msg)
    );
    wsRef.current = ws;
    ws.connect();

    const interval = setInterval(() => {
      setConnected(ws.connected);
    }, 1000);

    return () => {
      ws.disconnect();
      clearInterval(interval);
    };
  }, []);

  const startSession = useCallback((config: SessionConfig) => {
    wsRef.current?.sendStart(config);
    setTimeout(() => wsRef.current?.sendSubscribe(), 500);
  }, []);

  const stopSession = useCallback(() => {
    wsRef.current?.sendUnsubscribe();
    wsRef.current?.sendStop();
  }, []);

  const updateConfig = useCallback((config: SessionConfig) => {
    wsRef.current?.sendConfig(config);
  }, []);

  return (
    <WsContext.Provider
      value={{ frame, connected, statusMessage, startSession, stopSession, updateConfig }}
    >
      {children}
    </WsContext.Provider>
  );
}
