"use client";

import { useWs } from "./WebSocketProvider";

export default function StatusHUD() {
  const { frame, connected, statusMessage } = useWs();

  return (
    <div className="pointer-events-none">
      <div className="flex items-center gap-2">
        <span
          className={`inline-block w-2 h-2 rounded-full ${
            connected ? "bg-green-400 animate-pulse" : "bg-red-500"
          }`}
        />
        <span className="text-xs text-zinc-500 font-mono">
          {connected ? "LIVE" : "DISCONNECTED"}
        </span>
      </div>
      {statusMessage && (
        <p className="text-xs text-zinc-400 mt-1">{statusMessage}</p>
      )}
      {frame && (
        <>
          <p className="text-xs text-zinc-500 mt-2 font-mono">
            FREQ: {frame.t.toFixed(1)}s
          </p>
          <p className="text-xs text-zinc-500 font-mono">
            MODE: {frame.mode.toUpperCase()}
          </p>
          <p className="text-xs text-zinc-500 font-mono">
            WAVEFORM: {frame.waveform.toUpperCase()}
          </p>
          <p className="text-xs text-zinc-500 font-mono">
            BACKEND: {frame.backend}
          </p>
        </>
      )}
    </div>
  );
}
