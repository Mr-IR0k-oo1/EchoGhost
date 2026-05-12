from __future__ import annotations

import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from .bridge import SensingBridge
from .config import SensingFrame, SessionConfig

app = FastAPI(title="EchoGhost Hub Ultra — Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bridge = SensingBridge()


# ─── REST Endpoints ───────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return bridge.get_status().model_dump()


@app.post("/api/start")
async def start_session(config: SessionConfig):
    message = bridge.start(config)
    return {"message": message}


@app.post("/api/stop")
async def stop_session():
    message = bridge.stop()
    return {"message": message}


@app.post("/api/config")
async def update_config(config: SessionConfig):
    message = bridge.update_config(config)
    return {"message": message}


@app.get("/api/frame")
async def get_latest_frame():
    frame = bridge.latest_frame
    if frame is None:
        return JSONResponse(content={"t": 0.0, "status": "no data"}, status_code=200)
    return frame.model_dump()


# ─── WebSocket ────────────────────────────────────────────────────

@app.websocket("/ws/sensing")
async def websocket_sensing(websocket: WebSocket):
    await websocket.accept()
    print(f"[ws] client connected: {websocket.client}")

    try:
        while True:
            message = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            data = json.loads(message)

            if data.get("action") == "start":
                config = SessionConfig(**data.get("config", {}))
                msg = bridge.start(config)
                await websocket.send_json({"type": "status", "message": msg})

            elif data.get("action") == "stop":
                msg = bridge.stop()
                await websocket.send_json({"type": "status", "message": msg})

            elif data.get("action") == "config":
                config = SessionConfig(**data.get("config", {}))
                msg = bridge.update_config(config)
                await websocket.send_json({"type": "status", "message": msg})

            elif data.get("action") == "poll":
                frame = bridge.poll()
                if frame is not None:
                    await websocket.send_json({"type": "frame", **frame.model_dump()})

            elif data.get("action") == "subscribe":
                await _stream_frames(websocket)

    except asyncio.TimeoutError:
        pass
    except WebSocketDisconnect:
        print(f"[ws] client disconnected: {websocket.client}")
    except Exception as exc:
        print(f"[ws] error: {exc}")
    finally:
        pass


async def _stream_frames(websocket: WebSocket) -> None:
    """Stream frames at 30 FPS until the client disconnects or sends 'unsubscribe'."""
    while True:
        try:
            recv_task = asyncio.create_task(websocket.receive_text())
            poll_task = asyncio.create_task(asyncio.sleep(1.0 / 30.0))

            done, _ = await asyncio.wait(
                [recv_task, poll_task], return_when=asyncio.FIRST_COMPLETED
            )

            if recv_task in done:
                msg = json.loads(recv_task.result())
                if msg.get("action") in ("unsubscribe", "stop"):
                    return

            frame = bridge.poll()
            if frame is not None:
                await websocket.send_json({"type": "frame", **frame.model_dump()})

            await poll_task

        except WebSocketDisconnect:
            return
        except Exception as exc:
            print(f"[ws stream] error: {exc}")
            return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
