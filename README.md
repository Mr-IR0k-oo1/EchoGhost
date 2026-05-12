# EchoGhost Hub Ultra

**Multi-mode RF sensing platform** — one or more HackRF Ones turned into an adaptive motion detection, vital signs sensing, gesture recognition, and generative art system.

## Features

- **Motion Detection + Tracking** — baseline subtraction with phase-aware variance scoring
- **Breathing / Vital Signs** — slow-time phase FFT estimates respiratory rate (BPM)
- **Activity Classification** — 5-class: idle, micro-motion, gesture, walking, falling
- **Range-Doppler Heatmap** — rolling FFT spectrogram as a motion heatmap
- **7 Waveform Types** — tone, chirp, PRN, chaotic logistic map, Henon map, Lorenz attractor, Kuramoto–Sivashinsky
- **Adaptive Waveform Morphing** — real-time SNR-driven hill-climbing over spread, rate, and amplitude
- **Passive Mode** — RX-only ambient sensing using existing ISM-band signals (WiFi, etc.)
- **Multi-HackRF** — true full-duplex: one device TX, one or more devices RX simultaneously
- **Generative Art** — motion → audio synthesis (sounddevice) + colour palette output
- **3D Visualization** — Three.js room with motion particles via React/Next.js
- **Triple UI** — Dear PyGui (native), Streamlit (Plotly), and Next.js (Three.js + WebSocket)
- **Auto-detect Backend** — `python_hackrf` → SoapySDR → simulation fallback

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        FRONTENDS                                   │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Dear PyGui   │  │   Streamlit  │  │  Next.js 16 + Three.js   │  │
│  │  (app.py)     │  │  (plotly)    │  │  (3D room, particles)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                  │
│         │         Direct import          WebSocket (ws://8000)      │
│         │                 │                      │                  │
├─────────┴─────────────────┴──────────────────────┴────────────────┤
│                      BACKEND (Python)                              │
│                                                                    │
│  ┌────────────────────── FastAPI ─── /api/* + /ws/sensing ──────┐  │
│  │  bridge.py ←→ RFSession ←→ DashboardSnapshot @ 30 FPS       │  │
│  │  serializers.py: heatmap downsample + 3D position estimation │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┴────────────────────────────────────┐  │
│  │  RFSession (radio/session.py) — background thread             │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │  │
│  │  │ RadioBkend │ │ Waveforms  │ │ Processing │ │ Visualization│  │  │
│  │  │ sim/hackrf │ │ tone/prn   │ │ motion/    │ │ panels/art   │  │  │
│  │  │ multi      │ │ chaotic*   │ │ vitals/clf │ │ streamlit    │  │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Dependencies

- **Python 3.11+** — signal processing, SDR control, FastAPI backend
- **Node.js 18+** — Next.js frontend with Three.js
- **HackRF One** (optional) — hardware SDR

## Installation

### Python (Backend + Signal Processing)

```bash
# Core signal processing stack
uv pip install -r requirements.txt

# Backend for web UI
uv pip install fastapi uvicorn pydantic python-multipart

# Optional: native GUI
uv pip install dearpygui
uv pip install streamlit plotly

# Optional: hardware (requires HackRF SDK headers)
uv pip install python-hackrf
# SoapySDR must be installed manually (platform-dependent)
```

### Node.js (Frontend)

```bash
cd frontend
npm install
```

## Quick Start

### Simulation (no hardware needed)

```bash
# CLI headless mode
python -m echoghost_hub_ultra --headless --frames 20

# Breathing estimation (kicks in after ~24 frames)
python -m echoghost_hub_ultra --headless --frames 40

# Dear PyGui dashboard
python -m echoghost_hub_ultra

# Streamlit web dashboard
python -m streamlit run src/echoghost_hub_ultra/visualization/streamlit_app.py
```

### Web UI (FastAPI + Next.js + Three.js)

Terminal 1 — Start the FastAPI backend:
```bash
cd EchoGhost
$env:PYTHONPATH = "src"
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — Start the Next.js dev server:
```bash
cd EchoGhost/frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

## CLI Reference

```
usage: echoghost-hub-ultra [-h]
  --backend {simulation,hackrf,soapy,multi_hackrf}
  --mode {simulation,active,passive}
  --waveform {tone,chirp,chaotic,chaotic_henon,chaotic_lorenz,chaotic_ks,prn}
  --center-frequency CENTER_FREQUENCY    (default: 915000000)
  --sample-rate SAMPLE_RATE              (default: 2000000)
  --frame-size FRAME_SIZE                (default: 4096)
  --tx-gain TX_GAIN                      (default: 18.0)
  --rx-gain RX_GAIN                      (default: 24.0)
  --refresh-hz REFRESH_HZ                (default: 20.0)
  --headless                             run without GUI
  --frames FRAMES                        headless frame limit (0 = infinite)
  --adaptive                             enable waveform morphing
  --gui {dearpygui,streamlit}
```

### Examples by Use Case

| Use Case | Command |
|----------|---------|
| Default simulation | `python -m echoghost_hub_ultra` |
| Active sensing @ 2.4 GHz | `--mode active --center-frequency 2.4e9 --waveform chaotic_lorenz` |
| Passive ambient sensing | `--mode passive --center-frequency 2.4e9` |
| Adaptive chaotic TX | `--mode active --waveform chaotic --adaptive` |
| Multi-HackRF (2 devices) | `--backend multi_hackrf --mode active --waveform chaotic` |
| ISM band 5.8 GHz | `--center-frequency 5.8e9` |
| Higher range resolution | `--sample-rate 20e6 --frame-size 8192` |

## Web UI — Frontend (Next.js + Three.js)

```
frontend/
├── app/
│   ├── dashboard/page.tsx    # Main dashboard with tabs (3D, Spectrum, Breathing, Heatmap)
│   └── page.tsx              # Redirects to /dashboard
├── components/
│   ├── EchoGhost3D.tsx       # Three.js room + wireframe box + motion particles + OrbitControls
│   ├── WebSocketProvider.tsx # React context — connects to ws://localhost:8000/ws/sensing
│   ├── ControlPanel.tsx      # Sidebar: mode, waveform, freq, gains, adaptive toggle
│   └── StatusHUD.tsx         # Connection status + live metrics overlay
├── lib/
│   ├── types.ts              # TypeScript interfaces matching Python SensingFrame
│   └── websocket.ts          # WebSocket client with auto-reconnect
└── package.json
```

### 3D Room Visualization

- **Wireframe room** (10×6×10 units) with floor grid
- **Motion particles** — spheres at estimated 3D positions, coloured by intensity (hue 0–240°)
- **Emissive glow** scales with motion score — particles pulse brighter on detected movement
- **OrbitControls** — click-drag to rotate, scroll to zoom
- **Dark theme** — deep navy background (#050a18) with emerald accent (#00ff88)

## Web UI — Backend (FastAPI)

```
backend/
├── main.py           # FastAPI app — REST + WebSocket endpoints
├── bridge.py         # SensingBridge — wraps RFSession in async-safe interface
├── serializers.py    # DashboardSnapshot → JSON frame with downsampling + 3D position estimation
├── config.py         # Pydantic models: SensingFrame, SessionConfig, ServerStatus
└── requirements.txt
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Current session status (mode, waveform, uptime, frame count) |
| POST | `/api/start` | Start a sensing session with config body |
| POST | `/api/stop` | Stop the active session |
| POST | `/api/config` | Update session config without restarting |
| GET | `/api/frame` | Get the latest processed frame |
| WS | `/ws/sensing` | Bidirectional WebSocket — send actions, receive frame streams |

### WebSocket Protocol

```json
// Client → Server
{ "action": "start", "config": { "mode": "active", "waveform": "chaotic_lorenz", ... } }
{ "action": "stop" }
{ "action": "config", "config": { ... } }
{ "action": "subscribe" }       // Begin 30 FPS stream
{ "action": "unsubscribe" }

// Server → Client (after subscribe)
{ "type": "frame", "t": 123.4, "mode": "active", "waveform": "chaotic_lorenz",
  "motion": { "score": 0.042, "label": "gesture", "confidence": 0.87 },
  "breathing": { "bpm": 15.2, "confidence": 0.73 },
  "spectrum": [/* 256 dB values */],
  "heatmap_rows": 32, "heatmap_cols": 64, "heatmap_data": [/* downsampled */],
  "positions": [{ "x": 1.2, "y": 0.3, "z": 0.0, "intensity": 0.85, "label": "gesture" }]
}
```

## Data Flow

```
HackRF TX/RX (or simulation)
  │
  ▼  complex64 IQ samples
RFSession background thread (30 FPS)
  │
  ├── MotionDetector: baseline → residual energy + variance → score + label
  ├── BreathingEstimator: phase unwrap → FFT → peak BPM
  ├── ActivityClassifier: micro-Doppler features → 5-class label
  ├── RangeHeatmap: rolling FFT → 64×512 dB matrix
  └── WaveformAdapter: SNR → parameter tuning
  │
  ▼  DashboardSnapshot
serializers.py
  │
  ├── _downsample_spectrum(512 → 256)
  ├── _downsample_heatmap(64×512 → 32×64)
  └── _estimate_positions(spectrum peaks → x/y/z + intensity)
  │
  ▼  SensingFrame (JSON)
FastAPI WebSocket → Next.js client
  │
  ├── EchoGhost3D: positions → Three.js particles in room
  ├── SpectrumView: spectrum → bar chart
  ├── BreathingView: BPM + motion metrics
  └── HeatmapView: 32×64 grid → colour cells
```

## Waveform Types

| Waveform | Description |
|----------|-------------|
| `tone` | Continuous complex baseband tone at configurable frequency |
| `chirp` | Linear FMCW chirp sweeping between start/end frequencies |
| `chaotic` | Logistic map (r=3.92) — deterministic, noise-like, LPI |
| `chaotic_henon` | Henon map (a=1.4, b=0.3) — 2D chaotic attractor |
| `chaotic_lorenz` | Lorenz system (σ=10, ρ=28, β=8/3) — 3D chaos projected to I/Q |
| `chaotic_ks` | Kuramoto–Sivashinsky equation — spatiotemporal chaotic field |
| `prn` | Band-limited pseudo-random noise via moving-average of gaussian |

## Hardware Backend Auto-Detect

```
HackRFBackend.open()
  → Try python_hackrf (native libhackrf, fastest)
  → Fallback SoapySDR (if python_hackrf unavailable)
  → Fallback SimulationBackend (if no hardware at all)

MultiHackRFBackend.open()
  → Device 0 = TX (via same driver chain)
  → Devices 1+ = RX (simultaneous capture)
  → Returns combined IQ frames
```

## Multi-HackRF Setup

Connect two or more HackRF Ones. Device 0 transmits the active waveform while devices 1+ receive simultaneously — enabling true full-duplex operation impossible with a single HackRF.

```bash
python -m echoghost_hub_ultra --backend multi_hackrf --mode active --waveform lorenz
```
Or via the web UI: select `multi_hackrf` backend in ControlPanel.

## Processing Pipeline

```
IQ Frame
  │
  ├─→ MotionDetector: baseline EMA → residual energy + magnitude variance + phase variance → score + label
  ├─→ BreathingEstimator: mean phase → unwrap → detrend → FFT → peak in 0.08–0.7 Hz → BPM
  ├─→ ActivityClassifier: micro-Doppler BW + periodicity + ZCR + energy var → 5-class label
  ├─→ RangeHeatmap: FFT each frame → stack into rolling 64×512 matrix → dB spectrogram
  └─→ WaveformAdapter: SNR → hill-climb spread_hz, chaotic_rate, amplitude (if --adaptive)
```

## Project Structure

```
EchoGhost/
├── backend/                     # FastAPI web backend
│   ├── main.py                  #   REST + WebSocket server
│   ├── bridge.py                #   Async RFSession wrapper
│   ├── serializers.py           #   Frame → JSON + heatmap downsampling
│   └── config.py                #   Pydantic models
├── frontend/                    # Next.js 16 + Three.js
│   ├── app/dashboard/page.tsx   #   Dashboard layout with tabs
│   ├── components/              #   EchoGhost3D, ControlPanel, WebSocketProvider
│   └── lib/                     #   types.ts, websocket.ts
├── src/echoghost_hub_ultra/     # Python signal processing core
│   ├── app.py                   #   CLI + Dear PyGui launcher
│   ├── config/presets.py        #   Typed configuration dataclasses
│   ├── radio/                   #   Backends: sim, hackrf, multi, session
│   ├── waveforms/               #   Tone, chirp, PRN, 4 chaotic maps
│   ├── processing/              #   Motion, vitals, classifier, adaptive, range-doppler
│   └── visualization/           #   Panels, Streamlit dashboard, art generator
├── tests/                       # 15 unit tests
├── requirements.txt             # Python core dependencies
└── pyproject.toml               # Package build config
```

## Tests

```bash
python -m unittest discover tests -v
```

Tests cover: waveform generation (determinism, amplitude bounds), motion detection (score increases), breathing estimation (BPM recovery), range heatmap (dimension growth), simulator (frame production), classifier (label extraction, feature extraction), adapter (parameter bounds, metric tracking).
