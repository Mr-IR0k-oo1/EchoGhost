# EchoGhost Hub Ultra

EchoGhost Hub Ultra is a Python-based RF sensing dashboard for HackRF One hardware and simulation-first development.

## Current implementation

- Simulation-first radio backend
- Tone, chirp, pseudo-random, and chaotic waveform generators
- Baseline subtraction and phase-aware motion scoring
- Breathing-rate estimator from slow-time phase history
- Rolling spectrum and heatmap history for real-time visualization
- Hardware adapter stub for HackRF through SoapySDR

## Quick start

```bash
pip install -r requirements.txt
python -m echoghost_hub_ultra.app --headless --frames 5
```

For the GUI, install Dear PyGui and run without `--headless`.

## Hardware notes

The hardware path is designed to use HackRF through a Python-accessible backend, with SoapySDR as the default adapter in this starter implementation.
If you already have a preferred HackRF Python binding, the backend layer is isolated so it can be swapped in later.
