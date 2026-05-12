from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from echoghost_hub_ultra.config.presets import DashboardConfig, OperatingMode, ProcessingConfig, RadioConfig, SimulationConfig, WaveformConfig, WaveformKind
from echoghost_hub_ultra.processing.motion import MotionDetector
from echoghost_hub_ultra.processing.range_doppler import RangeHeatmap
from echoghost_hub_ultra.processing.vitals import BreathingEstimator
from echoghost_hub_ultra.radio.session import DashboardSnapshot, RFSession


@dataclass(slots=True)
class StreamlitState:
    session: RFSession | None = None
    lock: threading.Lock = threading.Lock()
    start_time: float = 0.0
    frame_count: int = 0


def _init_session_state() -> None:
    if "st_state" not in st.session_state:
        st.session_state.st_state = StreamlitState()


def _get_or_create_session(
    mode: OperatingMode,
    waveform_kind: WaveformKind,
    backend: str,
    center_freq: float,
    sample_rate: float,
    frame_size: int,
    tx_gain: float,
    rx_gain: float,
) -> RFSession:
    st_state: StreamlitState = st.session_state.st_state
    with st_state.lock:
        if st_state.session is not None:
            return st_state.session

        radio_cfg = RadioConfig(
            center_frequency_hz=center_freq,
            sample_rate_sps=sample_rate,
            tx_gain_db=tx_gain,
            rx_gain_db=rx_gain,
            frame_size=frame_size,
            backend=backend,
            mode=mode,
        )
        waveform_cfg = WaveformConfig(kind=waveform_kind)
        session = RFSession(
            radio_config=radio_cfg,
            waveform_config=waveform_cfg,
            processing_config=ProcessingConfig(),
            dashboard_config=DashboardConfig(refresh_hz=15.0),
            simulation_config=SimulationConfig(),
        )
        session.start()
        st_state.session = session
        st_state.start_time = time.monotonic()
        return session


def _stop_session() -> None:
    st_state: StreamlitState = st.session_state.st_state
    with st_state.lock:
        if st_state.session is not None:
            st_state.session.stop()
            st_state.session = None


def _build_spectrum_plot(snapshot: DashboardSnapshot) -> go.Figure:
    fig = go.Figure()
    if snapshot.spectrum_frequency_hz.size and snapshot.spectrum_db.size:
        freq_mhz = snapshot.spectrum_frequency_hz / 1e6
        fig.add_trace(go.Scatter(x=freq_mhz, y=snapshot.spectrum_db, mode="lines", name="Spectrum", line=dict(color="#00ff88", width=1)))
    fig.update_layout(
        title="Spectrum",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Magnitude (dB)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    return fig


def _build_motion_plot(snapshot: DashboardSnapshot) -> go.Figure:
    fig = go.Figure()
    if snapshot.motion_history.size:
        fig.add_trace(go.Scatter(y=snapshot.motion_history, mode="lines", name="Motion", line=dict(color="#ff6600", width=1)))
    fig.update_layout(
        title=f"Motion: {snapshot.motion_label} ({snapshot.motion_score:.6f})",
        xaxis_title="Frame",
        yaxis_title="Score",
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    return fig


def _build_breathing_plot(snapshot: DashboardSnapshot) -> go.Figure:
    fig = go.Figure()
    bpm_text = f"{snapshot.breathing_bpm:.1f} BPM" if snapshot.breathing_bpm is not None else "--"
    if snapshot.breathing_history_bpm.size:
        fig.add_trace(go.Scatter(y=snapshot.breathing_history_bpm, mode="lines", name="Breathing", line=dict(color="#00ccff", width=1)))
    fig.update_layout(
        title=f"Breathing: {bpm_text} (conf: {snapshot.breathing_confidence:.3f})",
        xaxis_title="Frame",
        yaxis_title="BPM",
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    return fig


def _build_heatmap_plot(snapshot: DashboardSnapshot) -> go.Figure:
    fig = go.Figure()
    matrix = snapshot.heatmap_result.matrix_db
    if matrix.size:
        freq_mhz = snapshot.heatmap_result.frequency_axis_hz / 1e6
        fig.add_trace(go.Heatmap(z=matrix, x=freq_mhz, colorscale="Viridis", showscale=True))
    fig.update_layout(
        title="Range-Doppler Heatmap",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Time (frame)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    return fig


_DARK_CSS = """
<style>
    .stApp { background-color: #0f0f23; color: #e0e0e0; }
    .stSidebar { background-color: #1a1a2e; }
    .stButton button { background-color: #00ff88; color: #0f0f23; font-weight: bold; }
    h1, h2, h3 { color: #00ff88; }
    .status-box { background: #1a1a2e; padding: 10px; border-radius: 8px; border-left: 3px solid #00ff88; margin: 4px 0; }
</style>
"""


def run_streamlit_dashboard() -> None:
    st.set_page_config(page_title="EchoGhost Hub Ultra", layout="wide", page_icon="")
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

    _init_session_state()

    st.title("EchoGhost Hub Ultra")
    st.caption("Multi-mode RF sensing platform")

    with st.sidebar:
        st.header("Controls")

        mode = st.selectbox("Mode", [m.value for m in OperatingMode], index=0)
        waveform = st.selectbox("Waveform", [w.value for w in WaveformKind], index=0)
        backend = st.selectbox("Backend", ["simulation", "hackrf", "soapy"], index=0)
        center_freq_mhz = st.slider("Center Frequency (MHz)", 100.0, 6000.0, 915.0, step=1.0)
        sample_rate_mhz = st.slider("Sample Rate (MSPS)", 0.5, 20.0, 2.0, step=0.5)
        tx_gain = st.slider("TX Gain (dB)", 0.0, 47.0, 18.0, step=1.0)
        rx_gain = st.slider("RX Gain (dB)", 0.0, 40.0, 24.0, step=1.0)
        frame_size = st.selectbox("Frame Size", [1024, 2048, 4096, 8192], index=2)

        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("Start Session", type="primary", use_container_width=True)
        with col2:
            stop_btn = st.button("Stop Session", use_container_width=True)

    if stop_btn:
        _stop_session()
        st.rerun()

    st_state: StreamlitState = st.session_state.st_state
    session: RFSession | None = st_state.session

    if start_btn and session is None:
        session = _get_or_create_session(
            mode=OperatingMode(mode),
            waveform_kind=WaveformKind(waveform),
            backend=backend,
            center_freq=center_freq_mhz * 1e6,
            sample_rate=sample_rate_mhz * 1e6,
            frame_size=frame_size,
            tx_gain=tx_gain,
            rx_gain=rx_gain,
        )

    if session is None:
        st.info("Configure settings and press **Start Session** to begin.")
        return

    session.set_mode(OperatingMode(mode))
    session.set_waveform_kind(WaveformKind(waveform))

    snapshot = session.snapshot()

    status_cols = st.columns(5)
    labels = [
        ("Status", snapshot.status_text, "#00ff88"),
        ("Backend", snapshot.backend_name, "#ffcc00"),
        ("Mode", snapshot.mode_name, "#ff6600"),
        ("Waveform", snapshot.waveform_name, "#00ccff"),
        ("Frequency", f"{snapshot.center_frequency_hz / 1e6:.1f} MHz", "#cc88ff"),
    ]
    for col, (label, value, color) in zip(status_cols, labels):
        col.markdown(f'<div class="status-box"><small>{label}</small><br><strong style="color:{color}">{value}</strong></div>', unsafe_allow_html=True)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        spectrum_plot = _build_spectrum_plot(snapshot)
        st.plotly_chart(spectrum_plot, use_container_width=True, key="spectrum")
        motion_plot = _build_motion_plot(snapshot)
        st.plotly_chart(motion_plot, use_container_width=True, key="motion")

    with plot_col2:
        breathing_plot = _build_breathing_plot(snapshot)
        st.plotly_chart(breathing_plot, use_container_width=True, key="breathing")
        heatmap_plot = _build_heatmap_plot(snapshot)
        st.plotly_chart(heatmap_plot, use_container_width=True, key="heatmap")

    motion_conf = st.markdown(
        f'<div class="status-box">'
        f'<strong>Motion:</strong> {snapshot.motion_label} '
        f'(score: {snapshot.motion_score:.6f}, conf: {snapshot.motion_confidence:.3f})'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Frame rate: ~{st_state.frame_count / max(time.monotonic() - st_state.start_time, 1):.1f} fps")
    st_state.frame_count += 1
    time.sleep(0.05)
    st.rerun()


if __name__ == "__main__":
    run_streamlit_dashboard()
