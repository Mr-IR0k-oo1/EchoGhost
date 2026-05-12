"""Main dashboard entrypoint for EchoGhost Hub Ultra."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

from .config.presets import DashboardConfig, OperatingMode, ProcessingConfig, RadioConfig, SimulationConfig, WaveformConfig, WaveformKind
from .radio.session import RFSession
from .visualization.panels import heatmap_to_rgba


@dataclass(slots=True)
class UiTags:
    status_text: str = "status_text"
    backend_text: str = "backend_text"
    mode_text: str = "mode_text"
    waveform_text: str = "waveform_text"
    motion_score_text: str = "motion_score_text"
    motion_label_text: str = "motion_label_text"
    motion_confidence_text: str = "motion_confidence_text"
    breathing_text: str = "breathing_text"
    breathing_confidence_text: str = "breathing_confidence_text"
    spectrum_series: str = "spectrum_series"
    motion_series: str = "motion_series"
    breathing_series: str = "breathing_series"
    heatmap_texture: str = "heatmap_texture"
    toggle_button: str = "toggle_button"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EchoGhost Hub Ultra dashboard")
    parser.add_argument("--backend", choices=["simulation", "hackrf", "soapy", "hardware"], default="simulation")
    parser.add_argument("--mode", choices=[mode.value for mode in OperatingMode], default=OperatingMode.SIMULATION.value)
    parser.add_argument("--waveform", choices=[kind.value for kind in WaveformKind], default=WaveformKind.TONE.value)
    parser.add_argument("--center-frequency", type=float, default=915_000_000.0)
    parser.add_argument("--sample-rate", type=float, default=2_000_000.0)
    parser.add_argument("--frame-size", type=int, default=4096)
    parser.add_argument("--tx-gain", type=float, default=18.0)
    parser.add_argument("--rx-gain", type=float, default=24.0)
    parser.add_argument("--refresh-hz", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true", help="Run the processing loop without the GUI.")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to print in headless mode; 0 means run until interrupted.")
    parser.add_argument("--application-title", default="EchoGhost Hub Ultra")
    return parser


def build_session(args: argparse.Namespace) -> RFSession:
    radio_config = RadioConfig(
        center_frequency_hz=args.center_frequency,
        sample_rate_sps=args.sample_rate,
        tx_gain_db=args.tx_gain,
        rx_gain_db=args.rx_gain,
        frame_size=args.frame_size,
        backend=args.backend,
        mode=OperatingMode(args.mode),
    )
    waveform_config = WaveformConfig(kind=WaveformKind(args.waveform))
    dashboard_config = DashboardConfig(refresh_hz=args.refresh_hz, application_title=args.application_title)
    processing_config = ProcessingConfig()
    simulation_config = SimulationConfig()
    return RFSession(
        radio_config=radio_config,
        waveform_config=waveform_config,
        processing_config=processing_config,
        dashboard_config=dashboard_config,
        simulation_config=simulation_config,
    )


def run_headless(session: RFSession, frame_limit: int = 0) -> int:
    session.start()
    try:
        frame_index = 0
        refresh_period_s = 1.0 / max(session.dashboard_config.refresh_hz, 1e-6)
        while True:
            snapshot = session.snapshot()
            breathing_text = "--" if snapshot.breathing_bpm is None else f"{snapshot.breathing_bpm:0.1f} bpm"
            print(
                f"[{frame_index:03d}] backend={snapshot.backend_name} mode={snapshot.mode_name} "
                f"waveform={snapshot.waveform_name} motion={snapshot.motion_label} "
                f"score={snapshot.motion_score:0.6f} breathing={breathing_text} "
                f"heatmap={snapshot.heatmap_result.matrix_db.shape}"
            )
            frame_index += 1
            if frame_limit > 0 and frame_index >= frame_limit:
                break
            time.sleep(refresh_period_s)
    except KeyboardInterrupt:
        return 130
    finally:
        session.stop()
    return 0


def _pad_heatmap(matrix: np.ndarray, rows: int, cols: int, fill_value: float) -> np.ndarray:
    padded = np.full((rows, cols), fill_value, dtype=np.float32)
    if matrix.size == 0:
        return padded
    source = np.asarray(matrix, dtype=np.float32)
    row_count = min(rows, source.shape[0])
    col_count = min(cols, source.shape[1])
    padded[-row_count:, :col_count] = source[-row_count:, :col_count]
    return padded


def run_gui(session: RFSession) -> int:
    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        print("Dear PyGui is not installed, falling back to headless mode.")
        return run_headless(session, frame_limit=5)

    tags = UiTags()
    heatmap_rows = session.processing_config.heatmap_history_size
    heatmap_cols = session.processing_config.heatmap_fft_size

    def on_mode_change(_sender: object, app_data: object, _user_data: object) -> None:
        session.set_mode(OperatingMode(str(app_data)))

    def on_waveform_change(_sender: object, app_data: object, _user_data: object) -> None:
        session.set_waveform_kind(WaveformKind(str(app_data)))

    def on_toggle(_sender: object, _app_data: object, _user_data: object) -> None:
        if session.is_running:
            session.stop()
            dpg.configure_item(tags.toggle_button, label="Start Session")
        else:
            session.start()
            dpg.configure_item(tags.toggle_button, label="Stop Session")

    def update_dashboard() -> None:
        snapshot = session.snapshot()
        dpg.set_value(tags.status_text, snapshot.status_text)
        dpg.set_value(tags.backend_text, f"Backend: {snapshot.backend_name}")
        dpg.set_value(tags.mode_text, f"Mode: {snapshot.mode_name}")
        dpg.set_value(tags.waveform_text, f"Waveform: {snapshot.waveform_name}")
        dpg.set_value(tags.motion_score_text, f"Motion score: {snapshot.motion_score:0.6f}")
        dpg.set_value(tags.motion_label_text, f"Motion class: {snapshot.motion_label}")
        dpg.set_value(tags.motion_confidence_text, f"Motion confidence: {snapshot.motion_confidence:0.3f}")
        breathing_text = "Breathing: --" if snapshot.breathing_bpm is None else f"Breathing: {snapshot.breathing_bpm:0.1f} bpm"
        dpg.set_value(tags.breathing_text, breathing_text)
        dpg.set_value(tags.breathing_confidence_text, f"Breathing confidence: {snapshot.breathing_confidence:0.3f}")

        if snapshot.spectrum_db.size:
            frequency_mhz = (snapshot.spectrum_frequency_hz / 1e6).astype(np.float32, copy=False)
            dpg.set_value(tags.spectrum_series, [frequency_mhz.tolist(), snapshot.spectrum_db.tolist()])

        if snapshot.motion_history.size:
            motion_x = np.arange(snapshot.motion_history.size, dtype=np.float32)
            dpg.set_value(tags.motion_series, [motion_x.tolist(), snapshot.motion_history.tolist()])

        if snapshot.breathing_history_bpm.size:
            breathing_x = np.arange(snapshot.breathing_history_bpm.size, dtype=np.float32)
            dpg.set_value(tags.breathing_series, [breathing_x.tolist(), snapshot.breathing_history_bpm.tolist()])

        heatmap_matrix = _pad_heatmap(
            snapshot.heatmap_result.matrix_db,
            rows=heatmap_rows,
            cols=heatmap_cols,
            fill_value=snapshot.heatmap_result.min_db,
        )
        heatmap_rgba = heatmap_to_rgba(
            heatmap_matrix,
            floor_db=snapshot.heatmap_result.min_db,
            ceiling_db=snapshot.heatmap_result.max_db,
        )
        dpg.set_value(tags.heatmap_texture, heatmap_rgba.tolist())

    session.start()
    dpg.create_context()
    try:
        default_texture = np.zeros((heatmap_rows, heatmap_cols, 4), dtype=np.float32).reshape(-1).tolist()
        dpg.create_viewport(title=session.dashboard_config.application_title, width=1600, height=960)
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=heatmap_cols, height=heatmap_rows, default_value=default_texture, tag=tags.heatmap_texture)

        with dpg.window(label="EchoGhost Hub Ultra", tag="main_window", width=-1, height=-1):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=340, height=-1):
                    dpg.add_text("Control Surface")
                    dpg.add_separator()
                    dpg.add_combo([mode.value for mode in OperatingMode], default_value=session.mode.value, label="Mode", callback=on_mode_change)
                    dpg.add_combo([kind.value for kind in WaveformKind], default_value=session.waveform_kind.value, label="Waveform", callback=on_waveform_change)
                    dpg.add_button(label="Stop Session", tag=tags.toggle_button, callback=on_toggle)
                    dpg.add_separator()
                    dpg.add_text(f"Center frequency: {session.radio_config.center_frequency_hz / 1e6:0.3f} MHz")
                    dpg.add_text(f"Sample rate: {session.radio_config.sample_rate_sps / 1e6:0.3f} MSPS")
                    dpg.add_text(f"Frame size: {session.radio_config.frame_size}")
                    dpg.add_separator()
                    dpg.add_text("Status")
                    dpg.add_text("Initializing", tag=tags.status_text)
                    dpg.add_text("Backend: --", tag=tags.backend_text)
                    dpg.add_text("Mode: --", tag=tags.mode_text)
                    dpg.add_text("Waveform: --", tag=tags.waveform_text)
                    dpg.add_text("Motion score: --", tag=tags.motion_score_text)
                    dpg.add_text("Motion class: --", tag=tags.motion_label_text)
                    dpg.add_text("Motion confidence: --", tag=tags.motion_confidence_text)
                    dpg.add_text("Breathing: --", tag=tags.breathing_text)
                    dpg.add_text("Breathing confidence: --", tag=tags.breathing_confidence_text)
                with dpg.child_window(width=-1, height=-1):
                    with dpg.plot(label="Spectrum", height=220, width=-1):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (MHz)")
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude (dB)")
                        dpg.add_line_series([], [], label="Spectrum", tag=tags.spectrum_series, parent=y_axis)

                    with dpg.plot(label="Motion History", height=180, width=-1):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame")
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Score")
                        dpg.add_line_series([], [], label="Motion", tag=tags.motion_series, parent=y_axis)

                    with dpg.plot(label="Breathing History", height=180, width=-1):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame")
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="BPM")
                        dpg.add_line_series([], [], label="Breathing", tag=tags.breathing_series, parent=y_axis)

                    dpg.add_separator()
                    dpg.add_text("Rolling heatmap / range-Doppler proxy")
                    dpg.add_image(tags.heatmap_texture)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        while dpg.is_dearpygui_running():
            update_dashboard()
            dpg.render_dearpygui_frame()
    finally:
        session.stop()
        dpg.destroy_context()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    session = build_session(args)
    if args.headless:
        return run_headless(session, frame_limit=args.frames)
    return run_gui(session)


if __name__ == "__main__":
    raise SystemExit(main())
