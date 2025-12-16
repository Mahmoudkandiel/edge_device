"""
Streamlit interface for the Enhanced Customer Foot Traffic Analyzer.

Run with: streamlit run streamlit_app.py
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional

import streamlit as st

from config.settings import Config
from modules.roi_selector import ROISelector
from modules.video_processor import VideoProcessor
from utils import capture_camera_frame, read_first_frame


st.set_page_config(
    page_title="Customer Foot Traffic Analyzer",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Customer Foot Traffic Analyzer – Streamlit Edition")
st.write(
    "Draw entry / exit / bypass regions directly on video frames or a live feed, "
    "then launch the analysis and review the results inside the browser."
)


def _save_uploaded_video(upload) -> str:
    suffix = os.path.splitext(upload.name)[-1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _make_report_filename(prefix: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def _launch_roi_selector_from_video(path: str) -> None:
    selector = ROISelector()
    try:
        roi = selector.select_roi_from_video_stream(path)
    except ValueError as exc:
        st.error(str(exc))
        return
    if roi.get("zones"):
        st.session_state["roi_config"] = roi
        st.success("ROI zones saved for this session.")
    else:
        st.warning("No ROI zones were stored.")


def _launch_roi_selector_from_camera(camera_index: int) -> None:
    selector = ROISelector()
    try:
        roi = selector.select_roi_from_camera_stream(camera_index)
    except ValueError as exc:
        st.error(str(exc))
        return
    if roi.get("zones"):
        st.session_state["roi_config"] = roi
        st.success("ROI zones saved for this session.")
    else:
        st.warning("No ROI zones were stored.")


def _launch_roi_selector_with_frame(frame, source_label: str) -> None:
    if frame is None:
        st.error(f"Unable to capture a frame from {source_label}.")
        return
    selector = ROISelector()
    roi = selector.select_roi_interactive(frame=frame)
    if roi.get("zones"):
        st.session_state["roi_config"] = roi
        st.success("ROI zones saved for this session.")
    else:
        st.warning("No ROI zones were stored.")


if "roi_config" not in st.session_state:
    st.session_state["roi_config"] = None


with st.sidebar:
    st.header("⚙️ Processing Options")
    frame_skip = st.slider("Frame skip", min_value=1, max_value=10, value=Config.VIDEO.FRAME_SKIP)
    save_video = st.checkbox("Save annotated video", value=Config.ANALYSIS.SAVE_VIDEO_OUTPUT)
    generate_reports = st.checkbox("Persist reports to disk", value=Config.ANALYSIS.GENERATE_REPORTS)
    st.caption("Note: OpenCV windows only appear when you launch ROI selection.")


roi_tab, video_tab, results_tab = st.tabs(["1️⃣ ROI Setup", "2️⃣ Video Analysis", "3️⃣ Results"])

with roi_tab:
    st.subheader("🎯 Define Regions of Interest")
    st.caption("Pick a video or live feed and a dedicated OpenCV window will let you draw the ROI lines.")

    roi_video_upload = st.file_uploader(
        "Video for ROI selection (mp4, avi, mov, mkv)",
        type=["mp4", "avi", "mov", "mkv"],
        key="roi_video_upload",
    )
    roi_video_path = st.text_input("…or use an existing server path:", value="", key="roi_video_path_input")
    if st.button("Launch ROI selector for this video", use_container_width=True):
        temp_path = None
        source_path = None
        if roi_video_upload is not None:
            temp_path = _save_uploaded_video(roi_video_upload)
            source_path = temp_path
        elif roi_video_path and os.path.exists(roi_video_path):
            source_path = roi_video_path
        else:
            st.error("No video was provided.")
        if source_path:
            try:
                _launch_roi_selector_from_video(source_path)
            except Exception:
                frame = read_first_frame(source_path)
                _launch_roi_selector_with_frame(frame, "the selected video")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    st.divider()
    st.subheader("🎥 Define ROI from a live camera")
    roi_camera_index = st.number_input("Camera index", value=0, step=1)
    if st.button("Capture from camera & draw ROI", use_container_width=True):
        try:
            _launch_roi_selector_from_camera(int(roi_camera_index))
        except Exception:
            frame = capture_camera_frame(int(roi_camera_index))
            _launch_roi_selector_with_frame(frame, "the camera feed")

    with st.expander("Advanced options"):
        st.caption("Optionally import/export ROI JSON.")
        roi_upload = st.file_uploader("Upload ROI JSON", type="json", key="roi_json_upload")
        if roi_upload is not None:
            try:
                st.session_state["roi_config"] = json.loads(roi_upload.getvalue().decode("utf-8"))
                st.success("ROI file loaded.")
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
        if st.session_state["roi_config"]:
            roi_bytes = json.dumps(st.session_state["roi_config"], ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "Download current ROI as JSON",
                data=roi_bytes,
                file_name="roi_config.json",
                mime="application/json",
            )

with video_tab:
    st.subheader("Upload or reference a video for analysis")
    video_upload = st.file_uploader(
        "Video file (mp4, avi, mov, mkv)",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )
    existing_video_path = st.text_input("…or server path:", value="", key="analysis_video_path_input")

with results_tab:
    st.subheader("Run analysis")
    run_button = st.button("🚀 Start processing", use_container_width=True)
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    report_placeholder = st.empty()

    if run_button:
        roi_config = st.session_state.get("roi_config")
        if roi_config is None:
            st.error("Please define an ROI first in the previous tab.")
        else:
            video_path = None
            temp_video_path = None
            if video_upload is not None:
                temp_video_path = _save_uploaded_video(video_upload)
                video_path = temp_video_path
            elif existing_video_path and os.path.exists(existing_video_path):
                video_path = existing_video_path

            if video_path is None:
                st.error("No video selected.")
            else:
                Config.VIDEO.FRAME_SKIP = frame_skip
                Config.ANALYSIS.SAVE_VIDEO_OUTPUT = save_video
                Config.ANALYSIS.GENERATE_REPORTS = generate_reports

                def progress_callback(processed: int, total: int) -> None:
                    if total <= 0:
                        return
                    ratio = min(processed / total, 1.0)
                    progress_bar.progress(ratio)
                    status_placeholder.write(f"Processed {processed} / {total} frames")

                processor = VideoProcessor(roi_config, display_windows=False)
                try:
                    with st.spinner("Processing video..."):
                        report = processor.process_video(
                            video_path,
                            progress_callback=progress_callback,
                        )
                    status_placeholder.success("📊 Analysis complete")
                    analytics = report.get("analytics", {})
                    processing_info = report.get("processing_info", {})

                    # Display IN/OUT counts prominently
                    st.markdown("### 📈 Entry/Exit Statistics")
                    cols = st.columns(3)
                    in_count = analytics.get("in_count", 0)
                    out_count = analytics.get("out_count", 0)
                    total_from_in_out = analytics.get("total_from_in_out", in_count + out_count)
                    cols[0].metric("IN (دخول)", in_count, delta=None)
                    cols[1].metric("OUT (خروج)", out_count, delta=None)
                    cols[2].metric("Total (إجمالي)", total_from_in_out, delta=None)

                    st.markdown("### 👥 Detailed Statistics")
                    cols = st.columns(3)
                    cols[0].metric("Total people", analytics.get("total_people", 0))
                    cols[1].metric("Bypassers", analytics.get("bypassers", 0))
                    cols[2].metric("Current Inside", analytics.get("current_inside", 0))

                    cols = st.columns(3)
                    cols[0].metric("Avg dwell time (s)", f"{analytics.get('avg_dwell_time', 0):.1f}")
                    cols[1].metric("Max concurrent", analytics.get("max_concurrent", 0))
                    cols[2].metric("Avg FPS", f"{processing_info.get('average_fps', 0):.1f}")

                    st.markdown("### Full report (JSON)")
                    report_placeholder.json(report)

                    json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button(
                        "⬇️ Download JSON report",
                        data=json_bytes,
                        file_name=_make_report_filename("traffic_report", "json"),
                        mime="application/json",
                    )

                    output_video_path = processing_info.get("output_video_path")
                    if output_video_path and os.path.exists(output_video_path):
                        with open(output_video_path, "rb") as video_file:
                            st.download_button(
                                "⬇️ Download annotated video",
                                data=video_file,
                                file_name=os.path.basename(output_video_path),
                                mime="video/x-msvideo",
                            )

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                finally:
                    # Clean up temporary file with retry logic
                    if temp_video_path and os.path.exists(temp_video_path):
                        import time as time_module
                        max_retries = 5
                        for attempt in range(max_retries):
                            try:
                                os.remove(temp_video_path)
                                break
                            except PermissionError:
                                if attempt < max_retries - 1:
                                    time_module.sleep(0.1)  # Wait 100ms before retry
                                else:
                                    # Last attempt failed, log but don't crash
                                    st.warning(f"Could not delete temporary file: {temp_video_path}")

