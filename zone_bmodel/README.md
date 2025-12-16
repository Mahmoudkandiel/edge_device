# CuTrack - Enhanced Customer Foot Traffic Analyzer

A comprehensive computer vision system for analyzing customer movement patterns around storefronts. It tracks people and classifies their movement patterns (entry/exit) using advanced zone-based detection with vector validation from recorded or live video feeds.

## 🎯 Features

- **Real-time Person Detection**: YOLOv8-based object detection with GPU acceleration support
- **Multi-Object Tracking**: Centroid-based tracking with trajectory analysis and history
- **Zone-Based Classification**: Double validation system using zone sequences (A→B, B→A) and movement vectors
- **Interactive ROI Selection**: Draw zone polygons (Zone A: outer, Zone B: inner) directly on video or live feed
- **Dual Interface**: CLI and Streamlit web UI for flexible usage
- **Comprehensive Reporting**: JSON, CSV, and visualization exports with detailed analytics
- **Edge Device Support**: Optimized deployment guides for Linux edge devices and SOPHON SE9
- **Automatic Backend Detection**: Automatically uses Sophon TPU (BModel) if available, falls back to PyTorch/YOLO
- **Sophon TPU Integration**: Native support for Sophon SE9 and other Sophon edge devices with optimized BModel inference

---

## 📚 Algorithms & Models

### Detection Models

- **YOLOv8 (Ultralytics)**: 
  - Model: `yolov8n.pt` (nano variant for speed)
  - Purpose: Person detection (COCO class 0)
  - Confidence Threshold: 0.5 (configurable)
  - Device: CUDA (if available) or CPU

### Tracking Algorithm

- **Centroid-Based Tracker**:
  - Algorithm: Euclidean distance-based association
  - Distance Metric: `scipy.spatial.distance.cdist` for centroid matching
  - Max Disappeared Frames: 50 (configurable)
  - Max Distance Threshold: 50 pixels (configurable)
  - Features:
    - Trajectory storage (deque with maxlen=1000)
    - Entry/exit point tracking
    - Dwell time calculation
    - First/last seen timestamps

### Classification Algorithm

- **Zone-Based Double Validation System**:
  - Algorithm: Zone Sequence (A→B or B→A) + Vector Validation (inward/outward direction)
  - Method: Uses two zones (Zone A: outer, Zone B: inner) with vector analysis
  - Classification Logic:
    - `inside`: A→B sequence + inward vector (entered inner zone)
    - `exited`: B→A sequence + outward vector (exited from inner zone)
    - `pending_inner`: In Zone B but entry not yet validated
    - `pending_outer`: In Zone A (outer zone)
    - `unknown`: No zone detected or insufficient trajectory data
  - Vector Analysis:
    - Analyzes last 10 trajectory points for movement direction
    - Inward vector: Primarily downward movement (positive Y)
    - Outward vector: Primarily upward movement (negative Y)
    - Minimum vector magnitude: 5.0 pixels to validate movement

### Analytics & Visualization

- **Statistical Aggregation**: Real-time counting and distribution analysis
- **Visualization Libraries**: 
  - Matplotlib for charts
  - Seaborn for styling
  - Pandas for data export

---

## 📁 Project Structure

```
cutrack/
├── __init__.py                 # Package initialization
├── main.py                     # CLI entry point
├── streamlit_app.py            # Streamlit web interface
├── requirements.txt            # Python dependencies (PyTorch/CUDA)
├── requirements_sophon.txt     # Python dependencies (Sophon optimized)
├── check_sophon.py             # Sophon deployment verification script
├── roi_config.json             # ROI zone configuration (JSON)
├── .gitignore                  # Git ignore rules (excludes models, outputs, etc.)
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration dataclasses (includes SophonConfig)
│
├── modules/
│   ├── __init__.py
│   ├── roi_selector.py         # Interactive zone polygon drawing
│   ├── tracker.py              # Person tracking engine
│   ├── classifier.py           # Zone-based movement classification
│   ├── demographics.py         # Person detection (auto-detects Sophon/PyTorch)
│   ├── demographics_sophon.py  # Sophon-specific person detector (BModel)
│   ├── video_processor.py      # Video file processing
│   ├── live_processor.py       # Live camera processing
│   └── analytics.py            # Statistics & reporting
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py              # Utility functions
│   └── sophon_utils.py         # Sophon SDK utilities and detection
│
├── models/                     # Model files directory (BModel, .pt, etc.)
│   └── yolov8n_bm1684x.bmodel # Sophon BModel (auto-detected if present)
│
├── outputs/
│   ├── videos/                 # Annotated video outputs
│   └── reports/                # JSON, CSV, PNG reports
│
└── docs/
    ├── deploy.md               # Linux edge device deployment guide
    └── README_SOPHON_SE9.md    # SOPHON SE9 deployment guide
```

---

## 🔄 Automatic Backend Detection

The system automatically detects and uses the best available detection backend:

### Detection Flow

```
PersonDetector.__init__()
    ↓
Check for Sophon BModel
    ├─ Found + Sophon SDK available → Use PersonDetectorSOPHON (TPU)
    └─ Not found → Check for PyTorch
        ├─ Available → Use YOLO (GPU/CPU)
        └─ Not available → Raise error with installation instructions
```

### Backend Priority

1. **Sophon TPU** (if available):
   - Checks for BModel in: `models/yolov8n_bm1684x.bmodel`, `models/yolov8n.bmodel`, etc.
   - Requires Sophon SDK installed separately
   - Fastest inference on Sophon edge devices
   - See `README_SOPHON_SE9.md` for setup

2. **PyTorch/YOLO** (fallback):
   - Uses `yolov8n.pt` (auto-downloaded on first run)
   - Works on CPU or CUDA GPU
   - Standard deployment option

### Status Display

On startup, the system displays which backend is being used:
```
✅ Sophon TPU detected - Using BModel: models/yolov8n_bm1684x.bmodel
```
or
```
ℹ️  Sophon SDK not available - using PyTorch/YOLO (if available)
```

### Verification

Check your setup:
```bash
python3 check_sophon.py
```

---

## 🔧 Module Functions

### `modules/roi_selector.py` - Zone Selection

**Class: `ROISelector`**

| Function | Description |
|----------|-------------|
| `__init__()` | Initialize ROI selector with empty state |
| `select_roi_interactive(video_path, frame)` | Main entry point for static frame or video-based zone selection |
| `select_roi_from_video_stream(video_path)` | Stream video and allow zone polygon drawing while playing |
| `select_roi_from_camera_stream(camera_index)` | Stream live camera feed for zone polygon drawing |
| `_run_static_loop()` | Handle static frame zone selection loop |
| `_run_stream_loop(frame_supplier, label)` | Handle streaming video/camera zone selection |
| `_reset_state()` | Clear all points and zones |
| `_mouse_callback(event, x, y, flags, param)` | Handle mouse clicks to add polygon points |
| `_draw_points_and_lines(image)` | Render zone polygon points and outlines on frame |
| `_draw_instructions(image, streaming)` | Draw keyboard shortcut overlay |
| `_handle_key(key, streaming)` | Process keyboard input for zone configuration |
| `_get_config()` | Export zone configuration as dict |
| `save_config(filename)` | Save zone config to JSON file (roi_config.json) |
| `load_config(filename)` | Load zone config from JSON file |

**Zone Configuration:**
- **Zone A (Outer)**: Define outer boundary polygon by clicking points
- **Zone B (Inner)**: Define inner boundary polygon (e.g., store entrance area)
- Zones are saved as polygons with multiple points in `roi_config.json`

**Keyboard Shortcuts:**
- Click points to define polygon vertices
- `c` - Undo last point
- `r` - Reset all points
- `s` - Save config to JSON
- `p` - Pause/resume (streaming only)
- `o` - Confirm and start processing
- `q` - Cancel/quit

---

### `modules/tracker.py` - Person Tracking

**Class: `PersonTracker`**

| Function | Description |
|----------|-------------|
| `__init__(max_disappeared, max_distance)` | Initialize tracker with thresholds |
| `update(detections, frame_count, timestamp)` | Main update loop: match detections to existing tracks |
| `_add_new_person(detection, frame_count, timestamp)` | Create new track for unmatched detection |
| `_update_person(person_id, detection, frame_count, timestamp)` | Update existing track with new detection |
| `_remove_person(person_id)` | Remove track and calculate final dwell time |
| `get_track_data(person_id)` | Get trajectory and metadata for a person |
| `get_all_tracks_data()` | Get all active track data |
| `classify_movement(person_id, roi_lines)` | Legacy classification (use MovementClassifier instead) |
| `draw_tracks(image)` | Draw bounding boxes, centroids, and trajectories |
| `_check_trajectory_intersection(trajectory, line_points)` | Check if trajectory intersects a line |
| `_line_intersection(...)` | Calculate line-segment intersection |

**Track Data Structure:**
```python
{
    "trajectory": deque([(x, y), ...]),  # Centroid history
    "entry_point": (x, y),                # First appearance
    "exit_point": (x, y),                 # Last appearance
    "first_seen": timestamp,
    "last_seen": timestamp,
    "entry_time": timestamp,
    "exit_time": timestamp,
    "dwell_time": seconds,
    "classification": "visitor" | "bypasser" | ...
}
```

---

### `modules/classifier.py` - Movement Classification

**Class: `MovementClassifier`**

| Function | Description |
|----------|-------------|
| `__init__(roi_config)` | Initialize with zone configuration from roi_config.json |
| `_process_roi_config()` | Extract Zone A (outer) and Zone B (inner) polygons from config |
| `classify_person_movement(tracker, person_id)` | Classify person movement using zone sequence + vector validation |
| `draw_classification_zones(image)` | Draw zone polygons on frame with thin outlines and labels |
| `_get_current_zone(point)` | Determine which zone (A, B, or None) a point is in |
| `_point_in_zone(point, zone)` | Check if a point is inside a zone polygon using cv2.pointPolygonTest |
| `_calculate_movement_vector(trajectory)` | Calculate movement vector from last N trajectory points |
| `_is_inward_vector(vector)` | Check if vector points inward (downward, positive Y) |
| `_is_outward_vector(vector)` | Check if vector points outward (upward, negative Y) |
| `_check_zone_sequence(zone_history, current_zone)` | Detect A→B or B→A zone transitions |

**Classification Rules:**
- `inside`: Zone A→B sequence detected AND inward vector validated (person entered inner zone)
- `exited`: Zone B→A sequence detected AND outward vector validated (person exited inner zone)
- `pending_inner`: Currently in Zone B but entry sequence not yet validated
- `pending_outer`: Currently in Zone A (outer zone)
- `unknown`: No zone detected or insufficient trajectory data

**Zone Configuration:**
- **Zone A (Outer)**: Light blue polygon (173, 216, 230) - outer boundary
- **Zone B (Inner)**: Orange polygon (255, 165, 0) - inner boundary (store entrance area)
- Zones are defined as polygons with multiple points in `roi_config.json`
- Vector validation parameters:
  - `VECTOR_HISTORY_SIZE`: 10 (number of trajectory points analyzed)
  - `MIN_VECTOR_MAGNITUDE`: 5.0 pixels (minimum movement to validate)

---

### `modules/demographics.py` - Person Detection

**Class: `PersonDetector`**

| Function | Description |
|----------|-------------|
| `__init__(detection_model_path=None)` | Auto-detect and load detection backend (Sophon BModel or PyTorch/YOLO) |
| `analyze_frame(frame)` | Detect people in frame and return bounding boxes and centroids |

**Automatic Backend Detection:**
- **Sophon TPU**: Automatically uses BModel if available (checks `models/yolov8n_bm1684x.bmodel` and other common paths)
- **PyTorch/YOLO**: Falls back to PyTorch/YOLO if Sophon unavailable
- **Error Handling**: Clear error messages if neither backend is available

**Detection Priority:**
1. Explicit model path provided in constructor
2. Sophon BModel in `models/` directory (if Sophon SDK available)
3. PyTorch/YOLO with `yolov8n.pt` (if PyTorch available)

---

### `modules/demographics_sophon.py` - Sophon TPU Detection

**Class: `PersonDetectorSOPHON`**

| Function | Description |
|----------|-------------|
| `__init__(bmodel_path, device_id=0)` | Load Sophon BModel for TPU inference |
| `analyze_frame(frame)` | Detect people using Sophon TPU and return bounding boxes and centroids |

**Features:**
- Native Sophon SAIL integration
- Optimized preprocessing for BModel input format
- YOLOv8 output format conversion
- Multi-device support (device_id parameter)

---

### `utils/sophon_utils.py` - Sophon Utilities

| Function | Description |
|----------|-------------|
| `is_sophon_available()` | Check if Sophon SDK is installed and importable |
| `find_bmodel(model_path=None)` | Locate BModel files in common locations |
| `get_sophon_info()` | Get Sophon SDK version and device information |
| `print_sophon_status()` | Display comprehensive Sophon status information |

---

### `modules/video_processor.py` - Video Processing

**Class: `VideoProcessor`**

| Function | Description |
|----------|-------------|
| `__init__(roi_config, display_windows=True)` | Initialize processor with ROI and components |
| `process_video(video_path, output_path, progress_callback)` | Process entire video file |
| `_process_frame(frame, frame_count, timestamp)` | Process single frame: detect, track, classify |
| `_draw_overlay(frame)` | Draw statistics overlay on frame |
| `_processing_fps()` | Calculate average processing FPS |
| `_final_report(processing_time, total_frames)` | Generate final report dictionary |
| `process_live(camera_index, duration)` | Process live camera feed (legacy, use LiveProcessor) |

**Processing Pipeline:**
1. Load video → Extract frame
2. `PersonDetector.analyze_frame()` → Get detections
3. `PersonTracker.update()` → Update tracks
4. `MovementClassifier.classify_person_movement()` → Classify each person
5. `Analytics.update_person_data()` → Update statistics
6. Draw annotations → Write to output video

---

### `modules/live_processor.py` - Live Processing

**Class: `LiveProcessor` (extends VideoProcessor)**

| Function | Description |
|----------|-------------|
| `__init__(roi_config, display_windows=True)` | Initialize live processor |
| `start_live_processing(camera_index, duration)` | Start live camera analysis |
| `stop_processing()` | Stop live processing |
| `get_processing_status()` | Get current processing state |

---

### `modules/analytics.py` - Analytics & Reporting

**Class: `Analytics`**

| Function | Description |
|----------|-------------|
| `__init__()` | Initialize analytics with empty stats |
| `update_person_data(person_id, track_data)` | Update stats for a person |
| `get_current_stats()` | Get real-time statistics |
| `get_final_stats()` | Get final stats with computed metrics |
| `save_report(report)` | Save JSON, CSV, and visualization |
| `_save_csv(timestamp)` | Export detailed CSV report |
| `_generate_visuals(timestamp)` | Generate PNG charts (pie, bar, metrics) |

**Statistics Tracked:**
- Total people detected
- People currently inside (Zone B)
- People who exited (completed B→A sequence)
- Max concurrent inside count
- Average dwell time (time spent in Zone B)
- Entry/exit event timestamps
- Trajectory data for each person

---

### `utils/helpers.py` - Utility Functions

| Function | Description |
|----------|-------------|
| `create_output_dir(path)` | Create directory if not exists |
| `calculate_centroid(bbox)` | Calculate bounding box center |
| `draw_text_with_background(image, text, position, ...)` | Draw text with background rectangle |
| `format_time(seconds)` | Format seconds as "Xs", "Xm", or "Xh" |
| `get_video_info(video_path)` | Get FPS, width, height, total frames |
| `check_camera_availability(camera_index)` | Check if camera is available |
| `read_first_frame(video_path)` | Read first frame from video |
| `capture_camera_frame(camera_index)` | Capture single frame from camera |
| `FPSCounter` | Class for FPS calculation |

---

### `config/settings.py` - Configuration

**Classes:**
- `VideoConfig`: Frame skip, confidence thresholds, ROI colors
- `AnalysisConfig`: Save video, generate reports flags
- `SophonConfig`: Sophon TPU configuration (model paths, device ID, performance settings)
- `Config`: Main config container with paths

**Default Settings:**
- `FRAME_SKIP`: 3 (process every 3rd frame)
- `CONFIDENCE_THRESHOLD`: 0.6
- `TRACKING_THRESHOLD`: 0.5
- `SAVE_VIDEO_OUTPUT`: True
- `GENERATE_REPORTS`: True

**Sophon Configuration:**
- `BMODEL_PATH`: None (auto-detect) or explicit path
- `MODEL_SEARCH_PATHS`: List of common BModel locations
- `DEVICE_ID`: 0 (for multi-TPU systems)
- `USE_FP16`: True (prefer FP16 for better accuracy)
- `FUSE_PREPROCESS`: False (enable if model compiled with --fuse_preprocess)

---

## 🚀 How to Run

### Quick Start

```bash
# 1. Clone or download the project
cd cutrack

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py

# 5. Or launch web UI
streamlit run streamlit_app.py
```

### Prerequisites

1. **Python 3.10+** (3.12 recommended)
2. **CUDA-capable GPU** (optional, for faster processing with PyTorch)
3. **Virtual Environment** (recommended)
4. **Camera or Video File** for analysis
5. **Detection Backend** (choose one):
   - **PyTorch/YOLO**: `yolov8n.pt` (automatically downloaded on first run)
   - **Sophon TPU**: `yolov8n_bm1684x.bmodel` (see `README_SOPHON_SE9.md` for conversion)

### Important: Git Repository Files

**Note**: This repository uses `.gitignore` to exclude large binary files and generated outputs:
- **Model files** (`.pt`, `.onnx`, `.bmodel`, `.mlir`, `.npz`) - These are large and should be downloaded/generated locally
- **Compiled outputs** (`.bmodel.json`, `.layer_group_*.json`, compiler profiles)
- **Output directories** (`outputs/`, `venv/`, `__pycache__/`)
- **Sophon compilation artifacts** (model directories, tensor files)

**To use the project:**
- **PyTorch/YOLO**: Models will be automatically downloaded on first run (`yolov8n.pt`)
- **Sophon TPU**: Convert and place `.bmodel` files in `models/` directory (see `README_SOPHON_SE9.md`)
- Generated outputs are saved to `outputs/` directory (excluded from git)

**Automatic Backend Selection:**
- The system automatically detects available backends in this order:
  1. Sophon BModel (if Sophon SDK installed and BModel found)
  2. PyTorch/YOLO (if PyTorch installed)
  3. Error message with installation instructions (if neither available)

### Edge Device Deployment

For deployment on edge devices, see:
- **Linux Edge Devices**: See `deploy.md` for Raspberry Pi, NVIDIA Jetson, Intel NUC, etc.
- **SOPHON SE9**: See `README_SOPHON_SE9.md` for SOPHON-specific deployment guide with BModel conversion

**Sophon Deployment Quick Check:**
```bash
# Verify Sophon setup before deployment
python3 check_sophon.py
```

This script checks:
- Sophon SDK installation status
- BModel file availability
- Directory structure
- Provides recommendations for setup

### Installation

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Note:** 
- The `requirements.txt` is configured for CUDA 12.4. For CPU-only systems, modify the PyTorch installation block in `requirements.txt` to use CPU wheels.
- For Sophon TPU deployment, use `requirements_sophon.txt` instead (excludes PyTorch, optimized for ARM64)
- Sophon SDK must be installed separately (not via pip) - see `README_SOPHON_SE9.md`

### CLI Usage

```powershell
python main.py
```

**Menu Options:**
1. **Configure ROI**: Draw zone polygons (Zone A: outer, Zone B: inner) on video or live feed
2. **Analyze Recorded Video**: Process a video file with zone-based tracking
3. **Live Camera Analysis**: Process live camera feed with real-time tracking
4. **View Saved Reports**: Browse generated reports (JSON, CSV, visualizations)
5. **Settings**: View current configuration (frame skip, thresholds, etc.)
6. **Exit**: Quit application

**Workflow:**
1. Start application → Menu appears
2. Select option 1 (Configure ROI) → Choose video or camera
3. OpenCV window opens → Draw Zone A (outer) polygon by clicking points
4. Draw Zone B (inner) polygon by clicking points
5. Press `o` to confirm and save configuration to `roi_config.json`
6. Select option 2 or 3 to process video/live feed
7. View real-time statistics in terminal
8. View results in `outputs/` directory (videos, reports, visualizations)

**ROI Configuration:**
- Zones are defined as polygons (multiple points forming a closed shape)
- **Zone A (Outer)**: Outer boundary zone - defines the monitoring area perimeter
- **Zone B (Inner)**: Inner boundary zone - defines the store entrance or target area
- Configuration is automatically saved to `roi_config.json`
- Can be loaded and reused for multiple video analyses

### Streamlit Web UI

```powershell
streamlit run streamlit_app.py
```

**Access:** Open browser to `http://localhost:8501`

**Features:**
- **Tab 1 - ROI Setup**: Upload video or use camera to draw zone polygons
- **Tab 2 - Video Analysis**: Upload video or specify server path
- **Tab 3 - Results**: Run analysis, view metrics, download reports

**Sidebar Options:**
- Frame skip slider (1-10)
- Save annotated video checkbox
- Generate reports checkbox
- Confidence threshold adjustment

---

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# Processing speed vs accuracy
Config.VIDEO.FRAME_SKIP = 3  # Process every Nth frame

# Detection sensitivity
Config.VIDEO.CONFIDENCE_THRESHOLD = 0.6

# Tracking parameters
Config.VIDEO.TRACKING_THRESHOLD = 0.5

# Output settings
Config.ANALYSIS.SAVE_VIDEO_OUTPUT = True
Config.ANALYSIS.GENERATE_REPORTS = True

# Sophon TPU configuration (optional)
Config.SOPHON.BMODEL_PATH = "models/custom.bmodel"  # Explicit path (None = auto-detect)
Config.SOPHON.DEVICE_ID = 0  # For multi-TPU systems
Config.SOPHON.USE_FP16 = True  # Prefer FP16 over INT8

# Zone colors (BGR format) - configured in classifier.py
# Zone A (outer): Light blue (173, 216, 230)
# Zone B (inner): Orange (255, 165, 0)
```

---

## 📊 Outputs

### Video Outputs
- **Location**: `outputs/videos/`
- **Format**: AVI (XVID codec) or MP4 (H.264)
- **Content**: Original video with annotations:
  - Bounding boxes around detected people
  - Trajectory lines showing movement paths
  - Person IDs and current classification status
  - Zone polygons (Zone A and Zone B) with thin outlines
  - Real-time statistics panel (total people, current inside, FPS, etc.)
  - Color coding: Different colors for different classification states

### Report Outputs
- **Location**: `outputs/reports/`
- **Files**:
  - `report_YYYYMMDD_HHMMSS.json`: Full report with all data
  - `detailed_data_YYYYMMDD_HHMMSS.csv`: Per-person detailed data
  - `visualization_YYYYMMDD_HHMMSS.png`: Charts and graphs

**Report Contents:**
- **Processing Metadata**: FPS, processing time, total frames, frame skip settings
- **Analytics Summary**: Total people detected, current inside count, exit count, average dwell time
- **Per-Person Tracking Data**: 
  - Person ID, classification status
  - Entry/exit timestamps
  - Dwell time
  - Trajectory points
  - Zone history
  - Entry/exit frame numbers

---

## 🔬 Technical Details

### Tracking Algorithm Flow

1. **Detection**: YOLOv8 detects people in frame → bounding boxes
2. **Centroid Calculation**: Compute center of each bounding box
3. **Distance Matrix**: Calculate Euclidean distances between existing tracks and new detections
4. **Hungarian-style Matching**: Greedy assignment based on minimum distance
5. **Track Update**: Update matched tracks, create new for unmatched, remove disappeared
6. **Trajectory Storage**: Append centroids to deque (max 1000 points)

### Classification Algorithm Flow

1. **Trajectory Extraction**: Get centroid history for person
2. **Zone Detection**: Check current zone (A, B, or None) using point-in-polygon test
3. **Zone History Tracking**: Maintain zone history for sequence detection
4. **Vector Calculation**: Analyze last N trajectory points for movement direction
5. **Double Validation**: 
   - Entry: A→B sequence AND inward vector
   - Exit: B→A sequence AND outward vector
6. **Final Classification**: Apply validated rules based on zone sequence and vector direction

### Performance Considerations

- **Frame Skipping**: Process every Nth frame (default: 3) to balance speed/accuracy
- **Trajectory Limiting**: Max 1000 points per track (configurable) to manage memory
- **GPU Acceleration**: YOLO runs on CUDA if available, falls back to CPU
- **Zone History Limiting**: Max 50 zone states per person to prevent memory issues
- **Vector Analysis**: Only analyzes last 10 trajectory points for efficiency
- **Batch Processing**: Detections processed in batches per frame
- **Edge Optimization**: See deployment guides for edge device optimizations

---

## 🛠️ Dependencies

### Standard Deployment (PyTorch/YOLO)
See `requirements.txt` for full list. Key dependencies:

- `opencv-python`: Video I/O and image processing
- `ultralytics`: YOLOv8 detection
- `torch`, `torchvision`: Deep learning backend (CUDA 12.4 or CPU)
- `numpy`, `scipy`: Numerical operations
- `pandas`, `matplotlib`, `seaborn`: Data analysis and visualization
- `streamlit`: Web UI framework

### Sophon TPU Deployment
See `requirements_sophon.txt` for optimized dependencies:

- `opencv-python`: Video I/O and image processing
- `numpy`, `scipy`: Numerical operations
- `pandas`, `matplotlib`, `seaborn`: Data analysis and visualization
- `streamlit`: Web UI framework (optional)
- **Sophon SDK**: Must be installed separately (not via pip) - see `README_SOPHON_SE9.md`
- **Note**: PyTorch/Ultralytics not required for Sophon deployment (code auto-detects backend)

---

## 📖 Additional Documentation

- **Edge Deployment**: See `deploy.md` for Linux edge device deployment (Raspberry Pi, Jetson, Intel NUC, etc.)
- **SOPHON SE9**: See `README_SOPHON_SE9.md` for SOPHON SE9-specific deployment with BModel conversion
- **Architecture**: See `new.md` for detailed system architecture and component breakdown
- **Arabic Documentation**: See `EXPLANATION_AR.md` for Arabic explanations and technical details
- **In/Out Tracking**: See `in_out_tracking.md` for detailed tracking algorithm explanations

---

## 📝 License

[Add your license here]

---

## 🤝 Contributing

[Add contribution guidelines here]

---

## 📧 Contact

[Add contact information here]

---

## 🔄 Version History

- **v1.1.0**: Sophon TPU Integration
  - Automatic backend detection (Sophon BModel or PyTorch/YOLO)
  - Native Sophon TPU support with optimized BModel inference
  - New `demographics_sophon.py` module for TPU detection
  - Sophon utilities (`sophon_utils.py`) for SDK detection and status
  - `check_sophon.py` script for deployment verification
  - `SophonConfig` in settings for TPU configuration
  - Backward compatible - existing code works without changes
  - Graceful fallback to PyTorch if Sophon unavailable

- **v1.0.0**: Initial release with zone-based double validation system
  - Zone A→B entry detection with vector validation
  - Zone B→A exit detection with vector validation
  - Enhanced trajectory analysis
  - Support for polygon-based zone configuration
  - Real-time tracking and classification
  - Comprehensive reporting (JSON, CSV, visualizations)
  - Dual interface (CLI + Streamlit web UI)
  - Edge device deployment guides (Linux + SOPHON SE9)

---

## 💡 Usage Tips

### Zone Configuration Best Practices

1. **Zone A (Outer)**: Should encompass the entire monitoring area
   - Include all possible entry/exit paths
   - Make it large enough to capture people before they enter Zone B

2. **Zone B (Inner)**: Should define the target area (e.g., store entrance)
   - Should be completely inside Zone A
   - Size should match your actual entrance/area of interest

3. **Camera Positioning**: 
   - Mount camera at an angle that shows clear entry/exit paths
   - Ensure both zones are clearly visible
   - Avoid occlusions (poles, signs, etc.)

### Performance Optimization

- **For Faster Processing**: Increase `FRAME_SKIP` to 5 or 10
- **For Better Accuracy**: Decrease `FRAME_SKIP` to 1 or 2
- **For Edge Devices**: See deployment guides for device-specific optimizations
- **Memory Issues**: Reduce trajectory maxlen or disable video output

### Troubleshooting

- **No detections**: Check camera/video path, adjust confidence threshold
- **Poor tracking**: Reduce frame skip, check lighting conditions
- **Zone not detected**: Ensure zones are properly configured in `roi_config.json`
- **Low FPS**: Enable GPU acceleration (PyTorch) or use Sophon TPU, increase frame skip, reduce resolution
- **Sophon not detected**: 
  - Run `python3 check_sophon.py` to verify setup
  - Check Sophon SDK installation and environment variables
  - Ensure BModel file is in `models/` directory
  - See `README_SOPHON_SE9.md` for detailed setup instructions
- **Backend selection issues**: Check console output on startup - system displays which backend is being used

---

---

## 📦 New Files in v1.1.0

### Sophon TPU Support Files

- **`modules/demographics_sophon.py`**: Sophon-specific person detector using BModel inference
- **`utils/sophon_utils.py`**: Utility functions for Sophon SDK detection and status checking
- **`check_sophon.py`**: Standalone script to verify Sophon deployment setup
- **`requirements_sophon.txt`**: Optimized dependencies for Sophon edge devices (excludes PyTorch)
- **`SOPHON_CHANGES.md`**: Technical documentation of all Sophon-related code changes

### Updated Files

- **`modules/demographics.py`**: Added automatic Sophon/PyTorch detection with graceful fallback
- **`config/settings.py`**: Added `SophonConfig` dataclass for TPU configuration
- **`main.py`**: Added Sophon status check on startup
- **`modules/__init__.py`**: Added optional Sophon imports
- **`utils/__init__.py`**: Added Sophon utilities

### Directory Structure Updates

- **`models/`**: New directory for model files (BModel, .pt, etc.) - auto-created
- All model files are automatically detected from this directory

---

**Happy Analyzing! 🎯**
