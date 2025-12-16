# Sophon Deployment - Code Changes Summary

This document summarizes all changes made to enable Sophon TPU deployment support.

## Overview

The codebase has been updated to automatically detect and use Sophon BModel when available, with graceful fallback to PyTorch/YOLO. This allows the same codebase to work on both Sophon edge devices and standard systems.

## New Files

### 1. `modules/demographics_sophon.py`
- Sophon-specific person detector implementation
- Uses `sophon.sail` for BModel inference
- Handles YOLOv8 output format conversion
- Includes preprocessing and postprocessing for BModel

### 2. `utils/sophon_utils.py`
- Utility functions for Sophon detection
- `is_sophon_available()` - Check if SDK is installed
- `find_bmodel()` - Locate BModel files
- `get_sophon_info()` - Get SDK information
- `print_sophon_status()` - Display status

### 3. `requirements_sophon.txt`
- Dependencies optimized for Sophon deployment
- Excludes PyTorch (not needed for Sophon)
- Includes installation notes

### 4. `check_sophon.py`
- Standalone script to verify Sophon setup
- Checks SDK availability, BModel presence, directories
- Provides recommendations

### 5. `DEPLOYMENT_SOPHON.md`
- Quick deployment guide
- Troubleshooting tips
- Configuration examples

## Modified Files

### 1. `modules/demographics.py`
**Changes:**
- Added automatic Sophon detection
- Falls back to PyTorch/YOLO if Sophon unavailable
- Maintains backward compatibility
- No API changes - same interface

**Key Features:**
- Auto-detects BModel in common locations
- Uses Sophon if available, PyTorch otherwise
- Clear error messages if neither available

### 2. `config/settings.py`
**Changes:**
- Added `SophonConfig` dataclass
- Added `MODELS_DIR` constant
- Configurable model paths and device ID
- Performance settings (FP16/INT8 preference)

### 3. `main.py`
**Changes:**
- Added Sophon status check on startup
- Displays which backend is being used
- Non-blocking (continues even if check fails)

### 4. `modules/__init__.py`
**Changes:**
- Added optional import for `PersonDetectorSOPHON`
- Graceful handling if Sophon module unavailable

### 5. `utils/__init__.py`
**Changes:**
- Added optional import for Sophon utilities
- Maintains backward compatibility

## How It Works

### Automatic Detection Flow

```
1. PersonDetector.__init__() called
   ↓
2. Check for Sophon BModel
   ├─ Found → Use PersonDetectorSOPHON
   └─ Not found → Check for PyTorch
       ├─ Available → Use YOLO
       └─ Not available → Raise error with instructions
```

### Model Detection

The system checks for BModel files in this order:
1. Explicit path provided in constructor
2. `models/yolov8n_bm1684x.bmodel`
3. `models/yolov8n.bmodel`
4. `models/yolov8_bm1684x.bmodel`
5. `models/yolov8.bmodel`

### Backward Compatibility

- ✅ Existing code works without changes
- ✅ Works on systems without Sophon SDK
- ✅ Works on systems without PyTorch (if Sophon available)
- ✅ No breaking API changes

## Usage Examples

### Basic Usage (Auto-detect)
```python
from modules.demographics import PersonDetector

# Automatically uses Sophon if BModel found, PyTorch otherwise
detector = PersonDetector()
detections = detector.analyze_frame(frame)
```

### Explicit Sophon Path
```python
detector = PersonDetector("models/my_model.bmodel")
```

### Check Status
```python
from utils.sophon_utils import is_sophon_available, find_bmodel

if is_sophon_available():
    bmodel = find_bmodel()
    if bmodel:
        print(f"Using Sophon with: {bmodel}")
```

## Testing

### Verify Setup
```bash
python3 check_sophon.py
```

### Test Detection
```python
from modules.demographics import PersonDetector
import cv2

detector = PersonDetector()
frame = cv2.imread("test.jpg")
detections = detector.analyze_frame(frame)
print(f"Found {len(detections)} people")
```

## Performance Considerations

### Sophon Advantages
- ✅ Faster inference on TPU
- ✅ Lower power consumption
- ✅ Optimized for edge deployment
- ✅ INT8 quantization support

### Fallback Behavior
- PyTorch/YOLO used if Sophon unavailable
- Same detection results
- May be slower on CPU

## Configuration

Edit `config/settings.py`:

```python
Config.SOPHON.BMODEL_PATH = "models/custom.bmodel"  # Optional
Config.SOPHON.DEVICE_ID = 0  # For multi-TPU
Config.SOPHON.USE_FP16 = True  # Prefer FP16
```

## Deployment Checklist

- [ ] Sophon SDK installed and configured
- [ ] Environment variables set (LD_LIBRARY_PATH, PYTHONPATH)
- [ ] BModel file converted and placed in `models/`
- [ ] Dependencies installed: `pip install -r requirements_sophon.txt`
- [ ] Verified with: `python3 check_sophon.py`
- [ ] Tested detection: `python3 main.py`

## Troubleshooting

See `DEPLOYMENT_SOPHON.md` for detailed troubleshooting guide.

## Notes

- Sophon SDK must be installed separately (not via pip)
- BModel conversion requires TPU-MLIR tools (see README_SOPHON_SE9.md)
- Code maintains full backward compatibility
- No changes needed to existing video processing or tracking code

