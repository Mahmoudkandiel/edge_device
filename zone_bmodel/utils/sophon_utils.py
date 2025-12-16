"""
Utility functions for Sophon TPU detection and configuration.
"""

import os
from typing import Optional


def is_sophon_available() -> bool:
    """
    Check if Sophon SDK is available.
    
    Returns:
        True if sophon.sail can be imported, False otherwise.
    """
    try:
        import sophon.sail as sail
        return True
    except ImportError:
        return False


def find_bmodel(model_path: Optional[str] = None, search_paths: Optional[list] = None) -> Optional[str]:
    """
    Find a BModel file in common locations.
    
    Args:
        model_path: Explicit path to check first
        search_paths: List of paths to search (defaults to common locations)
        
    Returns:
        Path to BModel file if found, None otherwise.
    """
    if search_paths is None:
        search_paths = [
            "models/yolov8n_bm1684x.bmodel",
            "models/yolov8n.bmodel",
            "models/yolov8_bm1684x.bmodel",
            "models/yolov8.bmodel",
        ]
    
    # Check explicit path first
    if model_path and os.path.exists(model_path):
        if model_path.endswith('.bmodel'):
            return model_path
    
    # Search common locations
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None


def get_sophon_info() -> dict:
    """
    Get information about Sophon SDK and available devices.
    
    Returns:
        Dictionary with Sophon SDK information.
    """
    info = {
        "available": False,
        "version": None,
        "devices": [],
        "error": None,
    }
    
    try:
        import sophon.sail as sail
        info["available"] = True
        
        # Try to get version (may not be available in all SDK versions)
        try:
            info["version"] = getattr(sail, "__version__", "unknown")
        except:
            pass
        
        # Try to detect devices
        try:
            # This may vary by SDK version
            # Some versions have sail.get_device_count() or similar
            info["devices"] = [0]  # Default to device 0
        except:
            pass
            
    except ImportError as e:
        info["error"] = str(e)
    except Exception as e:
        info["error"] = f"Unexpected error: {e}"
    
    return info


def print_sophon_status() -> None:
    """Print current Sophon SDK status."""
    info = get_sophon_info()
    
    print("=" * 60)
    print("SOPHON SDK Status")
    print("=" * 60)
    print(f"Available: {info['available']}")
    
    if info['available']:
        print(f"Version: {info.get('version', 'unknown')}")
        print(f"Devices: {info.get('devices', [])}")
        
        # Check for BModel
        bmodel = find_bmodel()
        if bmodel:
            print(f"✅ BModel found: {bmodel}")
        else:
            print("⚠️  No BModel found in common locations")
            print("   Place your .bmodel file in the 'models/' directory")
    else:
        print(f"Error: {info.get('error', 'Unknown error')}")
        print("\nTo install Sophon SDK:")
        print("1. Download from: https://developer.sophgo.com/")
        print("2. Follow installation instructions")
        print("3. Set environment variables:")
        print("   export LD_LIBRARY_PATH=/opt/sophon/libsophon/lib:$LD_LIBRARY_PATH")
        print("   export PYTHONPATH=/opt/sophon/sophon-sail/python3/lib:$PYTHONPATH")
    
    print("=" * 60)


if __name__ == "__main__":
    print_sophon_status()

