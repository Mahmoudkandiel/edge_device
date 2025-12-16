import cv2
import numpy as np
from sophon_engine import SophonInference

def run_yolo_logic():
    # 1. Initialize (Done once)
    engine = SophonInference(bmodel_path="engine/yolov8n_1688_f16.bmodel", device_id=0)
    input_meta = engine.get_input_details()
    # Assume generic YOLO single input named 'images' (or 'input' depending on export)
    input_name = list(input_meta.keys())[0] 
    req_h = input_meta[input_name]['height']
    req_w = input_meta[input_name]['width']

    # 2. Preprocess (Your Custom Logic)
    img = cv2.imread("test.jpg")
    
    # -- Your custom Letterbox / Resize logic goes here --
    img_resized = cv2.resize(img, (req_w, req_h)) 
    
    # Normalize & Transpose (HWC -> CHW)
    img_data = img_resized / 255.0
    img_data = img_data.transpose(2, 0, 1) 
    img_data = np.expand_dims(img_data, axis=0) 


    outputs = engine.infer({input_name: img_data})
    print("Output keys:", outputs.keys())
    print("First output shape:", list(outputs.values())[0].shape)

if __name__ == "__main__":
    run_yolo_logic()