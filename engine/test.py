import cv2
import numpy as np
import argparse
from sophon_engine import SophonInference

# Standard COCO classes for YOLO models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def postprocess_yolov8(outputs, conf_thres=0.25, iou_thres=0.45):
    """
    Parses YOLOv8 output.
    YOLOv8 output shape is typically (1, 84, 8400) -> (Batch, 4 box + 80 classes, anchors)
    """
    # 1. Get the main output array
    # Note: Depending on export, the key might vary. We take the first value.
    output0 = list(outputs.values())[0]  # Shape: (1, 84, 8400)
    
    # 2. Transpose to (1, 8400, 84) to make it easier to process rows
    # [batch, channels, anchors] -> [batch, anchors, channels]
    output0 = output0.transpose(0, 2, 1)
    prediction = output0[0]  # Take first batch: Shape (8400, 84)

    boxes = []
    confidences = []
    class_ids = []

    # 3. Iterate through anchors (vectorized for speed)
    # prediction[:, 4:] contains the class scores
    scores = prediction[:, 4:]
    max_scores = np.max(scores, axis=1)
    max_indices = np.argmax(scores, axis=1)

    # Filter by confidence threshold
    mask = max_scores > conf_thres
    
    # Apply filter
    valid_boxes = prediction[mask, :4]
    valid_scores = max_scores[mask]
    valid_indices = max_indices[mask]

    # 4. Prepare for NMS (Center-x, Center-y, W, H -> x, y, w, h for OpenCV)
    # The raw output is cx, cy, w, h normalized to model input size usually
    for i in range(len(valid_boxes)):
        cx, cy, w, h = valid_boxes[i]
        
        # Convert to top-left x, y
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        w = int(w)
        h = int(h)
        
        boxes.append([x, y, w, h])
        confidences.append(float(valid_scores[i]))
        class_ids.append(int(valid_indices[i]))

    # 5. Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],      # [x, y, w, h] in model coordinates
                "score": confidences[i],
                "class_id": class_ids[i]
            })
    return results

def run_yolo_logic():
    engine = SophonInference(bmodel_path="yolov8n_1688_f16.bmodel")
    input_meta = engine.get_input_details()
    print(f"Input Meta: {input_meta}")
    input_name = list(input_meta.keys())[0] 
    req_h = input_meta[input_name]['height']
    req_w = input_meta[input_name]['width']

    # --- 2. Preprocess ---
    image_path = "test.jpg"
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Could not load image {image_path}")
        return

    orig_h, orig_w = orig_img.shape[:2]

    # Simple Resize (Note: For best accuracy, use Letterbox padding, but simple resize works for testing)
    img_resized = cv2.resize(orig_img, (req_w, req_h)) 
    
    # Normalize & Transpose
    img_data = img_resized / 255.0
    img_data = img_data.transpose(2, 0, 1) 
    img_data = np.expand_dims(img_data, axis=0) 

    # --- 3. Inference ---
    outputs = engine.infer({input_name: img_data})
    
    # --- 4. Post-process & Rescale ---
    detections = postprocess_yolov8(outputs, conf_thres=0.5, iou_thres=0.5)

    # Calculate scaling factors to map back to original image
    # (Because we used simple resize, we just scale by the ratio)
    scale_x = orig_w / req_w
    scale_y = orig_h / req_h

    print(f"Found {len(detections)} objects.")

    # --- 5. Plotting ---
    for det in detections:
        x, y, w, h = det['box']
        score = det['score']
        class_id = det['class_id']
        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)

        # Scale coordinates back to original image size
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Draw Rectangle
        color = (0, 255, 0) # Green
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 2)

        # Draw Label
        caption = f"{label} {score:.2f}"
        cv2.putText(orig_img, caption, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- 6. Save Result ---
    output_path = "result.jpg"
    cv2.imwrite(output_path, orig_img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    run_yolo_logic()