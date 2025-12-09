import torch
import numpy as np
import cv2
import os
from mediapipe.python.solutions import holistic

# --- Import Configuration ---
# Ensure these constants exist in your config.py file
from .config import (
    NUM_JOINTS, SEQUENCE_LENGTH, POSE_INDICES,
    MODEL_PATH, NUM_CLASSES, DEVICE_STR,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE, MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    perfect_mapped_classes, reverse_label_map
)

# Assuming your model architecture class is in a sibling file named model_architecture.py
from .model_architecture import PoseCNN_LSTM_Attn 

# --- Model and Device Loading ---

def load_pose_model():
    """
    Initializes and loads the trained PoseCNN_LSTM_Attn model using constants from config.
    """
    
    device = torch.device(DEVICE_STR)
    model = PoseCNN_LSTM_Attn(num_classes=NUM_CLASSES)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}.")
        
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model, device

# --- Global Initialization (executed once on module import) ---
# Initialize the model and device
try:
    MODEL, DEVICE = load_pose_model()
    print(f"Model loaded successfully on {DEVICE} from {MODEL_PATH}.")
except Exception as e:
    print(f"FATAL: Model initialization failed: {e}")
    MODEL = None
    DEVICE = torch.device(DEVICE_STR)

# Initialize the MediaPipe Holistic model
try:
    holistic_model = holistic.Holistic(
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE, 
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
    )
    print("MediaPipe Holistic model initialized.")
except Exception as e:
    print(f"Warning: MediaPipe Holistic failed to initialize: {e}")
    holistic_model = None

# --- Preprocessing Functions ---

def extract_landmarks_from_frame(results):
    """
    Extracts relevant hand and pose landmarks into a fixed-size array.
    Order: Left Hand (0-20), Right Hand (21-41), Selected Pose (42-46).
    """
    if not holistic_model:
        raise RuntimeError("MediaPipe Holistic model is not initialized.")

    frame_coords = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    
    # Left Hand Landmarks (Indices 0-20)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_coords[i] = [lm.x, lm.y, lm.z]
    
    # Right Hand Landmarks (Indices 21-41)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_coords[i + 21] = [lm.x, lm.y, lm.z]
            
    # Selected Pose Landmarks (Indices 42-46)
    if results.pose_landmarks:
        for i, pose_index in enumerate(POSE_INDICES):
            lm = results.pose_landmarks.landmark[pose_index]
            frame_coords[i + 42] = [lm.x, lm.y, lm.z]
            
    return frame_coords

def preprocess_video(video_path):
    """Reads a video, extracts landmarks, and prepares the input tensor."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise ValueError("Video file contained no frames.")

    joint_seq = []
    # Select SEQUENCE_LENGTH frames evenly spaced from the video
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    
    for idx in indices:
        img = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        
        # Use the globally initialized holistic model
        results = holistic_model.process(img)
        coords = extract_landmarks_from_frame(results)
        joint_seq.append(coords)
    
    # Convert list of arrays to numpy array and transpose to (3, T, J)
    # Target shape: (Coords=3, Time=32, Joints=47)
    joint_seq = np.array(joint_seq, dtype=np.float32).transpose(2, 0, 1)
    
    # Convert to PyTorch tensor, add batch dimension, and move to device
    return torch.tensor(joint_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# --- Core Prediction Function ---

def predict_sign_language(video_path: str):
    """
    Processes a video file, runs inference on the loaded model, and returns 
    the predicted label and class ID.
    """
    if MODEL is None:
        return {"success": False, "predicted_class": "Model Error", "class_id": -1, "error": "Model failed to load."}
        
    try:
        # 1. Preprocess the video to get the input tensor
        x = preprocess_video(video_path)
        
        # 2. Run inference
        with torch.no_grad():
            output = MODEL(x)
            
            # 3. Apply masking for specific classes (perfect_mapped_classes)
            mask = torch.full_like(output, float("-inf"), device=DEVICE)
            mask[0, perfect_mapped_classes] = output[0, perfect_mapped_classes]
            
            # 4. Get prediction index
            pred_idx = int(torch.argmax(mask, dim=1).item())
            
            # 5. Get final label
            label = reverse_label_map.get(pred_idx, "unknown")
            
        return {"success": True, "predicted_class": label, "class_id": pred_idx}

    except Exception as e:
        error_message = f"Prediction failed during processing: {type(e).__name__} - {str(e)}"
        print(error_message)
        return {"success": False, "predicted_class": "Error", "class_id": -1, "error": error_message}