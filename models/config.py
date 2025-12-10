# config.py

# --- Model and Data Constants ---
NUM_JOINTS = 47
SEQUENCE_LENGTH = 32
POSE_INDICES = [0, 11, 12, 13, 14]

# --- Model Configuration ---
MODEL_PATH = "models/best_model.pt"
NUM_CLASSES = 67
DEVICE_STR = "cpu" # Use a string 'cpu' for clarity in config

# --- MediaPipe Configuration ---
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# --- Mapping and Label Constants ---
perfect_mapped_classes = [0, 2, 4, 6, 9, 10, 12, 13, 14, 18, 20, 21, 24, 25, 33, 37, 40, 41, 42, 45, 46, 47, 48, 49, 51, 53, 55, 58, 62, 63, 64]
reverse_label_map = {
    0:1, 2:3, 4:5, 6:7, 9:10, 10:11, 12:14, 13:15, 14:16, 18:21, 20:23, 21:24,
    24:27, 25:29, 33:39, 37:43, 40:48, 41:49, 42:50, 45:54, 46:55, 47:56,
    48:57, 49:58, 51:60, 53:62, 55:64, 58:67, 62:71, 63:72, 64:74
}


#The model was trained in the notebook using re-indexed, zero-based integer labels (e.g., $0, 1, 2, ...$) instead of
#  the original, non-contiguous class IDs to satisfy PyTorch's training requirements. When the model makes a prediction 
# in `app.py`, it outputs one of these internal re-indexed values. The `reverse_label_map` is therefore essential to 
# translate this predicted index back into the original, semantic class ID that corresponds to the actual sign in the dataset.