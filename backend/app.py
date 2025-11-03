from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import tempfile
import os
from model import PoseCNN_LSTM_Attn

# Use mediapipe.solutions instead of full import
from mediapipe.python.solutions import holistic
from mediapipe.python.solutions.drawing_utils import draw_landmarks

# Configuration
NUM_JOINTS = 47
SEQUENCE_LENGTH = 32
POSE_INDICES = [0, 11, 12, 13, 14]
perfect_mapped_classes = [0, 2, 4, 6, 9, 10, 12, 13, 14, 18, 20, 21, 24, 25, 33, 37, 40, 41, 42, 45, 46, 47, 48, 49, 51, 53, 55, 58, 62, 63, 64]
reverse_label_map = {
    0:1, 2:3, 4:5, 6:7, 9:10, 10:11, 12:14, 13:15, 14:16, 18:21, 20:23, 21:24,
    24:27, 25:29, 33:39, 37:43, 40:48, 41:49, 42:50, 45:54, 46:55, 47:56,
    48:57, 49:58, 51:60, 53:62, 55:64, 58:67, 62:71, 63:72, 64:74
}

# Initialize FastAPI app
app = FastAPI()

# --- START: CRITICAL CORS SETUP FOR VERCEL/RENDER COMMUNICATION ---

# Get Vercel frontend URL from environment (set in Render dashboard)
VERCEL_FRONTEND_URL = os.environ.get("VITE_FRONTEND_URL") 
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "https://koala-sign-learn.vercel.app"  # Add your Vercel domain directly
]

if VERCEL_FRONTEND_URL:
    ALLOWED_ORIGINS.append(VERCEL_FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    # In production, this list restricts access to only the Vercel app.
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- END: CRITICAL CORS SETUP ---


# Load model
# Render runs on CPU, so we map the model to CPU
device = torch.device("cpu")
model = PoseCNN_LSTM_Attn(num_classes=67)
# Ensure best_model.pt is in the same directory as app.py (the root of the backend service)
state = torch.load("best_model.pt", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# MediaPipe setup
# Initialize Holistic model once for better performance
holistic_model = holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks_from_frame(results):
    frame_coords = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_coords[i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_coords[i + 21] = [lm.x, lm.y, lm.z]
    if results.pose_landmarks:
        for i, pose_index in enumerate(POSE_INDICES):
            lm = results.pose_landmarks.landmark[pose_index]
            frame_coords[i + 42] = [lm.x, lm.y, lm.z]
    return frame_coords

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    joint_seq = []
    # Ensure there are frames before attempting to sample indices
    if not frames:
        raise ValueError("Video file contained no frames.")

    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    for idx in indices:
        img = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = holistic_model.process(img)
        coords = extract_landmarks_from_frame(results)
        joint_seq.append(coords)
    
    joint_seq = np.array(joint_seq, dtype=np.float32)
    joint_seq = joint_seq.transpose(2, 0, 1)  # (C, T, V)
    return torch.tensor(joint_seq, dtype=torch.float32).unsqueeze(0).to(device)

@app.get("/")
def root():
    return {"message": "KSL Sign Recognition API is operational!"}

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        # NOTE: Since this is an ML service, you must use tempfile.NamedTemporaryFile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            contents = await video.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Preprocess and predict
        x = preprocess_video(tmp_path)
        with torch.no_grad():
            output = model(x)
            # Apply mask to perfect_mapped_classes
            mask = torch.full_like(output, float("-inf"), device=device)
            mask[0, perfect_mapped_classes] = output[0, perfect_mapped_classes]
            pred_idx = int(torch.argmax(mask, dim=1).item())
            label = reverse_label_map.get(pred_idx, "unknown")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "predicted_class": label,
            "class_id": pred_idx
        }
    
    except Exception as e:
        # Better error handling for file reading or processing issues
        error_message = f"Prediction failed: {type(e).__name__} - {str(e)}"
        
        # Clean up temp file if it was created
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        raise HTTPException(
            status_code=500,
            detail=error_message
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
