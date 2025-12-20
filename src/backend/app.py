from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import tempfile
import os
from model import PoseCNN_LSTM_Attn
import mediapipe as mp
holistic = mp.solutions.holistic

# --- Configuration ---
NUM_JOINTS = 47
SEQUENCE_LENGTH = 32
POSE_INDICES = [0, 11, 12, 13, 14]
perfect_mapped_classes = [
    0, 2, 4, 6, 10, 12, 14, 16, 17, 18, 19, 
    20, 22, 24, 33, 37, 39, 40, 41, 42, 46, 
    47, 48, 49, 51, 52, 53, 55, 58, 59, 62
]

# The keys are the 0-66 indices from your model's argmax
# The values are the FOLDER NUMBERS from your provided list
reverse_label_map = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 20, 18: 21, 19: 22,
    20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 29, 26: 30, 27: 31, 28: 32, 29: 34,
    30: 36, 31: 37, 32: 38, 33: 39, 34: 40, 35: 41, 36: 42, 37: 43, 38: 44, 39: 47,
    40: 48, 41: 49, 42: 50, 43: 51, 44: 52, 45: 54, 46: 55, 47: 56, 48: 57, 49: 58,
    50: 59, 51: 60, 52: 61, 53: 62, 54: 63, 55: 64, 56: 65, 57: 66, 58: 67, 59: 68,
    60: 69, 61: 70, 62: 71, 63: 72, 64: 74, 65: 76, 66: 77
}

# --- Allowed origins for CORS ---
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://koala-sign-learn.vercel.app",
    "http://localhost:8080",
    "http://localhost:8000"
]

# Optionally include environment variable
VERCEL_FRONTEND_URL = os.environ.get("VITE_FRONTEND_URL")
if VERCEL_FRONTEND_URL:
    ALLOWED_ORIGINS.append(VERCEL_FRONTEND_URL)

# --- Initialize FastAPI ---
app = FastAPI()

# --- Add CORSMiddleware (robust solution) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # exact origins only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the model ---
device = torch.device("cpu")
model = PoseCNN_LSTM_Attn(num_classes=67)
state = torch.load("best_model.pt", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# --- MediaPipe setup ---
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
    
    if not frames:
        raise ValueError("Video file contained no frames.")

    joint_seq = []
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    for idx in indices:
        img = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = holistic_model.process(img)
        coords = extract_landmarks_from_frame(results)
        joint_seq.append(coords)
    
    joint_seq = np.array(joint_seq, dtype=np.float32).transpose(2, 0, 1)
    return torch.tensor(joint_seq, dtype=torch.float32).unsqueeze(0).to(device)

# --- Routes ---
@app.get("/")
def root():
    return {"message": "KSL Sign Recognition API is operational!"}

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    tmp_path = None
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            contents = await video.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Preprocess and predict
        x = preprocess_video(tmp_path)
        with torch.no_grad():
            output = model(x)
            mask = torch.full_like(output, float("-inf"), device=device)
            mask[0, perfect_mapped_classes] = output[0, perfect_mapped_classes]
            pred_idx = int(torch.argmax(mask, dim=1).item())
            label = reverse_label_map.get(pred_idx, "unknown")
        
        return {"success": True, "predicted_class": label, "class_id": pred_idx}
    
    except Exception as e:
        error_message = f"Prediction failed: {type(e).__name__} - {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# --- Run locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
