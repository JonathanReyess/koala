# 🇰🇷 Koala (코아라) - Korean Sign Language (KSL) Recognition System

This is the repository for the Koala (코아라), a full-stack application that uses a deep learning model to **classify dynamic Korean Sign Language (KSL) words** from user video input. The system provides real-time AI feedback to help users practice KSL signs.

---

## What it Does

Koala addresses the need for accessible KSL learning tools by utilizing a vision-based approach. The system extracts 47 3D joint coordinates from video frames using MediaPipe Holistic and feeds this sequence data into a specialized CNN-LSTM-Attention model. 

This model analyzes the spatial and temporal patterns of the signs to classify them against 67 distinct KSL words. The system is deployed as a user-friendly web application with a FastAPI backend serving the PyTorch model and a React/TypeScript frontend for recording and displaying AI-driven results. 

---

## Quick Start

This guide explains how to run the full-stack Koala application locally.

### Prerequisites
* Python 3.9+
* Node.js / npm
* The trained model file (`best_model.pt`) is required in the `backend/` directory.

### 1. Backend Setup (FastAPI)

1.  **Clone the repository and navigate to the backend:**
    ```bash
    git clone [https://github.com/JonathanReyess/koala-sign-learn.git](https://github.com/JonathanReyess/koala-sign-learn.git)
    cd koala-sign-learn/backend
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the API server:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`.

### 2. Frontend Setup (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend
    ```
2.  **Install Node dependencies:**
    ```bash
    npm install
    ```
3.  **Configure API URL:** Ensure your `.env` file or environment variables are set correctly to point to the backend (e.g., `VITE_API_URL=http://localhost:8000`).
4.  **Run the web application:**
    ```bash
    npm run dev
    ```
    The web application will typically open in your browser at `http://localhost:5173`.

---

## Evaluation

The model was trained and evaluated on a subset of the **KSL77 dataset** (67 classes).

### Model Performance

| Metric | Result |
| :--- | :--- |
| Test Accuracy | 86.18% |
| Classes | 67  |
| Framework | PyTorch |

### Model Architecture (`PoseCNN_LSTM_Attn`)

The architecture is based on the principles in the paper: ["Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention‑Based Neural Network"](https://ieeexplore.ieee.org/document/10360810).

1.  **Feature Extraction:** MediaPipe Holistic extracts a fixed-length sequence of 47 keypoints (normalized 3D coordinates) per frame for a 32-frame clip.
2.  **Spatial Modeling (CNN):** `Conv1d` layers process the 47 joint features to extract spatial relationships.
3.  **Temporal Modeling (LSTM):** A Bidirectional LSTM processes the sequence of spatial features for long-range temporal dependencies.
4.  **Feature Aggregation (Attention):** An Attention Pooling mechanism computes a weighted context vector over all time steps to create a single, discriminative feature vector for classification.

---

## Video Links

### Demo Video
* [Direct link to your project demonstration video]

### Technical Walkthrough
* [Direct link to your technical explanation video]

---

## Technical Stack & Development Details

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model** | PyTorch, NumPy, Scikit-learn | Trained on KSL77 dataset, achieving 86.18% test accuracy. |
| **Feature Extraction** | MediaPipe Holistic, OpenCV | Extracts 47 3D joint coordinates across a fixed sequence length (32 frames). |
| **Backend API** | Python, FastAPI | Serves the trained PyTorch model and handles video uploads and preprocessing. |
| **Frontend** | Vite, React, TypeScript, Tailwind CSS | Provides a user-friendly interface for recording/uploading videos and displaying AI feedback. |

### Development Details
A detailed Jupyter Notebook is included in the [`notebook/`](notebook/) folder, providing a full walkthrough of video preprocessing, feature extraction, and model training (run on Google Colab using an NVIDIA T4 GPU).

> **Dataset Source:** Original KSL77 dataset and labels obtained from [Yangseung/KSL](https://github.com/Yangseung/KSL).
> **Reference Paper:** ["Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention‑Based Neural Network"](https://ieeexplore.ieee.org/document/10360810) by Jungpil Shin et al.
