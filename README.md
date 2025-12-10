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

1. **Clone the repository and navigate to the backend:**
   ```bash
   git clone [https://github.com/JonathanReyess/koala-sign-learn.git](https://github.com/JonathanReyess/koala-sign-learn.git)
   cd koala-sign-learn/backend
````

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the API server:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`.

### 2\. Frontend Setup (React)

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

-----

## Evaluation

The model was trained and evaluated on a subset of the **KSL77 dataset** (67 classes).

### Model Performance

| Metric | Result |
| :--- | :--- |
| Test Accuracy | **88.21%** |
| Classes | 67 |
| Framework | PyTorch |
| Best Hyperparameters | `lr`: 0.00034, `cnn_hidden`: 64, `lstm_hidden`: 256, `dropout_rate`: 0.222, `optimizer`: Adam |

### Model Architecture (`PoseCNN_LSTM_Attn`)

The architecture is based on the principles in the paper: ["Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention‑Based Neural Network"](https://ieeexplore.ieee.org/document/10360810).

1.  **Feature Extraction:** MediaPipe Holistic extracts a fixed-length sequence of 47 keypoints (normalized 3D coordinates) per frame for a 32-frame clip.
2.  **Spatial Modeling (CNN):** `Conv1d` layers process the 47 joint features to extract spatial relationships.
3.  **Temporal Modeling (LSTM):** A Bidirectional LSTM processes the sequence of spatial features for long-range temporal dependencies.
4.  **Feature Aggregation (Attention):** An Attention Pooling mechanism computes a weighted context vector over all time steps to create a single, discriminative feature vector for classification.

-----

## Detailed Results

### Training Curves

The following plots illustrate the model's accuracy and loss convergence over the training and validation epochs.
![Training Curves for KSL Recognition Model](images/training_curves.png)

### Confusion Matrix

This matrix visualizes the performance of the final model across all 67 KSL classes on the test set. The strong diagonal indicates good performance, with off-diagonal elements highlighting specific misclassifications.
![Confusion Matrix for KSL Recognition Model](images/confusion_matrix.png)

### Full Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 1.00 | 1.00 | 1.00 | 4 |
| 1 | 1.00 | 0.75 | 0.86 | 4 |
| 2 | 1.00 | 1.00 | 1.00 | 4 |
| 3 | 1.00 | 0.50 | 0.67 | 4 |
| 4 | 1.00 | 1.00 | 1.00 | 4 |
| 5 | 0.67 | 1.00 | 0.80 | 4 |
| 6 | 1.00 | 1.00 | 1.00 | 4 |
| 7 | 1.00 | 0.75 | 0.86 | 4 |
| 8 | 1.00 | 0.75 | 0.86 | 4 |
| 9 | 1.00 | 0.67 | 0.80 | 3 |
| 10 | 1.00 | 1.00 | 1.00 | 3 |
| 11 | 0.33 | 0.25 | 0.29 | 4 |
| 12 | 1.00 | 1.00 | 1.00 | 3 |
| 13 | 0.67 | 0.67 | 0.67 | 3 |
| 14 | 1.00 | 1.00 | 1.00 | 3 |
| 15 | 0.67 | 1.00 | 0.80 | 4 |
| 16 | 1.00 | 1.00 | 1.00 | 3 |
| 17 | 1.00 | 1.00 | 1.00 | 4 |
| 18 | 1.00 | 1.00 | 1.00 | 4 |
| 19 | 1.00 | 1.00 | 1.00 | 3 |
| 20 | 1.00 | 1.00 | 1.00 | 4 |
| 21 | 0.60 | 1.00 | 0.75 | 3 |
| 22 | 1.00 | 1.00 | 1.00 | 3 |
| 23 | 0.80 | 1.00 | 0.89 | 4 |
| 24 | 1.00 | 1.00 | 1.00 | 3 |
| 25 | 1.00 | 0.75 | 0.86 | 4 |
| 26 | 0.67 | 1.00 | 0.80 | 4 |
| 27 | 0.80 | 1.00 | 0.89 | 4 |
| 28 | 1.00 | 0.67 | 0.80 | 3 |
| 29 | 1.00 | 0.75 | 0.86 | 4 |
| 30 | 0.75 | 0.75 | 0.75 | 4 |
| 31 | 1.00 | 0.25 | 0.40 | 4 |
| 32 | 1.00 | 0.75 | 0.86 | 4 |
| 33 | 1.00 | 1.00 | 1.00 | 3 |
| 34 | 1.00 | 0.67 | 0.80 | 3 |
| 35 | 1.00 | 0.75 | 0.86 | 4 |
| 36 | 0.67 | 0.67 | 0.67 | 3 |
| 37 | 1.00 | 1.00 | 1.00 | 4 |
| 38 | 1.00 | 0.50 | 0.67 | 4 |
| 39 | 1.00 | 1.00 | 1.00 | 4 |
| 40 | 1.00 | 1.00 | 1.00 | 4 |
| 41 | 1.00 | 1.00 | 1.00 | 4 |
| 42 | 1.00 | 1.00 | 1.00 | 4 |
| 43 | 1.00 | 0.75 | 0.86 | 4 |
| 44 | 0.75 | 1.00 | 0.86 | 3 |
| 45 | 0.75 | 1.00 | 0.86 | 3 |
| 46 | 1.00 | 1.00 | 1.00 | 4 |
| 47 | 1.00 | 1.00 | 1.00 | 3 |
| 48 | 1.00 | 1.00 | 1.00 | 3 |
| 49 | 1.00 | 1.00 | 1.00 | 4 |
| 50 | 0.80 | 1.00 | 0.89 | 4 |
| 51 | 1.00 | 1.00 | 1.00 | 3 |
| 52 | 1.00 | 1.00 | 1.00 | 3 |
| 53 | 1.00 | 1.00 | 1.00 | 4 |
| 54 | 0.60 | 0.75 | 0.67 | 4 |
| 55 | 1.00 | 1.00 | 1.00 | 4 |
| 56 | 0.57 | 1.00 | 0.73 | 4 |
| 57 | 1.00 | 0.67 | 0.80 | 3 |
| 58 | 1.00 | 1.00 | 1.00 | 4 |
| 59 | 1.00 | 1.00 | 1.00 | 4 |
| 60 | 0.80 | 1.00 | 0.89 | 4 |
| 61 | 0.80 | 1.00 | 0.89 | 4 |
| 62 | 1.00 | 1.00 | 1.00 | 3 |
| 63 | 0.80 | 1.00 | 0.89 | 4 |
| 64 | 0.80 | 1.00 | 0.89 | 4 |
| 65 | 0.50 | 0.50 | 0.50 | 4 |
| 66 | 1.00 | 0.75 | 0.86 | 4 |
| **Accuracy** | | | **0.88** | 246 |
| **Macro Avg** | **0.91** | **0.88** | **0.88** | 246 |
| **Weighted Avg** | **0.91** | **0.88** | **0.88** | 246 |

The table highlights that classes like **11** (`F1-Score: 0.29`), **31** (`Recall: 0.25`), **38** (`Recall: 0.50`), and **65** (`F1-Score: 0.50`) are the most challenging for the current model.

-----

## Video Links

### Demo Video

  * [Direct link to your project demonstration video]

### Technical Walkthrough

  * [Direct link to your technical explanation video]

-----

## Technical Stack & Development Details

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model** | PyTorch, NumPy, Scikit-learn | Trained on KSL77 dataset, achieving 88.21% test accuracy. |
| **Feature Extraction** | MediaPipe Holistic, OpenCV | Extracts 47 3D joint coordinates across a fixed sequence length (32 frames). |
| **Backend API** | Python, FastAPI | Serves the trained PyTorch model and handles video uploads and preprocessing. |
| **Frontend** | Vite, React, TypeScript, Tailwind CSS | Provides a user-friendly interface for recording/uploading videos and displaying AI feedback. |

### Development Details

A detailed Jupyter Notebook is included in the [`notebook/`](https://www.google.com/search?q=notebook/) folder, providing a full walkthrough of video preprocessing, feature extraction, and model training (run on Google Colab using an NVIDIA T4 GPU).

> **Dataset Source:** Original KSL77 dataset and labels obtained from [Yangseung/KSL](https://github.com/Yangseung/KSL).
> **Reference Paper:** ["Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention‑Based Neural Network"](https://ieeexplore.ieee.org/document/10360810) by Jungpil Shin et al.

```
