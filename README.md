
## KSL Sign Language Recognition System

This is the repository for the **Koala Sign Language (KSL) recognition project**, which utilizes a deep learning model to classify dynamic Korean Sign Language words from video input. The system is deployed as a full-stack application with a **FastAPI** backend and a **React** frontend, allowing users to practice KSL signs and receive real-time AI feedback.

---

### Project Overview

The core of this project is a dynamic sign language recognition model trained on skeleton-based features extracted from videos. The system addresses the need for accessible and portable KSL learning tools by utilizing a vision-based approach that can run on any camera-equipped device.

The model architecture is based on the principles outlined in the paper ["Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention‑Based Neural Network"](https://ieeexplore.ieee.org/document/10360810) by Jungpil Shin *et al.* (IEEE Access, Vol. 11, pp. 143501–143513, 2023, DOI: [10.1109/ACCESS.2023.3343404](https://doi.org/10.1109/ACCESS.2023.3343404)).


---

### Key Features

* **Dynamic Sign Recognition:** Designed to accurately classify dynamic, time-series sign words.
* **Skeleton-Based Input:** Uses **MediaPipe Holistic** to extract **47 joint skeleton points** (hands, body, and face) from video frames, reducing computational complexity, enhancing privacy, and improving robustness to background and lighting variations.
* **Deep Learning Architecture:** The `PoseCNN_LSTM_Attn` model combines a **Convolutional Neural Network (CNN)** for feature extraction across joints, a **Long Short-Term Memory (LSTM)** network for temporal modeling, and an **Attention mechanism** to focus on the most relevant frames.

---

### Jupyter Notebook

A detailed **Jupyter Notebook** is included in the [`notebook/`](notebook/) folder. The notebook provides a full walkthrough of processing, feature extraction, and model training:

* **Environment Setup:** Installs required packages (`torch`, `numpy`, `scikit-learn`, `mediapipe`, `opencv`, etc.) and configures GPU support in Google Colab.
* **Data Preparation:** Loads KSL77 videos, extracts **47 joint skeleton points** per frame using **MediaPipe Holistic**, and saves processed sequences as `.pkl` files.
* **Dataset & DataLoader:** Converts features and labels into PyTorch `Dataset` and `DataLoader` objects with train/validation/test splits and optional data augmentation.
* **Model Definition:** Implements the `PoseCNN_LSTM_Attn` network with CNN, LSTM, and Attention layers.
* **Training & Evaluation:** Trains the model with early stopping, evaluates test accuracy, generates classification reports, and visualizes the confusion matrix.
* **Model Saving:** Saves the trained model for inference in the FastAPI backend.

> **Dataset Source:** Original KSL77 dataset and labels obtained from [Yangseung/KSL](https://github.com/Yangseung/KSL).

---

### Technical Stack

| Component              | Technology                         | Description                                                                                   |
| :--------------------- | :--------------------------------- | :-------------------------------------------------------------------------------------------- |
| **Model**              | PyTorch, NumPy, Scikit-learn       | Trained on KSL77 dataset, achieving **86.18% test accuracy**.                                 |
| **Feature Extraction** | MediaPipe Holistic, OpenCV (`cv2`) | Extracts 47 3D joint coordinates across a fixed sequence length (32 frames).                  |
| **Backend API**        | Python, **FastAPI**                | Serves the trained PyTorch model and handles video uploads and preprocessing.                 |
| **Frontend**           | React, TypeScript, Tailwind CSS    | Provides a user-friendly interface for recording/uploading videos and displaying AI feedback. |

---

### Model Details

The implemented `PoseCNN_LSTM_Attn` model structure:

1. **Input:** `(Batch, 3, 32, 47)` (Channel, Time, Joints)
2. **CNN over Joints:** `Conv1d` layers reduce the joint dimension (47 → 23).
3. **Temporal CNN:** `Conv2d` layers enhance temporal features.
4. **LSTM:** **Bidirectional LSTM** processes sequential data `(B, T, C*J)` for temporal modeling.
5. **Attention Pooling:** Computes weighted context over time steps for a single feature vector.
6. **Classifier:** Fully connected layers output logits for **67 classes**.

---

### Setup and Deployment

This project requires **Python** for the backend and **Node.js/npm** for the frontend.

#### Backend (FastAPI)

1. **Clone the repository:**

   ```bash
   git clone [repository-url]
   cd [repository-folder]/backend
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > Note: The model file (`best_model.pt`) and data file (`KSL77_joint_stream_47pt.pkl`) are required for the application to run.
3. **Run the API:**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`.

#### Frontend (React)

1. **Navigate to the frontend directory:**

   ```bash
   cd [repository-folder]/frontend
   ```
2. **Install dependencies:**

   ```bash
   npm install
   ```
3. **Configure API URL:** Ensure your `.env` file or environment variables are set correctly for `VITE_API_URL` (e.g., `http://localhost:8000`).
4. **Run the application:**

   ```bash
   npm run dev
   ```

   The learning interface will be available at `http://localhost:5173`.

