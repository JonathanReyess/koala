## KSL Sign Language Recognition System

This is the repository for the Koala Sign Language (KSL) recognition project, which utilizes a deep learning model to classify dynamic Korean Sign Language words from video input. The system is deployed as a full-stack application with a FastAPI backend and a React frontend, allowing users to practice KSL signs and receive real-time AI feedback.

### Project Overview

The core of this project is a dynamic sign language recognition model trained on skeleton-based features extracted from videos. The system addresses the need for accessible and portable KSL learning tools by utilizing a vision-based approach that can run on any camera-equipped device.

The model architecture is based on the principles outlined in the paper **"Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention-Based Neural Network"** by Jungpil Shin et al. (IEEE Access, Volume: 11, Date of Publication: 15 December 2023, DOI: 10.1109/ACCESS.2023.3343404).

-----

### Key Features

  * **Dynamic Sign Recognition:** Unlike many static recognition systems, this project is designed to accurately classify dynamic, time-series sign words.
  * **Skeleton-Based Input:** The system uses **MediaPipe Holistic** to extract **47 joint skeleton points** (from hands, body, and face) from video frames, which significantly reduces computational complexity, enhances privacy, and offers robustness against varying backgrounds and lighting conditions.
  * **Deep Learning Architecture:** The model, `PoseCNN_LSTM_Attn`, is a custom implementation inspired by advanced attention-based neural networks for video classification. It combines a **Convolutional Neural Network (CNN)** for feature extraction across joints, a **Long Short-Term Memory (LSTM)** network for temporal modeling, and an **Attention mechanism** for focusing on the most relevant frames.

-----

### Technical Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model** | PyTorch, NumPy, Scikit-learn | Trained using the KSL77 dataset, achieving **86.18% test accuracy** in this implementation. |
| **Feature Extraction** | MediaPipe Holistic, OpenCV (via `cv2`) | Extracts and preprocesses 47 3D joint coordinates across a fixed sequence length (32 frames). |
| **Backend API** | Python, **FastAPI** | Serves the trained PyTorch model for prediction and handles video file uploads and processing. |
| **Frontend** | React, TypeScript, Tailwind CSS | A user-friendly interface (`LearningCard` component) for recording and uploading video signs and displaying real-time success/failure feedback. |

-----

### Model Details

The implemented `PoseCNN_LSTM_Attn` model structure is as follows:

1.  **Input:** `(Batch, 3, 32, 47)` (Channel, Time, Joints)
2.  **CNN over Joints:** `Conv1d` layers reduce the joint dimension (47 -\> 23).
3.  **Temporal CNN:** A `Conv2d` layer enhances temporal features.
4.  **LSTM:** A **Bidirectional LSTM** processes the sequential data (`(B, T, C*J)`) for temporal modeling.
5.  **Attention Pooling:** An attention layer computes weights across the time steps to create a single, context-rich feature vector.
6.  **Classifier:** A Fully Connected (`fc`) block provides the final classification logits for **67 classes**.

-----

### Setup and Deployment

This project requires Python for the backend and Node.js/npm for the frontend development environment.

#### Backend (FastAPI)

1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    cd [repository-folder]/backend
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The actual model file (`best_model.pt`) and data file (`KSL77_joint_stream_47pt.pkl`) are required for the application to run.*
3.  **Run the API:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`.

#### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd [repository-folder]/frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Configure API URL:** Ensure your `.env` file or environment variables are set correctly for `VITE_API_URL` (e.g., `http://localhost:8000`).
4.  **Run the application:**
    ```bash
    npm run dev
    ```
    The learning interface will be available at `http://localhost:5173`.