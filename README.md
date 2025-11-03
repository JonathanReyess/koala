
## 코아라 - Korean Sign Language (KSL) Recognition System

This is the repository for the **Koala (코아라)**, which utilizes a deep learning model to classify dynamic Korean Sign Language words from video input. 

The system is deployed as a full-stack application with a **FastAPI** backend and a **React** frontend, allowing users to practice KSL signs and receive real-time AI feedback.

---

### Project Overview

This project addresses the need for accessible KSL learning tools and resources by utilizing a vision-based approach that can run on any camera-equipped device.

The model architecture is based on the principles outlined in the paper:
 ["Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention‑Based Neural Network"](https://ieeexplore.ieee.org/document/10360810) by Jungpil Shin et al. (IEEE Access, Volume: 11, Date of Publication: 15 December 2023, DOI: 10.1109/ACCESS.2023.3343404).

---

### Jupyter Notebook

A detailed **Jupyter Notebook** is included in the [`notebook/`](notebook/) folder. 

The notebook provides a full walkthrough of video preprocessing, feature extraction, and model training, and was run on Google Colab using an NVIDIA T4 GPU.

> **Dataset Source:** Original KSL77 dataset and labels obtained from [Yangseung/KSL](https://github.com/Yangseung/KSL).

---

### Technical Stack

| Component              | Technology                         | Description                                                                                   |
| :--------------------- | :--------------------------------- | :-------------------------------------------------------------------------------------------- |
| **Model**              | PyTorch, NumPy, Scikit-learn       | Trained on KSL77 dataset, achieving **86.18%** test accuracy.                                 |
| **Feature Extraction** | MediaPipe Holistic, OpenCV | Extracts 47 3D joint coordinates across a fixed sequence length (32 frames).                  |
| **Backend API**        | Python, FastAPI                | Serves the trained PyTorch model and handles video uploads and preprocessing.                 |
| **Frontend**           | Vite, React, TypeScript, Tailwind CSS    | Provides a user-friendly interface for recording/uploading videos and displaying AI feedback. |

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

This project requires Python for the backend and Node.js/npm for the frontend.

#### Backend (FastAPI)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JonathanReyess/koala-sign-learn.git
   cd koala-sign-learn/backend
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > Note: The model file (`best_model.pt`) is required for the application to run locally.
3. **Run the API:**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`.

#### Frontend (React)

1. **Navigate to the frontend directory:**

   ```bash
   cd koala-sign-learn/frontend
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


