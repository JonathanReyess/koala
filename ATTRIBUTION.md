# Attributions

This document details the external resources, libraries, datasets, and methodologies used in the project.

---

## 1. Datasets

### A. Korean Sign Language (KSL) Dataset

The core training data for this project is derived from the Korean Sign Language (KSL) Dataset.

* **Source:** The dataset was proposed and released by researchers at KAIST and Samsung Electronics.
* **Original Paper (Required Citation):**
    > Yang, S., Jung, S., Kang, H., & Kim, C. (2020). The Korean Sign Language Dataset for Action Recognition. In *Proceedings of the 11th ACM Multimedia Systems Conference (MMSys '20)* (pp. 532–542). Springer, Cham.
* **Source Link:** [Source Research Paper (Springer)](https://link.springer.com/content/pdf/10.1007/978-3-030-37731-1_43.pdf)
* **Data License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

---

## 2. Methodology and Model Architecture

### A. PoseCNN_LSTM_Attn
The model architecture, PoseCNN_LSTM_Attn (Pose Convolutional Neural Network with Long Short-Term Memory and Attention), is based on established research combining human pose estimation with temporal deep learning. The methodology combines concepts from pose estimation (e.g., OpenPose, MediaPipe) with recurrent neural networks and attention mechanisms for sequential sign language recognition.

* **Model Inspiration:** This model is inspired by deep learning approaches to Korean Sign Language (KSL) recognition, specifically those utilizing LSTM and modern attention mechanisms.
  * **Transformer-Based KSL Recognition:** Lee, J., et al. (2023). Korean sign language recognition using transformer-based deep neural network. *Applied Sciences*, *13*(5), 3029. https://www.mdpi.com/2076-3417/13/5/3029
  * **LSTM-Based KSL Recognition:** Han, S., et al. (2020). Korean sign language recognition using LSTM and video datasets. *Journal of KIPS*, *27*(1), 356. https://xml.jips-k.org/full-text/view?doi=10.3745/JIPS.04.0356
  * **Pose Estimation Based and Attention-Based Neural Network:** Shin, J., Miah, A. S. M., Suzuki, K., Hirooka, K., & Hasan, M. A. M. (2023). Dynamic Korean sign language recognition using pose estimation based and attention-based neural network. *IEEE Access*, *11*, 121299-121311. https://ieeexplore.ieee.org/document/10360810

---
## 3. Software and Libraries

### A. Backend (Python/FastAPI)
| Dependency | Version/Notes | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.8+ | Core execution environment. |
| **FastAPI** | Latest | High-performance Python web framework for the API. |
| **Uvicorn** | [standard] | ASGI server used to run the FastAPI application. |
| **PyTorch** | 2.3.1+cpu | Deep learning framework used for model training and inference (`best_model.pt`). |
| **NumPy** | Latest | Numerical computing and array operations. |
| **Pandas** | Latest | Data manipulation and processing (used heavily in the notebook). |
| **OpenCV** | Latest | Computer vision library for image/video processing. |
| **MediaPipe** | Latest | Framework for pose estimation and landmark detection. |

### B. Frontend (JavaScript/Vite.js)
| Dependency | Purpose |
| :--- | :--- |
| **Node.js/npm** | Core execution environment and package management. |
| **Vite** | Modern frontend build tool. |
| **React** | JavaScript library for building the user interface. |
| **Axios** | HTTP client for communicating with the FastAPI backend. |
---

## 4. Hosting and Infrastructure

| Service | Purpose |
| :--- | :--- |
| **AWS Lightsail** | Hosting for the **FastAPI** backend API service. |
| **Vercel** | Hosting for the **Vite.js** frontend application. |
| **Google Colab (T4 GPU)** | Execution environment used for training the original model and running the study notebook. |

---

## 5. AI-Generated Content

This section discloses the use of large language models (LLMs) in the project's development:

* **Model Used:** Gemini 3
* **Scope of Assistance:** Gemini 3 was used for generating **preliminary versions** of code across both the frontend and backend, including:
    * **Frontend Components:** Drafting initial React components, layout structures, and API integration logic.
    * **Model Processing:** Generating code for initial data preprocessing steps and segments of the model training pipeline within the Jupyter notebook.
    * **Documentation:** Assisting with the structuring and drafting of project documentation (e.g., `README.md`, `SETUP.md`, and this `ATTRIBUTION.md`).
