# ✍️ ATTRIBUTION

This document details the external resources, libraries, datasets, and methodologies used in the **PoseCNN_LSTM_Attn** project.

---

## 1. Project Datasets

### A. Korean Sign Language (KSL) Dataset

The core training data for this project is derived from the Korean Sign Language (KSL) Dataset.

* **Source:** The dataset was proposed and released by researchers at KAIST and Samsung Electronics.
* **Original Paper (Required Citation):**
    > Yang, S., Jung, S., Kang, H., & Kim, C. (2020). The Korean Sign Language Dataset for Action Recognition. In *Proceedings of the 11th ACM Multimedia Systems Conference (MMSys '20)* (pp. 532–542). Springer, Cham.
* **Source Link:** [Source Research Paper (Springer)](https://link.springer.com/content/pdf/10.1007/978-3-030-37731-1_43.pdf)
* **License (Data Package):** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

---

## 2. Methodology and Model Architecture

### A. PoseCNN\_LSTM\_Attn

The model architecture, **PoseCNN\_LSTM\_Attn** (Pose Convolutional Neural Network with Long Short-Term Memory and Attention), is based on established research combining human pose estimation with temporal deep learning.

* **Foundation:** The methodology combines concepts from pose estimation (e.g., OpenPose, MediaPipe) with recurrent neural networks and attention mechanisms for sequential sign language recognition.
* **Specific Model Inspiration:** *[**Crucial:** If your architecture is a direct implementation of a specific paper (e.g., "A specific paper for attention in sign language recognition"), cite it here.]*
    * **[Citation/Link to Key Research Paper on Model Idea]**
* **License (Model Code):** MIT License

---

## 3. Software and Libraries

### A. Backend (Python/FastAPI)

| Dependency | Purpose | License |
| :--- | :--- | :--- |
| **Python** | Core execution environment. | Python Software Foundation License (PSF) |
| **FastAPI** | High-performance Python web framework for the API. | MIT License |
| **Uvicorn** | ASGI server used to run the FastAPI application. | BSD 3-Clause "New" or "Revised" License |
| **PyTorch** | Deep learning framework used for model training and inference (`best_model.pt`). | BSD-style license |
| **NumPy / Pandas** | Data manipulation and processing (used heavily in the notebook). | BSD License |
| **Other Dependencies** | *[List other key packages from your `requirements.txt` here, e.g., Scikit-learn, OpenCV, etc.]* | *[List License for each]* |

### B. Frontend (JavaScript/Vite.js)

| Dependency | Purpose | License |
| :--- | :--- | :--- |
| **Node.js/npm** | Core execution environment and package management. | MIT License |
| **Vite** | Modern frontend build tool. | MIT License |
| **React** | JavaScript library for building the user interface. | MIT License |
| **Axios** | HTTP client for communicating with the FastAPI backend. | MIT License |

---

## 4. Hosting and Infrastructure

| Service | Purpose | License/Terms |
| :--- | :--- | :--- |
| **AWS Lightsail** | Hosting for the FastAPI backend API service. | AWS Customer Agreement |
| **Vercel** | Hosting for the Vite.js frontend application. | Vercel Terms of Service |
| **Google Colab (T4 GPU)** | Execution environment used for training the original model and running the study notebook. | Google Terms of Service |

---

## 5. AI-Generated Content

This section discloses the use of large language models (LLMs) in the project's development:

* **Model Used:** Gemini 3
* **Scope of Assistance:** Gemini 3 was used for generating **preliminary versions** of code across both the frontend and backend, including:
    * **Frontend Components:** Drafting initial React components, layout structures, and API integration logic.
    * **Model Processing:** Generating code for initial data preprocessing steps and segments of the model training pipeline within the Jupyter notebook.
    * **Documentation:** Assisting with the structuring and drafting of project documentation (e.g., `README.md`, `SETUP.md`, and this `ATTRIBUTION.md`).
