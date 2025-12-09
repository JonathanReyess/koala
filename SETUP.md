# Setup and Installation Guide 

This document provides instructions for accessing the deployed version of the project, hosting the application locally, and replicating the original study.

---

## 1. Accessing the Deployed Application

For immediate testing, the project is deployed across two services:

* **Frontend (Vite.js)** - Hosted on Vercel: [https://koala-sign-learn.vercel.app/](https://koala-sign-learn.vercel.app/)
* **Backend (FastAPI)** - Hosted on an AWS Lightsail 2 GB RAM, 2 vCPUs, 60 GB SSD instance

The frontend is configured to communicate with the live backend API.

---

## 2. Local Setup and Deployment

### Prerequisites

* **Python 3.8+** (for the FastAPI backend)
* **Node.js / npm** (for the Vite.js frontend)

### A. Clone the Repository

First, clone the project and navigate into the root directory:

```bash
git clone https://github.com/JonathanReyess/koala-sign-learn.git
cd koala-sign-learn/
````

### B. Backend Setup (FastAPI)

The backend requires the trained model file to function.

1.  **Navigate to the backend directory:**

    ```bash
    cd backend
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Required Model File:**

    > **NOTE:** The model file (`best_model.pt`) is required for the application to run locally. Ensure this file is placed inside the `backend/` directory.

4.  **Run the API service:**

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

    The API service will now be available at **http://localhost:8000**.

### C. Frontend Setup (React/Vite.js)

1.  **Navigate to the frontend directory:**

    ```bash
    cd ../frontend
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

3.  **Configure API URL:**
    To ensure the frontend communicates with your local backend, check or create a `.env` file in the `frontend/` directory with the following setting:

    ```
    VITE_API_URL=http://localhost:8000
    ```

4.  **Run the application:**

    ```bash
    npm run dev
    ```

    The frontend application will start and provide a local access URL (typically `http://localhost:5173`).

-----

## 3. Study Replication and Analysis

To replicate the original study, which was performed in a Google Colab Jupyter environment using a T4 GPU, follow these steps:

1.  **Follow the Notebook:** Open and execute the instructions in the main notebook:
      * `/notebook/KSL.ipynb`
2.  **Access Data:** Review the data requirements and setup instructions in the data documentation:
      * `/data/DATA_README.md`
3.  **Skip Preprocessing (Optional):** If you wish to bypass the initial data preprocessing steps, the final pre-processed data can be found and loaded directly from:
      * `/data/KSL77_joint_stream_47pt.pkl`


<!-- end list -->
