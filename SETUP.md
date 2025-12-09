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

If you wish to host the application and API locally, follow these steps.

### Prerequisites

* **Python 3.8+** (for the FastAPI backend)
* **Node.js / npm** (for the Vite.js frontend)

### A. Clone the Repository

First, clone the project and navigate into the root directory:

```bash
git clone [https://github.com/JonathanReyess/koala-sign-learn.git](https://github.com/JonathanReyess/koala-sign-learn.git)
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

## 🔬 3. Study Replication and Analysis

If you wish to replicate the study, train the model, or perform in-depth analysis on the data, it is advised to use the provided Jupyter Notebook.

1.  **Install Notebook Dependencies:** Ensure you have Jupyter installed in your backend environment:
    ```bash
    pip install jupyter
    ```
2.  **Access Data:** The necessary raw and pre-processed data are located in the `data/` directory.
3.  **Follow the Notebook:** Navigate to the `notebook/` directory and open the provided Jupyter Notebook file to follow the step-by-step instructions for data loading, training, and evaluation.

<!-- end list -->

```
