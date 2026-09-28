# ⚠️ Deprecated

This FastAPI backend (`app.py`, `model.py`, `best_model.pt`) is **deprecated** and no longer used.
It was replaced by in-browser inference in PR 3: the frontend now runs MediaPipe landmark
extraction and the exported ONNX model (`src/frontend/public/models/ksl_f/`) client-side, so no
server or API URL is needed. The Lightsail instance that hosted this service has been shut down.

The code is kept in the repository for history only. See the top-level `README.md` for how to run
the app.
