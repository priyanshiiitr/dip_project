# Facial Emotion Recognition (starter scaffold)

Project scaffold for a small facial emotion recognition demo.

Structure

- `data/` — optional sample images for testing
- `models/` — saved models or logs
- `src/` — main source files
  - `__init__.py`
  - `app.py` — CLI app for image & webcam (OpenCV + FER)
  - `streamlit_app.py` — Streamlit UI (upload image + camera input)
  - `preprocessing.py` — DIP utilities: blur/noise/contrast detection & fixes
  - `detector.py` — face detection & cropping helpers
  - `classifier.py` — wrapper around pretrained FER classifier
- `requirements.txt` — recommended packages
- `utils/helpers.py` — small helpers (drawing, logging)

Quickstart

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run CLI on an image:

```powershell
python -m src.app --image data/example.jpg
```

4. Run the Streamlit UI:

```powershell
streamlit run src/streamlit_app.py
```

Notes

- The scaffold uses the `fer` package if available. If you prefer a different model (PyTorch/TensorFlow), replace `src/classifier.py` with your model wrapper.
- This is a minimal starter; adjust thresholds, pre/post-processing, and model selection for production use.
- MIT License
