# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import io
from PIL import Image
import tensorflow as tf
import uvicorn
import os

MODEL_PATH = "mnist-model.h5"

app = FastAPI(title="MNIST FastAPI")

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found in cwd: {MODEL_PATH}. Download it first.")

model = tf.keras.models.load_model(MODEL_PATH)  # tf.keras load
model._make_predict_function() if hasattr(model, "_make_predict_function") else None

class Array28(BaseModel):
    # expects a 28x28 nested list (or flattened 784 list) of pixel values 0-255 or 0-1
    data: list

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    # convert to grayscale 28x28 normalized float32
    img = pil_img.convert("L").resize((28, 28))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)  # model expects (1,28,28,1)
    return arr

def preprocess_array(arr_input):
    arr = np.array(arr_input, dtype="float32")
    if arr.size == 784:
        arr = arr.reshape(28,28)
    if arr.shape != (28,28):
        raise ValueError("Array must be 28x28 or length 784")
    arr = arr.astype("float32") / 255.0
    arr = arr.reshape(1,28,28,1)
    return arr

@app.post("/predict-from-file")
async def predict_from_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    x = preprocess_image(pil)
    preds = model.predict(x)
    digit = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {"digit": digit, "confidence": confidence}

@app.post("/predict-from-array")
async def predict_from_array(payload: Array28):
    try:
        x = preprocess_array(payload.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    preds = model.predict(x)
    digit = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {"digit": digit, "confidence": confidence}

@app.get("/")
def root():
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
