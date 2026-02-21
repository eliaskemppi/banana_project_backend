# API

from fastapi import FastAPI, UploadFile, File
from model_loading import load_models
from utils import transform_image
import torch

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, use ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once on startup to save memory
custom_model, mobilenet = load_models()
classes = ["Overripe", "Ripe"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image
    img_bytes = await file.read()
    tensor = transform_image(img_bytes)

    # 2. Run Inference
    with torch.no_grad():
        out_custom = custom_model(tensor)
        out_mobile = mobilenet(tensor)

    # 3. Get results (Probabilities)
    prob_custom = torch.nn.functional.softmax(out_custom, dim=1)
    prob_mobile = torch.nn.functional.softmax(out_mobile, dim=1)

    return {
        "custom_cnn": {
            "label": classes[torch.argmax(prob_custom)],
            "confidence": float(torch.max(prob_custom))
        },
        "mobilenet": {
            "label": classes[torch.argmax(prob_mobile)],
            "confidence": float(torch.max(prob_mobile))
        }
    }