from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model_loading import load_models
from utils import transform_image
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, use ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once on startup to save memory
custom_model, mobilenet = load_models()
classes = ["Overripe", "Ripe", "Spotty", "Underripe"]

@app.get("/")
def health():
    return {"status": "awake"}

# API endpoint to handle image uploads and return predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Read image
    img_bytes = await file.read()
    tensor = transform_image(img_bytes)

    # Run Inference
    with torch.no_grad():
        out_custom = custom_model(tensor)
        out_mobile = mobilenet(tensor)

    # Get probabilities
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