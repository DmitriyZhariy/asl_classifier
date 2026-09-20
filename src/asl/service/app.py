import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO

import albumentations as A
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from asl import db
from asl.config import settings


@dataclass
class InferenceBundle:
    model: torch.nn.Module
    transform: A.Compose
    device: torch.device
    class_names: list[str]
    model_version: str


class Prediction(BaseModel):
    request_id: uuid.UUID
    model_version: str
    prediction_class: int
    confidence: float
    all_probabilities: dict
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.shufflenet_v2_x1_5(
        weights=None,
        num_classes=settings.num_classes,
    )

    checkpoint = torch.load(
        settings.weights_path, 
        map_location=device, 
        weights_only=True,
        )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    transform = A.Compose(
        [
            A.Resize(settings.input_height, settings.input_width),
            A.Normalize(
                mean=settings.normalize_mean, 
                std=settings.normalize_std
            ),
            ToTensorV2(),
        ]
    )

    app.state.bundle = InferenceBundle(
        model=model,
        transform=transform,
        device=device,
        class_names=settings.class_names,
        model_version=settings.model_version
    )

    db.init()
    yield

    del app.state.bundle


app = FastAPI(title="asl", version="1.0.0", lifespan=lifespan)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'inference': bool(app.state.bundle),
    }


@app.get('/ready')
def ready():
    try:
        _ = app.state.bundle.model
    except AttributeError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model_version = app.state.bundle.model_version
    return {
        'status': 'ok',
        'model_version': model_version,
    }


@app.post('/v1/predict')
async def predict(
    bg: BackgroundTasks,
    file: UploadFile,
    ) -> Prediction:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    input_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
    }

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = app.state.bundle.transform(image=np.array(image))["image"]

    image_tensor = image_tensor.unsqueeze(0).to(app.state.bundle.device)

    with torch.inference_mode():
        logits = app.state.bundle.model(image_tensor)
        all_probabilities = torch.softmax(logits, dim=1)
        prediction_class = all_probabilities.argmax(dim=1).item()
        confidence = all_probabilities[0, prediction_class].item()

    all_probabilities = all_probabilities[0].tolist()
    all_probabilities = dict(enumerate(all_probabilities))

    latency_ms = (time.perf_counter() - t0) * 1000

    bg.add_task(
        db.save_prediction, 
        request_id, 
        app.state.bundle.model_version,
        prediction_class, 
        all_probabilities,
        input_metadata,
        latency_ms,
        )
    
    return Prediction(
        request_id=request_id,
        model_version=app.state.bundle.model_version,
        prediction_class=prediction_class,
        confidence=confidence,
        all_probabilities=all_probabilities,
        latency_ms=latency_ms,
    )