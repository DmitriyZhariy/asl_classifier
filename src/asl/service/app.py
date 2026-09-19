import time
import uuid

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, BackgroundTasks, HTTPException

from pydantic import BaseModel, Field

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision

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
    prediction_class: int
    all_probabilities: list[float]
    asl: bool
    model_version: str
    request_id: int
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

    yield

    del app.state.bundle


app = FastAPI(title="asl", version="1.0.0", lifespan=lifespan)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'device': app.state.bundle.model_version,
    }