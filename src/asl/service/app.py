import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import onnxruntime as ort
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from asl import db
from asl.config import settings
from asl.inference.preprocessing import preprocess_image
from asl.inference.postprocessing import predict_image


@dataclass
class InferenceBundle:
    model: ort.InferenceSession
    model_version: str


class Prediction(BaseModel):
    request_id: uuid.UUID
    model_version: str
    prediction_class: str
    confidence: float
    all_probabilities: dict
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = ort.InferenceSession(settings.weights_path, providers=["CPUExecutionProvider"])

    app.state.bundle = InferenceBundle(
        model=model,
        model_version=settings.model_version,
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

    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail="Поддерживаются только jpeg, png и webp файлы",
        )

    try:
        image_tensor = await preprocess_image(file)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail="Файл изображения некорректен"
        )

    request_id = str(uuid.uuid4())

    input_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
    }

    all_probabilities, prediction_class, confidence = predict_image(app.state.bundle.model, image_tensor)

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