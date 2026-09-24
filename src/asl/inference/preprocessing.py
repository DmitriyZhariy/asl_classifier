from io import BytesIO

import numpy as np
from fastapi import UploadFile
from PIL import Image

from asl.config import settings


async def preprocess_image(file: UploadFile) -> np.ndarray:
    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes))
    image.load()
    image = image.convert("RGB")

    image = image.resize((settings.input_height, settings.input_width))

    array = np.asarray(image, dtype=np.float32)

    array = array / 255.0
    array = np.transpose(array, (2, 0, 1))

    mean = np.array(settings.mean, dtype=np.float32)
    std = np.array(settings.std, dtype=np.float32)

    array = (array - mean[:, None, None]) / std[:, None, None]

    return np.expand_dims(array, axis=0).astype(np.float32)