from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from asl.service.app import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def valid_image_file() -> dict:

    image = Image.new(
        mode="RGB",
        size=(516, 516),
        color=(255, 0, 0),
    )

    input_image = BytesIO()
    image.save(input_image, format="PNG")

    return {
        "file": (
            "test_name.png",
            input_image,
            "image/png",
        ),
    }

@pytest.fixture()
def invalid_image_file() -> dict:
    return {
        "file": (
            "test_name.png",
            b'not a valid image',
            "image/png",
        ),
    }

@pytest.fixture()
def unsupported_file() -> dict:
    return {
        "file": (
            "document.txt",
            b"Hello, this is not an image",
            "text/plain",
        ),
    }
