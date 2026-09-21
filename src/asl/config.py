from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    weights_path: str = "artifacts/shufflenet_v1.onnx"
    database_url: str | None = None
    log_level: str = "INFO"
    num_classes: int = 3
    class_names: list[str] = ["A", "B", "C"]

    input_height: int = 224
    input_width: int = 224

    mean: list[float, float, float] = [
        0.485, 
        0.456, 
        0.406,
    ]
    std: list[float, float, float] = [
        0.229, 
        0.224, 
        0.225,
    ]
    model_version: str = "1.0.0"

    allowed_content_types: list = (
        "image/jpeg",
        "image/png",
        "image/webp"
    )


settings = Settings()