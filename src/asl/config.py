from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    weights_path: str = "artifacts/shufflenet_v1.pt"
    database_url: str | None = None
    log_level: str = "INFO"
    num_classes: int = 3
    class_names: list[str] = ["A", "B", "C"]

    input_height: int = 224
    input_width: int = 224

    normalize_mean: tuple[float, float, float] = (
        0.485, 
        0.456, 
        0.406,
    )
    normalize_std: tuple[float, float, float] = (
        0.229, 
        0.224, 
        0.225,
    )
    model_version: str = "1.0.0"

    allowed_content_types: list = (
        "image/jpeg",
        "image/png",
        "image/webp"
    )


settings = Settings()