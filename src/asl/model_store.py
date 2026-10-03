import json
from pathlib import Path

import onnxruntime as ort

from asl.config import settings


def load_model() -> tuple[object, dict, str]:
    if not settings.model_name:
        ...

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    client = MlflowClient()

    mv = client.get_model_version_by_alias(
        name=settings.model_name,
        alias=settings.model_alias,
    )

    version = str(mv.version)

    model_uri = f"models:/{settings.model_name}/{version}"

    download_dir = Path("artifacts/downloaded") / settings.model_name / version
    download_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(download_dir),
        )
    )

    onnx_path = model_dir / "model.onnx"
    metadata_path = model_dir / "extra_files" / "metadata.json"

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX-модель не найдена: {onnx_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata не найдены: {metadata_path}")

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    print(session, metadata, version)
    return session, metadata, version

    # return ModelBundle(
    #     session=session,
    #     metadata=metadata,
    #     version=version,
    #     model_dir=model_dir,
    # )