from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torchvision.models import shufflenet_v2_x1_5


PT_PATH = Path("weights/shufflenet_v1.pt")
ONNX_PATH = Path("weights/shufflenet_v1.onnx")

NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 1


def load_model() -> torch.nn.Module:
    model = shufflenet_v2_x1_5(
        weights=None,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        PT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model


def main() -> None:
    model = load_model()

    sample = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
    )

    batch_dim = torch.export.Dim(
        "batch_size",
        min=1,
        max=64,
    )

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample,),
            ONNX_PATH,
            input_names=["images"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            export_params=True,
            dynamic_shapes=(
                {
                    0: batch_dim,
                },
            ),
            verify=True,
        )

    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    import numpy as np

    with torch.inference_mode():
        pytorch_logits = model(sample).cpu().numpy()

    onnx_logits = session.run(
        ["logits"],
        {"images": sample.numpy()},
    )[0]

    np.testing.assert_allclose(
        pytorch_logits,
        onnx_logits,
        rtol=1e-3,
        atol=1e-5,
    )

    print("PyTorch and ONNX predictions match.")

    print(f"ONNX model saved: {ONNX_PATH}")
    print("Inputs:", [(x.name, x.shape, x.type) for x in session.get_inputs()])
    print("Outputs:", [(x.name, x.shape, x.type) for x in session.get_outputs()])


if __name__ == "__main__":
    main()
    