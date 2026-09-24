import numpy as np
import onnxruntime as ort

from asl.config import settings


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

def predict_image(
        model: ort.InferenceSession,
        image: np.ndarray
        ) -> tuple[list, str, float]:
    
    model_input = model.get_inputs()[0]
    model_output = model.get_outputs()[0]

    logits = model.run(
        [model_output.name],
        {model_input.name: image},
    )[0]

    all_probabilities = softmax(logits)
    prediction_class = int(np.argmax(all_probabilities[0]))
    confidence = float(all_probabilities[0, prediction_class])

    all_probabilities = all_probabilities[0].tolist()
    all_probabilities = dict(enumerate(all_probabilities))

    prediction_class = settings.class_names[prediction_class]

    return all_probabilities, prediction_class, confidence