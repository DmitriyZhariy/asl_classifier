import numpy as np

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