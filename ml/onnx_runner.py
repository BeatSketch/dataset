from typing import cast
import onnxruntime as rt
import numpy as np


# Specify the model path
def run_model(model_path: str, X: np.ndarray, y: np.ndarray):
    print("==> Testing model", model_path)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    with open(model_path, "rb") as f:
        model_data = f.read()
        session = rt.InferenceSession(
            model_data, providers=rt.get_available_providers()
        )

    pred = cast(np.ndarray, session.run(None, {"X": X})[0])
    score = 0
    for j in range(pred.size):
        if pred[j] == y[j]:
            score += 1
    print("  --> Correctness score for model", model_path, "is", score / len(y))
