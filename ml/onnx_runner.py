from typing import cast
import onnxruntime as rt
import numpy as np


# Specify the model path
def run_model(model_path: str, X: np.ndarray, y: np.ndarray):
    print("==> Testing model", model_path, "with ONNX runtime")
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    session = rt.InferenceSession(model_path, providers=rt.get_available_providers())
    input_name = session.get_inputs()[0].name
    print(session.get_inputs()[0].shape)

    pred = cast(np.ndarray, session.run(None, {input_name: X})[0])
    score = 0
    for j in range(pred.size):
        if pred[j] == y[j]:
            score += 1
    print("  --> Correctness score for model", model_path, "is", score / len(y))
