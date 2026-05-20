from typing import Tuple
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator
import numpy as np


def save(clf: BaseEstimator, X: np.ndarray, file: str):
    onx = to_onnx(clf, X[:1], target_opset=12)
    with open(file, "wb") as f:
        if not isinstance(onx, Tuple) and not isinstance(onx, BaseEstimator):
            f.write(onx.SerializeToString())
        else:
            print("  --> WARNING: Model export failed")
        f.close()
