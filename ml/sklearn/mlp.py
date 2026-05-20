from time import time
from sklearn.neural_network import MLPClassifier
from ml import save_onnx
from ml.preprocess import DATASET_TYPE


def train_model(dataset: DATASET_TYPE) -> str:
    print("  -> Training sklearn.neural_network.MLPClassifier")

    X, X_test, y, y_test = dataset

    # TODO: Tuning
    clf = MLPClassifier()
    start = time()
    clf.fit(X, y)

    # Validation
    print("  --> Training completed in", time() - start, "seconds. Evaluating")
    pred = clf.predict(X_test)
    score = 0
    for j in range(pred.size):
        if pred[j] == y_test[j]:
            score += 1
    print("  --> Correctness score for MLP model is", score / len(y_test))

    # EXPORT
    print("  --> Exporting model")
    FILE = "models/mlp.onnx"
    save_onnx.save(clf, X, FILE)
    print("  --> Training complete\n")

    return FILE
