# Model training infrastructure
This folder contains everything needed to train the classifiers used in BeatSketch. We are currently using `sklearn` and `PyTorch`. The goal is that every model is exportable as `.onnx`.

## Models

### sklearn/mlp.py

Simple model using `sklearn.neural_network.MLPClassifier` without tuning. This is the default fallback in the CLI.

### torch/torch_mlp.py

Simple MLP model in torch using `torch.nn`. Uses weighted assymetric loss to counteract Block/Non-Block inbalance in dataset.

**Note**: This model outputs a tensor. Use `torch.argmax(output, dim=1)` to get the class.