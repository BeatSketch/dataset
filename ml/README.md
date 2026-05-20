# Model training infrastructure
This folder contains everything needed to train the classifiers used in BeatSketch.
We will likely train models using both sklearn and pytorch, see the separate subfolders.

Currently though, only sklearn.

We aim to export every possible model as ONNX to simplify runtime and keep the application size down
for the end-user software
