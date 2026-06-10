# dataset
Dataset pre-processing & model training scripts for [BeatSketch](https://github.com/BeatSketch/BeatSketch), the in-VR Beat Saber level editor.

## BSOR Files
BeatSketch models are trained using BSOR replay files. This repository handles all needed preprocessing.

If you can't code but want to help out, [uploading BSOR files](https://polybox.ethz.ch/index.php/s/RbRFRgc7WnmotAg) is an easy way.


# Contributing

## Checkout
Please note that all `.onnx` model files are stored using `git-lfs`. (see `man git-lfs`)

Your IDE should handle this, but `git` by itself does not handle this by default.

In short, install `git-lfs` and run:
```bash
git lfs install
```
Now, any `git` command (`pull`, `push`, etc.) will also handle LFS files.

Alternatively, run `git lfs pull` to pull the LFS files explicitly.

## Dependencies

The `requirements.txt` covers all dependencies, except PyTorch. We recommend using a `venv` and running:

```
pip install -r requirements.txt
```

For PyTorch, choose & install the version appropriate for your hardware [here](https://pytorch.org/get-started/locally/). 

## Usage
The script `beatsketch_dataset.py` has a CLI for common tasks.

The CLI features three primary commands:
- Create dataset file: 
    ```bash
    ./beatsketch_dataset.py preprocess folder /path/to/folder path/to/dataset.json
    ```
- Train model from folder after preprocessing: (`-m` selects model, if `-o` specified, save dataset)
    ```bash
    ./beatsketch_dataset.py train folder /path/to/folder -m mlp [-o path/to/dataset.json]
    ```
- Train models using pre-saved dataset file: 
    ```bash
    ./beatsketch_dataset.py train dataset path/to/dataset.json -m mlp
    ```

For other commands, see `./beatsketch_dataset.py help`
