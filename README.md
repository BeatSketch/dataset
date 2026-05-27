# dataset
Dataset pre-processing scripts for [BeatSketch](https://github.com/BeatSketch/BeatSketch)

It also contains all the scripts necessary to train the models.

If you can't code and still want to help out, [uploading BSOR files](https://polybox.ethz.ch/index.php/s/RbRFRgc7WnmotAg) is an easy way.


# Contributing
## Checkout
Please note that all model files are stored using `git-lfs`, see the corresponding man page (`man git-lfs`) for it.
In short, install `git-lfs`, then run `git lfs install`, then any `git pull`, `git push` and `git commit` command will automatically also handle the LFS files.
Alternatively, run `git lfs pull` to pull down the models.

## Usage
There are three primary commands to know:
- Create dataset file: 
    ```bash
    ./beatsketch_dataset.py preprocess folder /path/to/folder path/to/dataset.json
    ```
- Train model (and if `-o` specified, save dataset) from folder after preprocessing: 
    ```bash
    ./beatsketch_dataset.py train folder /path/to/folder [-o path/to/dataset.json]
    ```
- Train models using pre-saved dataset file: 
    ```bash
    ./beatsketch_dataset.py train dataset path/to/dataset.json
    ```

For other commands, see the help pages, accessed using `./beatsketch_dataset.py help`
