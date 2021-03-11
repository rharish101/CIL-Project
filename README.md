# CIL Project

This is a repository for the project submission of team SHAJ for the Computational Intelligence Lab (263-0008-00L) at ETH Zürich offered in the spring semester of 2021.

## Group Members
* **S**habnam Ghasemirad
* **H**arish Rajagopal
* **A**li Gorji
* **J**ohannes Dollinger

## Instructions

All Python scripts use argparse to parse commandline arguments.
To view the list of all positional and optional arguments for a script `script.py`, run:
```sh
./script.py --help
```

### Setup
[Poetry](https://python-poetry.org/) is used for conveniently installing and managing dependencies.
[pre-commit](https://pre-commit.com/) is used for managing hooks that run before each commit, to ensure code quality and run some basic tests.

1. *[Optional]* Create and activate a virtual environment with Python >= 3.6.2.

2. Install Poetry globally (recommended), or in a virtual environment.
    Please refer to [Poetry's installation guide](https://python-poetry.org/docs/#installation) for recommended installation options.

    You can use pip to install it:
    ```sh
    pip install poetry
    ```

3. Install all dependencies, including extra dependencies for development, with Poetry:
    ```sh
    poetry install
    ```

    If you didn't create and activate a virtual environment in step 1, Poetry creates one for you and installs all dependencies there.
    To use this virtual environment, run:
    ```sh
    poetry shell
    ```

4. Install pre-commit hooks:
    ```sh
    pre-commit install
    ```

**NOTE**: You need to be inside the virtual environment where you installed the above dependencies every time you commit.
However, this is not required if you have installed pre-commit globally.

### Hyper-Parameter Configuration
Hyper-parameters can be specified through [TOML](https://toml.io/en/) configs.
For example, to specify a training batch size of 32 and a learning rate of 0.001, use the following config:
```toml
learn_rate = 0.001
train_batch_size = 32
```

You can store configs in a directory named `configs` located in the root of this repository.
It has an entry in the [`.gitignore`](./.gitignore) file so that custom configs aren't picked up by git.

The available hyper-parameters, their documentation and default values are specified in the `Config` class in the file [`src/config.py`](./src/config.py).

### Training
Run the script `train.py`:
```sh
./train.py /path/to/CIL/dataset/
```

The weights of trained models are saved with the `.pt` extension to the directory given by the `--save-dir` argument.
By default, this directory is `saved_models`.

Training logs are by default stored inside an ISO 8601 timestamped subdirectory, which is stored in a parent directory (as given by the `--log-dir` argument).
By default, the parent directory is `logs`.

The hyper-parameter config (including defaults) is saved as a TOML file named `config.toml` in both the saved models directory and the timestamped log directory.

#### Mixed Precision Training
This implementation supports mixed-precision training.
This can be enabled by setting the `mixed_precision` hyper-parameter in a config.
Note that this will only provide significant speed-ups if your GPU(s) have special support for mixed-precision compute.

### Inference
The script `inference.py` generates predictions on the test data using a trained model.
Run it as follows:
```sh
./inference.py /path/to/image/dir/
```

The output images are saved in the directory given by the `--output-dir` argument.
By default, this directory is `outputs`.
The images will be saved as PNG images with the file names corresponding to the input images.
