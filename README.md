# CIL Project

This is a repository for the project submission of team SHAJ for the Computational Intelligence Lab (263-0008-00L) at ETH Zürich offered in the spring semester of 2021.

## Group Members
* **S**habnam Ghasemirad
* **H**arish Rajagopal
* **A**li Gorji
* **J**ohannes Dollinger

## Instructions

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
For example, to specify a batch size of 32 and a learning rate of 0.001, use the following config:
```toml
learn_rate = 0.001
batch_size = 32
```

You can store configs in a directory named `configs` located in the root of this repository.
It has an entry in the [`.gitignore`](./.gitignore) file so that custom configs aren't picked up by git.

The available hyper-parameters, their documentation and default values are specified in the `Config` class in the file [`src/config.py`](./src/config.py).
