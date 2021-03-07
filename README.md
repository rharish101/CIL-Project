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
