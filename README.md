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
    
5. If you intend to run the visualization script, install tkinter

**NOTE:** You need to be inside the virtual environment where you installed the above dependencies every time you commit.
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

**NOTE:** You do not need to mention every single hyper-parameter in a config.
In such a case, the missing ones will use their default values.

### Training
Run the script `train.py`:
```sh
./train.py /path/to/CIL/dataset/
```

The weights of trained models are saved with the `.pt` extension inside an ISO 8601 timestamped subdirectory, which is stored in a parent directory (as given by the `--save-dir` argument).
By default, the parent directory is `saved_models`.

Training logs are by default stored inside an ISO 8601 timestamped subdirectory, which is stored in a parent directory (as given by the `--log-dir` argument).
The timestamp will be the same one as the one where the model weights are saved.
By default, the parent directory is `logs`.

The hyper-parameter config (including defaults) is saved as a TOML file named `config.toml` in both the saved models directory and the timestamped log directory.

#### Multi-GPU Training
This implementation supports multi-GPU training on a single machine using PyTorch's [`torch.nn.DataParallel`](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).

For choosing which GPUs to train on, set the `CUDA_VISIBLE_DEVICES` environment variable when running a script as follows:
```sh
CUDA_VISIBLE_DEVICES=0,1,3 ./script.py
```
This selects the GPUs 0, 1 and 3 for training.
By default, all available GPUs are chosen.

#### Mixed Precision Training
This implementation supports mixed-precision training.
This can be enabled by setting the `mixed_precision` hyper-parameter in a config.
Note that this will only provide significant speed-ups if your GPU(s) have special support for mixed-precision compute.

### Inference
The script `inference.py` generates predictions on the test data using a trained model.
Run it as follows:
```sh
./inference.py /path/to/image/dir/ /path/to/save/dir/
```

**NOTE:** Here, you need to give the path to the directory containing the input images, as opposed to the root directory of the CIL dataset.
Also, you need to give the path to the directory containing the saved model's weights.

The output images are saved in the directory given by the `--output-dir` argument.
By default, this directory is `outputs`.
The images will be saved as PNG images with the file names corresponding to the input images.

### Evaluation
The script `evaluate.py` uses the generated predictions and prints metrics on the training and validation data.
Run it as follows:
```sh
./evaluate.py /path/to/CIL/dataset/ /path/to/predictions/dir/
```

### Post-Processing

#### SPP
To run the SPP algorithm on output masks and create enhanced masks, run the `spp.py` script:
```sh
./spp.py
```
The path to input masks and output directory are given by `--input-dir` and `--output-dir` arguments to the script, respectively.

The metrics are printed to `stdout`.

#### Graph-Cut
To run Graph-Cut algorithm on output masks and create enhanced masks, run the `graph_cut.py` script:
```sh
./graph_cut.py
```
The path to input masks and output directory are given by `--mask-dir` and `--output-dir` arguments to the script, respectively.
The number of iteration of Grabcut algorithm also can be specified by the `--iter` argument. 

In case of using the script to ensemble the output of different models, `--ensemble` argument is given.
In this case, it is assumed that the mask directory includes directories in which individual masks are provided.

There is also a Jupyter notebook for visualizing the results of applying the GrabCut algorithm on a given test image as well as its corresponding mask.
It is in `notebooks/visualize_graph_cut.ipynb` and can be used for tuning hyper-parameters.

### Submissions
To create a submission from the inference outputs on the test data, run the script `submission.py`:
```sh
./submission.py
```

The inference images are loaded from the directory given by the `--image-dir` argument.
The output CSV is saved in the directory given by the `--output-dir` argument.
By default, both directories are `outputs`.

### Visualization
The script `visualizer.py` provides a GUI that layers black-and-white segmentations on top of the corresponding images.
Additionally, it can also visualize the model's predictions in green to compare them with the ground truth.
Run this as follows:
```sh
./visualizer.py /path/to/CIL/dataset
```

By default, this visualizes the training data.
To visualize the inference outputs for the training data, run it as follows:
```sh
./visualizer.py /path/to/CIL/dataset --pred-dir /path/to/model/outputs
```
Here, the inference images are loaded from the directory given by the `--pred-dir` argument.

To visualize the inference outputs for the test data, run it as follows:
```sh
./visualizer.py /path/to/CIL/dataset --mode test --pred-dir /path/to/model/outputs
```
For the test data, you can't visualize the ground truth (because it doesn't exist!), and hence the inference outputs are used as "ground truth".
Therefore, the `--pred-dir` argument is necessary in test mode.
