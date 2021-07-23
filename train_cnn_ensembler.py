"""Trains an cnn-based enssembler."""
import glob
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import List, Tuple

import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision.io import read_image
from tqdm import tqdm

from src.ensembler import EnsemblerCNN

TRAIN_IMAGE_FOLDER = "training/training/images/"
TRAIN_GROUNDTRUTH_FOLDER = "training/training/groundtruth/"
TEST_IMAGE_FOLDER = "test_images/test_images/"

DEVICE = torch.device("cuda")
INPUT_IMAGE = False
# DEVICE = torch.device('cpu')


def load_data(
    outputs_folder: str, dataset: str, mode: str, input_image: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads data from paths."""
    x: List = []
    y = []
    models = list(glob.glob(outputs_folder + mode + "/*"))
    for file in glob.glob(models[0] + "/*.png"):
        image = []
        file_name = file.split("/")[-1]
        for model in models:
            image.append(read_image(model + "/" + file_name))

        image = torch.cat(image)

        if input_image:
            if mode == "train":
                image = torch.cat(
                    (
                        image,
                        read_image(dataset + TRAIN_IMAGE_FOLDER + file_name),
                    )
                )
            else:
                image = torch.cat(
                    (
                        image,
                        read_image(dataset + TEST_IMAGE_FOLDER + file_name),
                    )
                )

        x.append(image)

        if mode == "train":
            y.append(
                read_image(dataset + TRAIN_GROUNDTRUTH_FOLDER + file_name)
            )

    x_tensor: torch.Tensor = torch.stack(x)
    if mode == "train":
        y = torch.stack(y) / 255

    return x_tensor / 255, y


def train_cnn(
    cnn: EnsemblerCNN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    iterations: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Trains the cnn."""
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.01
    )

    loss = BCEWithLogitsLoss()
    optim = Adam(
        cnn.parameters(),
        lr=learning_rate,
        weight_decay=5e-6,
    )

    with tqdm(
        total=int(iterations / len(train_x)) * len(train_x)
    ) as progress_bar:
        total_iterations = 0
        test_loss = 0
        lowest_train_loss: float = 1.0

        cnn.to(DEVICE)
        val_x = val_x.to(DEVICE)
        val_y = val_y.to(DEVICE)

        avg_train_loss = 0.0
        for epoch in range(int(iterations / len(train_x))):

            perm = torch.randperm(len(train_x))
            train_x = train_x[perm]
            train_y = train_y[perm]

            cum_train_loss = 0.0
            num_batches = 0
            for i in range(0, len(train_x), batch_size):

                optim.zero_grad()
                x = train_x[i : i + batch_size]
                y = train_y[i : i + batch_size]

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred_y = cnn(x)
                loss_value = loss(pred_y, y)
                loss_value.backward()
                optim.step()

                if total_iterations % 500 == 0:
                    cnn.eval()
                    pred_val_y = cnn(val_x)
                    test_loss = loss(pred_val_y, val_y)
                    cnn.train()

                cum_train_loss += loss_value
                progress_bar.set_description(
                    "Loss: "
                    + str(float(loss_value))
                    + " Test Loss "
                    + str(float(test_loss))
                    + " Avg Train Loss "
                    + str(float(avg_train_loss))
                )
                progress_bar.update(batch_size)
                total_iterations += batch_size
                num_batches += 1

            avg_train_loss = cum_train_loss / num_batches
            if avg_train_loss < lowest_train_loss:
                lowest_train_loss = float(avg_train_loss)
                file_names = [
                    file.split("/")[-1] for file in glob.glob("outputs/*.png")
                ]
                cnn.eval()
                apply_cnn(cnn, test_x, "ensembler_outputs", file_names)
                cnn.train()

    print("Lowest avg train loss:", lowest_train_loss)


def apply_cnn(
    cnn: EnsemblerCNN,
    test_x: torch.Tensor,
    output_folder: str,
    file_names: List[str],
) -> None:
    """Applies the cnn and saves test predictions."""
    try:
        os.mkdir(output_folder)
    except Exception:
        pass

    cnn.eval()
    for idx in range(len(test_x)):
        image = test_x[idx : idx + 1].to(DEVICE)
        image = cnn(image).detach().cpu()
        # image = (image.numpy() > 0).astype(int)
        image = torch.sigmoid(image).numpy()
        cv2.imwrite(output_folder + "/" + file_names[idx], image[0][0] * 255)


def main(args: Namespace) -> None:
    """Main method."""
    num_inputs = len(os.listdir("ensembler/new-outputs/test"))
    train_x, train_y = load_data(
        args.output_dir, args.image_dir, "train", INPUT_IMAGE
    )
    test_x, _ = load_data(args.output_dir, args.image_dir, "test", INPUT_IMAGE)
    print("Train set:", train_x.shape)
    print("Test set:", test_x.shape)
    cnn = EnsemblerCNN(INPUT_IMAGE, num_inputs)
    train_cnn(cnn, train_x, train_y, test_x, 400000, 4, 0.00001)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to infer with the model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        metavar="IMAGE_DIR",
        type=str,
        default="cil-road-segmentation-2021/",
        help="Path to the directory containing the input images",
    )
    parser.add_argument(
        "--output_dir",
        metavar="OUTPUT_DIR",
        type=str,
        default="ensembler/new-outputs/",
        help="Path to the directory containing outputs"
        "from the models that should be ensembled",
    )

    main(parser.parse_args())
