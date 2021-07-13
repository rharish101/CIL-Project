"""Here's some slime for you."""
import tkinter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from random import randint, shuffle

import numpy
from PIL import Image, ImageTk
from torch import cat, float, maximum, minimum, ones_like, zeros_like
from torchvision.io import read_image

from src.data import get_file_paths
from src.slime import Slime

GUI_UPDATE_STEPS = 40
VISUALIZE = True


def get_classification_paths(classification_dir):
    """Get all classification paths."""
    file_names = [path.name for path in Path(classification_dir).glob("*")]
    if "submission.csv" in file_names:
        file_names.remove("submission.csv")
    file_names.sort()
    return [
        str(Path(classification_dir) / file_name) for file_name in file_names
    ]


def classify_image(
    image_name, x, y, start_positions, slime, root=None, visualize=False
):
    """Classify image."""
    stack = [start_positions]
    already_visited = zeros_like(y, dtype=float)

    gui_step_counter = 0
    if visualize:
        root = root
        initial_image = numpy.array(already_visited[0]).astype(int)
        initial_image = Image.fromarray(initial_image)
        initial_image = ImageTk.PhotoImage(initial_image)

        label1 = tkinter.Label(image=initial_image)
        label1.place(x=0, y=0)

        root.geometry(
            f"{initial_image.width() + 4}x{initial_image.height() + 4}"
        )
        root.update()

    while not len(stack) == 0:
        next_posses = stack[randint(0, len(stack) - 1)]
        stack.remove(next_posses)
        shuffle(next_posses)
        for pos in next_posses:
            pos_x = pos[0]
            pos_y = pos[1]
            classified = slime.classify(pos_x, pos_y, x, already_visited, y)
            if classified:
                already_visited[0][pos_x][pos_y] = 1
                next_posses_list = []
                if (
                    pos_x < len(y[0][0]) - 1
                    and already_visited[0][pos_x + 1][pos_y] == 0
                ):
                    next_posses_list.append((pos_x + 1, pos_y))
                if 0 < pos_x and already_visited[0][pos_x - 1][pos_y] == 0:
                    next_posses_list.append((pos_x - 1, pos_y))
                if (
                    pos_y < len(y[0][0]) - 1
                    and already_visited[0][pos_x][pos_y + 1] == 0
                ):
                    next_posses_list.append((pos_x, pos_y + 1))
                if 0 < pos_y and already_visited[0][pos_x][pos_y - 1] == 0:
                    next_posses_list.append((pos_x, pos_y - 1))
                if not next_posses_list == []:
                    stack.append(next_posses_list)
            else:
                already_visited[0][pos_x][pos_y] = -1

        gui_step_counter += 1
        if visualize and gui_step_counter > GUI_UPDATE_STEPS:
            root.title(image_name.split("\\")[-1] + " - " + str(len(stack)))
            next_im = ((numpy.array(already_visited[0]) + 1) * 127).astype(int)
            next_im = Image.fromarray(next_im)
            next_im = ImageTk.PhotoImage(next_im)
            label1.configure(image=next_im)
            label1.image = next_im
            root.update()
            gui_step_counter = 0

    already_visited = maximum(already_visited, zeros_like(already_visited))
    if visualize:
        root.title("Result")
        next_im = numpy.array(already_visited[0]).astype(int) * 255
        next_im = Image.fromarray(next_im)
        next_im = ImageTk.PhotoImage(next_im)
        label1.configure(image=next_im)
        label1.image = next_im
        root.update()

    if slime.inference:
        print(int(already_visited.sum()), "were positively classified")
        print(
            "Previously missclassified:",
            int(((x[3] / 255) - minimum(y[0], ones_like(y[0]))).abs().sum()),
            "(Not sure if this actually work correctly)",
        )
        print(
            "Now missclassified:",
            int(
                (already_visited[0] - minimum(y[0], ones_like(y[0])))
                .abs()
                .sum()
            ),
            "(Not sure if this actually work correctly)",
        )

    return already_visited


def main(args: Namespace) -> None:
    """Slimey main."""
    print("Loading data")
    training_path_list, ground_truth_path_list = get_file_paths(
        Path("cil-road-segmentation-2021")
    )
    classification_path_list = get_classification_paths(
        "cil-road-segmentation-2021/training/training/train_outputs"
    )
    x = []
    y = []
    for idx in range(len(training_path_list)):
        img = read_image(training_path_list[idx]).float()
        img = (img - img.mean()) / img.std()
        y.append(read_image(ground_truth_path_list[idx]))
        classif = read_image(classification_path_list[idx]).float()
        classif = (classif - classif.mean()) / classif.std()
        x.append(cat((img, classif)))

    start_positions = []
    for pic in x:
        sp = []
        for j in range(len(pic[3])):
            if pic[3][0][j] > 0:
                sp.append((0, j))
            if pic[3][len(pic[0]) - 1][j] > 0:
                sp.append((len(pic[3]) - 1, j))
            if pic[3][j][0] > 0:
                sp.append((j, 0))
            if pic[3][j][len(pic[3]) - 1] > 0:
                sp.append((j, len(pic[0]) - 1))
        start_positions.append(sp)

    print("Initializing slime")
    slime = Slime(
        state_file_path=args.slime_vision, inference=(args.mode == "inference")
    )

    if args.mode == "train":
        print("Training slime")
    else:
        print("Doing inference")
    while True:
        if args.mode == "train":
            slime.save()


if __name__ == "__main__":
    description = "Script to run the slime."
    formatter_class = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description, formatter_class)  # type: ignore
    parser.add_argument(
        "--slime_vision",
        type=str,
        default=None,
        help="File with the weights of the slime vision.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "inference"],
        default="train",
        help="Train or only infer",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        nargs="?",
        const=True,
        help="Train or only infer",
    )
    main(parser.parse_args())
