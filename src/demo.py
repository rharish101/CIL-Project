"""
The MIT License (MIT)
Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time

import imageio
import numpy as np
import torch
from torch.autograd import Variable

import conv_crf as convcrf
from utils import pascal_visualizer as vis
from utils import synthetic

# import scipy as scp
# import scipy.misc






try:
    import matplotlib.pyplot as plt

    figure = plt.figure()
    matplotlib = True
    plt.close(figure)
except:
    matplotlib = False
    pass

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def do_crf_inference(images, unaries, args):

    if args.pyinn or not hasattr(torch.nn.functional, "unfold"):
        # pytorch 0.3 or older requires pyinn.
        args.pyinn = True
        # Cheap and easy trick to make sure that pyinn is loadable.
        import pyinn

    # get basic hyperparameters
    num_classes = unaries[0].shape[2]
    shape = images[0].shape[0:2]
    config = convcrf.default_conf
    config["filter_size"] = 7
    config["pyinn"] = args.pyinn

    if args.normalize:
        # Warning, applying image normalization affects CRF computation.
        # The parameter 'col_feats::schan' needs to be adapted.

        for i in range(len(images)):
            # Normalize image range
            #     This changes the image features and influences CRF output
            image = images[i] / 255
            # mean substraction
            #    CRF is invariant to mean subtraction, output is NOT affected
            image = image - 0.5
            # std normalization
            #       Affect CRF computation
            images[i] = image / 0.3

        # schan = 0.1 is a good starting value for normalized images.
        # The relation is f_i = image * schan
        config["col_feats"]["schan"] = 0.1

    # make input pytorch compatible
    image = torch.from_numpy(images).permute(
        0, 3, 1, 2
    )  # shape: [N, 3, height, width]
    img_var = Variable(torch.Tensor(image))

    unary = torch.from_numpy(unaries).permute(
        0, 3, 1, 2
    )  # shape: [N, 3, height, width]
    unary_var = Variable(torch.Tensor(unary))

    logging.info("Build ConvCRF.")
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(
        conf=config, shape=shape, nclasses=num_classes, use_gpu=not args.cpu
    )

    # move to GPU if requested
    if not args.cpu:
        img_var = img_var.cuda()
        unary_var = unary_var.cuda()
        gausscrf.cuda()

    logging.info("Start Computation.")
    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    if args.nospeed:
        # Evaluate inference speed
        logging.info("Doing speed evaluation.")
        start_time = time.time()
        for i in range(10):
            # Running ConvCRF 10 times and average total time
            pred = gausscrf.forward(unary=unary_var, img=img_var)

        pred.cpu()  # wait for all GPU computations to finish

        duration = (time.time() - start_time) * 1000 / 10

        logging.info("Finished running 10 predictions.")
        logging.info("Avg. Computation time: {} ms".format(duration))

    return prediction.data.cpu().numpy()


def plot_results(image, unary, prediction, label, args):

    logging.info("Plot results.")

    # Create visualizer
    myvis = vis.PascalVisualizer()

    # Transform id image to coloured labels
    coloured_label = myvis.id2color(id_image=label)

    unary_hard = np.argmax(unary, axis=2)
    coloured_unary = myvis.id2color(id_image=unary_hard)

    prediction = prediction[0]  # Remove Batch dimension
    prediction_hard = np.argmax(prediction, axis=0)
    coloured_crf = myvis.id2color(id_image=prediction_hard)

    if matplotlib:
        # Plot results using matplotlib
        figure = plt.figure()
        figure.tight_layout()

        # Plot parameters
        num_rows = 2
        num_cols = 2

        ax = figure.add_subplot(num_rows, num_cols, 1)
        # img_name = os.path.basename(args.image)
        ax.set_title("Image ")
        ax.axis("off")
        ax.imshow(image)

        ax = figure.add_subplot(num_rows, num_cols, 2)
        ax.set_title("Label")
        ax.axis("off")
        ax.imshow(coloured_label.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 3)
        ax.set_title("Unary")
        ax.axis("off")
        ax.imshow(coloured_unary.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 4)
        ax.set_title("CRF Output")
        ax.axis("off")
        ax.imshow(coloured_crf.astype(np.uint8))

        plt.show()
    else:
        if args.output is None:
            args.output = "out.png"

        logging.warning("Matplotlib not found.")
        logging.info("Saving output to {} instead".format(args.output))

    if args.output is not None:
        # Save results to disk
        out_img = np.concatenate(
            (image, coloured_label, coloured_unary, coloured_crf), axis=1
        )

        imageio.imwrite(args.output, out_img.astype(np.uint8))

        logging.info("Plot has been saved to {}".format(args.output))

    return


def get_parser():
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("image", type=str, help="input image")

    parser.add_argument("label", type=str, help="Label file.")

    parser.add_argument("unary", type=str, help="Prediction file.")

    parser.add_argument(
        "--gpu", type=str, default="0", help="which gpu to use"
    )

    parser.add_argument(
        "--output", type=str, help="Optionally save output as img."
    )

    parser.add_argument(
        "--nospeed", action="store_false", help="Skip speed evaluation."
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize input image before inference.",
    )

    parser.add_argument(
        "--pyinn",
        action="store_true",
        help="Use pyinn based Cuda implementation" "for message passing.",
    )

    parser.add_argument(
        "--cpu", action="store_true", help="Run on CPU instead of GPU."
    )

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Load data
    images = []
    labels = []
    unaries = []
    for im in os.listdir(args.image):
        images.append(imageio.imread(f"{args.image}\\{im}"))
        label = imageio.imread(f"{args.label}\\{im}")
        label = np.uint8(label > 127.5)
        labels.append(label)

        # Produce unary by adding noise to label
        unary = imageio.imread(f"{args.unary}\\{im}")
        unary = unary / 255
        unary = np.stack([1 - unary, unary], -1)
        # unary = synthetic.augment_label(label, num_classes=2)
        unaries.append(unary)

    images = np.array(images).astype(np.float32)
    unaries = np.array(unaries).astype(np.float32)

    # Compute CRF inference
    predictions = do_crf_inference(images, unaries, args)
    # Plot output
    plot_results(images[0], unaries[0], predictions, labels[0], args)
    logging.info("Thank you for trying ConvCRFs.")
