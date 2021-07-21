"""Inference over ultiple models."""
import glob
import os


def main():
    """Main."""
    for model in glob.glob("ensembler/models/*"):
        print(model)
        try:
            os.mkdir(model + "/train_inference")
        except Exception:
            pass
        os.system(
            "python inference.py --output-dir "
            + model
            + "/train_inference -c "
            + model
            + "/config.toml "
            + "cil-road_segmentation-2021/training/training/images "
            + model
        )

        try:
            os.mkdir(model + "/test_inference")
        except Exception:
            pass
        os.system(
            "python inference.py --output-dir "
            + model
            + "/est_inference -c "
            + model
            + "/config.toml "
            + "cil-road_segmentation-2021/test_images/test_images/ "
            + model
        )


if __name__ == "__main__":
    main()
