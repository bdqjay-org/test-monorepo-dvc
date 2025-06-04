import argparse
import json
from typing import Text

import yaml
from fastai.vision.all import CategoryBlock
from fastai.vision.all import DataBlock
from fastai.vision.all import error_rate
from fastai.vision.all import get_image_files
from fastai.vision.all import ImageBlock
from fastai.vision.all import parent_label
from fastai.vision.all import RandomSplitter
from fastai.vision.all import Resize
from fastai.vision.all import resnet18
from fastai.vision.all import vision_learner

from shared.src.utils import get_git_root


def data_load_train(config_path: Text) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    raw_data_path = config["data"]["raw_data_path"]
    random_state = config["base"]["random_state"]
    validation_split = config["data_block"]["validation_size"]
    img_resize_value = config["data_block"]["img_resize_value"]
    batch_size = config["data_block"]["batch_size"]

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=validation_split, seed=random_state),
        get_y=parent_label,
        item_tfms=[Resize(img_resize_value, method="squish")],
    ).dataloaders(raw_data_path, bs=batch_size)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    print("\n📈 Final Metrics on Validation Set:")
    results = learn.validate()
    print(f"  - Loss: {results[0]:.4f}")
    print(f"  - Error Rate: {results[1]:.4f}")

    root = get_git_root()
    metrics_file_path = root / "bird_or_forest" / "reports" / "metrics.json"
    with open(metrics_file_path, "w") as f:
        json.dump({"loss": results[0], "error rate": results[1]}, f)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_load_train(config_path=args.config)
