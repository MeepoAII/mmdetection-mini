import argparse
import os
from pathlib import Path

from mmdet import cv_core
from mmdet.cv_core import Config
from mmdet.datasets.builder import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='learn dataset piplines')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--out-dir',
                        default=None,
                        type=str,
                        help='output path')
    args = parser.parse_args()
    return args


def get_dataset_cfg(config_path):
    cfg = Config.fromfile(config_path)
    return cfg.data.train


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_cfg(args.config)
    print(cfg)

    dataset = build_dataset(cfg)

    progress_bar = cv_core.ProgressBar(len(dataset))

    for item in dataset:
        for key, value in item.items():
            print(f"key is {key}, type(value) is {type(value)}")
        break
