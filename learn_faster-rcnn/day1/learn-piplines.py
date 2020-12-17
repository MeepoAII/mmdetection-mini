import argparse
import os
from pathlib import Path

from mmdet import cv_core
from mmdet.cv_core import Config
from mmdet.datasets.builder import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='learn dataset piplines')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--out_dir',
                        default=None,
                        type=str,
                        help='output path')
    args = parser.parse_args()
    return args


# def get_dataset_cfg(config_path, skip_type):
#     cfg = Config.fromfile(config_path)
#     if skip_type:
#         cfg.train_pipeline = [
#             x for x in cfg.train_pipeline if x['type'] not in skip_type
#         ]
#
#
#     return cfg


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)

    progress_bar = cv_core.ProgressBar(len(dataset))
    print("print out the last pipeline\n")
    i = 1
    for item in dataset:
        filename = os.path.join(args.out_dir,
                                Path(item['filename']).name) if args.out_dir is not None else None
        # print(filename)
        # for key, value in item.items():
        #     print(f"{key}")
        cv_core.imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
            class_names=dataset.CLASSES,
            show=False,
            out_file=filename
        )
        i += 1
        if i == 100:
            break

        progress_bar.update()
