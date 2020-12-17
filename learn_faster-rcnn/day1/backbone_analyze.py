import argparse
import os
import cv2
from mmdet.cv_core import (Config, load_checkpoint)
from mmdet.models import build_detector
from mmdet.datasets.builder import build_dataset
from mmdet.datasets.pipelines import Compose

from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(
        description='Backbone analyze'
    )
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path')

    args = parser.parse_args()

    return args


def forward(self, img, img_metas=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    return outs

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # print model
    for key, value in cfg.model.items():
        print(key)

    # create model only use cpu
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    model.forward_train()



    return


if __name__ == '__main__':
    main()