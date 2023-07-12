#!/bin/bash
mkdir -p /mnt/data_volume/vaw
python tools/vision/prepare_data.py \
    vision=train_rgb \
    vision.data.path=/mnt/data_volume/vaw
