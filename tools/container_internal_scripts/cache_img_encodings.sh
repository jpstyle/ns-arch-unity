#!/bin/bash
mkdir -p /mnt/data_volume/cache
python tools/vision/cache_image_encodings.py \
    vision=train_rgb \
    vision.data.path=/mnt/data_volume/vaw \
    paths.cache_dir=/mnt/data_volume/cache
