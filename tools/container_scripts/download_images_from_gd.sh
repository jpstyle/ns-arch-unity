#!/bin/bash
mkdir -p /mnt/data_volume/vaw
gdown -O /mnt/data_volume/vaw/ https://drive.google.com/uc?id=$VAW_GD_LINK
tar -xvzf /mnt/data_volume/vaw/vaw.tgz -C /mnt/data_volume/vaw
rm /mnt/data_volume/vaw/vaw.tgz