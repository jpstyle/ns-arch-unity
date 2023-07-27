#!/bin/bash
bash tools/container_internal_scripts/start_xvfb.sh
eval $*
rsync --archive --update --compress --info=progress2 outputs/ /mnt/data_volume/outputs