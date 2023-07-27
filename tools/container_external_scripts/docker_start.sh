#!/bin/bash
# arg1: Name of the container
# arg2: A directory to bind mount for access from the container
# arg3: Path to dotenv file containing environment variables (wandb secret, etc.)
# Remaining args: Command to run with the container

# Needed to expose Nvidia GPUs to Vulkan renderer
# Locations of Nvidia icd files may vary across hosts and cannot be baked in 
ICD_SEARCH_LOCATIONS=(
    /usr/local/etc/vulkan/icd.d
    /usr/local/share/vulkan/icd.d
    /etc/vulkan/icd.d
    /usr/share/vulkan/icd.d
    /etc/glvnd/egl_vendor.d
    /usr/share/glvnd/egl_vendor.d
)
ICD_MOUNTS=( )
for filename in $(find "${ICD_SEARCH_LOCATIONS[@]}" -name "*nvidia*.json" 2> /dev/null); do
    ICD_MOUNTS+=( --volume "${filename}":"${filename}":ro )
done

docker run -d --rm --gpus "device=0" --name $1 \
    --mount type=bind,source=$2,target=/mnt/host ${ICD_MOUNTS[@]} \
    --mount type=bind,source=$3,target=/home/nonroot/ns-arch-unity/.env \
    jpstyle92/ns-arch-unity "${@:4}"
