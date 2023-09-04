#!/bin/bash
# arg1: Name of the container
# arg2: CUDA device index to use
# arg3: Docker volume to mount
# arg4: Path to dotenv file containing environment variables (wandb secret, etc.)
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

# Uncomment to debug only with virtual display
docker run -d --name $1 --gpus "device=$2" \
    --volume $3:/mnt/data_volume \
    --volume $4:/home/nonroot/ns-arch-unity/.env \
    ${ICD_MOUNTS[@]} \
    jpstyle92/ns-arch-unity "${@:5}"

# Uncomment to debug with local (linux) machine display
# docker run -d --name $1 --gpus "device=$2" \
#     --volume $3:/mnt/data_volume \
#     --volume $4:/home/nonroot/ns-arch-unity/.env \
#     --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix \
#     ${ICD_MOUNTS[@]} \
#     jpstyle92/ns-arch-unity "${@:5}"
