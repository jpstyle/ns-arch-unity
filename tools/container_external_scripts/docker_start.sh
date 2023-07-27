#!/bin/bash
# arg1: A directory to bind mount for access from the dev container

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

docker run -d --rm --name ns-dev --mount type=bind,source=$1,target=/mnt/host ${ICD_MOUNTS[@]} --gpus "device=0" --entrypoint sleep jpstyle92/ns-arch-unity infinity
