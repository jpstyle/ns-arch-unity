#!/bin/bash

# Make sure to have an argument provided
if [ $# -eq 0 ] ; then
    echo "Error: Provide yaml file to modify"
    exit 0
fi

# Make sure to use python3
py_ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).*/\1/')
if [ $py_ver == "3" ]; then
    py_cmd=(python)
else
    py_cmd=(python3)
fi

py_cmd+=(tools/container_external_scripts/modify_k8s_yaml.py $1)

${py_cmd[@]} | kubectl create -f -
