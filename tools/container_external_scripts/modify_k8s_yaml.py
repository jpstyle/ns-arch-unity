"""
Modify the given K8s object yaml file to add hostPath volumeMounts configurations,
so that Nvidia icd files on worker nodes are exposed and can be accessed for
Vulkan rendering with Nvidia GPUs. Print modified yaml content to standard input.
"""
import os
import sys
import yaml


if __name__ == "__main__":
    with open(sys.argv[1], "r") as yml_f:
        k8s_obj = yaml.safe_load(yml_f)

    if "spec" not in k8s_obj: k8s_obj["spec"] = {}
    if "volumes" not in k8s_obj["spec"]: k8s_obj["spec"]["volumes"] = []
    if "containers" not in k8s_obj["spec"]: k8s_obj["spec"]["containers"] = []

    icd_search_locations = [
        "/usr/local/etc/vulkan/icd.d",
        "/usr/local/share/vulkan/icd.d",
        "/etc/vulkan/icd.d",
        "/usr/share/vulkan/icd.d",
        "/etc/glvnd/egl_vendor.d",
        "/usr/share/glvnd/egl_vendor.d"
    ]

    for i, path in enumerate(icd_search_locations):

        k8s_obj["spec"]["volumes"].append({
            "name": f"icd{i}",
            "hostPath": { "path": path, "type": "FileOrCreate" }
        })

        for c_spec in k8s_obj["spec"]["containers"]:
            if "volumeMounts" not in c_spec: c_spec["volumeMounts"] = []

            c_spec["volumeMounts"].append({
                "mountPath": path,
                "name": f"icd{i}",
                "readOnly": True
            })

    print(yaml.dump(k8s_obj))
