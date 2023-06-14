"""
Helper methods for preparing VAW dataset for training/validation/testing vision
module, to be called by prepare_data() in PyTorch Lightning DataModule.
"""
import os
import json
import logging
import requests
import multiprocessing
from urllib.request import urlretrieve
from urllib.error import URLError
from collections import defaultdict
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_from_url(url, target_path):
    """
    Download a single file from the given url to the target path, with a progress bar shown
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))

    with open(target_path, 'wb') as file, tqdm(
        desc=target_path.split("/")[-1],
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_vg_images(images_path, vg_img_data, imgs_to_download):
    """
    Download image files designated by the `imgs_to_download` argument, which is a list
    of image IDs per Visual Genome v1.4
    """
    url_path_pairs = [
        (img["url"], os.path.join(images_path, img["url"].split("/")[-1]))
        for img in vg_img_data if img["image_id"] in imgs_to_download
    ]

    num_workers = multiprocessing.cpu_count()
    dl_queues = ThreadPool(num_workers).imap_unordered(
        _download_indiv_image, url_path_pairs
    )

    pbar = tqdm(enumerate(dl_queues), total=len(imgs_to_download))
    pbar.set_description("Downloading images")

    downloaded = 0
    for result in pbar:
        downloaded += result[1]
    logger.info(f"{downloaded} images downloaded")


def _download_indiv_image(url_and_path):
    """
    Helper method for downloading individual images, to be called from download_images()
    by multiple threads in parallel
    """
    url, path = url_and_path

    if not os.path.exists(path):
        try:
            urlretrieve(url, path)
        except URLError as e:
            logger.info(f"{e}: Retrying download {url}...")
            _download_indiv_image(url_and_path)
        except e:
            raise e

        return True
    else:
        return False


def extract_metadata(dataset_path):
    """
    Process VAW annotations for the train/val/test splits to do the following:
        1) Index annotation entries by class/attribute concept and save as metadata
        2) Collect image IDs (as per VG v1.4) for downloading as needed (return value)
    """
    # Return value; set of all image IDs (by Visual Genome v1.4 index)
    image_ids = set()

    # Indexing object instances by class/attribute concepts, per split
    index_by_cls = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list)
    }
    index_by_att_pos = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list)
    }

    if os.path.exists(f"{dataset_path}/metadata.json"):
        # Dataset is already processed and metadata is extracted, just return
        # list of image IDs that need to be present
        with open(f"{dataset_path}/metadata.json") as meta_f:
            image_ids = {
                img
                for per_conc_type in json.load(meta_f).values()
                for per_split in per_conc_type.values()
                for per_conc in per_split.values()
                for img, _ in per_conc
            }
    else:
        # Metadata not present, annotation data must be processed to extract and
        # store metadata

        # Process annotation data per split
        for data_name in ["train_part1", "train_part2", "val", "test"]:
            # Data split: one of 'train', 'val', 'test'
            spl = data_name.split("_")[0]

            # Load annotation file
            with open(f"{dataset_path}/{data_name}.json") as anno_f:
                anno_data = json.load(anno_f)

            # Process each entry, extracting image IDs to download and indexing entries
            for entry in anno_data:
                # Some entry has null value for the segmentation field, ignore them
                if entry["instance_polygon"] is None: continue

                img_id = int(entry["image_id"])
                instance_id = int(entry["instance_id"])

                image_ids.add(img_id)
                index_by_cls[spl][entry["object_name"]].append((img_id, instance_id))
                for att_pos in entry["positive_attributes"]:
                    index_by_att_pos[spl][att_pos].append((img_id, instance_id))
                # We don't make any use of negative attribute labels here

        # Write metadata to JSON file
        with open(f"{dataset_path}/metadata.json", "w") as meta_f:
            metadata = {
                "instances_class": {
                    k: dict(v) for k, v in index_by_cls.items()
                },
                "instances_attribute": {
                    k: dict(v) for k, v in index_by_att_pos.items()
                }
            }
            json.dump(metadata, meta_f)

    return image_ids
