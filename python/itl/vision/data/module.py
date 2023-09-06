import os
import copy
import json
import gzip
import pickle
import random
import logging
from PIL import Image
from functools import reduce
from collections import defaultdict

import torch
import numpy as np
import pytorch_lightning as pl
from pycocotools import mask
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from .prepare import download_from_url, download_vg_images, extract_metadata

logger = logging.getLogger(__name__)


VAW_URL = "https://github.com/adobe-research/vaw_dataset/raw/main/data"

class FewShotDataModule(pl.LightningDataModule):
    """
    DataModule for preparing & loading data for few-shot tasks concerning visual
    object classes and attributes. Responsible for downloading, pre-processing
    data, managing train/val/test split, and shipping appropriate dataloaders.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Create data directory at the specified path if not exists
        dataset_path = self.cfg.vision.data.path
        images_path = f"{dataset_path}/images"
        cache_path = self.cfg.paths.cache_dir
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)

        if self.cfg.vision.data.name == "vaw":
            # VAW (Visual Attributes in the Wild) dataset; images are from VG
            # datasets, with higher-quality (i.e., denser, less noisy, provides
            # negative labels as well) annotations on visual attributes

            # Note: Turns out Visual Genome API is somewhat unstable lately, so
            # the VG image metadata file is salvaged and included as part of repo.
            # In the meantime, image files themselves are hosted on stanford server
            # and seem to be affected less, fortunately.
            with open(f"{self.cfg.paths.data_dir}/vg_image_data.json") as vg_imgs_f:
                vg_img_data = json.load(vg_imgs_f)

            # Download annotation JSON files
            for data_name in ["train_part1", "train_part2", "val", "test"]:
                json_filename = f"{data_name}.json"
                data_file_url = f"{VAW_URL}/{json_filename}"
                target_path = f"{dataset_path}/{json_filename}"

                if os.path.exists(target_path):
                    logger.info(f"{json_filename} already exists, skip download")
                else:
                    download_from_url(data_file_url, target_path)

            # Download image files, reformat & save VG annotations
            imgs_to_download = extract_metadata(dataset_path)
            download_vg_images(
                vg_img_data, imgs_to_download, images_path, cache_path, "vaw"
            )

    def setup(self, stage):
        # Prepare dataset & samplers according to task type
        dataset_path = self.cfg.vision.data.path
        annotations = {
            "train": defaultdict(dict),
            "val": defaultdict(dict),
            "test": defaultdict(dict)
        }
        self.datasets = {}; self.samplers = {}

        if self.cfg.vision.task.startswith("rgb"):
            # Load VAW metadata
            with open(f"{dataset_path}/metadata.json") as meta_f:
                metadata = json.load(meta_f)

            # For loading annotation data for train/val/test splits
            def load_annotation_data(file_name, anno_dict):
                with open(f"{dataset_path}/{file_name}") as ann_f:
                    for entry in json.load(ann_f):
                        img_id = int(entry["image_id"])
                        instance_id = int(entry["instance_id"])
                        anno_dict[img_id][instance_id] = {
                            "class": entry["object_name"],
                            "attributes_pos": entry["positive_attributes"],
                            "attributes_neg": entry["negative_attributes"],
                            "instance_bbox": entry["instance_bbox"],
                            "instance_mask": entry["instance_polygon"]
                        }

            # Create _FewShotDataset & _FewShotDataSampler instances per split
            # as required by `stage`
            def add_dataset_and_sampler(spl):
                B = self.cfg.vision.data.batch_size
                K = self.cfg.vision.data.num_exs_per_conc
                self.datasets[spl] = _FewShot2DDataset(
                    dict(annotations[spl]), dataset_path,
                    self.cfg.paths.cache_dir, self.cfg.vision.data.name
                )
                self.samplers[spl] = {
                    conc_type: _FewShot2DDataSampler(metadata, conc_type, spl, B, K)
                    for conc_type in ["class", "attribute"]
                }
            if stage in ["fit"]:
                # Training set required for "fit" stage setup
                load_annotation_data("train_part1.json", annotations["train"])
                load_annotation_data("train_part2.json", annotations["train"])
                add_dataset_and_sampler("train")
            if stage in ["fit", "validate"]:
                # Validation set required for "fit"/"validate" stage setup
                load_annotation_data("val.json", annotations["val"])
                add_dataset_and_sampler("val")
            if stage in ["test"]:
                # Test set required for "fit"/"test"/"predict" stage setup
                load_annotation_data("test.json", annotations["test"])
                add_dataset_and_sampler("test")

    def train_dataloader(self):
        return _ChainedLoader(*self._return_dataloaders("train"))
    
    def val_dataloader(self):
        return self._return_dataloaders("val")

    def test_dataloader(self):
        return self._return_dataloaders("test")

    def _return_dataloaders(self, spl):
        return [
            self._fetch_dataloader(spl, conc_type)
            for conc_type in ["class", "attribute"]
        ]

    def _fetch_dataloader(self, spl, conc_type):
        return DataLoader(
            dataset=self.datasets[spl],
            batch_sampler=self.samplers[spl][conc_type],
            num_workers=self.cfg.vision.data.num_loader_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(data):
        """ Custom collate_fn to pass to dataloaders """        
        assert all(len(d)==2 for d in data)
        conc_type = data[0][1]

        assert all(isinstance(d[0], dict) for d in data)
        assert all(conc_type == d[1] for d in data)

        collated = {}
        for field in data[0][0]:
            assert all(field in d[0] for d in data)

            if field in ["image", "concept_labels"]:
                # Leave em as-is; 1) "image": PyTorch default collate function cannot
                # recognize PIL Images, 2) "concept_labels": string values
                collated[field] = [d[0][field] for d in data]
            elif field in ["instance_mask", "concept_masks"]:
                # Data may be of varying sizes
                collated[field] = [torch.tensor(d[0][field]) for d in data]
            else:
                # Otherwise, process with default collate fn
                collated[field] = default_collate([d[0][field] for d in data])

        return collated, conc_type


class _ChainedLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
    
    def __iter__(self):
        iters = [iter(dl) for dl in self.dataloaders]
        while True:
            for it in iters:
                yield next(it)


class _FewShot2DDataset(Dataset):
    def __init__(self, annotations, dataset_path, cache_path, dataset_name):
        super().__init__()
        self.annotations = annotations
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.name = dataset_name

    def __getitem__(self, idx):
        assert len(idx) == 5
        img_id, inst_id, conc_type, conc, all_concs = idx

        # Values to return: raw image or pre-computed image encodings + metadata,
        # segmentation mask for the specified instance, and ground-truth segmentation
        # binary map to predict as specified by conc
        data_dict = {}

        images_path = f"{self.dataset_path}/images"
        image_file = f"{images_path}/{img_id}.jpg"
        cached_encoding_file = f"{self.cache_path}/{self.name}_{img_id}.gz"

        if os.path.exists(cached_encoding_file):
            # Found pre-computed image encoding files
            with gzip.open(cached_encoding_file) as enc_f:
                processed_image = pickle.load(enc_f)
            
            data_dict["image_embedding"] = processed_image["embedding"][0]
            data_dict["original_sizes"] = np.array(processed_image["original_sizes"])
            data_dict["reshaped_input_sizes"] = np.array(processed_image["reshaped_input_sizes"])
            img_size = {
                "h": data_dict["original_sizes"][0],
                "w": data_dict["original_sizes"][1]
            }

        else:
            # Load raw image
            image_raw = Image.open(image_file)

            if image_raw.mode != "RGB":
                # Cast non-RGB images (e.g. grayscale) into RGB format
                old_image_raw = image_raw
                image_raw = Image.new("RGB", old_image_raw.size)
                image_raw.paste(old_image_raw)

            data_dict["image"] = image_raw
            img_size = { "h": image_raw.height, "w": image_raw.width }

        # Segmentation mask for the specified instance
        instance_mask = self.poly_to_mask(
            self.annotations[img_id][inst_id]["instance_mask"], img_size
        )
        data_dict["instance_mask"] = instance_mask

        if instance_mask.sum() > 0:
            # Additional training signals for augmenting each training data point with
            # two additional hybrid prompts: 1) support examples + bbox and 2) support
            # examples + (near)-centroid coordinate
            nz_inds_y, nz_inds_x = instance_mask.nonzero()

            # Axis-aligned bounding box from min/max indices for nonzero value in mask
            data_dict["instance_bbox"] = np.array([[
                nz_inds_x.min(), nz_inds_y.min(),
                nz_inds_x.max(), nz_inds_y.max()
            ]])

            # (Near-)Centroid, as a point in mask closest to 'center of mass'
            mass_center = (nz_inds_x.mean(), nz_inds_y.mean())
            distances_to_center = np.transpose([nz_inds_x, nz_inds_y])
            distances_to_center = np.linalg.norm(distances_to_center - mass_center, axis=1)
            centroid_ind = distances_to_center.argmin()
            data_dict["instance_point"] = np.array([[[
                nz_inds_x[centroid_ind], nz_inds_y[centroid_ind]
            ]]])
        else:
            # instance_mask.sum() == 0; segmentation mask too small that the decoded
            # binary mask doesn't have any nonzero entry... This item doesn't serve
            # as a valid training signal
            data_dict["instance_bbox"] = np.zeros((1, 4), dtype=np.int64)
            data_dict["instance_point"] = np.zeros((1, 1, 2), dtype=np.int64)

        # Ground-truth segmentation binary maps for any instances of the specified
        # 'focus' concept, 
        if conc_type == "class":
            all_conc_insts = [
                [
                    data for data in self.annotations[img_id].values()
                    if c == data["class"]
                ]
                for c in all_concs
            ]
        else:
            assert conc_type == "attribute"
            all_conc_insts = [
                [
                    data for data in self.annotations[img_id].values()
                    if c in data["attributes_pos"]
                ]
                for c in all_concs
            ]

        # Sort by area in descending order then pick top 3, as SAM model is set to predict
        # at most 3 'valid' segmentation masks for each (non-hybrid) prompt; we are sorting
        # by area in order to pick the most prominent instances if there are more than 3
        all_conc_inst_masks = [
            sorted([
                self.poly_to_mask(data["instance_mask"], img_size)
                for data in per_conc if data["instance_mask"] is not None
            ], key=np.sum, reverse=True)
            for per_conc in all_conc_insts
        ]

        # Only return top 3 masks with largest areas; if fewer than 3 masks, pad with
        # 'null' masks
        pad_mask = np.full(data_dict["original_sizes"], -1)
        data_dict["concept_masks"] = [
            per_conc + [pad_mask for _ in range(3-len(per_conc))]
                if len(per_conc) < 3
                else per_conc[:3]
            for per_conc in all_conc_inst_masks
        ]

        # Positive concept labels; needed for discerning positive pairs (within batch
        # and memory queue for LooK loss computation). Record as a binary tuple (c, C),
        # where c is the 'focus' concept label and C is the list of any other applicable
        # concept labels (C would be always empty for "class" batch)
        if conc_type == "class":
            data_dict["concept_labels"] = (conc, [])
        else:
            assert conc_type == "attribute"
            other_attrs = [
                attr for attr in self.annotations[img_id][inst_id]["attributes_pos"]
                if attr != conc
            ]
            data_dict["concept_labels"] = (conc, other_attrs)

        return data_dict, conc_type

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def poly_to_mask(multipoly, img_size):
        decoded = [
            # pycoco needs polygon vertex x/y-coordinates to be flattened into 1-dim
            [c for point in poly for c in point] for poly in multipoly
        ]
        decoded = mask.decode(mask.frPyObjects(decoded, img_size["h"], img_size["w"]))
        decoded = reduce(np.bitwise_or, [
            decoded[..., i] for i in range(decoded.shape[-1])
        ])      # Merging multiple regions into a single mask

        return decoded


class _FewShot2DDataSampler(Sampler):
    """
    Prepare batch sampler that returns few-shot 'episodes' consisting of K instances
    of N concepts (similar to typical N-way K-shot episodes for few-shot learning,
    but without support-query set distinction); each instance is passed as a pair
    of indices, namely (image id, instance id)
    """
    def __init__(self, metadata, conc_type, spl, batch_size, num_exs_per_conc):
        assert spl in ["train", "val", "test"]

        self.conc_type = conc_type
        self.spl = spl
        self.is_eval = spl != "train"
        self.batch_size = batch_size
        self.num_exs_per_conc = num_exs_per_conc
        self.num_concepts = self.batch_size // self.num_exs_per_conc

        # Extract instance indexing by concept for provided concept type and split
        # from metadata
        self.index_by_conc = {
            k: v for k, v in metadata[f"instances_{conc_type}"][spl].items()
            if len(v) >= self.num_exs_per_conc
        }

        # Let's keep things simple by making batch size divisible by number of
        # exemplars per concept (equivalently, number of concepts or 'ways')
        assert self.batch_size % self.num_exs_per_conc == 0

        # If is_eval, will terminate and can sample in advance; primarily for
        # consistency and getting total length info
        if self.is_eval:
            self.sampled_in_advance = list(self.sample())

    def __iter__(self):
        if self.is_eval:
            yield from self.sampled_in_advance
        else:
            yield from self.sample()

    def sample(self):
        # Concept-based sampling of instances for metric learning (by concept
        # instance classification task) and segmentation search

        # Shortcut abbreviations
        K, N = self.num_exs_per_conc, self.num_concepts

        # For maintaining lists of concepts & instances as sampling candidates
        index_by_conc = copy.deepcopy(self.index_by_conc)

        while True:
            # Sequentially sample K instances of N concepts
            candidate_concs = [
                conc for conc, insts in index_by_conc.items() if len(insts) >= K
            ]

            if len(candidate_concs) < N:
                # Exhausted, stop sampling and return
                assert self.is_eval     # Shouldn't happen when not eval
                return

            sampled_concs = random.sample(candidate_concs, N)
            sampled_insts = [
                random.sample(index_by_conc[conc], K) for conc in sampled_concs
            ]

            # Sampling success, yield the sampled concepts and instances
            batch = [
                (inst[0], inst[1], self.conc_type, conc, sampled_concs)
                    # sampled_concs included for reference, for Dataset.__get__ later
                for conc, insts in zip(sampled_concs, sampled_insts)
                for inst in insts
            ]
            yield batch

            if self.is_eval:
                # Sampling w/o replacement; remove the sampled instances from the sets
                for conc, insts in zip(sampled_concs, sampled_insts):
                    for inst in insts:
                        index_by_conc[conc].remove(inst)

    def __len__(self):
        if self.is_eval:
            return len(self.sampled_in_advance)
        else:
            return NotImplementedError
