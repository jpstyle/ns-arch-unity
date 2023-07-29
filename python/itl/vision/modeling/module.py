import os
import gzip
import pickle
from PIL import Image
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.optim import AdamW
from transformers import SamModel, SamProcessor
from transformers.activations import ACT2FN
from transformers.models.sam.modeling_sam import SamFeedForward, SamLayerNorm

from .process_data import (
    process_batch, preprocess_input, shape_guided_roi_align, compute_conc_embs
)
from ..utils import masks_bounding_boxes, flatten_cfg


class VisualSceneAnalyzer(pl.LightningModule):
    """
    Few-shot visual object detection (concept recognition & segmentation) model,
    implemented by attaching lightweight MLP blocks to pre-trained SAM (Meta's
    Segment Anything Model) model.
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        sam_model = self.cfg.vision.model.sam_model
        assets_dir = self.cfg.paths.assets_dir

        # Loading pre-trained SAM to use as basis model
        self.sam_processor = SamProcessor.from_pretrained(
            sam_model, cache_dir=os.path.join(assets_dir, "vision_models", "sam")
        )
        self.sam = SamModel.from_pretrained(
            sam_model, cache_dir=os.path.join(assets_dir, "vision_models", "sam")
        )

        D = self.sam.config.vision_config.output_channels

        # Feature extractors consisting of two levels of Conv2d layers (with LayerNorm
        # and activation in between), for obtaining embeddings specific to concept types
        # (class/attribute). To be applied after shape-guided RoIAlign outputs.
        self.kernel_size = 5; self.stride = 2
        self.roi_align_out = self.kernel_size + self.stride * (self.kernel_size-1)
        def conv2dExtractor():
            return nn.Sequential(
                nn.Conv2d(D, D, kernel_size=self.kernel_size, stride=self.stride),
                SamLayerNorm(D, data_format="channels_first"),
                ACT2FN[self.sam.config.vision_config.hidden_act],
                nn.Conv2d(D, D, kernel_size=self.kernel_size, stride=self.stride),
                SamLayerNorm(D, data_format="channels_first"),
                ACT2FN[self.sam.config.vision_config.hidden_act],
                nn.Flatten(),
                nn.Linear(D, D)
            )
        self.embed_cls = conv2dExtractor()
        self.embed_att = conv2dExtractor()
        # For feature-wise conditioning of RoI embeddings by class-centric embeddings,
        # needed for obtaining attribute-centric embeddings
        self.condition_cls_mult = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )
        self.condition_cls_add = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )

        # Maintaining consistent indexing by concept string name
        self.conc2ind = {
            "class": defaultdict(lambda: len(self.conc2ind["class"])),
            "attribute": defaultdict(lambda: len(self.conc2ind["attribute"]))
        }

        # MLP heads to encode sets of class/attibute concept exemplars into sparse
        # prompts for SAM mask decoding, and associated 'tag' embeddings
        self.exs_prompt_encode_cls = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )
        self.exs_prompt_encode_att = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )
        self.exs_prompt_tag_cls = nn.Embedding(1, D)
        self.exs_prompt_tag_att = nn.Embedding(1, D)

        if self.training:
            # Freeze all parameters except ones that need training
            self.to_train_prefixes = [
                "embed_cls.", "embed_att.",
                "exs_prompt_encode_cls.", "exs_prompt_encode_att.",
                "exs_prompt_tag_cls.", "exs_prompt_tag_att.",
                "condition_cls_mult.", "condition_cls_add.",
                "sam.mask_decoder."
            ]
            for name, param in self.named_parameters():
                param.requires_grad = any(
                    name.startswith(train_param)
                    for train_param in self.to_train_prefixes
                )
        else:
            self.to_train_prefixes = []

        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

        # Loss component weights
        self.loss_weights = { "nca": 2, "focal": 20, "dice": 1, "iou": 1 }

        flattened_cfg = flatten_cfg(OmegaConf.to_container(self.cfg, resolve=True))
        self.save_hyperparameters(flattened_cfg)

        # For caching image embeddings for __forward__ inputs
        self.processed_img_cached = None

    def training_step(self, batch, *_):
        losses, metrics = process_batch(self, batch)

        conc_type = batch[1]

        # Log loss values per type
        for name, val in losses.items():
            self.log(f"train_loss_{name}_{conc_type}", val)

        # Aggregate loss for the batch
        total_loss = sum(
            weight * losses[name] for name, weight in self.loss_weights.items()
            if name in losses
        )
        self.log(f"train_loss_{conc_type}", total_loss)

        # Log metric values per type
        for name, val in metrics.items():
            self.log(f"train_metric_{name}_{conc_type}", val)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, metrics = process_batch(self, batch)
        pred = loss, metrics, batch[1]
        self.validation_step_outputs[dataloader_idx].append(pred)
        return pred

    def on_validation_epoch_end(self):
        outputs = list(self.validation_step_outputs.values())

        avg_total_losses = []
        for outputs_per_dataloader in outputs:
            if len(outputs_per_dataloader) == 0:
                continue

            conc_type = outputs_per_dataloader[0][2]

            # Log epoch average loss
            avg_losses = defaultdict(list)
            for loss_type in outputs_per_dataloader[0][0]:
                for losses, _, _ in outputs_per_dataloader:
                    avg_losses[loss_type].append(losses[loss_type])

            for loss_type, vals in avg_losses.items():
                avg_val = sum(vals) / len(vals)
                avg_losses[loss_type] = avg_val
                self.log(
                    f"val_loss_{loss_type}_{conc_type}", avg_val,
                    add_dataloader_idx=False
                )

            avg_total_loss = sum(
                weight * avg_losses[name] for name, weight in self.loss_weights.items()
                if name in avg_losses
            )
            self.log(
                f"val_loss_{conc_type}", avg_total_loss.item(), add_dataloader_idx=False
            )
            avg_total_losses.append(avg_total_loss)

            # Log epoch average metrics
            avg_metrics = defaultdict(list)
            for metric_type in outputs_per_dataloader[0][1]:
                for _, metrics, _ in outputs_per_dataloader:
                    avg_metrics[metric_type].append(metrics[metric_type])

            for metric_type, vals in avg_metrics.items():
                avg_val = sum(vals) / len(vals)
                self.log(
                    f"val_metric_{metric_type}_{conc_type}", avg_val,
                    add_dataloader_idx=False
                )

        # Total validation loss
        final_avg_loss = sum(avg_total_losses) / (len(avg_total_losses) / len(outputs))
        self.log(f"val_loss", final_avg_loss, add_dataloader_idx=False)

    def test_step(self, batch, dataloader_idx):
        _, metrics = process_batch(self, batch)
        pred = metrics, batch[1]
        self.test_step_outputs[dataloader_idx].append(pred)
        return pred
    
    def test_epoch_end(self):
        outputs = list(self.test_step_outputs.values())

        for outputs_per_dataloader in outputs:
            if len(outputs_per_dataloader) == 0:
                continue

            conc_type = outputs_per_dataloader[0][1]

            # Log epoch average metrics
            avg_metrics = defaultdict(list)
            for metric_type in outputs_per_dataloader[0][0]:
                for metrics, _ in outputs_per_dataloader:
                    avg_metrics[metric_type].append(metrics[metric_type])
            
            for metric_type, vals in avg_metrics.items():
                avg_val = sum(vals) / len(vals)
                self.log(
                    f"test_{metric_type}_{conc_type}", avg_val, add_dataloader_idx=False
                )

    def configure_optimizers(self):
        # Populate optimizer configs
        optim_kwargs = {}
        if "init_lr" in self.cfg.vision.optim:
            optim_kwargs["lr"] = self.cfg.vision.optim.init_lr
        if "beta1_1m" in self.cfg.vision.optim and "beta2_1m" in self.cfg.vision.optim:
            optim_kwargs["betas"] = (
                1-self.cfg.vision.optim.beta1_1m, 1-self.cfg.vision.optim.beta2_1m
            )
        if "eps" in self.cfg.vision.optim:
            optim_kwargs["eps"] = self.cfg.vision.optim.eps

        # Construct optimizer instance; faster learning rates for mask decoder params
        default_params = {
            "params": [
                param for name, param in self.named_parameters()
                if not name.startswith("sam.mask_decoder.")
            ]
        }
        faster_params = {
            "params": [
                param for name, param in self.named_parameters()
                if name.startswith("sam.mask_decoder.")
            ],
            "lr": self.cfg.vision.optim.init_lr * 1.5
        }            
        optim = AdamW([default_params, faster_params], **optim_kwargs)

        # Populate LR scheduler configs
        sched_kwargs = {}
        if "lr_scheduler_milestones" in self.cfg.vision.optim:
            sched_kwargs["milestones"] = [
                int(s * self.cfg.vision.optim.max_steps)
                for s in self.cfg.vision.optim.lr_scheduler_milestones
            ]
        if "lr_scheduler_gamma" in self.cfg.vision.optim:
            sched_kwargs["gamma"] = self.cfg.vision.optim.lr_scheduler_gamma

        # Construct LR scheduler instance
        Scheduler = getattr(
            torch.optim.lr_scheduler, self.cfg.vision.optim.lr_scheduler
        )
        sched = Scheduler(optim, **sched_kwargs)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
        }

    def on_save_checkpoint(self, checkpoint):
        """
        No need to save weights for SAM image & propmt encoder components; del their
        weights to leave params for the newly added components only
        """
        state_dict_filtered = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("sam.vision_encoder."):
                continue
            if k.startswith("sam.prompt_encoder."):
                continue
            state_dict_filtered[k] = v
        checkpoint["state_dict"] = state_dict_filtered

    def forward(self, image, grid=None, masks=None):
        """
        Purposed as the most general endpoint of the vision module for inference,
        which takes an image as input and returns 'raw output' consisting of the
        following types of data for each recognized instance:
            1) a class-centric embedding
            2) an attribute-centric embedding
            3) an instance segmentation mask
            4) an objectness score (estimated IoU)
        for each object (candidate) detected.

        Can be optionally provided with a 2-d grid specification (as int-int pair),
        for prompt-free 'ensemble recognition'; essentially predictions on point
        prompts scattered as 2-d grid.

        Can be optionally provided with a list of additional segmentation masks
        as references to regions which are guaranteed to enclose an object each
        -- so that their concept identities can be classified.

        Either grid or masks should be provided.
        """
        assert grid is not None or masks is not None

        # Preprocessing input image & prompts
        img_provided = image is not None
        grid_provided = grid is not None

        input_data = {}
        if img_provided:
            orig_size = (image.width, image.height)
            input_data["image"] = image
        else:
            orig_size = self.processed_img_cached[1]
            input_data["original_sizes"] = torch.tensor([orig_size])

        if grid_provided:
            # Prepare prompts for grid (if provided)
            assert isinstance(grid, tuple)
            assert isinstance(grid[0], int) and isinstance(grid[1], int)
            y_count, x_count = grid
            input_data["instance_point"] = torch.tensor([[
                [[orig_size[0] / (x_count+1) * (i+1), orig_size[1] / (y_count+1) * (j+1)]]
                for i in range(x_count) for j in range(y_count)
            ]])

        processed_input = preprocess_input(self, input_data, img_provided)

        # Obtain image embedding computed by SAM image encoder; compute from scratch
        # if image input is new, fetch cached embs otherwise
        if img_provided:
            # Compute image embedding and cache for later use
            img_embs = self.sam.get_image_embeddings(processed_input["pixel_values"])
            reshaped_size = processed_input["reshaped_input_sizes"]
            self.processed_img_cached = (img_embs, orig_size, reshaped_size)
        else:
            # No image provided, used cached image embedding
            assert self.processed_img_cached is not None
            img_embs = self.processed_img_cached[0]
            reshaped_size = self.processed_img_cached[2]

        if grid_provided:
            # Run SAM to obtain segmentation mask estimations on grid point prompts
            
            # Prepare image-positional embeddings
            img_positional_embs = self.sam.get_image_wide_positional_embeddings()
            # No-mask embeddings; Dense embedding in SAM when mask prompts are not provided
            no_mask_dense_embs = self.sam.prompt_encoder.no_mask_embed.weight
            no_mask_dense_embs = no_mask_dense_embs[...,None,None].expand_as(img_embs)

            # Call prompt encoder to get grid point prompts encoding
            grid_tokens, _ = self.sam.prompt_encoder(
                processed_input["input_points"],
                processed_input["input_labels"],
                None,
                None
            )

            # Call mask decoder with the image & prompt encodings
            low_res_masks, iou_predictions, _ = self.sam.mask_decoder(
                image_embeddings=img_embs,
                image_positional_embeddings=img_positional_embs,
                sparse_prompt_embeddings=grid_tokens,
                dense_prompt_embeddings=no_mask_dense_embs,
                multimask_output=True
            )

            # Binarize the mask predictions
            high_res_masks = self.sam_processor.image_processor.post_process_masks(
                low_res_masks.cpu(),
                processed_input["original_sizes"].cpu(),
                processed_input["reshaped_input_sizes"].cpu()
            )

            # Flatten & rank predictions by scores
            preds_sorted = sorted([
                (mask_inst, score_inst.item())
                for masks_per_pt, scores_per_pt in zip(high_res_masks[0], iou_predictions[0])
                for mask_inst, score_inst in zip(masks_per_pt, scores_per_pt)
            ], key= lambda x: x[1], reverse=True)
            masks_all = [msk for msk, _ in preds_sorted]
            scores_all = [scr for _, scr in preds_sorted]
        else:
            # No predictions on grid points
            masks_all = []
            scores_all = []

        if masks is not None:
            # Add the provided masks along with the recognized masks to be processed
            # for [concept]-centric embeddings. Treat the provided masks as golden,
            # i.e. having scores of 1.0 and prepending to the mask/score lists
            masks_all += masks
            scores_all += [1.0] * len(masks)

        # Prepare tensor inputs for shape-guided RoIAlign, filtering out invalid ones
        # (i.e., masks with zero area)
        masks_tensor = torch.stack(masks_all)
        valid_masks = masks_tensor.sum(dim=(-2,-1)) > 0
        masks_tensor = masks_tensor[valid_masks]
        boxes_tensor = masks_bounding_boxes(masks_tensor.cpu().numpy())
        boxes_tensor = torch.tensor(boxes_tensor, dtype=torch.float)

        # Filter mask/score outputs accordingly
        masks_all = [masks_all[i] for i, is_valid in enumerate(valid_masks) if is_valid]
        scores_all = [scores_all[i] for i, is_valid in enumerate(valid_masks) if is_valid]

        # Shape-guided RoIAlign
        sg_roi_embs = shape_guided_roi_align(
            self, img_embs.expand(len(masks_all), -1, -1, -1),
            masks_tensor, boxes_tensor[:,None],
            [(orig_size[1], orig_size[0])] * len(masks_all)     # Needs flipping (H, W)
        )

        # Obtain class/attribute-centric feature vectors for the masks
        cls_embs, att_embs = compute_conc_embs(self, sg_roi_embs, "attribute")

        # Output tensors to numpy arrays
        cls_embs = cls_embs.cpu().numpy()
        att_embs = att_embs.cpu().numpy()
        masks_all = np.stack([msk.numpy() for msk in masks_all])

        return cls_embs, att_embs, masks_all, scores_all

    def search(self, image, conds_lists, k=None):
        """
        For few-shot search (exemplar-based conditioned detection). Given an image
        and a list of (concept type, exemplar vector set) pairs, find and return
        the top k **candidate** region proposals. The proposals do not intend to be
        highly accurate at this stage, as they will be further processed by the full
        model and tested again.
        """
        enc_out = detr_enc_outputs(self.detr, image, self.feature_extractor)
        _, _, outputs_coords, outputs_scores = \
            few_shot_search_img(self, enc_out, conds_lists)

        # Relative to absolute bbox dimensions, then center- to corner-format
        outputs_coords = torch.stack([
            outputs_coords[:,:,0] * image.width, outputs_coords[:,:,1] * image.height,
            outputs_coords[:,:,2] * image.width, outputs_coords[:,:,3] * image.height
        ], dim=-1)
        outputs_coords = box_convert(outputs_coords, "cxcywh", "xyxy")
        outputs_coords = clip_boxes_to_image(
            outputs_coords, (image.height, image.width)
        )

        if k is None:
            k = len(outputs_coords)
        topk_inds = outputs_scores.max(dim=-1).values.topk(k).indices

        return outputs_coords[topk_inds], outputs_scores[topk_inds]

    def cache_image_encodings(self):
        """
        Preprocess dataset images with the image encoder and store to local disk
        in advance, so that image encoding is not bottlenecked by the computationally
        demanding process (but rather than by file I/O, which can be mitigated by
        using multiple workers in dataloaders)
        """
        # Data input directory
        dataset_path = self.cfg.vision.data.path
        images_path = os.path.join(dataset_path, "images")

        # Embedding cache output directory
        cache_path = self.cfg.paths.cache_dir
        os.makedirs(cache_path, exist_ok=True)

        # Dataset name (e.g., 'vaw')
        d_name = self.cfg.vision.data.name

        # Need to move to cuda as this method is not handled by pytorch lightning Trainer
        if torch.cuda.is_available():
            self.cuda()

        # Process each image in images_path
        all_images = os.listdir(images_path)
        pbar = tqdm(all_images, total=len(all_images))
        pbar.set_description("Pre-computing image embs")

        for img in pbar:
            # Load raw image
            image_raw = Image.open(f"{images_path}/{img}")
            image_id = img.split(".")[0]

            # Load and preprocess raw image
            processed_input = self.sam_processor(
                image_raw,
                return_tensors="pt"
            ).to(self.device)

            # Obtain image embeddings from the raw pixel values
            with torch.no_grad():
                img_embs = self.sam.get_image_embeddings(processed_input["pixel_values"])
                img_embs = img_embs.cpu().numpy()

            # Save embedding (along with metadata like original size, reshaped size)
            # as compressed file
            with gzip.open(f"{cache_path}/{d_name}_{image_id}.gz", "wb") as enc_f:
                # Replace raw pixel values with the computed embeddings, then save
                processed_input["embedding"] = img_embs
                processed_input["original_sizes"] = \
                    processed_input["original_sizes"][0].tolist()
                processed_input["reshaped_input_sizes"] = \
                    processed_input["reshaped_input_sizes"][0].tolist()
                del processed_input["pixel_values"]
                pickle.dump(processed_input, enc_f)
