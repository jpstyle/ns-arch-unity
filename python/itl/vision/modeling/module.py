import os
import bz2
import pickle
from PIL import Image
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from torch.optim import AdamW
from torchvision.ops import box_convert, clip_boxes_to_image
from transformers import SamModel, SamProcessor
from transformers.activations import ACT2FN
from transformers.models.sam.modeling_sam import SamFeedForward, SamLayerNorm

from .process_batch import process_batch


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
        # For feature-wise gating of RoI embeddings by class-centric embeddings,
        # needed for obtaining attribute-centric embeddings; essentially conditioning
        # by object class identity
        # (cf. [Learning to Predict Visual Attributes in the Wild], Pham et al. 2021)
        self.gate_compute = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )

        # Momentum encoder for maintaining memory keys computed from stably updating
        # encoder; clone the base encoders' architecture and copy weights
        self.momentum = 0.99
        self.embed_cls_momentum = conv2dExtractor()
        self.embed_att_momentum = conv2dExtractor()
        self.gate_compute_momentum = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )
        self.base_mmt_pairs = [
            (self.embed_cls, self.embed_cls_momentum),
            (self.embed_att, self.embed_att_momentum),
            (self.gate_compute, self.gate_compute_momentum)
        ]
        for base_module, mmt_module in self.base_mmt_pairs:
            for base_param, mmt_param in zip(
                base_module.parameters(), mmt_module.parameters()
            ):
                mmt_param.data.copy_(base_param.data)

        # Additional MLP modules for predicting momentum encoder outputs
        self.mmt_predict_cls = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )
        self.mmt_predict_att = SamFeedForward(
            input_dim=D, hidden_dim=D, output_dim=D, num_layers=2
        )

        # Create memory queue for computing LooK loss
        M = 65536 - self.cfg.vision.data.batch_size
        self.register_buffer("queue_embs_cls", torch.randn(M, D))
        self.register_buffer("queue_embs_att", torch.randn(M, D))
        self.register_buffer("queue_labels_cls", torch.full((M, 1), -1))
        self.register_buffer("queue_labels_att", torch.full((M, 1), -1))
        self.conc2ind = {
            "class": defaultdict(lambda: len(self.conc2ind["class"])),
            "attribute": defaultdict(lambda: len(self.conc2ind["attribute"]))
        }

        # For annealing the number of neighbors to consider when computing LooK loss
        self.num_neighbors_range = (400, 40)

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
            self.to_train = [
                "embed_cls.", "embed_att.",
                "exs_prompt_encode_cls.", "exs_prompt_encode_att.",
                "exs_prompt_tag_cls.", "exs_prompt_tag_att.",
                "mmt_predict_cls.", "mmt_predict_att.",
                "sam.mask_decoder."
            ]
            for name, param in self.named_parameters():
                param.requires_grad = any(
                    name.startswith(train_param)
                    for train_param in self.to_train
                )
        else:
            self.to_train = []
        
        # Loss component weights
        self.loss_weights = { "look": 2, "focal": 20, "dice": 1, "iou": 1 }

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, *_):
        losses, metrics = process_batch(self, batch, batch_idx)

        conc_type = batch[1]

        # Log loss values per type
        for name, val in losses.items():
            self.log(f"train_loss_{name}_{conc_type}", val)

        # Aggregate loss for the batch
        total_loss = sum(
            weight * losses[name] for name, weight in self.loss_weights.items()
        )
        self.log(f"train_loss_{conc_type}", total_loss)

        # Log metric values per type
        for name, val in metrics.items():
            self.log(f"train_metric_{name}_{conc_type}", val)

        return total_loss

    def validation_step(self, batch, batch_idx, *_):
        loss, metrics = process_batch(self, batch, batch_idx)
        return loss, metrics, batch[1]

    def validation_epoch_end(self, outputs):
        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]

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

    def test_step(self, batch, batch_idx, *_):
        _, metrics = process_batch(self, batch, batch_idx)
        return metrics, batch[1]
    
    def test_epoch_end(self, outputs):
        if len(self.trainer.test_dataloaders) == 1:
            outputs = [outputs]

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

        # Construct optimizer instance
        optim = AdamW(self.parameters(), **optim_kwargs)

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

    def forward(self, image, bboxes=None, lock_provided_boxes=True):
        """
        Purposed as the most general endpoint of the vision module for inference,
        which takes an image as input and returns 'raw output' consisting of the
        following types of data:
            1) a class-centric embedding (projection from DETR decoder output)
            2) an attribute-centric embedding (projection from DETR decoder output)
            3) a bounding box (from DETR decoder)
        for each object (candidate) detected.
        
        Can be optionally provided with a list of additional bounding boxes as
        references to regions which are guaranteed to enclose an object each --
        so that their concept identities can be classified.
        """
        # Corner- to center-format, then absolute to relative bbox dimensions
        if bboxes is not None:
            bboxes = box_convert(bboxes, "xywh", "cxcywh")
            bboxes = torch.stack([
                bboxes[:,0] / image.width, bboxes[:,1] / image.height,
                bboxes[:,2] / image.width, bboxes[:,3] / image.height,
            ], dim=-1)
        else:
            bboxes = torch.tensor([]).view(0, 4).to(self.device)

        encoder_outputs_all = detr_enc_outputs(
            self.detr, image, self.feature_extractor
        )
        encoder_outputs, valid_ratios, spatial_shapes, \
            level_start_index, mask_flatten = encoder_outputs_all

        decoder_outputs, last_reference_points, enc_objectness_scores = \
            detr_dec_outputs(
                self.detr, encoder_outputs, bboxes, lock_provided_boxes,
                valid_ratios, spatial_shapes, level_start_index, mask_flatten
            )

        # Class/attribute-centric feature vectors
        cls_embeddings = self.fs_embed_cls(decoder_outputs[0])
        att_embeddings = self.fs_embed_att(
            torch.cat([decoder_outputs[0], cls_embeddings], dim=-1)
        )

        # Obtain final bbox estimates
        dec_last_layer_ind = self.detr.config.decoder_layers
        last_bbox_embed = self.detr.bbox_embed[dec_last_layer_ind-1]
        delta_bbox = last_bbox_embed(decoder_outputs[0])

        final_bboxes = delta_bbox + torch.logit(last_reference_points[0])
        final_bboxes = final_bboxes.sigmoid()
        final_bboxes = torch.cat([bboxes, final_bboxes[bboxes.shape[0]:]])

        # Relative to absolute bbox dimensions, then center- to corner-format
        final_bboxes = torch.stack([
            final_bboxes[:,0] * image.width, final_bboxes[:,1] * image.height,
            final_bboxes[:,2] * image.width, final_bboxes[:,3] * image.height,
        ], dim=-1)
        final_bboxes = box_convert(final_bboxes, "cxcywh", "xywh")

        return cls_embeddings, att_embeddings, final_bboxes, enc_objectness_scores[0]

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

        # Process each image in images_path
        all_images = os.listdir(images_path)
        pbar = tqdm(all_images, total=len(all_images))
        pbar.set_description("Pre-computing image embs")

        # Need to move to cuda as this method is not handled by pytorch lightning Trainer
        if torch.cuda.is_available():
            self.cuda()

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
                img_emb = self.sam.get_image_embeddings(processed_input["pixel_values"])
                img_emb = img_emb.cpu().numpy()

            # Save embedding (along with metadata like original size, reshaped size)
            # as compressed file
            with bz2.BZ2File(f"{cache_path}/{d_name}_{image_id}.pbz2", "wb") as enc_f:
                # Replace raw pixel values with the computed embeddings, then save
                processed_input["embedding"] = img_emb
                processed_input["original_sizes"] = \
                    processed_input["original_sizes"][0].tolist()
                processed_input["reshaped_input_sizes"] = \
                    processed_input["reshaped_input_sizes"][0].tolist()
                del processed_input["pixel_values"]
                pickle.dump(processed_input, enc_f)
