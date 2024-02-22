import os
import gzip
import pickle
from PIL import Image
from itertools import product
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import ImageFilter
from scipy.optimize import linear_sum_assignment
from skimage.morphology import opening, closing, dilation
from skimage.measure import label
from sklearn.cluster import KMeans
from torch.optim import AdamW
from torchvision.ops import batched_nms, masks_to_boxes, box_convert
from transformers import AutoImageProcessor, Dinov2Model, SamProcessor, SamModel

from .process_data import process_batch, preprocess_input
from ..utils import flatten_cfg


BLUR_RADIUS = 5                 # Gaussian blur kernel radius for background image

class VisualSceneAnalyzer(pl.LightningModule):
    """
    Few-shot visual object detection (concept recognition & segmentation) model
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        dino_model = self.cfg.vision.model.dino_model
        sam_model = self.cfg.vision.model.sam_model
        assets_dir = self.cfg.paths.assets_dir

        # Loading pre-trained models to use as basis encoder & segmentation models
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_model, cache_dir=os.path.join(assets_dir, "vision_models", "dino"),
            do_resize=False, do_center_crop=False
        )
        self.dino = Dinov2Model.from_pretrained(
            dino_model, cache_dir=os.path.join(assets_dir, "vision_models", "dino")
        )
        self.sam_processor = SamProcessor.from_pretrained(
            sam_model, cache_dir=os.path.join(assets_dir, "vision_models", "sam")
        )
        self.sam = SamModel.from_pretrained(
            sam_model, cache_dir=os.path.join(assets_dir, "vision_models", "sam")
        )

        # if "task" in self.cfg.vision:
        #     if self.training:
        #         # Freeze all parameters except those that need training
        #         if self.cfg.vision.task == "rgb":
        #             self.to_train_prefixes = [
        #                 "embed_cls.", "embed_att.",
        #                 "exs_prompt_encode_cls.", "exs_prompt_encode_att.",
        #                 "exs_prompt_tag_cls.", "exs_prompt_tag_att.",
        #                 "condition_cls_mult.", "condition_cls_add.",
        #                 "sam.mask_decoder."
        #             ]
        #         elif self.cfg.vision.task == "rgb_segm_only":
        #             self.to_train_prefixes = [
        #                 "exs_prompt_encode_cls.", "exs_prompt_encode_att.",
        #                 "exs_prompt_tag_cls.", "exs_prompt_tag_att.",
        #                 "sam.mask_decoder."
        #             ]
        #         else:
        #             raise NotImplementedError

        #         for name, param in self.named_parameters():
        #             param.requires_grad = any(
        #                 name.startswith(train_param)
        #                 for train_param in self.to_train_prefixes
        #             )
        #     else:
        #         self.to_train_prefixes = []

        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

        flattened_cfg = flatten_cfg(OmegaConf.to_container(self.cfg, resolve=True))
        self.save_hyperparameters(flattened_cfg)

        # For caching image embeddings for __forward__ inputs
        self.processed_img_cached = None

    def training_step(self, batch, *_):
        segm_only = self.cfg.vision.task == "rgb_segm_only"
        losses, metrics = process_batch(self, batch, segm_only)

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
        segm_only = self.cfg.vision.task == "rgb_segm_only"
        loss, metrics = process_batch(self, batch, segm_only)
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
        segm_only = self.cfg.vision.task == "rgb_segm_only"
        _, metrics = process_batch(self, batch, segm_only)
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
        No need to save weights for pretrained components; del their weights to
        leave params for the newly added components only
        """
        # state_dict_filtered = OrderedDict()
        # for k, v in checkpoint["state_dict"].items():
        #     if k.startswith("sam.vision_encoder."):
        #         continue
        #     if k.startswith("sam.prompt_encoder."):
        #         continue
        #     state_dict_filtered[k] = v
        # checkpoint["state_dict"] = state_dict_filtered
        pass

    def forward(self, image, masks=None, search_conds=None):
        """
        Purposed as the most general endpoint of the vision module for inference,
        which takes an image as input and returns 'raw output' consisting of the
        following types of data for each recognized instance:
            1) a visual feature embedding
            2) an instance segmentation mask
            3) a 'confidence' score, as to how certain the module is about the
               quality of the mask returned 
        for each object (candidate) detected.

        Can be optionally provided with a list of additional segmentation masks
        as references to objects, in which case visual embeddings corresponding to
        each mask will be returned without segmentation prediction.

        Can be optionally provided with a list of (exemplar vector set, binary
        concept classifier) pair set, where each set represents a disjunction of
        concepts, and each concept is represented by its positive exemplar vectors
        and a binary classifier. (In that regard, `search_conds` can be seen to be in
        'conjunctive normal form'.) In this case, conditioned segmentation prediction
        will be performed first and object instances are recognized as connected
        components. Then visual embeddings are extracted, and the confidence scores
        are computed with the provided binary classifiers.

        If neither are provided, run ensemble prediction instead; i.e., first run
        segmentation prediction conditioned with the general description of "an
        object", recognize object instances as connected components, then extract
        visual embeddings.
        """
        # Preprocessing input image & prompts
        img_provided = image is not None
        masks_provided = masks is not None
        search_conds_provided = search_conds is not None

        assert not (masks_provided and search_conds_provided), \
            "Cannot provide both masks and search conditions"

        # Obtain image embedding computed by pre-trained basis models; compute from
        # scratch if image input is new, fetch cached features otherwise
        if img_provided:
            # Process image and pass through vision encoder

            orig_size = (image.width, image.height)     # Self-explanatory

            # Processing for image encoder
            dino_processed_input = self.dino_processor(images=image, return_tensors="pt")
            pixel_values = dino_processed_input.pixel_values.to(self.device)
            dino_processed_input = self.dino(
                pixel_values=pixel_values, return_dict=True
            )
            dino_embs = dino_processed_input.last_hidden_state[:,1:]

            # Prepare grid prompt for ensemble prediction
            w_g = 3; h_g = 2
            grid_points = [
                [[orig_size[0] * i/(w_g+1), orig_size[1] * j/(h_g+1)]]
                for i, j in product(range(1,w_g+1), range(1,h_g+1))
            ]

            # Processing for promptable segmenter
            sam_processed_input = self.sam_processor(
                images=image, input_points=grid_points, return_tensors="pt"
            )
            pixel_values = sam_processed_input.pixel_values.to(self.device)
            sam_grid_prompts = sam_processed_input.input_points.to(self.device)
            sam_reshaped_size = sam_processed_input.reshaped_input_sizes.to(self.device)
            sam_embs = self.sam.get_image_embeddings(
                pixel_values=pixel_values, return_dict=True
            )

            # Background image prepared via greyscale + gaussian blur
            bg_image = image.convert("L").convert("RGB")
            bg_image = bg_image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))

            # Cache image and processing results
            self.processed_img_cached = (
                image, bg_image, dino_embs, sam_embs, sam_grid_prompts,
                orig_size, sam_reshaped_size
            )
        else:
            # No image provided, used cached data
            assert self.processed_img_cached is not None
            image = self.processed_img_cached[0]
            bg_image = self.processed_img_cached[1]
            sam_embs = self.processed_img_cached[3]
            sam_grid_prompts = self.processed_img_cached[4]
            orig_size = self.processed_img_cached[5]
            sam_reshaped_size = self.processed_img_cached[6]

        # Obtain masks in different manners according to the provided info
        if masks_provided:
            # Simply use the provided masks
            masks_all = masks

        elif search_conds_provided:
            # Condition with the provided search condition; run mask decoder per 'conjunct'
            # and then find intersection to obtain masks
            masks_all = [np.ones((image.height, image.width))]
            for conjunct in search_conds:
                masks_cnjt = []
                for pos_exs_info, _ in conjunct:
                    masks_cnjt += self._conditioned_segment(pos_exs_info)

                # Find every possible intersection between the returned instances vs. the
                # intersection outputs accumulated so far, obtained from each binary cross
                # product; take logical_and for masks and minimums for scores
                masks_int = [msk_a * msk_c for msk_a, msk_c in product(masks_all, masks_cnjt)]

            masks_all = masks_int

        else:
            # We don't really need ensemble prediction in our experiments though... Let's
            # skip this for a while
            masks_all = []

            # # Condition with the grid of points prepared above to obtain a batch of mask
            # # proposals, then postprocess by NMS to remove any duplicates
            # w_o, h_o = orig_size
            
            # # Run SAM on grid
            # sam_preds = self.sam(
            #     image_embeddings=sam_embs.expand(len(sam_grid_prompts),-1,-1,-1),
            #     input_points=sam_grid_prompts
            # )
            # sam_masks = self.sam_processor.image_processor.post_process_masks(
            #     masks=sam_preds.pred_masks.cpu(),
            #     original_sizes=[(h_o, w_o)]*len(sam_preds.pred_masks),
            #     reshaped_input_sizes=sam_reshaped_size.expand(len(sam_preds.pred_masks), -1)
            # )
            # sam_masks = torch.stack(sam_masks).view(-1, h_o, w_o)
            # valid_inds = torch.stack([msk.sum() > 0 for msk in sam_masks])
            # sam_masks = sam_masks[valid_inds]

            # # Collect masks and keep the largest chunks among the disconnected bits
            # sam_masks = [label(msk.numpy(), return_num=True) for msk in sam_masks]
            # sam_masks = [
            #     max([msk==i+1 for i in range(num_chunks)], key=lambda x: x.sum())
            #     for msk, num_chunks in sam_masks
            # ]
            # sam_masks = torch.stack([torch.tensor(msk) for msk in sam_masks])

            # # Run NMS to remove duplicates with worse qualities
            # sam_boxes = masks_to_boxes(sam_masks)
            # sam_scores = sam_preds.iou_scores.cpu().view(-1)[valid_inds]
            # kept_inds = batched_nms(
            #     sam_boxes, sam_scores, torch.zeros(sam_boxes.shape[0]), 0.5
            # )
            # masks_all = [opening(closing(msk.numpy())) for msk in sam_masks[kept_inds]]

        # Filter out invalid (empty or extremely small) masks
        masks_all = [msk.astype(bool) for msk in masks_all if msk.sum() > 500]

        # If we have valid masks, obtain visual feature embeddings corresponding to each
        # mask instance by applying a series of 'visual prompt engineering' process, and 
        # assing through CLIP visual transformer; if none found, can return early
        if len(masks_all) > 0:
            # Non-empty list of segmentation masks
            masks_all = np.stack(masks_all)
            visual_prompts = self._visual_prompt_by_mask(image, bg_image, masks_all)

            # Pass through vision encoder to obtain visual embeddings corresponding to each mask
            vis_embs = []
            for vp in visual_prompts:
                vp_processed = self.dino_processor(images=vp, return_tensors="pt")
                vp_pixel_values = vp_processed.pixel_values.to(self.device)
                vp_dino_out = self.dino(pixel_values=vp_pixel_values, return_dict=True)
                vis_embs.append(vp_dino_out.pooler_output.cpu().numpy())
            vis_embs = np.concatenate(vis_embs)
        else:
            # Empty list of segmentation masks; occurs when search returned zero matches
            vis_embs = np.zeros((0, self.dino.config.hidden_size), dtype="float32")
            masks_all = np.zeros((0, image.height, image.width), dtype="float32")
            scores_all = np.zeros((0,), dtype="float32")

            return vis_embs, masks_all, scores_all

        # Obtain the confidence scores, again in different manners according to the info
        # provided
        if masks_provided:
            # Ground truths provided by external source (i.e., user), treat the provided
            # masks as golden, i.e. having scores of 1.0
            scores_all = [1.0] * len(masks)

        elif search_conds_provided:
            # Assess the quality of the masks found using the provided binary concept
            # classifiers
            scores_all = []
            for emb in vis_embs:
                agg_score = 1.0     # Start from 1.0 and keep taking t-norms (min here)
                for conjunct in search_conds:
                    # Obtain t-conorms (max here) across disjunct concepts in the conjunct
                    conjunct_score = max(
                        bin_clf.predict_proba(emb[None])[0][1].item() \
                            if bin_clf is not None else 0.5
                        for _, bin_clf in conjunct
                    )
                    # Obtain t-norm between the conjunct score and aggregate score
                    agg_score = min(conjunct_score, agg_score)

                # Append final score for the embedding
                scores_all.append(agg_score)

        else:
            # See above for skipping ensemble prediction
            scores_all = []

            # # For ensemble prediction with the general text prompt, use the maximum float
            # # values in each mask as confidence score
            # scores_all = sam_scores[kept_inds].numpy().tolist()

        return vis_embs, masks_all, scores_all

    def _conditioned_segment(self, exemplar_prompts):
        """
        Conditional image segmentation factored out for readability; condition with
        exemplar prompts to obtain segmentation. Return found masks along with scores.
        """
        assert self.processed_img_cached is not None
        image = self.processed_img_cached[0]
        dino_embs = self.processed_img_cached[2]
        sam_embs = self.processed_img_cached[3]
        orig_size = self.processed_img_cached[5]

        # Obtain cropped images and masks
        cropped_images, cropped_masks = _crop_exemplars_by_masks(exemplar_prompts)

        # Matching between cropped exemplar (reference) images w/ masks vs. current
        # scene view, select low resolution patches
        matched_patches = _patch_matching(
            cropped_images, cropped_masks, orig_size, self.device,
            self.dino, self.dino_processor, dino_embs, self.dino.config
        )

        # Obtain segmentation input prompts, run segmentation model and postprocess
        final_masks = _prompted_segmentation(
            matched_patches, image, orig_size, self.device,
            self.sam, self.sam_processor, sam_embs, self.dino.config
        )

        return final_masks

    def _visual_prompt_by_mask(self, image, bg_image, masks):
        """
        'Visual prompt engineering' (cf. CLIPSeg paper) factored out for readability
        """
        # Obtain visual prompts to process by mixing image & bg_image per each mask...
        images_mixed = [
            (image * msk[:,:,None] + bg_image * (1-msk[:,:,None])).astype("uint8")
            for msk in masks
        ]

        # ... then cropping with some context & square pad as needed
        visual_prompts = []
        for i, msk in enumerate(masks):
            nonzero_y, nonzero_x = msk.nonzero()
            x1 = nonzero_x.min(); x2 = nonzero_x.max(); w = x2-x1
            y1 = nonzero_y.min(); y2 = nonzero_y.max(); h = y2-y1

            pad_ratio = 1/16          # Relative size to one side
            if w >= h:
                w_pad = int(w*pad_ratio)
                target_size = w + 2*w_pad
                h_pad = (target_size-h) // 2
                # w_pad = int(w*pad_ratio)
                # h_pad = w_pad
            else:
                h_pad = int(h*pad_ratio)
                target_size = h + 2*h_pad
                w_pad = (target_size-w) // 2
                # h_pad = int(h*pad_ratio)
                # w_pad = h_pad

            x1_crp = max(0, x1-w_pad); x2_crp = min(image.width, x2+w_pad)
            y1_crp = max(0, y1-h_pad); y2_crp = min(image.height, y2+h_pad)
            pad_spec = (
                -min(0, x1-w_pad), max(image.width, x2+w_pad) - image.width,
                -min(0, y1-h_pad), max(image.height, y2+h_pad) - image.height
            )

            # Draw contour as guided by mask
            contour = dilation(dilation(msk)) & ~msk
            images_mixed[i][contour] = [255,0,0]

            cropped = torch.tensor(images_mixed[i][y1_crp:y2_crp, x1_crp:x2_crp])
            cropped = F.pad(cropped.permute(2,0,1), pad_spec)
            visual_prompts.append(cropped)
        
        return visual_prompts

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


def _crop_exemplars_by_masks(exemplar_prompts):
    """
    Helper method factored out for cropping original scene images & masks so
    that they are centered
    """
    # Process exemplar scene image + mask info to obtain square-cropped images
    # with center focus on the exemplars. Don't make the cropping boxes too tight
    # so that enough visual contexts are included; about twice the max(width, height)
    # w.r.t. the bounding box for each exemplar?
    scene_imgs_wh = torch.tensor(
        [scene_img.size for scene_img, _, _ in exemplar_prompts]
    )
    ex_boxes_xyxy = masks_to_boxes(
        torch.tensor(np.stack([msk for _, msk, _ in exemplar_prompts]))
    ).to(torch.long)
    ex_boxes_cwh = box_convert(ex_boxes_xyxy, "xyxy", "cxcywh").to(torch.long)
    ex_max_dims = ex_boxes_cwh[:,2:].max(dim=1).values[:,None]
    ex_crops_xyxy = torch.cat([
        (ex_boxes_cwh[:,:2] - ex_max_dims),
        (ex_boxes_cwh[:,:2] + ex_max_dims)
    ], dim=1)
    ex_crops_xyxy_clamped = torch.cat([
        ex_crops_xyxy[:,:2].clamp(min=0),
        ex_crops_xyxy[:,2:].clamp(max=scene_imgs_wh)
    ], dim=1)
    clamp_diffs = ex_crops_xyxy_clamped - ex_crops_xyxy

    # Cropping images and masks; images are easier thanks to PIL crop method,
    # but masks take more work :/
    cropped_images = [
        scene_img.crop(box.tolist())
        for (scene_img, _, _), box in zip(exemplar_prompts, ex_crops_xyxy)
    ]
    cropped_masks = [np.zeros((d*2,d*2)) for d in ex_max_dims]
    for i in range(len(cropped_masks)):
        src_mask = exemplar_prompts[i][1]
        cr_w, cr_h = ex_crops_xyxy[i][2:] - ex_crops_xyxy[i][:2]
        src_slice_w = slice(
            int(ex_crops_xyxy_clamped[i][0]), int(ex_crops_xyxy_clamped[i][2])
        )
        src_slice_h = slice(
            int(ex_crops_xyxy_clamped[i][1]), int(ex_crops_xyxy_clamped[i][3])
        )
        tgt_slice_w = slice(int(clamp_diffs[i][0]), int(cr_w+clamp_diffs[i][2]))
        tgt_slice_h = slice(int(clamp_diffs[i][1]), int(cr_h+clamp_diffs[i][3]))
        cropped_masks[i][tgt_slice_h, tgt_slice_w] = src_mask[src_slice_h, src_slice_w]

    return cropped_images, cropped_masks



def _patch_matching(
        cropped_images, cropped_masks, orig_size, device,
        dino_model, dino_processor, dino_embs, dino_config
    ):
    """
    Helper method factored out for 'patch-level feature matching (cf. Matcher
    by Liu et al., 2024)'
    """
    w_d = orig_size[0] // dino_config.patch_size

    # Process the cropped exemplar images with the image encoder module to obtain
    # patch-level embeddings
    lr_dim = 32; resize_target = lr_dim * dino_config.patch_size
    cropped_images_processed = dino_processor.preprocess(
        images=cropped_images,
        do_resize=True, size={ "shortest_edge": resize_target }, return_tensors="pt"
    )
    ex_dino_out = dino_model(
        cropped_images_processed.pixel_values.to(device), return_dict=True
    )
    ex_patch_embs = ex_dino_out.last_hidden_state[:,1:]

    # Compute cosine similarities between patches, exemplars vs. current scene view
    dino_embs_nrm = F.normalize(dino_embs.reshape(-1, dino_config.hidden_size))
    exs_embs_nrm = F.normalize(ex_patch_embs.reshape(-1, dino_config.hidden_size))
    S = (exs_embs_nrm @ dino_embs_nrm.t()).cpu()

    # Flatten exemplar masks to 1 dimension
    ex_masks_flattened = [
        cv2.resize(msk, (lr_dim, lr_dim), interpolation=cv2.INTER_NEAREST_EXACT)
        for msk in cropped_masks
    ]
    ex_masks_flattened = np.concatenate([
        msk_resized.reshape(-1).astype(bool)
        for msk_resized in ex_masks_flattened
    ])

    # Forward matching
    match_forward = linear_sum_assignment(S[ex_masks_flattened], maximize=True)
    # Reverse matching
    match_reverse = linear_sum_assignment(S.t()[match_forward[1]], maximize=True)
    # Mask filtering
    retain_inds = np.isin(match_reverse[1], ex_masks_flattened.nonzero()[0])
    # Fetch and unflatten matched patch indices
    matched_patches = np.stack([
        match_forward[1][retain_inds] % w_d, match_forward[1][retain_inds] // w_d
    ], axis=1)

    return matched_patches


def _prompted_segmentation(
        matched_patches, image, orig_size, device,
        sam_model, sam_processor, sam_embs, dino_config
    ):
    """
    Helper method factored out for running segmentation model; obtain point & box
    prompt inputs based on the matched patches provided
    """
    if len(matched_patches) < 4: return []

    w_o, h_o = orig_size
    w_d = w_o // dino_config.patch_size
    h_d = h_o // dino_config.patch_size

    # Obtain segmentation input prompts; point prompts by clustering, box prompts
    # by finding bounding box.
    cluster_means = KMeans(n_clusters=4).fit(matched_patches).cluster_centers_
    cluster_means = cluster_means * dino_config.patch_size
    point_prompts = cluster_means.round().astype(int).tolist()
        # Mapping back to original input resolution
    lr_msk = torch.zeros((1, h_d, w_d))
    lr_msk[0][(matched_patches[:,1], matched_patches[:,0])] = 1
    box_prompts = masks_to_boxes(lr_msk)
    box_prompts = (box_prompts.numpy() * dino_config.patch_size).tolist()

    # Process the input prompts and run the segmentation model
    input_processed = sam_processor(
        image, input_points=[point_prompts], input_boxes=[box_prompts]
    )
    sam_out = sam_model(
        image_embeddings=sam_embs,
        input_points=torch.tensor(input_processed.input_points[None]).to(device),
        input_boxes=torch.tensor(input_processed.input_boxes).to(device),
        multimask_output=False
    )

    # Postprocessing to obtain the final refined mask in original resolution space;
    # upscale, morphological processing, then separate disconnected masks
    mask_upscaled = sam_processor.image_processor.post_process_masks(
        sam_out.pred_masks.cpu(),
        input_processed.original_sizes,
        input_processed.reshaped_input_sizes
    )[0][0][0].numpy()
    mask_upscaled = opening(closing(mask_upscaled))
    chunks, count = label(mask_upscaled, return_num=True)
    final_masks = [chunks==i+1 for i in range(count)]
    final_masks = [msk for msk in final_masks if msk.sum() > 500]
        # Size thresholding by some arbitrary area criterion

    return final_masks
