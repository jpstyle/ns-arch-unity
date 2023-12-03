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
from PIL import ImageFilter, ImageEnhance
from skimage.morphology import opening, closing
from skimage.measure import label
from sklearn.cluster import HDBSCAN
from torch.optim import AdamW
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from .process_data import process_batch, preprocess_input
from ..utils import flatten_cfg


BLUR_RADIUS = 5                 # Gaussian blur kernel radius for background image
INTENSITY_RATIO = 0.3           # Brightness multiplier for background image
MASK_THRES = 0.6                # Segmentation mask binarization threshold

class VisualSceneAnalyzer(pl.LightningModule):
    """
    Few-shot visual object detection (concept recognition & segmentation) model,
    implemented by attaching lightweight MLP blocks to pre-trained SAM (Meta's
    Segment Anything Model) model.
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        clipseg_model = self.cfg.vision.model.clipseg_model
        assets_dir = self.cfg.paths.assets_dir

        # Loading pre-trained CLIPSeg to use as basis model
        self.clipseg_processor = CLIPSegProcessor.from_pretrained(
            clipseg_model, cache_dir=os.path.join(assets_dir, "vision_models", "clipseg")
        )
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(
            clipseg_model, cache_dir=os.path.join(assets_dir, "vision_models", "clipseg")
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
        No need to save weights for SAM image & propmt encoder components; del their
        weights to leave params for the newly added components only
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

        # Obtain image embedding computed by CLIP visual transformer; compute from scratch
        # if image input is new, fetch cached features otherwise
        if img_provided:
            # Process image and pass through vision encoder
            image_processed = self.clipseg_processor(images=image, return_tensors="pt")
            image_processed = image_processed["pixel_values"].to(self.device)
            vision_outputs = self.clipseg.clip.vision_model(
                pixel_values=image_processed,
                output_hidden_states=True,
                return_dict=True
            )

            # Extract 'activation' features needed for the CLIPSeg decoder
            hidden_states = vision_outputs.hidden_states
            activations = [hidden_states[i + 1] for i in self.clipseg.config.extract_layers]
            orig_size = (image.width, image.height)

            # Background image prepared via gaussian blur + decreased intensity
            bg_image = image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
            bg_image = ImageEnhance.Brightness(bg_image).enhance(INTENSITY_RATIO)

            # Cache image and processing results
            self.processed_img_cached = (image, bg_image, activations, orig_size)
        else:
            # No image provided, used cached data
            assert self.processed_img_cached is not None
            image = self.processed_img_cached[0]
            bg_image = self.processed_img_cached[1]

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
                for pos_exs_vecs, _ in conjunct:
                    # Run HDBSCAN clustering to obtain a set of prototypes, to be used
                    # as exemplar prompts to CLIPSeg. Taking the global mean as a single
                    # prototype is highly likely to misrepresent the concept since we
                    # don't have any guarantee that the distribution of embeddings is
                    # unimodal. Note that we would rather 'over-cluster' than 'under-
                    # cluster' by the same logic.
                    hdb = HDBSCAN(min_cluster_size=2, store_centers="centroid")
                    clusters = hdb.fit_predict(pos_exs_vecs)

                    # Cluster centroids + 'noise' points as prototypes
                    prototypes = np.concatenate([hdb.centroids_, pos_exs_vecs[clusters==-1]])
                    prototypes = torch.tensor(prototypes, dtype=torch.float, device=self.device)
                    masks_cnjt += self._conditioned_segment(exemplar_prompts=prototypes)

                # Find every possible intersection between the returned instances vs. the
                # intersection outputs accumulated so far, obtained from each binary cross
                # product; take logical_and for masks and minimums for scores
                masks_int = [msk_a * msk_c for msk_a, msk_c in product(masks_all, masks_cnjt)]

                # Filter out invalid (empty or extremely small) masks
                valid_inds = [i for i, msk in enumerate(masks_int) if msk.sum() > 200]
                masks_all = [masks_int[i] for i in valid_inds]

        else:
            # Condition with a general text prompt ("an object") and recognize instances
            # by connected component analysis to obtain masks
            masks_all = self._conditioned_segment(text_prompts=["an object"])

        # If we have valid masks, obtain visual feature embeddings corresponding to each
        # mask instance by applying a series of 'visual prompt engineering' process, and 
        # assing through CLIP visual transformer; if none found, can return early
        if len(masks_all) > 0:
            # Non-empty list of segmentation masks
            visual_prompts = self._visual_prompt_by_mask(image, bg_image, masks_all)

            # Pass through vision encoder to obtain visual embeddings corresponding to each mask
            visual_prompts_processed = self.clipseg_processor(
                images=visual_prompts, return_tensors="pt"
            )
            visual_prompts_processed = visual_prompts_processed["pixel_values"].to(self.device)
            vis_embs = self.clipseg.clip.get_image_features(visual_prompts_processed)
        else:
            # Empty list of segmentation masks; occurs when search returned zero matches
            vis_embs = np.zeros((0, self.clipseg.config.projection_dim), dtype="float32")
            masks_all = np.zeros((0, image.height, image.width), dtype="float32")
            scores_all = np.zeros((0,), dtype="float32")

            return vis_embs, masks_all, scores_all

        # Output tensors to numpy arrays
        vis_embs = vis_embs.cpu().numpy()
        masks_all = np.stack(masks_all)

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
                        bin_clf.predict_proba(emb[None])[0][1].item()
                        for _, bin_clf in conjunct
                    )
                    # Obtain t-norm between the conjunct score and aggregate score
                    agg_score = min(conjunct_score, agg_score)

                # Append final score for the embedding
                scores_all.append(agg_score)

        else:
            # For ensemble prediction with the general text prompt, use the maximum float
            # values in each mask as confidence score
            scores_all = [msk.max().item() for msk in masks_all]

        return vis_embs, masks_all, scores_all

    def _conditioned_segment(self, text_prompts=None, exemplar_prompts=None):
        """
        Conditional image segmentation factored out for readability and reusability;
        condition with either a text prompt or exemplar feature set prompt to obtain
        segmentation. Return found masks along with scores.
        """
        condition_on_text = text_prompts is not None
        condition_on_exemplar = exemplar_prompts is not None

        assert self.processed_img_cached is not None
        activations = self.processed_img_cached[2]
        orig_size = self.processed_img_cached[3]

        if condition_on_text:
            # Process the conditioning text prompt
            text_processed = [
                self.clipseg_processor(text=txt_prt, return_tensors="pt")
                for txt_prt in text_prompts
            ]
            text_processed = [
                txt_prt_processed["input_ids"].to(self.device)
                for txt_prt_processed in text_processed
            ]
            cond_embs = torch.cat([
                self.clipseg.clip.get_text_features(txt_prt_processed)
                for txt_prt_processed in text_processed
            ])
        else:
            assert condition_on_exemplar
            cond_embs = exemplar_prompts

        # Expand activations according to the number of prompts provided
        activations = [actv.expand(cond_embs.shape[0], -1, -1) for actv in activations]

        # Call mask decoder with the prompt encoding
        decoder_outputs = self.clipseg.decoder(activations, cond_embs, return_dict=True)
        decoder_outputs = torch.sigmoid(decoder_outputs.logits)
        if len(decoder_outputs.shape) == 2:
            decoder_outputs = decoder_outputs[None]

        # Binarize, and some morphological processing to clean noises
        low_res_masks = decoder_outputs.cpu().numpy() > MASK_THRES
        low_res_masks = [opening(closing(lr_msk)) for lr_msk in low_res_masks]

        # Split the processed decoder output into a collection of masks for individual
        # instances by connected component analysis
        low_res_masks = [label(lr_msk, return_num=True) for lr_msk in low_res_masks]
        low_res_masks = [
            lr_msk==i+1 for lr_msk, num_insts in low_res_masks for i in range(num_insts)
        ]           # Split by connected components and flatten, ignoring empty masks

        # Obtain masks resized to original image size, and 'objectness scores' for each
        # instance by its max score per mask in original decoder output
        masks = [
            cv2.resize(
                lr_msk.astype("float32"), orig_size, interpolation=cv2.INTER_NEAREST_EXACT
            )
            for lr_msk in low_res_masks
        ]

        return masks

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

            if w >= h:
                x1_crp = max(0, x1-w//4); x2_crp = min(image.width, x2+w//4)
                y1_crp = max(0, y1-w*3//4+h//2); y2_crp = min(image.height, y2+w*3//4-h//2)
                pad_spec = (
                    -min(0, x1-w//4), max(image.width, x2+w//4) - image.width,
                    -min(0, y1-w*3//4+h//2), max(image.height, y2+w*3//4-h//2) - image.height
                )
            else:
                x1_crp = max(0, x1-h*3//4+w//2); x2_crp = min(image.width, x2+h*3//4-w//2)
                y1_crp = max(0, y1-h//4); y2_crp = min(image.height, y2+h//4)
                pad_spec = (
                    -min(0, x1-h*3//4+w//2), max(image.width, x2+h*3//4-w//2) - image.width,
                    -min(0, y1-h//4), max(image.height, y2+h//4) - image.height
                )

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
