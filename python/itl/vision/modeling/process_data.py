"""
Helper methods factored out, which are needed for performing computations for two
tasks with 'opposite' directions:
    1) embedding (to be classified later) instances in input image given their
        references by segmentation
    2) searching instances of specified concept in input image given its support
        example set
using relevant modules in a VisualSceneAnalyzer instance (cf. module.py), which
include pre-trained SAM and newly added lightweight prediction heads.
"""
import gc

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import roi_align
from torchvision.transforms.functional import resize
from transformers.image_processing_utils import BatchFeature


EPS = 1e-8              # Value used for numerical stabilization

def process_batch(model, batch, segm_only):
    """
    Shared subroutine for processing batch to obtain loss & performance metric
    """
    batch_data, conc_type = batch

    conc_labels = batch_data["concept_labels"]

    # Shortcuts for batch size and number of examples per concept in batch
    B = model.cfg.vision.data.batch_size
    K = model.cfg.vision.data.num_exs_per_conc
    N = B // K

    # Dicts to store loss & metric values
    losses = {}
    metrics = {}

    ## Process images & instances to obtain embeddings that are used in both
    ## encoding & decoding direction computations

    if "image_embedding" in batch_data:
        # Passed pre-computed image encodings
        processed_input = preprocess_input(model, batch_data, False)
        img_embs = batch_data["image_embedding"]

    else:
        # Passed raw images, need to compute their encodings
        processed_input = preprocess_input(model, batch_data, True)

        # Obtain image embeddings from the raw pixel values (one-by-one, due to harsh
        # memory consumption of the image encoder)
        img_embs = []
        with torch.no_grad():
            for pixel_values in processed_input["pixel_values"]:
                img_embs.append(model.sam.get_image_embeddings(pixel_values[None]))
            gc.collect(); torch.cuda.empty_cache()      # Some stress relief for GPU
        img_embs = torch.cat(img_embs, dim=0)

    # If a box's dimensions are all zeros, it means it's an invalid entity (primarily
    # due to having a segmentation mask so small or invalid that its decoded binary
    # mask has no nonzero values); use this boolean flag array later to screen out
    # invalid values
    valid_entries = batch_data["instance_bbox"].sum(dim=(-2,-1)) > 0

    # roi_align needs consistent dtypes
    processed_input["input_boxes"] = processed_input["input_boxes"].to(img_embs.dtype)

    # Shape-guided RoIAlign
    sg_roi_embs = shape_guided_roi_align(
        model, img_embs,
        batch_data["instance_mask"],
        processed_input["input_boxes"],
        processed_input["original_sizes"]
    )

    # Embed the fixed-sized feature maps into 1-D [conc_type]-centric embeddings;
    # class-centric embeddings are needed for batches of both conc_types
    cls_embs, att_embs = compute_conc_embs(model, sg_roi_embs, conc_type)

    # Determine appropriate values and modules to use in computation by conc_type
    if conc_type == "class":
        inst_embs = cls_embs
        exs_prompt_encode = model.exs_prompt_encode_cls
        exs_prompt_tag = model.exs_prompt_tag_cls
    else:
        assert conc_type == "attribute"
        inst_embs = att_embs
        exs_prompt_encode = model.exs_prompt_encode_att
        exs_prompt_tag = model.exs_prompt_tag_att

    ## Encoding-direction (i.e., <Image, Instance => Concepts>)
    if not segm_only:
        # Instance label sets for the batch, with string names mapped to int indices
        inst_labels = [
            [model.conc2ind[conc_type][conc] for conc in [fc]+aux]
            for fc, aux in conc_labels
        ]
        # Pad labels for entries by current maximum label count (cf. for VAW, will
        # be consistent for classes with only one label per entry, but may differ
        # for attributes)
        max_label_count = max(len(ls) for ls in inst_labels)
        inst_labels = torch.tensor(
            [ls+[-1]*(max_label_count-len(ls)) for ls in inst_labels],
            dtype=torch.long, device=model.device
        )

        # Screen out labels for invalid entries by replacing with 'null' values
        inst_labels[~valid_entries] = -1

        # Obtain pairwise distances (squared) between examples
        dists_sq = torch.cdist(inst_embs, inst_embs, p=2.0) ** 2

        # Replace the diagonals with inf so as not to consider reflexive distances
        # in NCA loss
        dists_sq.fill_diagonal_(float("inf"))

        # Positive pair mask needed for NCA loss computation; the more concept labels
        # agree, the higher weights such pairs receive
        nonpad_intersections = [
            [torch.cat([labs_i[labs_i!=-1], labs_j[labs_j!=-1]]) for labs_j in inst_labels]
            for labs_i in inst_labels
        ]
        nonpad_intersections = torch.tensor([
            [(pair.unique(return_counts=True)[1] > 1).sum() for pair in per_entry]
            for per_entry in nonpad_intersections
        ], device=model.device)
        pos_mask = nonpad_intersections > 0

        # Compute NCA loss adjusted for multilabel scenarios: i.e., softmax weight in
        # a way such that entries sharing more labels are pulled stronger
        exps = (-dists_sq + dists_sq.min(dim=-1).values).exp()
        exp_weights = torch.max(nonpad_intersections, torch.ones_like(nonpad_intersections))
        weighted_exps_all = exps * exp_weights
        weighted_exps_pos = exps * nonpad_intersections
        losses["nca"] = weighted_exps_pos.sum(dim=-1) / weighted_exps_all.sum(dim=-1)
        losses["nca"] = -(losses["nca"] + EPS).log()
        losses["nca"] = losses["nca"][valid_entries].mean()

        # Compute precision@K metric
        topk_closest_exs = dists_sq.topk(K-1, largest=False, dim=-1).indices
        metrics["precision@K"] = torch.stack([
            pm[inds].float().mean()
            for pm, inds in zip(pos_mask, topk_closest_exs)
        ])[valid_entries].mean()

    ## Decoding-direction (i.e., <Image, Concept => Instances>)

    # Mask decoding to obtain segmentations; three runs for each image with
    # different types of prompts; one with support examples only, one with
    # support + centroid point, and one with support + bounding box

    # Prepare image-positional embeddings
    img_positional_embs = model.sam.get_image_wide_positional_embeddings()
    img_positional_embs = img_positional_embs.repeat(B, 1, 1, 1)

    # Obtain two types of prototypes: 1) leave-one-out averages of (valid) support
    # example vectors and 2) normal averages of (valid) vectors. 1) for predictions
    # without self as support example, 2 for normal predictions.
    inst_embs_per_conc = inst_embs.view(N, K, -1)
    valid_entries_per_conc = valid_entries.view(N, K)
    # 1) Leave-one-out prototypes
    loo_prototypes = [
        per_conc[valid_entries_per_conc[i]].sum(dim=0, keepdims=True) - per_conc
        for i, per_conc in enumerate(inst_embs_per_conc)
    ]
    loo_prototypes = [
        per_conc / (K-1-(~valid_entries_per_conc[i]).sum().item())
        for i, per_conc in enumerate(loo_prototypes)
    ]
    loo_prototypes = torch.cat(loo_prototypes)          # Shape (B, D)
    # 2) Normal prototypes
    nrm_prototypes = [
        per_conc[valid_entries_per_conc[i]].sum(dim=0, keepdims=True)
        for i, per_conc in enumerate(inst_embs_per_conc)
    ]
    nrm_prototypes = [
        per_conc / (K-(~valid_entries_per_conc[i]).sum().item())
        for i, per_conc in enumerate(nrm_prototypes)
    ]
    nrm_prototypes = torch.cat(nrm_prototypes)          # Shape (N, D)

    # Encode the prototypes into decoder prompt tokens
    loo_proto_tokens = exs_prompt_encode(loo_prototypes) + exs_prompt_tag.weight
    nrm_proto_tokens = exs_prompt_encode(nrm_prototypes) + exs_prompt_tag.weight
    all_proto_tokens = torch.stack([
        torch.stack([
            # Use leave-one-out for the 'focus' concept, normal ones for else
            loo_proto_tokens[i] if i // K == j else nrm_proto_tokens[j]
            for j in range(N)
        ])
        for i in range(B)
    ])
    all_proto_tokens = all_proto_tokens[:,:,None]       # Shape (B, N, 1, D)

    # No-mask embeddings; Dense embedding in SAM when mask prompts are not provided
    no_mask_dense_embs = model.sam.prompt_encoder.no_mask_embed.weight
    no_mask_dense_embs = no_mask_dense_embs[...,None,None].expand_as(img_embs)

    # Ground truth masks resized (again) to match decoder output sizes
    # (4 for low_size corresponds to two 2x deconvolutions in mask decoder)
    patch_size = model.sam.config.vision_config.patch_size
    resize_size = model.sam.config.vision_config.image_size
    low_size = resize_size // patch_size * 4
    pad_target_sizes = [max(msk.shape) for msk in batch_data["instance_mask"]]
    pad_specs = [
        (0,0,0,tgt_size-org_size[0].item())
            if org_size[0] < tgt_size else (0,tgt_size-org_size[1].item())
        for tgt_size, org_size in zip(pad_target_sizes, processed_input["original_sizes"])
    ]
    # Ground truth from batch_data["concept_masks"], for multimask prediction
    gt_masks_conc = [
        F.pad(masks, p_spec, value=0)
        for masks, p_spec in zip(batch_data["concept_masks"], pad_specs)
    ]
    gt_masks_conc = torch.stack([resize(masks, low_size) for masks in gt_masks_conc])
    # Ground truth from batch_data["instance_masks"], for single-mask prediction
    gt_masks_inst = [
        F.pad(mask, p_spec, value=0)
        for mask, p_spec in zip(batch_data["instance_mask"], pad_specs)
    ]
    gt_masks_inst = torch.cat([resize(masks[None], low_size) for masks in gt_masks_inst])

    # Binary masks corresponding to image paddings, for screening out loss values
    # at padded pixels
    img_pad_masks = torch.stack([
        torch.arange(low_size)[:,None].expand(low_size,low_size) >= h.item()
            if h<w else torch.arange(low_size)[None].expand(low_size,low_size) >= w.item()
        for h, w in (processed_input["reshaped_input_sizes"] // 4)
    ]).to(model.device)

    # Binary masks corresponding to valid ground-truth paddings, i.e. for addressing
    # variable numbers of valid mask predictions per image
    valid_pad_masks = gt_masks_conc[:,:,:,0,0] != -1

    # Needs casting into float for computations below
    gt_masks_conc = gt_masks_conc.to(torch.float)

    # The mask decoder call, invoked repeatedly, is quite bulky; factor out as lambda
    def mask_decode(prompt_tokens, multimask):
        low_res_masks, iou_predictions, _ = model.sam.mask_decoder(
            image_embeddings=img_embs,
            image_positional_embeddings=img_positional_embs,
            sparse_prompt_embeddings=prompt_tokens,
            dense_prompt_embeddings=no_mask_dense_embs,
            multimask_output=multimask
        )
        return low_res_masks, iou_predictions

    # Also factor out segmentation task loss & metric computation logic
    def segmentation_loss_metric_update(prediction, multimask):
        pred_masks, pred_ious = prediction
        B = pred_masks.shape[0]; N = pred_masks.shape[1]

        if multimask:
            num_msks = model.sam.config.mask_decoder_config.num_multimask_outputs

            # Compute pairwise dice coefficients between predictions vs. ground truths
            dice_coeffs = _pairwise_dice(pred_masks, gt_masks_conc, img_pad_masks)

            # Hungarian matching between predictions vs. valid ground-truths based on
            # the pairwise dice coefficients computed, per batch entry * concept, to
            # find most compatible guesses for each valid ground truth
            best_matching = [
                linear_sum_assignment(coeffs[:, masks].detach().cpu().numpy(), maximize=True)
                for coeffs, masks in zip(dice_coeffs.view(B*N,3,3), valid_pad_masks.view(B*N,3))
            ]
            best_matching_cmpl = [
                np.array(list(set(range(num_msks))-set(match[0])), dtype=np.int64)
                for match in best_matching
            ]

            # Collect & rearrange predictions and ground truths according to the matches
            pred_masks_matched = [
                pred_masks[i//N,i%N,match[0]] for i, match in enumerate(best_matching)
            ]
            gt_masks_matched = [
                gt_masks_conc[i//N,i%N,match[1]].to(torch.float32)
                for i, match in enumerate(best_matching)
            ]
            pred_ious_matched = [       # And padded for non-instances
                torch.cat([pred_ious[i//N,i%N,match[0]], pred_ious[i//N,i%N,cmpl]])
                for i, (match, cmpl) in enumerate(zip(best_matching, best_matching_cmpl))
            ]
            # Gather (possibly duplicate) copies of image pad masks as needed
            img_pad_masks_matched = torch.stack([
                img_pad_masks[i//N]
                for i, match in enumerate(best_matching) for _ in match[0]
            ])

            # Fetch appropriate dice coefficients to be cast into dice loss
            # Compute pairwise dice coefficients between predictions vs. ground truths
            dice_coeffs_matched = torch.cat([
                dice_coeffs[i//N,i%N,match[0],match[1]]
                for i, match in enumerate(best_matching)
            ])

            # And don't forget the valid entry mask
            valid_entries_matched = torch.stack([
                valid_entries[i//N]
                for i, match in enumerate(best_matching) for _ in match[0]
            ])
        else:
            num_msks = 1

            # Compute pairwise dice coefficients between predictions vs. ground truths
            # (though we are not dealing with multiple possible pred-gt pairs here, as
            # we only make one mask prediction against one specific ground-truth mask)
            dice_coeffs = _pairwise_dice(pred_masks, gt_masks_inst[:,None,None], img_pad_masks)

            # Compute dice coefficients between the sole predictions vs. first ground
            # truths, per batch entry

            # 'Focus' concepts always have match, others always don't
            pred_masks_matched = [
                per_conc if j==i//K else per_conc[:0]
                for i, masks in enumerate(pred_masks)
                for j, per_conc in enumerate(masks)
            ]
            gt_masks_matched = [
                masks[None] if j==i//K else masks[None][:0]
                for i, masks in enumerate(gt_masks_inst.to(torch.float32))
                for j in range(N)
            ]
            pred_ious_matched = list(pred_ious.view(-1))
            img_pad_masks_matched = img_pad_masks

            # All the dice coefficients computed will be cast into dice loss; slide in
            # empty tensors for consistent processing
            dice_coeffs_matched = torch.cat([
                per_conc[0] if j==i//K else per_conc[0,:0]
                for i, coeffs in enumerate(dice_coeffs)
                for j, per_conc in enumerate(coeffs)
            ])

            # Ditto
            valid_entries_matched = valid_entries

        # Compute dice & focal loss values, on valid pred-gt mask pairs
        dice_values = dice_coeffs_matched[valid_entries_matched]
        dice_loss = (-dice_values.log()).mean()
        dice_metric = dice_values.mean()

        focal_loss = _sigmoid_focal_loss(
            torch.cat(pred_masks_matched),
            torch.cat(gt_masks_matched),
            img_pad_masks_matched
        )
        focal_loss = focal_loss[valid_entries_matched].mean()

        # Compute IoUs between predictions vs. ground truths, cast into both loss and
        # metric values
        with torch.no_grad():
            pred_probs = [masks.sigmoid() for masks in pred_masks_matched]
            soft_intersections = [
                torch.min(prd, gt).sum(dim=(-2,-1))
                for prd, gt in zip(pred_probs, gt_masks_matched)
            ]
            soft_unions = [
                torch.max(prd, gt).sum(dim=(-2,-1))
                for prd, gt in zip(pred_probs, gt_masks_matched)
            ]
            gt_ious = [
                its/unn for its, unn in zip(soft_intersections, soft_unions)
            ]
            gt_ious_padded = [
                F.pad(ious, (0,num_msks-len(ious)), value=0) for ious in gt_ious
            ]

        iou_loss = F.mse_loss(
            torch.stack(pred_ious_matched), torch.stack(gt_ious_padded)
        )
        iou_metric = torch.cat(gt_ious)[valid_entries_matched].mean()

        # Update losses & metrics
        if multimask:
            # Twice the importance
            losses["focal"] += focal_loss * 3
            losses["dice"] += dice_loss * 3
            losses["iou"] += iou_loss * 3
            metrics["dice"] += dice_metric * 3
            metrics["iou"] += iou_metric * 3
        else:
            losses["focal"] += focal_loss
            losses["dice"] += dice_loss
            losses["iou"] += iou_loss
            metrics["dice"] += dice_metric
            metrics["iou"] += iou_metric

    # To be updated by increments
    losses["focal"] = 0; losses["dice"] = 0; losses["iou"] = 0
    metrics["dice"] = 0; metrics["iou"] = 0

    # First prompt: Support examples only, predict at most 3 masks if applicable
    segm_preds_1 = mask_decode(all_proto_tokens, True)
    segmentation_loss_metric_update(segm_preds_1, True)

    # Second prompt: Support + centroid (hybrid), predict 1 mask
    centroid_tokens, _ = model.sam.prompt_encoder(
        processed_input["input_points"],
        processed_input["input_labels"],
        None,
        None
    )
    centroid_tokens = centroid_tokens.expand(-1, N, -1, -1)
    hybrid_tokens_2 = torch.cat([all_proto_tokens, centroid_tokens], dim=2)
    segm_preds_2 = mask_decode(hybrid_tokens_2, False)
    segmentation_loss_metric_update(segm_preds_2, False)

    # Third prompt: Support + bounding box (hybrid), predict 1 mask
    bbox_tokens, _ = model.sam.prompt_encoder(
        None,
        None,
        processed_input["input_boxes"],
        None
    )
    bbox_tokens = bbox_tokens.expand(-1, N, -1, -1)
    hybrid_tokens_3 = torch.cat([all_proto_tokens, bbox_tokens], dim=2)
    segm_preds_3 = mask_decode(hybrid_tokens_3, False)
    segmentation_loss_metric_update(segm_preds_3, False)

    # Divide the segmentation loss & metric values by three (one for each prompt)
    # to obtain average
    losses["focal"] /= 5; losses["dice"] /= 5; losses["iou"] /= 5
    metrics["dice"] /= 5; metrics["iou"] /= 5

    return losses, metrics


def preprocess_input(model, batch_data, img_prompt_together):
    """ Input preprocessing logic factored out and exposed """
    # Massage input prompts into appropriate formats
    if "instance_point" in batch_data:
        B = batch_data["instance_point"].shape[0]
        N_O = batch_data["instance_point"].shape[1]
        N_P = batch_data["instance_point"].shape[2]
        input_points = batch_data["instance_point"].cpu()
        input_labels = torch.ones(B, N_O, N_P, dtype=torch.long)
    else:
        input_points = input_labels = None
    
    if "instance_bbox" in batch_data:
        input_boxes = batch_data["instance_bbox"].cpu()
    else:
        input_boxes = None

    if img_prompt_together:
        # Passed raw images, need to compute their encodings
        assert "image" in batch_data

        processed_input = model.sam_processor(
            batch_data["image"],
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(model.device)

    else:    
        # Passed pre-computed image encodings
        assert "original_sizes" in batch_data

        # Run part of SamProcessor.__call__ method except image processing with
        # provided original size info
        processed_points = model.sam_processor._check_and_preprocess_points(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )

        base_features = {
            "original_sizes": batch_data["original_sizes"]
        }
        if "reshaped_input_sizes" in batch_data:
            base_features["reshaped_input_sizes"] = batch_data["reshaped_input_sizes"]
        processed_input = model.sam_processor._normalize_and_convert(
            BatchFeature(base_features),
            batch_data["original_sizes"].cpu().numpy(),
            input_points=processed_points[0],
            input_labels=processed_points[1],
            input_boxes=processed_points[2],
            return_tensors="pt",
        ).to(model.device)

    return processed_input


def shape_guided_roi_align(model, img_embs, masks, boxes, orig_sizes):
    """
    Shape-guided RoIAlign; amplify feature map areas corresponding to the masks,
    then run roi_align to obtain fixed-sized feature maps for each instance
    (cf. [Deeply Shape-guided Cascade for Instance Segmentation], Ding et al. 2021)
    """
    # Sync devices and to list
    masks = [msk.to(img_embs.device) for msk in masks]
    boxes = [box.to(img_embs.device) for box in boxes]

    # Pad and resize instance masks to match the downsampled feature map size,
    # needed for shape-guided RoIAlign
    pad_target_sizes = [max(msk.shape) for msk in masks]
    pad_specs = [
        (0, 0, 0, tgt_size-org_size[0])
            if org_size[0] < tgt_size else (0, tgt_size-org_size[1])
        for tgt_size, org_size in zip(pad_target_sizes, orig_sizes)
    ]
    masks_resized = [
        F.pad(msk, p_spec, value=0).to(torch.float)
        for msk, p_spec in zip(masks, pad_specs)
    ]
    masks_resized = torch.stack([
        resize(msk[None], img_emb.shape[1:])
        for msk, img_emb in zip(masks_resized, img_embs)
    ])

    # RoI-align on feature maps where features in regions covered by the provided
    # masks are further amplified
    bg_mult = 0.1
    img_embs_amplified = img_embs * bg_mult + img_embs * masks_resized
    patch_size = model.sam.config.vision_config.patch_size
    resize_size = model.sam.config.vision_config.image_size
    sg_roi_embs = roi_align(
        img_embs_amplified, boxes,
        output_size=model.roi_align_out, spatial_scale=patch_size/resize_size
    )

    return sg_roi_embs


def compute_conc_embs(model, sg_roi_embs, conc_type):
    """ [conc_type]-specific embedding computation logic factored out and exposed """
    # Compute class-centric embeddings
    cls_embs = model.embed_cls(sg_roi_embs)

    # Compute attribute-centric embeddings (as needed)
    if conc_type == "class":
        att_embs = None
    else:
        assert conc_type == "attribute"

        # Obtain conditioned RoI embeddings
        modulator_multiplicative = model.condition_cls_mult(cls_embs).sigmoid()
        modulator_additive = model.condition_cls_add(cls_embs)
        sg_roi_embs_conditioned = sg_roi_embs * modulator_multiplicative[...,None,None]
        sg_roi_embs_conditioned = sg_roi_embs_conditioned + modulator_additive[...,None,None]

        # Compute attribute-centric embeddings
        att_embs = model.embed_att(sg_roi_embs_conditioned)

    return cls_embs, att_embs


def _sigmoid_focal_loss(inputs, targets, img_pad_masks, alpha=0.25, gamma=2):
    """ Taken & modified from transformers.models.detr.modeling_detr """
    # Logits to probs
    probs = inputs.sigmoid()

    # Compute per-pixel focal loss values
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Screen out loss values at padded pixels by zeroing out
    loss[img_pad_masks] = 0

    # Take sum and divide by numbers of pixels to obtain per-entry average losses,
    # then take mean across batch to obtain final loss
    focal_loss = loss.sum(dim=(-2,-1)) / (~img_pad_masks).sum(dim=(-2,-1))

    return focal_loss


def _pairwise_dice(inputs, targets, img_pad_masks):
    """
    Compute pairwise DICE coefficients between predicted masks vs. ground-truth masks;
    used for Hungarian matching, and also as loss value
    """
    # Logits to probs
    probs = inputs.sigmoid()

    # Numerators, with img_pad_masks applied
    numerators = probs[:,:,:,None] * targets[:,:,None]
    numerators[img_pad_masks[:,None,None,None].expand_as(numerators)] = 0
    numerators = 2 * numerators.sum(dim=(-2,-1))

    # Denominators, with img_pad_masks applied
    denominators = probs[:,:,:,None] + targets[:,:,None]
    denominators[img_pad_masks[:,None,None,None].expand_as(denominators)] = 0
    denominators = denominators.sum(dim=(-2,-1))

    # Compute dice coefficients (smoothing by 1 in num and denom)
    dice_coeffs = (numerators+1) / (denominators+1)

    # Output shape: batch_size * num_multimask_outputs * num_ground_truth
    return dice_coeffs
