"""
Helper methods factored out, which perform computations required for two tasks
with 'opposite' directions:
    1) embedding (to be classified later) instances in input image given their
        references by segmentation
    2) searching instances of specified concept in input image given its support
        example set
using relevant modules in a VisualSceneAnalyzer instance (cf. module.py), which
include pre-trained SAM and newly added lightweight prediction heads.
"""
import gc

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import roi_align
from torchvision.transforms.functional import resize
from transformers.image_processing_utils import BatchFeature


def process_batch(model, batch, batch_idx):
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
        assert "original_sizes" in batch_data
        assert "reshaped_input_sizes" in batch_data

        # Run part of SamProcessor.__call__ method except image processing with
        # provided original size info
        processed_points = model.sam_processor._check_and_preprocess_points(
            input_points=batch_data["instance_centroid"].cpu(),
            input_labels=torch.ones(B, dtype=torch.long),
            input_boxes=batch_data["instance_bbox"].cpu(),
        )

        processed_input = model.sam_processor._normalize_and_convert(
            BatchFeature({
                "original_sizes": batch_data["original_sizes"],
                "reshaped_input_sizes": batch_data["reshaped_input_sizes"]
            }),
            batch_data["original_sizes"].cpu().numpy(),
            input_points=processed_points[0],
            input_labels=processed_points[1],
            input_boxes=processed_points[2],
            return_tensors="pt",
        ).to(model.device)

        img_embs = batch_data["image_embedding"]

    else:
        # Passed raw images, need to compute their encodings
        assert "image" in batch_data

        # Preprocess input images and points & box annotations
        processed_input = model.sam_processor(
            batch_data["image"],
            input_points=batch_data["instance_centroid"].cpu(),
            input_labels=torch.ones(B, dtype=torch.long),
            input_boxes=batch_data["instance_bbox"].cpu(),
            return_tensors="pt"
        ).to(model.device)

        # Obtain image embeddings from the raw pixel values (one-by-one, due to harsh
        # memory consumption of the image encoder)
        img_embs = []
        with torch.no_grad():
            for pixel_values in processed_input["pixel_values"]:
                img_embs.append(model.sam.get_image_embeddings(pixel_values[None]))
            gc.collect(); torch.cuda.empty_cache()      # Some stress relief for GPU
        img_embs = torch.cat(img_embs, dim=0)

    # roi_align needs consistent dtypes
    processed_input["input_boxes"] = processed_input["input_boxes"].to(img_embs.dtype)

    # Pad and resize instance masks to match the downsampled feature map size,
    # needed for shape-guided RoIAlign
    pad_target_sizes = [max(mask.shape) for mask in batch_data["instance_mask"]]
    pad_specs = [
        (0,0,0,tgt_size-org_size[0].item())
            if org_size[0] < tgt_size else (0,tgt_size-org_size[1].item())
        for tgt_size, org_size in zip(
            pad_target_sizes, processed_input["original_sizes"]
        )
    ]
    masks_resized = [
        F.pad(mask, p_spec, value=0).to(torch.float)
        for mask, p_spec in zip(batch_data["instance_mask"], pad_specs)
    ]
    masks_resized = torch.stack([
        resize(mask[None], img_emb.shape[1:])
        for mask, img_emb in zip(masks_resized, img_embs)
    ])

    # Shape-guided RoIAlign; amplify feature map areas corresponding to the masks,
    # then run roi_align to obtain fixed-sized feature maps for each instance
    # (cf. [Deeply Shape-guided Cascade for Instance Segmentation], Ding et al. 2021)
    img_embs_amplified = img_embs * (1 + masks_resized)
    patch_size = model.sam.config.vision_config.patch_size
    resize_size = model.sam.config.vision_config.image_size
    sg_roi_embs = roi_align(
        img_embs_amplified, list(processed_input["input_boxes"]),
        output_size=model.roi_align_out,
        spatial_scale=patch_size / resize_size,
    )

    # Embed the fixed-sized feature maps into 1-D [conc_type]-centric embeddings;
    # class-centric embeddings are needed for batches of both conc_types
    cls_embs, att_embs = _compute_conc_embs(model, sg_roi_embs, conc_type)

    # Momentum encoder processing under no_grad
    with torch.no_grad():
        if model.training:
            # Update momentum encoder weights, by mixing with ratio defined by momentum
            for base_module, mmt_module in model.base_mmt_pairs:
                for base_param, mmt_param in zip(
                    base_module.parameters(), mmt_module.parameters()
                ):
                    mmt_param.data = mmt_param.data * model.momentum + \
                        base_param.data * (1 - model.momentum)

        # Compute embeddings with momendum encoder as well
        cls_embs_mmt, att_embs_mmt = _compute_conc_embs(
            model, sg_roi_embs, conc_type, momentum=True
        )

    # Determine appropriate values and modules to use in computation by conc_type
    if conc_type == "class":
        inst_embs = cls_embs
        inst_embs_mmt = cls_embs_mmt
        mmt_predict = model.mmt_predict_cls
        queue_embs = model.queue_embs_cls
        queue_labels = model.queue_labels_cls
        exs_prompt_encode = model.exs_prompt_encode_cls
        exs_prompt_tag = model.exs_prompt_tag_cls
    else:
        assert conc_type == "attribute"
        inst_embs = att_embs
        inst_embs_mmt = att_embs_mmt
        mmt_predict = model.mmt_predict_att
        queue_embs = model.queue_embs_att
        queue_labels = model.queue_labels_att
        exs_prompt_encode = model.exs_prompt_encode_att
        exs_prompt_tag = model.exs_prompt_tag_att

    ## Encoding-direction (i.e., <Image, Instance => Concepts>)

    # Number of nearest neighbors to consider when computing LooK loss
    if model.training:
        # Annealed by progress (0~1)
        progress = batch_idx / model.cfg.vision.optim.max_steps
        num_neighbors = model.num_neighbors_range[0] + \
            progress * (model.num_neighbors_range[1]-model.num_neighbors_range[0])
        num_neighbors = int(num_neighbors)
    else:
        # End value
        num_neighbors = model.num_neighbors_range[1]

    # Instance label sets for the batch, with string names mapped to int indices
    inst_labels = [
        [model.conc2ind[conc_type][conc] for conc in [fc]+aux]
        for fc, aux in conc_labels
    ]
    # Pad labels for both batch & memory queue entries by current maximum label
    # count (cf. for VAW, will be consistent for classes with only one label per
    # entry, but may differ for attributes)
    max_label_count = max(
        max(len(ls) for ls in inst_labels), queue_labels.shape[-1]
    )
    inst_labels = torch.tensor(
        [ls+[-1]*(max_label_count-len(ls)) for ls in inst_labels],
        dtype=torch.long, device=model.device
    )
    if queue_labels.shape[-1] < max_label_count:
        queue_labels = F.pad(
            queue_labels, (0,max_label_count-queue_labels.shape[-1]), value=-1
        )

    # Pass the embeddings from the base encoder through the projector module
    predicted_inst_embs_mmt = mmt_predict(inst_embs)

    # Find nearest neighbors from the memory queue for computing LooK loss
    # value, alongside the current input batch
    search_space_embs = torch.cat([inst_embs_mmt, queue_embs])
    search_space_labels = torch.cat([inst_labels, queue_labels])
    dists_sq = torch.cdist(predicted_inst_embs_mmt, search_space_embs, p=2.0) ** 2
    k_nearest_dists_sq, k_nearest_inds = torch.topk(
        dists_sq, num_neighbors, dim=-1, largest=False
    )
    # Positive pair mask; (batch instance, memory queue instance) pair is positive
    # iff the memory instance's label set includes the 'focus' label of the batch
    # instance
    gather_input_size = [B]+list(search_space_labels.shape)
    gather_index_size = list(k_nearest_inds.shape)+[search_space_labels.shape[-1]]
    k_nearest_labels = torch.gather(
        search_space_labels[None].expand(gather_input_size),
        1,      # dim [1] spans across whole memory bank
        k_nearest_inds[...,None].expand(gather_index_size)
    )
    nonpad_intersections = [
        # For each batch entry, concat lists of non-pad labels between instance vs.
        # each of its k-nearest neighbors fetched
        [torch.cat([i_ls[i_ls!=-1], n_ls[n_ls!=-1]]) for n_ls in knn_ls]
        for i_ls, knn_ls in zip(inst_labels, k_nearest_labels)
    ]
    pos_mask = torch.tensor([
        # If the occurrence count is larger than 1 for any label, we have non-zero
        # intersections (i.e. shared concept labels), thus positive pairs
        [torch.any(pair.unique(return_counts=True)[1] > 1) for pair in per_entry]
        for per_entry in nonpad_intersections
    ], device=model.device)

    # Compute LooK loss and precision@K metric
    losses["look"] = F.softmax(-k_nearest_dists_sq) * pos_mask
    losses["look"] = -losses["look"].sum(dim=-1).log().mean()
    metrics["precision@K"] = (pos_mask[:,:K].sum(dim=-1) / K).mean()

    # Update memory queue
    with torch.no_grad():
        new_queue_embs = torch.cat([inst_embs_mmt, queue_embs[:-B]])
        new_queue_labels = torch.cat([inst_labels, queue_labels[:-B]])
        if conc_type == "class":
            model.queue_embs_cls = new_queue_embs
            model.queue_labels_cls = new_queue_labels
        else:
            assert conc_type == "attribute"
            model.queue_embs_att = new_queue_embs
            model.queue_labels_att = new_queue_labels

    ## Decoding-direction (i.e., <Image, Concept => Instances>)

    # Mask decoding to obtain segmentations; three runs for each image with
    # different types of prompts; one with support examples only, one with
    # support + centroid point, and one with support + bounding box

    # Prepare image-positional embeddings
    img_positional_embs = model.sam.get_image_wide_positional_embeddings()
    img_positional_embs = img_positional_embs.repeat(B, 1, 1, 1)

    # Obtain prototypes as leave-one-out averages of support example vectors
    inst_embs_per_conc = inst_embs.view(N, K, -1)
    prototypes = inst_embs_per_conc.sum(dim=1, keepdims=True) - inst_embs_per_conc
    prototypes = (prototypes / (K-1)).view(B, -1)

    # Encode the prototypes into decoder prompt tokens
    proto_tokens = exs_prompt_encode(prototypes) + exs_prompt_tag.weight
    proto_tokens = proto_tokens[:,None,None]

    # No-mask embeddings; Dense embedding in SAM when mask prompts are not provided
    no_mask_dense_embs = model.sam.prompt_encoder.no_mask_embed.weight
    no_mask_dense_embs = no_mask_dense_embs[...,None,None].expand_as(img_embs)

    # Ground truth masks resized (again) to match decoder output sizes
    # (4 corresponds to two 2x deconvolutions in mask decoder)
    low_size = resize_size // patch_size * 4
    gt_masks = [
        [F.pad(mask, p_spec, value=0) for mask in masks]
        for masks, p_spec in zip(batch_data["concept_masks"], pad_specs)
    ]
    gt_masks = torch.stack([
        torch.cat([resize(mask[None], low_size) for mask in masks])
        for masks in gt_masks
    ])

    # Binary masks corresponding to image paddings, for screening out loss values
    # at padded pixels
    img_pad_masks = torch.stack([
        torch.arange(low_size)[:,None].expand(low_size,low_size) >= h.item()
            if h<w else torch.arange(low_size)[None].expand(low_size,low_size) >= w.item()
        for h, w in (processed_input["reshaped_input_sizes"] // 4)
    ]).to(model.device)

    # Binary masks corresponding to valid ground-truth paddings, i.e. for addressing
    # variable numbers of valid mask predictions per image
    valid_pad_masks = gt_masks[:,:,0,0] != -1

    # Needs casting into float for computations below
    gt_masks = gt_masks.to(torch.float)

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

        # Compute pairwise dice coefficients between predictions vs. ground truths
        dice_coeffs = _pairwise_dice(pred_masks[:,0], gt_masks, img_pad_masks)

        if multimask:
            # Hungarian matching between predictions vs. valid ground-truths based on
            # the pairwise dice coefficients computed, per batch entry, to find most
            # compatible guesses for each valid ground truth
            best_matching = [
                linear_sum_assignment(coeffs[:,masks].detach().cpu().numpy(), maximize=True)
                for coeffs, masks in zip(dice_coeffs, valid_pad_masks)
            ]

            # Collect & rearrange predictions and ground truths according to the matches
            pred_masks_fetched = torch.cat([
                pred_masks[i,0,match[0]] for i, match in enumerate(best_matching)
            ])
            pred_ious_fetched = torch.cat([
                pred_ious[i,0,match[0]] for i, match in enumerate(best_matching)
            ])
            gt_masks_fetched = torch.cat([
                gt_masks[i,match[1]] for i, match in enumerate(best_matching)
            ])
            # Gather (possibly duplicate) copies of image pad masks as needed
            img_pad_masks_fetched = torch.stack([
                img_pad_masks[i] for i, match in enumerate(best_matching) for _ in match[0]
            ])

            # Fetch appropriate dice coefficients to be cast into dice loss
            # Compute pairwise dice coefficients between predictions vs. ground truths
            dice_coeffs_fetched = torch.cat([
                dice_coeffs[i,match[0],match[1]] for i, match in enumerate(best_matching)
            ])
        else:
            # Compute dice coefficients between the sole predictions vs. first ground
            # truths, per batch entry

            # All used as-is, just readjust dimensions for downstream computations
            pred_masks_fetched = pred_masks[:,0,0]
            pred_ious_fetched = pred_ious[:,0,0]
            gt_masks_fetched = gt_masks[:,0]
            img_pad_masks_fetched = img_pad_masks

            # All the dice coefficients computed will be cast into dice loss
            dice_coeffs_fetched = dice_coeffs

        # Compute dice & focal loss values
        dice_loss = -dice_coeffs_fetched.log().mean()
        focal_loss = _sigmoid_focal_loss(
            pred_masks_fetched, gt_masks_fetched, img_pad_masks_fetched
        )

        # Compute IoUs between predictions vs. ground truths, cast into both loss and
        # metric values
        with torch.no_grad():
            pred_probs = pred_masks_fetched.sigmoid()
            soft_intersections = torch.min(pred_probs, gt_masks_fetched).sum(dim=(-2,-1))
            soft_unions = torch.max(pred_probs, gt_masks_fetched).sum(dim=(-2,-1))
            gt_ious = soft_intersections / soft_unions

            iou_loss = F.mse_loss(pred_ious_fetched, gt_ious)
            iou_metric = gt_ious.mean()

        # Update losses & metrics
        losses["focal"] += focal_loss
        losses["dice"] += dice_loss
        losses["iou"] += iou_loss
        metrics["IoU"] += iou_metric

    # To be updated by increments
    losses["focal"] = 0; losses["dice"] = 0; losses["iou"] = 0
    metrics["IoU"] = 0

    # First prompt: Support examples only, predict at most 3 masks if applicable
    segm_preds_1 = mask_decode(proto_tokens, True)
    segmentation_loss_metric_update(segm_preds_1, True)

    # Second prompt: Support + centroid (hybrid), predict 1 mask
    centroid_tokens, _ = model.sam.prompt_encoder(
        processed_input["input_points"][:,:,None],
        processed_input["input_labels"][:,:,None],
        None,
        None
    )
    hybrid_tokens_2 = torch.cat([centroid_tokens, proto_tokens], dim=2)
    segm_preds_2 = mask_decode(hybrid_tokens_2, False)
    segmentation_loss_metric_update(segm_preds_2, False)

    # Third prompt: Support + bounding box (hybrid), predict 1 mask
    bbox_tokens, _ = model.sam.prompt_encoder(
        None,
        None,
        processed_input["input_boxes"],
        None
    )
    hybrid_tokens_3 = torch.cat([bbox_tokens, proto_tokens], dim=2)
    segm_preds_3 = mask_decode(hybrid_tokens_3, False)
    segmentation_loss_metric_update(segm_preds_3, False)

    # Divide the segmentation loss & metric values by three (one for each prompt)
    # to obtain average
    losses["focal"] /= 3; losses["dice"] /= 3; losses["iou"] /= 3
    metrics["IoU"] /= 3

    return losses, metrics


def _compute_conc_embs(model, sg_roi_embs, conc_type, momentum=False):
    """
    Factored out [conc_type]-specific embedding computation logic, which is shared
    by both base & momentum encoders
    """
    # Choosse encoder modules
    if momentum:
        embed_cls = model.embed_cls_momentum
        embed_att = model.embed_att_momentum
        gate_compute = model.gate_compute_momentum
    else:
        embed_cls = model.embed_cls
        embed_att = model.embed_att
        gate_compute = model.gate_compute

    # Compute class-centric embeddings
    cls_embs = embed_cls(sg_roi_embs)

    # Compute attribute-centric embeddings (as needed)
    if conc_type == "class":
        att_embs = None
    else:
        assert conc_type == "attribute"

        # Obtain gated RoI embeddings
        feature_gate = gate_compute(cls_embs).sigmoid()
        sg_roi_embs_gated = sg_roi_embs * feature_gate[...,None,None]

        # Compute attribute-centric embeddings
        att_embs = embed_att(sg_roi_embs_gated)

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
    focal_loss = focal_loss.mean()

    return focal_loss


def _pairwise_dice(inputs, targets, img_pad_masks):
    """
    Compute pairwise DICE coefficients between predicted masks vs. ground-truth masks;
    used for Hungarian matching, and also as loss value
    """
    # Logits to probs
    probs = inputs.sigmoid()

    # Numerators, with img_pad_masks applied
    numerators = probs[:,:,None] * targets[:,None]
    numerators[img_pad_masks[:,None,None].expand_as(numerators)] = 0
    numerators = 2 * numerators.sum(dim=(-2,-1))

    # Denominators, with img_pad_masks applied
    denominators = probs[:,:,None] + targets[:,None]
    denominators[img_pad_masks[:,None,None].expand_as(denominators)] = 0
    denominators = denominators.sum(dim=(-2,-1))

    # Compute dice coefficients (smoothing by 1 in num and denom)
    dice_coeffs = (numerators+1) / (denominators+1)

    # Output shape: batch_size * num_multimask_outputs * num_ground_truth
    return dice_coeffs
