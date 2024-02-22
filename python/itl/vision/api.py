"""
Vision processing module API that exposes the high-level functionalities required
by the ITL agent inference (concept classification of object instances in image,
concept instance search in image). Implemented by building upon the pretrained
Segment Anything Model (SAM).
"""
import os
import logging
from PIL import Image
from itertools import product

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from pycocotools import mask
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from .data import FewShotDataModule
from .modeling import VisualSceneAnalyzer
from .utils.visualize import visualize_sg_predictions

logger = logging.getLogger(__name__)


WB_PREFIX = "wandb://"
DEF_CON = 0.6       # Default confidence value in absence of binary concept classifier

class VisionModule:

    K = 0               # Top-k detections to leave in ensemble prediction mode
    NMS_THRES = 0.65    # IoU threshold for post-detection NMS

    def __init__(self, cfg):
        self.cfg = cfg

        self.scene = None
        self.latest_input = None        # Latest raw input
        self.previous_input = None      # Any input *before* current self.latest_input

        self.fs_model_path = None

        # Inventory of distinct visual concepts that the module (and thus the agent
        # equipped with this module) is aware of, one per concept category. Right now
        # I cannot think of any specific type of information that has to be stored in
        # this module (exemplars are stored in long term memory), so let's just keep
        # only integer sizes of inventories for now...
        self.inventories = VisualConceptInventory()

        self.model = VisualSceneAnalyzer(self.cfg)

        # Reading W&B config environment variables, if exists
        try:
            load_dotenv(find_dotenv(raise_error_if_not_found=True))
        except OSError as e:
            logger.warn(f"While reading dotenv: {e}")

        # If pre-trained vision model is specified, download and load weights
        if "fs_model" in self.cfg.vision.model:
            self.fs_model_path = self.cfg.vision.model.fs_model
            self.load_weights()
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.confusions = set()

    def load_weights(self):
        """ Load model parameter weights from specified source (self.fs_model) """
        # Path to trained weights few-shot prediciton head is provided as either
        # W&B run id or local path to checkpoint file
        assert self.fs_model_path is not None

        if self.fs_model_path.startswith(WB_PREFIX):
            wb_entity = os.environ.get("WANDB_ENTITY")
            wb_project = os.environ.get("WANDB_PROJECT")
            wb_path = self.fs_model_path[len(WB_PREFIX):].split(":")

            if len(wb_path) == 1:
                wb_run_id = wb_path[0]
                wb_alias = "best_k"
            else:
                assert len(wb_path) == 2
                wb_run_id, wb_alias = wb_path

            wb_full_path = f"{wb_entity}/{wb_project}/model-{wb_run_id}:{wb_alias}"
            local_model_root_path = os.path.join(
                self.cfg.paths.assets_dir, "vision_models", "wandb", wb_run_id
            )
            local_ckpt_path = os.path.join(local_model_root_path, f"model_{wb_alias}.ckpt")
            if not os.path.exists(local_ckpt_path):
                # Download if needed (remove current file if want to re-download)
                local_ckpt_path = wandb.Api().artifact(wb_full_path).download(
                    root=local_model_root_path
                )
                os.rename(
                    os.path.join(local_ckpt_path, f"model.ckpt"),
                    os.path.join(local_ckpt_path, f"model_{wb_alias}.ckpt")
                )
                local_ckpt_path = os.path.join(local_ckpt_path, f"model_{wb_alias}.ckpt")
            logger.info(f"Loading few-shot component weights from {wb_full_path}")
        else:
            local_ckpt_path = self.fs_model_path
            logger.info(f"Loading few-shot component weights from {local_ckpt_path}")

        ckpt = torch.load(local_ckpt_path)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)


    def predict(
        self, image, exemplars, reclassify=False, masks=None, specs=None,
        visualize=True, lexicon=None
    ):
        """
        Model inference in either one of three modes:
            1) full scene graph generation mode, where the module is only given
                an image and needs to return its estimation of the full scene
                graph for the input
            2) concept reclassification mode, where the scene objects and their
                visual embeddings are preserved intact and only the concept tests
                are run again (most likely called when agent exemplar base is updated)
            3) instance classification mode, where a number of masks are given
                along with the image and category predictions are made for only
                those instances
            4) instance search mode, where a specification is provided in the
                form of FOL formula with a variable and best fitting instance(s)
                should be searched

        3) and 4) are 'incremental' in the sense that they should add to an existing
        scene graph which is already generated with some previous execution of this
        method.
        
        Provide reclassify=True to run in 2) mode. Provide masks arg to run in 3) mode,
        or spec arg to run in 4) mode. All prediction modes are mutually exclusive in
        the sense that only one of their corresponding 'flags' should be active.
        """
        masks_provided = masks is not None
        specs_provided = specs is not None
        ensemble = not (reclassify or masks_provided or specs_provided)

        # Image must be provided for ensemble prediction
        if ensemble: assert image is not None

        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            else:
                assert isinstance(image, Image.Image)

        self.model.eval()
        with torch.no_grad():
            # Prediction modes
            if ensemble:
                # Full (ensemble) prediction
                vis_embs, masks_out, scores = self.model(image)

                self.latest_input = image         # Cache last input image

                # Newly compose a scene graph with the output; filter patches to leave top-k
                # detections
                self.scene = {
                    f"o{i}": {
                        "vis_emb": vis_embs[i],
                        "pred_mask": masks_out[i],
                        "pred_objectness": scores[i],
                        "pred_cls": self._fs_conc_pred(exemplars, vis_embs[i], "cls"),
                        "pred_att": self._fs_conc_pred(exemplars, vis_embs[i], "att"),
                        "pred_rel": {
                            f"o{j}": np.zeros(self.inventories.rel)
                            for j in range(self.K) if i != j
                        },
                        "exemplar_ind": None       # Store exemplar storage index, if applicable
                    }
                    for i in range(self.K)
                }

                for oi, obj_i in self.scene.items():
                    oi_msk = obj_i["pred_mask"]

                    # Relation concepts (Only for "have" concept, manually determined
                    # by the geomtrics of the bounding boxes; note that this is quite
                    # an abstraction. In distant future, relation concepts may also be
                    # open-vocabulary and neurally predicted...)
                    for oj, obj_j in self.scene.items():
                        if oi==oj: continue     # Dismiss self-self object pairs
                        oj_msk = obj_j["pred_mask"]

                        intersection_A = np.minimum(oi_msk, oj_msk).sum()
                        mask2_A = oj_msk.sum()

                        obj_i["pred_rel"][oj][0] = intersection_A / mask2_A

            elif reclassify:
                # Concept reclassification with the same set of objects
                for obj_i in self.scene.values():
                    vis_emb = obj_i["vis_emb"]
                    obj_i["pred_cls"] = self._fs_conc_pred(exemplars, vis_emb, "cls")
                    obj_i["pred_att"] = self._fs_conc_pred(exemplars, vis_emb, "att")

            else:
                # Incremental scene graph expansion
                if masks_provided:
                    # Instance classification mode
                    incr_preds = self.model(image, masks=list(masks.values()))

                else:
                    # Instance search mode
                    assert specs_provided
                    incr_preds = self._instance_search(image, exemplars, specs)

                # Selecting names of new objects to be added to the scene
                if masks_provided:
                    # Already provided with appropriate object identifier
                    new_objs = list(masks.keys())
                else:
                    # Come up with new object identifiers for valid search matches
                    incr_masks_out = incr_preds[1]

                    new_objs = []; ind_offset = 0
                    for msk in incr_masks_out:
                        if msk is None:
                            # Null match
                            new_objs.append(None)
                        else:
                            new_objs.append(f"o{len(self.scene)+ind_offset}")
                            ind_offset += 1

                # Update visual scene
                self._incremental_scene_update(incr_preds, new_objs, exemplars)

        if visualize:
            if lexicon is not None:
                lexicon = {
                    conc_type: {
                        ci: lexicon.d2s[(ci, conc_type)][0][0].split("/")[0]
                        for ci in range(getattr(self.inventories, conc_type))
                    }
                    for conc_type in ["cls", "att"]
                }
            self.summ = visualize_sg_predictions(self.latest_input, self.scene, lexicon)

    def _fs_conc_pred(self, exemplars, emb, conc_type):
        """
        Helper method factored out for few-shot concept probability estimation
        """
        conc_inventory_count = getattr(self.inventories, conc_type)
        if conc_inventory_count > 0:
            # Non-empty concept inventory, most cases
            predictions = []
            for ci in range(conc_inventory_count):
                if exemplars.binary_classifiers[conc_type][ci] is not None:
                    # Binary classifier induced from pos/neg exemplars exists
                    clf = exemplars.binary_classifiers[conc_type][ci]
                    pred = clf.predict_proba(emb[None])[0]
                    pred = max(min(pred[1], 0.99), 0.01)    # Soften absolute certainty)
                    predictions.append(pred)
                else:
                    # No binary classifier exists due to lack of either positive
                    # or negative exemplars, fall back to default estimation
                    predictions.append(DEF_CON)
            return np.stack(predictions)
        else:
            # Empty concept inventory, likely at the very beginning of training
            # an agent from scratch,
            return np.empty(0, dtype=np.float32)

    def _instance_search(self, image, exemplars, search_specs):
        """
        Helper method factored out for running model in incremental search mode
        """
        # Mapping between existing entity IDs and their numeric indexing
        exs_idx_map = { i: ent for i, ent in enumerate(self.scene) }
        exs_idx_map_inv = { ent: i for i, ent in enumerate(self.scene) }

        # Return values
        incr_vis_embs = []
        incr_masks_out = []
        incr_scores = []

        for s_vars, description, pred_glossary in search_specs:
            # Prepare search conditions to feed into model.forward()
            search_conds = []
            for d_lit in description:
                if d_lit.name.startswith("disj"):
                    # A predicate standing for a disjunction of elementary concepts
                    # (as listed in `pred_glossary`), fetch all positive exemplar sets
                    # and binary classifiers for each disjunct concept
                    disj_cond = []
                    for conc in pred_glossary[d_lit.name][1]:
                        conc_type, conc_ind = conc.split("_")
                        conc_ind = int(conc_ind)

                        if conc_type == "cls" or conc_type == "att":
                            # Fetch set of positive exemplars
                            pos_exs_inds = exemplars.exemplars_pos[conc_type][conc_ind]
                            pos_exs_info = [
                                (
                                    exemplars.scenes[scene_id][0],
                                    mask.decode(
                                        exemplars.scenes[scene_id][1][obj_id]["mask"]
                                    ).astype(bool),
                                    exemplars.scenes[scene_id][1][obj_id]["f_vec"]
                                )
                                for scene_id, obj_id in pos_exs_inds
                            ]
                            bin_clf = exemplars.binary_classifiers[conc_type][conc_ind]
                            disj_cond.append((pos_exs_info, bin_clf))
                        else:
                            assert conc_type == "rel"

                            # Relations are not neurally predicted, nor used for search
                            # (for now, at least...)
                            continue

                    search_conds.append(disj_cond)

                else:
                    # Single elementary predicate, fetch positive exemplars for the
                    # corresponding concept
                    conc_type, conc_ind = d_lit.name.split("_")
                    conc_ind = int(conc_ind)

                    if conc_type == "cls" or conc_type == "att":
                        # Fetch set of positive exemplars
                        pos_exs_inds = exemplars.exemplars_pos[conc_type][conc_ind]
                        pos_exs_info = [
                            (
                                exemplars.scenes[scene_id][0],
                                mask.decode(
                                    exemplars.scenes[scene_id][1][obj_id]["mask"]
                                ).astype(bool),
                                exemplars.scenes[scene_id][1][obj_id]["f_vec"]
                            )
                            for scene_id, obj_id in pos_exs_inds
                        ]
                        bin_clf = exemplars.binary_classifiers[conc_type][conc_ind]
                        search_conds.append([(pos_exs_info, bin_clf)])
                    else:
                        assert conc_type == "rel"

                        # Relations are not neurally predicted, nor used for search
                        # (for now, at least...)
                        continue

            # Run search conditioned on the exemplars
            vis_embs, masks_out, scores = self.model(image, search_conds=search_conds)

            if len(masks_out) == 0:
                # Search didn't return any match
                incr_vis_embs.append(None)
                incr_masks_out.append(None)
                incr_scores.append(None)
                continue

            # If len(masks_out) > 0, search returned some matches

            # Test each candidate to select the one that's most compatible
            # to the search spec
            agg_compatibility_scores = torch.ones(
                len(masks_out), device=self.model.device
            )
            for d_lit in description:
                if d_lit.name.startswith("disj_"):
                    # Use the returned scores as compatibility scores
                    comp_scores = torch.tensor(scores, device=self.model.device)

                else:
                    conc_type, conc_ind = d_lit.name.split("_")
                    conc_ind = int(conc_ind)

                    if conc_type == "cls" or conc_type == "att":
                        # Use the returned scores as compatibility scores
                        comp_scores = torch.tensor(scores, device=self.model.device)

                    else:
                        # Compatibility scores geometric relations, which are not neurally
                        # predicted int the current module implementation
                        assert conc_type == "rel"

                        # Cannot process relations other than "have" for now...
                        assert conc_ind == 0

                        # Cannot process search specs with more than one variables for
                        # now (not planning to address that for a good while!)
                        assert len(s_vars) == 1

                        # Handles to literal args; either search target variable or
                        # previously identified entity
                        arg_handles = [
                            ("v", s_vars.index(arg[0]))
                                if arg[0] in s_vars
                                else ("e", exs_idx_map_inv[arg[0]])
                            for arg in d_lit.args
                        ]

                        # Mask areas for all candidates
                        masks_out_A = masks_out.sum(axis=(-2,-1))

                        # Fetch bbox of reference entity, against which bbox area
                        # ratios will be calculated among candidates
                        reference_ent = [
                            arg_ind for arg_type, arg_ind in arg_handles
                            if arg_type=="e"
                        ][0]
                        reference_ent = exs_idx_map[reference_ent]
                        reference_mask = self.scene[reference_ent]["pred_mask"]

                        # Compute area ratio between the reference mask and all proposals
                        intersections = masks_out * reference_mask[None]
                        intersections_A = intersections.sum(axis=(-2,-1))

                        comp_scores = torch.tensor(
                            intersections_A / masks_out_A, device=self.model.device
                        )

                # Update aggregate compatibility score; using min function
                # as the t-norm (other options: product, ...)
                agg_compatibility_scores = torch.minimum(
                    agg_compatibility_scores, comp_scores
                )

            # Finally choose and keep the best search output
            best_match_ind = agg_compatibility_scores.max(dim=0).indices
            incr_vis_embs.append(vis_embs[best_match_ind])
            incr_masks_out.append(masks_out[best_match_ind])
            incr_scores.append(scores[best_match_ind])

        return incr_vis_embs, incr_masks_out, incr_scores

    def _incremental_scene_update(self, incr_preds, new_objs, exemplars):
        """
        Helper method factored out updating current scene with new incrementally
        predicted instances (by masks or search specs)
        """
        incr_vis_embs = incr_preds[0]
        incr_masks_out = incr_preds[1]
        incr_scores = incr_preds[2]

        # Incrementally update the existing scene graph with the output with the
        # detections best complying with the conditions provided
        existing_objs = list(self.scene)

        for oi, oj in product(existing_objs, new_objs):
            # Add new relation score slots for existing objects
            self.scene[oi]["pred_rel"][oj] = np.zeros(self.inventories.rel)

        update_data = list(zip(
            new_objs, incr_vis_embs, incr_masks_out, incr_scores
        ))
        for oi, vis_emb, msk, score in update_data:
            # Pass null entry
            if oi is None: continue

            # Register new objects into the existing scene
            self.scene[oi] = {
                "vis_emb": vis_emb,
                "pred_mask": msk,
                "pred_objectness": score,
                "pred_cls": self._fs_conc_pred(exemplars, vis_emb, "cls"),
                "pred_att": self._fs_conc_pred(exemplars, vis_emb, "att"),
                "pred_rel": {
                    **{
                        oj: np.zeros(self.inventories.rel)
                        for oj in existing_objs
                    },
                    **{
                        oj: np.zeros(self.inventories.rel)
                        for oj in new_objs if oi != oj
                    }
                },
                "exemplar_ind": None
            }

        for oi in new_objs:
            # Pass null entry
            if oi is None: continue

            oi_msk = self.scene[oi]["pred_mask"]

            # Relation concepts (Within new detections)
            for oj in new_objs:
                # Pass null entry
                if oj is None: continue

                if oi==oj: continue     # Dismiss self-self object pairs
                oj_msk = self.scene[oj]["pred_mask"]

                intersection_A = np.minimum(oi_msk, oj_msk).sum()
                mask2_A = oj_msk.sum()

                self.scene[oi]["pred_rel"][oj][0] = intersection_A / mask2_A

            # Relation concepts (Between existing detections)
            for oj in existing_objs:
                oj_msk = self.scene[oj]["pred_mask"]

                intersection_A = np.minimum(oi_msk, oj_msk).sum()
                mask1_A = oi_msk.sum()
                mask2_A = oj_msk.sum()

                self.scene[oi]["pred_rel"][oj][0] = intersection_A / mask2_A
                self.scene[oj]["pred_rel"][oi][0] = intersection_A / mask1_A

    def train(self):
        """
        Training few-shot visual object detection & class/attribute classification
        model with specified dataset. Uses a pre-trained Deformable DETR as feature
        extraction backbone and learns lightweight MLP blocks (one each for class
        and attribute prediction) for embedding raw feature vectors onto a metric
        space where instances of the same concepts are placed closer. (Mostly likely
        not called by end user.)
        """
        # Prepare DataModule from data config
        dm = FewShotDataModule(self.cfg)

        # Configure W&B logger
        wb_kwargs = {
            "project": os.environ.get("WANDB_PROJECT"),
            "entity": os.environ.get("WANDB_ENTITY"),
            "save_dir": self.cfg.paths.outputs_dir
        }
        # Whether to run offline
        if self.cfg.vision.offline:
            wb_kwargs["offline"] = True
        else:
            wb_kwargs["log_model"] = True
        # Whether to resume training from previous few-shot components
        if "fs_model" in self.cfg.vision.model:
            if (self.cfg.vision.model.fs_model.startswith(WB_PREFIX) and
                self.cfg.vision.optim.resume):
                wb_path = self.cfg.vision.model.fs_model[len(WB_PREFIX):].split(":")
                wb_kwargs["id"] = wb_path[0]
                wb_kwargs["resume"] = "must"
        wb_logger = WandbLogger(**wb_kwargs)

        # Configure and run trainer
        trainer = pl.Trainer(
            accelerator="auto",
            max_steps=self.cfg.vision.optim.max_steps,
            # check_val_every_n_epoch=None,       # Iteration-based val
            log_every_n_steps=self.cfg.vision.optim.log_interval,
            val_check_interval=self.cfg.vision.optim.val_interval,
            num_sanity_val_steps=0,
            logger=wb_logger,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    every_n_train_steps=self.cfg.vision.optim.val_interval,
                    save_last=True
                ),
                LearningRateMonitor(logging_interval='step')
            ]
        )
        # trainer.validate(self.model, datamodule=dm)
        trainer.fit(self.model, datamodule=dm)

    def evaluate(self):
        """
        Evaluate best model from a run on test dataset
        """
        # Prepare DataModule from data config
        dm = FewShotDataModule(self.cfg)

        if "fs_model" in self.cfg.vision.model:
            if self.cfg.vision.model.fs_model.startswith(WB_PREFIX):
                wb_path = self.cfg.vision.model.fs_model[len(WB_PREFIX):].split(":")

                # Configure W&B logger
                wb_kwargs = {
                    "project": os.environ.get("WANDB_PROJECT"),
                    "entity": os.environ.get("WANDB_ENTITY"),
                    "save_dir": self.cfg.paths.outputs_dir,
                    "id": wb_path[0],
                    "resume": "must"
                }
                # Whether to run offline
                if self.cfg.vision.offline:
                    wb_kwargs["offline"] = True

                logger = WandbLogger(**wb_kwargs)
            else:
                logger = False
        else:
            logger = False

        trainer = pl.Trainer(accelerator="auto", logger=logger)
        trainer.test(self.model, datamodule=dm)

    def add_concept(self, conc_type):
        """
        Register a novel visual concept to the model, expanding the concept inventory of
        corresponding category type (class/attribute/relation). Note that visual concepts
        are not inseparably attached to some linguistic symbols; such connections are rather
        incidental and should be established independently (consider synonyms, homonyms).
        Plus, this should allow more flexibility for, say, multilingual agents, though there
        is no plan to address that for now...

        Returns the index of the newly added concept.
        """
        C = getattr(self.inventories, conc_type)
        setattr(self.inventories, conc_type, C+1)
        return C


class VisualConceptInventory:
    def __init__(self):
        # self.cls = self.att = self.rel = 0

        # (Temp) Inventory of relation concept is a fixed singleton set, containing "have"
        self.cls = self.att = 0
        self.rel = 1
