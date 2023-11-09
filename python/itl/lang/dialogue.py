import copy

import numpy as np

from ..vision.utils import mask_iou


IOU_THRES = 0.8

class DialogueManager:
    """ Maintain dialogue state and handle NLU, NLG in context """

    def __init__(self):

        self.referents = {
            "env": {},  # Sensed via physical perception
            "dis": {}   # Introduced by dialogue
        }
        self.referents["env"]["_self"] = None
                # Self always included as environment entity

        self.assignment_hard = {}   # Store fixed assignment by demonstrative+pointing, names, etc.
        self.referent_names = {}    # Store mapping from symbolic name to entity
        self.clause_info = {}       # Store any important information re. individual clauses

        # Each record is a 3-tuple of:
        #   1) speaker: user ("U") or agent ("A")
        #   2) logical form of utterance content
        #   3) original user input string
        self.record = []

        # Sensemaking results after processing each dialogue input, indexed by dialogue
        # turns
        self.sensemaking_v_snaps = {}
        self.sensemaking_vl_snaps = {}

        self.unanswered_Qs = set()
        if hasattr(self, "acknowledged_stms"):
            # If acknowledgement info exists for current record when refreshing, outdate
            # by re-indexing "curr" entries with "prev"
            self.acknowledged_stms = {
                ("prev", ti, ci): data
                for (prev_or_curr, ti, ci), data in self.acknowledged_stms.items()
                if prev_or_curr=="curr"
            }
        else:
            self.acknowledged_stms = {}

        # Buffer of utterances to generate
        self.to_generate = []

    def refresh(self):
        """ Clear the current dialogue state to start fresh in a new situation """
        self.__init__()
    
    def export_as_dict(self):
        """ Export the current dialogue information state as a dict """
        return copy.deepcopy(vars(self))

    def dem_point(self, dem_mask):
        """
        Provided a segmentation mask specification, return reference (by object id)
        to an appropriate environment entity recognized, potentially creating one
        if not already explicitly aware of it as object.
        """
        env_entities = {
            k: v for k, v in self.referents["env"].items()
            if k != "_self"
        }       # Entities except self

        if len(env_entities) > 0:
            # First check if there's any existing high-IoU segmentation mask; by
            # 'high' we refer to the threshold we set as global constant above
            env_ref_masks = np.stack([e["mask"] for e in env_entities.values()])
            ious = mask_iou(dem_mask[None], env_ref_masks)[0]
            best_match = ious.argmax()

            if ious[best_match] > IOU_THRES:
                # Presume the 'pointed' entity is actually this one
                pointed = list(env_entities.keys())[best_match]
            else:
                # Register the entity as a novel environment referent
                pointed = f"o{len(env_ref_masks)}"

                self.referents["env"][pointed] = {
                    "mask": dem_mask,
                    "area": dem_mask.sum().item()
                }
                self.referent_names[pointed] = pointed
        else:
            # Register the entity as a novel environment referent
            pointed = "o0"

            self.referents["env"][pointed] = {
                "mask": dem_mask,
                "area": dem_mask.sum().item()
            }
            self.referent_names[pointed] = pointed

        return pointed
