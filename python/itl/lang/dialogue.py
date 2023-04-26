import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


class DialogueManager:
    """Maintain dialogue state and handle NLU, NLG in context"""

    def __init__(self):

        self.referents = {
            "env": {},  # Sensed via physical perception
            "dis": {}   # Introduced by dialogue
        }

        self.assignment_hard = {}  # Store fixed assignment by demonstrative+pointing, names, etc.
        self.referent_names = {}   # Store mapping from symbolic name to entity

        # Each record is a 3-tuple of:
        #   1) speaker: user ("U") or agent ("A")
        #   2) logical form of utterance content
        #   3) original user input string
        self.record = []

        self.unanswered_Q = set()

        # Buffer of utterances to generate
        self.to_generate = []

    def refresh(self):
        """ Clear the current dialogue state to start fresh in a new situation """
        self.__init__()
    
    def export_as_dict(self):
        """ Export the current dialogue information state as a dict """
        return vars(self)

    def dem_point(self, dem_bbox):
        """
        Provided a bounding box specification, return reference (by object id) to
        an appropriate environment entity recognized, potentially creating one if
        not already explicitly aware of it as object.
        """
        if len(self.referents["env"]) > 0:
            # First check if there's any existing high-IoU bounding box; by 'high'
            # we refer to some arbitrary threshold -- let's use 0.8 here
            env_ref_bboxes = torch.stack(
                [torch.tensor(e["bbox"]) for e in self.referents["env"].values()]
            )

            iou_thresh = 0.7
            ious = torchvision.ops.box_iou(
                torch.tensor(dem_bbox)[None,:], env_ref_bboxes
            )
            best_match = ious.max(dim=-1)

            if best_match.values.item() > iou_thresh:
                # Assume the 'pointed' entity is actually this one
                max_ind = best_match.indices.item()
                pointed = list(self.referents["env"].keys())[max_ind]
            else:
                # Register the entity as a novel environment referent
                pointed = f"o{len(env_ref_bboxes)}"

                self.referents["env"][pointed] = {
                    "bbox": dem_bbox,
                    "area": (dem_bbox[2]-dem_bbox[0]) * (dem_bbox[3]-dem_bbox[1])
                }
                self.referent_names[pointed] = pointed
        else:
            # Register the entity as a novel environment referent
            pointed = "o0"

            self.referents["env"][pointed] = {
                "bbox": dem_bbox,
                "area": (dem_bbox[2]-dem_bbox[0]) * (dem_bbox[3]-dem_bbox[1])
            }
            self.referent_names[pointed] = pointed

        return pointed
