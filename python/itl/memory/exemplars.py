from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from pycocotools import mask


class Exemplars:
    """
    Inventory of exemplars encountered, stored as scenes comprising the raw image
    along with object masks & feature vectors (output from vision module's feature
    encoder component). Primarily used for incremental few-shot registration of novel
    concepts.
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Storage of scenes; each scene consists of a raw image and a list of objects,
        # where each object is specified by a binary segmentation mask and annotated
        # with the feature vector extracted with the vision encoder module
        self.scenes = []

        # Labeling as dict from visual concept to list of pointers to vectors
        self.exemplars_pos = {
            "cls": defaultdict(set), "att": defaultdict(set), "rel": defaultdict(set)
        }
        self.exemplars_neg = {
            "cls": defaultdict(set), "att": defaultdict(set), "rel": defaultdict(set)
        }

        # Keep classifiers trained from current storage of positive/negative exemplars
        self.binary_classifiers = { "cls": {}, "att": {}, "rel": {} }

    def __repr__(self):
        conc_desc = f"concepts={len(self.exemplars_pos['cls'])}" \
            f"/{len(self.exemplars_pos['att'])}" \
            f"/{len(self.exemplars_pos['rel'])}"
        return f"Exemplars({conc_desc})"

    def add_exs(self, scene_img, exemplars, pointers):

        # Return value; storage indices ((scene_id, object_id)) of any new added exemplars
        added_inds = []

        # Add new scene image and initialize empty object list
        if scene_img is not None:
            self.scenes.append((scene_img, []))

        # Add exemplars to appropriate scene object lists
        for ex_info in exemplars:
            obj_info = {
                "mask": mask.encode(np.asfortranarray(ex_info["mask"])),
                "f_vec": ex_info["f_vec"]
            }

            if ex_info["scene_id"] is None:
                # Objects in the new provided scene; scene_img must have been provided
                assert scene_img is not None
                scene_id = len(self.scenes) - 1
            else:
                # Objects in a scene already stored
                assert isinstance(ex_info["scene_id"], int)
                scene_id = ex_info["scene_id"]

            # Add to object list and log indices
            self.scenes[scene_id][1].append(obj_info)
            obj_id = len(self.scenes[scene_id][1]) - 1
            added_inds.append((scene_id, obj_id))

        # Iterate through pointers to update concept exemplar index sets
        xb_updated = set()      # Collection of concepts with updated exemplar sets
        for (conc_type, conc_ind, pol), objs in pointers.items():
            for is_new, xi in objs:
                if is_new:
                    # Referring to one of the provided list of new exemplars (`exemplars`)
                    scene_id, obj_id = added_inds[xi]
                else:
                    # Referring to an already existing exemplar
                    scene_id, obj_id = xi

                if pol == "pos":
                    self.exemplars_pos[conc_type][conc_ind].add((scene_id, obj_id))
                elif pol == "neg":
                    self.exemplars_neg[conc_type][conc_ind].add((scene_id, obj_id))
                else:
                    raise ValueError("Bad concept polarity value")

                xb_updated.add((conc_type, conc_ind))

        for conc_type, conc_ind in xb_updated:
            # Update binary classifier if needed
            if len(self.exemplars_pos[conc_type][conc_ind]) > 0 and \
                len(self.exemplars_neg[conc_type][conc_ind]) > 0:
                # If we have at least one positive & negative exemplars each,
                # (re-)train a binary classifier and store it

                # Prepare training data (X, y) from exemplar storage
                pos_inds = self.exemplars_pos[conc_type][conc_ind]
                neg_inds = self.exemplars_neg[conc_type][conc_ind]
                X = np.stack([
                    self.scenes[scene_id][1][obj_id]["f_vec"]
                    for scene_id, obj_id in pos_inds
                ] + [
                    self.scenes[scene_id][1][obj_id]["f_vec"]
                    for scene_id, obj_id in neg_inds
                ])
                y = ([1] * len(pos_inds)) + ([0] * len(neg_inds))

                # Induce binary decision boundary by fitting a classifier and
                # update
                # bin_clf = SVC(C=1000, probability=True)
                bin_clf = KNeighborsClassifier(n_neighbors=min(len(X), 5), weights="distance")
                bin_clf.fit(X, y)
                self.binary_classifiers[conc_type][conc_ind] = bin_clf

            else:
                # Cannot induce any decision boundary with either positive or
                # negative examples only
                self.binary_classifiers[conc_type][conc_ind] = None

        return added_inds
