from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC


class Exemplars:
    """
    Inventory of exemplars encountered, stored as feature vectors (output from vision
    module's feature extractor component). Primarily used for incremental few-shot
    registration of novel concepts.
    """
    TARGET_MAX_WH = 80

    def __init__(self, cfg):
        self.cfg = cfg

        # Storage of vectors, and pointers to their source
        self.storage_vec = {
            "cls": np.array([], dtype=np.float32),
            "att": np.array([], dtype=np.float32),
            "rel": np.array([], dtype=np.float32)
        }

        # Labeling as dict from visual concept to list of pointers to vectors
        self.exemplars_pos = {
            "cls": defaultdict(set), "att": defaultdict(set), "rel": defaultdict(set)
        }
        self.exemplars_neg = {
            "cls": defaultdict(set), "att": defaultdict(set), "rel": defaultdict(set)
        }

        # Also keep classifiers trained from current storage of positive/negative exemplars
        self.binary_classifiers = { "cls": {}, "att": {}, "rel": {} }

        # Track numbers of successful concept membership predictions; used for modulating
        # (epistemic) uncertainty estimated by feature vector density
        self.success_counts = {
            "cls": defaultdict(int), "att": defaultdict(int), "rel": defaultdict(int)
        }

    def __repr__(self):
        conc_desc = f"concepts={len(self.exemplars_pos['cls'])}" \
            f"/{len(self.exemplars_pos['att'])}" \
            f"/{len(self.exemplars_pos['rel'])}"
        return f"Exemplars({conc_desc})"

    def __getitem__(self, item):
        conc_ind, conc_type = item

        pos_exs = self.exemplars_pos[conc_type][conc_ind]
        pos_exs = self.storage_vec[conc_type][list(pos_exs)]
        neg_exs = self.exemplars_neg[conc_type][conc_ind]
        neg_exs = self.storage_vec[conc_type][list(neg_exs)]

        return { "pos": pos_exs, "neg": neg_exs }

    def refresh(self):
        # Not expected to be called by user; for compact exemplar storage during injection
        self.__init__()
    
    def concepts_covered_gen(self):
        for conc_type in ["cls", "att", "rel"]:
            pos_covered = set(self.exemplars_pos[conc_type])
            neg_covered = set(self.exemplars_neg[conc_type])

            for conc_ind in pos_covered | neg_covered:
                yield (conc_ind, conc_type)

    def add_exs(self, f_vecs, pointers):

        for conc_type in ["cls", "att", "rel"]:
            if conc_type in pointers:
                assert conc_type in f_vecs
                N_C = len(self.storage_vec[conc_type])

                # Add to feature vector matrix
                self.storage_vec[conc_type] = np.concatenate([
                    self.storage_vec[conc_type].reshape(-1, f_vecs[conc_type].shape[-1]),
                    f_vecs[conc_type]
                ])

                # Positive/negative exemplar tagging by vector row index
                for conc_ind, (exs_pos, exs_neg) in pointers[conc_type].items():
                    exs_pos = {fv_id+N_C for fv_id in exs_pos}
                    exs_neg = {fv_id+N_C for fv_id in exs_neg}
                    self.exemplars_pos[conc_type][conc_ind] |= exs_pos
                    self.exemplars_neg[conc_type][conc_ind] |= exs_neg

                    # If we have at least one positive & negative exemplars each,
                    # (re-)train a binary classifier and store it
                    if len(self.exemplars_pos[conc_type][conc_ind]) > 0 and \
                        len(self.exemplars_neg[conc_type][conc_ind]) > 0:
                        # Prepare training data (X, y) from exemplar storage
                        pos_inds = list(self.exemplars_pos[conc_type][conc_ind])
                        neg_inds = list(self.exemplars_neg[conc_type][conc_ind])
                        X = self.storage_vec[conc_type][pos_inds + neg_inds]
                        y = ([1] * len(pos_inds)) + ([0] * len(neg_inds))

                        # Fit classifier and update
                        # bin_clf = SVC(C=1000, probability=True)
                        bin_clf = KNeighborsClassifier(n_neighbors=min(len(X), 10), weights="distance")
                        bin_clf.fit(X, y)
                        self.binary_classifiers[conc_type][conc_ind] = bin_clf
                    else:
                        self.binary_classifiers[conc_type][conc_ind] = None
