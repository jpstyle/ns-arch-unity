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

        # Keep classifiers trained from current storage of positive/negative exemplars
        self.binary_classifiers = { "cls": {}, "att": {}, "rel": {} }

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

        # Return value; storage indices of the added feature vectors per concept type
        added_inds = {}

        for conc_type in ["cls", "att", "rel"]:
            if conc_type in pointers:
                assert conc_type in f_vecs
                new_vecs = isinstance(f_vecs[conc_type], np.ndarray)

                if new_vecs:
                    # Given raw vectors, most likely not witnessed by the exemplar
                    # base yet; add to feature vector matrix
                    N_C = len(self.storage_vec[conc_type])
                    D = f_vecs[conc_type].shape[-1]

                    self.storage_vec[conc_type] = np.concatenate([
                        self.storage_vec[conc_type].reshape(-1, D),
                        f_vecs[conc_type]
                    ])
                    added_inds[conc_type] = (N_C, N_C+f_vecs[conc_type].shape[0])
                else:
                    # Given pointers to existing vectors in storage; don't add redundant
                    # entries
                    pass

                # Positive/negative exemplar tagging by vector row index
                for conc_ind, (exs_pos, exs_neg) in pointers[conc_type].items():
                    if new_vecs:
                        exs_pos = {fv_id+N_C for fv_id in exs_pos}
                        exs_neg = {fv_id+N_C for fv_id in exs_neg}
                    else:
                        exs_pos = {f_vecs[conc_type][fv_id] for fv_id in exs_pos}
                        exs_neg = {f_vecs[conc_type][fv_id] for fv_id in exs_neg}
                    self.exemplars_pos[conc_type][conc_ind] |= exs_pos
                    self.exemplars_neg[conc_type][conc_ind] |= exs_neg

                    if len(self.exemplars_pos[conc_type][conc_ind]) > 0 and \
                        len(self.exemplars_neg[conc_type][conc_ind]) > 0:
                        # If we have at least one positive & negative exemplars each,
                        # (re-)train a binary classifier and store it

                        # Prepare training data (X, y) from exemplar storage
                        pos_inds = list(self.exemplars_pos[conc_type][conc_ind])
                        neg_inds = list(self.exemplars_neg[conc_type][conc_ind])
                        X = self.storage_vec[conc_type][pos_inds + neg_inds]
                        y = ([1] * len(pos_inds)) + ([0] * len(neg_inds))

                        # Induce binary decision boundary by fitting a classifier and
                        # update
                        # bin_clf = SVC(C=1000, probability=True)
                        bin_clf = KNeighborsClassifier(n_neighbors=min(len(X), 10), weights="distance")
                        bin_clf.fit(X, y)
                        self.binary_classifiers[conc_type][conc_ind] = bin_clf

                    else:
                        # Cannot induce any decision boundary with either positive or
                        # negative examples only
                        self.binary_classifiers[conc_type][conc_ind] = None

        return added_inds
