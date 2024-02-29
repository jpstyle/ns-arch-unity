"""
Visual analysis of concept exemplar vectors in low-dimensional space
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import uuid
import random
import warnings
warnings.filterwarnings("ignore")

import umap
import umap.plot
import hydra
import tqdm as tqdm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from bokeh.io import save
from bokeh.models import Title
from bokeh.layouts import column

from python.itl import ITLAgent


TAB = "\t"

OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../python/itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Set seed
    pl.seed_everything(cfg.seed)

    # Set up agent
    agent = ITLAgent(cfg)
    exemplars = agent.lt_mem.exemplars

    accs = {}
    conc_types = ["cls"] #, "att", "rel"]
    for conc_type in conc_types:
        if conc_type == "cls" or conc_type == "att":
            pos_exs_inds = exemplars.exemplars_pos[conc_type]
            neg_exs_inds = exemplars.exemplars_neg[conc_type]
            all_exs_inds = list(set.union(*pos_exs_inds.values(), *neg_exs_inds.values()))
            ind_map = { xi: i for i, xi in enumerate(all_exs_inds) }
            vectors = np.stack([
                exemplars.scenes[scene_id][1][obj_id]["f_vec"]
                for scene_id, obj_id in all_exs_inds
            ])

            # Dimensionality reduction down to 2D by UMAP, for visual inspection
            mapper = umap.UMAP().fit(vectors)

            # Plot for each concept
            umap.plot.output_file(
                os.path.join(cfg.paths.outputs_dir, f"{conc_type}_embs.html")
            )

            plots = []
            for c in tqdm.tqdm(pos_exs_inds, total=len(pos_exs_inds), desc=f"{conc_type}_embs"):
                concept_name = agent.lt_mem.lexicon.d2s[(c, conc_type)][0][0]
                concept_name = concept_name.replace("/", "_")

                # Mapping pair indices to union vector matrix index
                pos_exs_inds_mapped = [ind_map[xi] for xi in pos_exs_inds[c]]
                neg_exs_inds_mapped = [ind_map[xi] for xi in neg_exs_inds[c]]

                # Evaluating exemplar sets by binary classification performance on
                # random 80:20 train/test split
                pos_shuffled = random.sample(pos_exs_inds_mapped, len(pos_exs_inds[c]))
                pos_train = pos_shuffled[:int(0.8*len(pos_shuffled))]
                pos_test = pos_shuffled[int(0.8*len(pos_shuffled)):]
                neg_shuffled = random.sample(neg_exs_inds_mapped, len(neg_exs_inds[c]))
                neg_train = neg_shuffled[:int(0.8*len(neg_shuffled))]
                neg_test = neg_shuffled[int(0.8*len(neg_shuffled)):]

                X = vectors[pos_train + neg_train]
                y = ([1] * len(pos_train)) + ([0] * len(neg_train))

                # Fit classifier and run on test set
                bin_clf = KNeighborsClassifier(n_neighbors=min(len(X), 10), weights="distance")
                # bin_clf = SVC(C=1000, probability=True, random_state=cfg.seed)
                bin_clf.fit(X, y)
                true_pos = bin_clf.predict_proba(vectors[pos_test])[:,1] > 0.5
                true_neg = bin_clf.predict_proba(vectors[neg_test])[:,0] > 0.5
                false_pos = bin_clf.predict_proba(vectors[neg_test])[:,1] > 0.5
                false_neg = bin_clf.predict_proba(vectors[pos_test])[:,0] > 0.5

                accs[(concept_name, conc_type)] = {
                    "precision": true_pos.sum() / (true_pos.sum()+false_pos.sum()),
                    "recall": true_pos.sum() / (true_pos.sum()+false_neg.sum()),
                    "accuracy": (true_pos.sum()+true_neg.sum()) / (len(pos_test)+len(neg_test))
                }

                labels = [
                    "p" if fv_i in pos_exs_inds_mapped
                        else "n" if fv_i in neg_exs_inds_mapped else "."
                    for fv_i in range(len(vectors))
                ]
                hover_data = pd.DataFrame(
                    {
                        "index": np.arange(len(vectors)),
                        "label": labels
                    }
                )

                # Plot data and save
                p = umap.plot.interactive(
                    mapper, labels=labels, hover_data=hover_data,
                    color_key={ "p": "#3333FF", "n": "#CC0000", ".": "#A0A0A0" },
                    point_size=5
                )
                p.add_layout(Title(text=concept_name, align="center"), "above")
                plots.append(p)

            save(column(*plots))
    
    for (concept_name, conc_type), acc in accs.items():
        accuracy = acc["accuracy"]
        precision = acc["precision"]
        recall = acc["recall"]
        f1 = 2 * precision * recall / (precision+recall)

        print(f"Scores for {concept_name} ({conc_type}):")
        print(f"{TAB}Accuracy: {accuracy}")
        print(f"{TAB}Precision: {precision}")
        print(f"{TAB}Recall: {recall}")
        print(f"{TAB}F1: {f1}")

if __name__ == "__main__":
    main()
