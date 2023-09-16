"""
Script for dry-running the ITL environment with a student and a user-controlled
teacher (in Unity lingo, Behavior Type: Heuristics for teacher agent)
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import re
import csv
import uuid
import logging
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

import tqdm
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../python/itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dirs_with_results = []
    outputs_root_dir = os.path.join(cfg.paths.root_dir, "outputs")
    for date_dir in os.listdir(outputs_root_dir):
        if not os.path.isdir(os.path.join(outputs_root_dir, date_dir)): continue

        for time_dir in os.listdir(os.path.join(outputs_root_dir, date_dir)):
            full_out_dir = os.path.join(outputs_root_dir, date_dir, time_dir)

            if not os.path.isdir(full_out_dir):
                continue
            
            if "results" in os.listdir(full_out_dir):
                dirs_with_results.append((date_dir, time_dir))

    results_cumulReg = {}
    # results_confMat = defaultdict(lambda: defaultdict(dict))
    # results_learningCurve = defaultdict(dict)

    num_concepts = {}

    # Collect results
    for res_dir in tqdm.tqdm(dirs_with_results, total=len(dirs_with_results)):
        res_dir = os.path.join(outputs_root_dir, *res_dir, "results")
        dir_contents = list(os.listdir(res_dir))

        for data in dir_contents:
            name_parse = re.findall(r"(.+)_(.+)_(.+)_(\d+)", data)[0]
            data_type, *exp_config, seed = name_parse
            strat_fb, strat_gn = exp_config

            if data_type == "cumulReg":
                # Cumulative regret curve data
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)

                    # Header column; 'episode,cumulative_regret'
                    _ = next(reader)

                    curve = np.array([[int(row[0]), int(row[1])] for row in reader])

                    if (strat_fb, strat_gn) in results_cumulReg:
                        stats_agg = results_cumulReg[(strat_fb, strat_gn)]
                        for ep_num, regret in curve:
                            if ep_num in stats_agg:
                                stats_agg[ep_num].append(regret)
                            else:
                                stats_agg[ep_num] = [regret]
                    else:
                        results_cumulReg[(strat_fb, strat_gn)] = {
                            ep_num: [regret] for ep_num, regret in curve
                        }

            elif data_type.startswith("confMat"):
                # Confusion matrix data
                num_exs = int(data_type.strip("confMat"))

                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)
                    concepts = next(reader)

                    confMat = np.array([[float(d) for d in row] for row in reader])

                    if (strat_fb, strat_gn) in results_confMat[diff][num_exs]:
                        stats_agg = results_confMat[diff][num_exs][(strat_fb, strat_gn)]
                        stats_agg["matrix"].append(confMat)
                        stats_agg["num_test_suites"] += 1
                        assert stats_agg["concepts"] == concepts
                    else:
                        results_confMat[diff][num_exs][(strat_fb, strat_gn)] = {
                            "matrix": [confMat],
                            "num_test_suites": 1,
                            "concepts": concepts
                        }

            elif data_type.startswith("mAPs"):
                # Learning curve data (performance measured by mAP)
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)
                    _ = next(reader)

                    curve = [(int(row[0]), float(row[1])) for row in reader]

                    if (strat_fb, strat_gn) in results_learningCurve[diff]:
                        stats_agg = results_learningCurve[diff][(strat_fb, strat_gn)]
                        for num_exs, mAP in curve:
                            if num_exs in stats_agg:
                                stats_agg[num_exs].append(mAP)
                            else:
                                stats_agg[i] = [mAP]
                    else:
                        results_learningCurve[diff][(strat_fb, strat_gn)] = {
                            num_exs: [mAP] for num_exs, mAP in curve
                        }

            else:
                continue

    # Pre-defined ordering for listing legends
    config_ord = [
        "semOnly_minHelp", "semOnly_medHelp", "semOnly_maxHelp",
        "semNeg_maxHelp",
        "semNegScal_maxHelp"
    ]
    config_aliases = {
        "semOnly_minHelp": "minHelp",
        "semOnly_medHelp": "medHelp",
        "semOnly_maxHelp": "maxHelp_semOnly",
        "semNeg_maxHelp": "maxHelp_semNeg",
        "semNegScal_maxHelp": "maxHelp_semNegScal"
    }   # To be actually displayed in legend
    config_colors = {
        "semOnly_minHelp": "tab:red",
        "semOnly_medHelp": "tab:orange",
        "semOnly_maxHelp": "tab:green",
        "semNeg_maxHelp": "tab:blue",
        "semNegScal_maxHelp": "tab:purple"
    }

    # Aggregate and visualize: cumulative regret curve
    _, ax = plt.subplots()
    ymax = 0

    for exp_config, data in results_cumulReg.items():
        strat_fb, strat_gn = exp_config
        stats = [
            (i, np.mean(rgs), 1.96 * np.std(rgs)/np.sqrt(len(rgs)))
            for i, rgs in data.items()
        ]
        ymax = max(ymax, max(mrg+cl for _, mrg, cl in stats))

        # Plot mean curve
        ax.plot(
            [i+1 for i, _, _ in stats],
            [mrg for _, mrg, _ in stats],
            label=f"{strat_gn}_{strat_fb}",
            color=config_colors[f"{strat_gn}_{strat_fb}"]
        )
        # Plot confidence intervals
        ax.fill_between(
            [i+1 for i, _, _ in stats],
            [mrg-cl for _, mrg, cl in stats],
            [mrg+cl for _, mrg, cl in stats],
            color=config_colors[f"{strat_gn}_{strat_fb}"], alpha=0.2
        )

    # Plot curve
    ax.set_xlabel("# training episodes")
    ax.set_ylabel("cumulative regret")
    ax.set_ylim(0, ymax * 1.2)
    ax.grid()

    # Ordering legends according to the prespecified ordering above
    handles, labels = ax.get_legend_handles_labels()
    hls_sorted = sorted(
        [(h, l) for h, l in zip(handles, labels)],
        key=lambda x: config_ord.index(x[1])
    )
    handles = [hl[0] for hl in hls_sorted]
    labels = [config_aliases.get(hl[1], hl[1]) for hl in hls_sorted]
    ax.legend(handles, labels)
    
    ax.set_title(f"Cumulative regret curve (N={len(data[0])} per config)")
    plt.savefig(os.path.join(cfg.paths.outputs_dir, f"cumulReg.png"))

    # # Aggregate and visualize: confusion matrices
    # for diff, per_diff in results_confMat.items():
    #     last_num_exs = max(per_diff)
    #     for num_exs, per_config in per_diff.items():
    #         for exp_config, data in per_config.items():
    #             strat_fb, strat_gn = exp_config
    #             config_label = f"{strat_gn}_{strat_fb}"

    #             num_concepts[diff] = len(data["matrix"][0])

    #             if num_exs == last_num_exs:
    #                 # Draw confusion matrix
    #                 fig = plt.figure()
    #                 draw_matrix(
    #                     sum(data["matrix"]) / data["num_test_suites"],
    #                     data["concepts"], data["concepts"], fig.gca()    # Binary choice mode
    #                 )
    #                 plt.suptitle(f"Confusion matrix for ({diff}; N={len(data['matrix'])} per config)", fontsize=16)
    #                 plt.title(f"{config_label} agent", pad=18)
    #                 plt.tight_layout()
    #                 plt.savefig(os.path.join(cfg.paths.outputs_dir, f"confMat_{diff}_{config_label}.png"))

    # # Aggregate and visualize: learning curve
    # for diff, agg_stats in results_learningCurve.items():
    #     fig, ax = plt.subplots()

    #     for exp_config, data in agg_stats.items():
    #         strat_fb, strat_gn = exp_config
    #         data = sorted(data.items())
    #         stats = [
    #             (num_exs, np.mean(mAPs), 1.96 * np.std(mAPs)/np.sqrt(len(mAPs)))
    #             for num_exs, mAPs in data
    #         ]

    #         # Temporary adjustment
    #         if diff=="fineEasy":
    #             stats = [s for s in stats if s[0] <= 30]

    #         # Plot mean curve
    #         ax.plot(
    #             [num_exs for num_exs, _, _ in stats],
    #             [mmAP for _, mmAP, _ in stats],
    #             label=f"{strat_gn}_{strat_fb}",
    #             color=config_colors[f"{strat_gn}_{strat_fb}"]
    #         )
    #         ax.plot(
    #             [0, stats[0][0]], [1/num_concepts[diff], stats[0][1]],
    #             color=config_colors[f"{strat_gn}_{strat_fb}"], linestyle="dashed"
    #         )
    #         # Plot confidence intervals
    #         ax.fill_between(
    #             [0]+[num_exs for num_exs, _, _ in stats],
    #             [1/num_concepts[diff]]+[mmAP-cl for _, mmAP, cl in stats],
    #             [1/num_concepts[diff]]+[mmAP+cl for _, mmAP, cl in stats],
    #             color=config_colors[f"{strat_gn}_{strat_fb}"], alpha=0.2
    #         )

    #     # Plot curve
    #     ax.set_xlabel("# training examples")
    #     ax.set_ylabel("mAP score")
    #     ax.set_xlim(0, stats[-1][0])
    #     ax.set_ylim(0, 1)
    #     ax.grid()

    #     # Ordering legends according to the prespecified ordering above
    #     handles, labels = ax.get_legend_handles_labels()
    #     hls_sorted = sorted(
    #         [(h, l) for h, l in zip(handles, labels)],
    #         key=lambda x: config_ord.index(x[1])
    #     )
    #     handles = [hl[0] for hl in hls_sorted]
    #     labels = [config_aliases.get(hl[1], hl[1]) for hl in hls_sorted]
    #     plt.legend(handles, labels)
        
    #     plt.title(f"Learning curve ({diff}; N={len(data[0][1])} per config)")
    #     plt.savefig(os.path.join(cfg.paths.outputs_dir, f"learningCurve_{diff}.png"))

    # # Report final mAP scores on terminal
    # for diff, agg_stats in results_learningCurve.items():
    #     first_num_exs = sorted(list(agg_stats.values())[0])[0]
    #     middle_num_exs = sorted(list(agg_stats.values())[0])[2]
    #     last_num_exs = sorted(list(agg_stats.values())[0], reverse=True)[0]

    #     # Temporary adjustment
    #     if diff=="fineEasy":
    #         last_num_exs = 30

    #     for num_exs in [first_num_exs, middle_num_exs, last_num_exs]:
    #         print("")
    #         logger.info(f"mAP scores after {num_exs} examples ({diff}):")

    #         final_mAPs = {
    #             f"{strat_gn}_{strat_fb}": data[num_exs]
    #             for (strat_fb, strat_gn), data in agg_stats.items()
    #         }
    #         for cfg in sorted(final_mAPs, key=lambda x: config_ord.index(x)):
    #             logger.info("\t"+f"{config_aliases.get(cfg, cfg)}: {float(np.mean(final_mAPs[cfg]))}")


if __name__ == "__main__":
    main()
