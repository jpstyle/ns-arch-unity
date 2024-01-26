"""
Script for collecting and summarizing statistics recorded from the exp_run.py script.
Any results existing in the outputs folder will be gathered and summarized, as long
as they exist in the right arrangement (i.e., that expected after running exp_run.py
scripts in appropriate settings).
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
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
    for group_dir in os.listdir(outputs_root_dir):
        if not os.path.isdir(os.path.join(outputs_root_dir, group_dir)): continue
        if group_dir.startswith("_"): continue      # Ignore groups prefixed with "_"

        for run_dir in os.listdir(os.path.join(outputs_root_dir, group_dir)):
            full_out_dir = os.path.join(outputs_root_dir, group_dir, run_dir)

            if not os.path.isdir(full_out_dir):
                continue
            
            if "results" in os.listdir(full_out_dir):
                dirs_with_results.append((group_dir, run_dir))

    results_cumulReg = {}
    results_learningCurve = defaultdict(lambda: defaultdict(list))
    results_confMat = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )               # This is one ugly nesting but bear with me, myself...

    all_seeds = set()

    # Collect results
    for res_dir in tqdm.tqdm(dirs_with_results, total=len(dirs_with_results)):
        res_dir = os.path.join(outputs_root_dir, *res_dir, "results")
        dir_contents = list(os.listdir(res_dir))

        for data in dir_contents:
            name_parse = data.split(".")[0].split("_")
            data_type = name_parse[0]
            strat_fb, strat_gn, strat_as = name_parse[1], name_parse[2], name_parse[3]
            seed = name_parse[4]

            all_seeds.add(seed)

            if data_type == "cumulReg":
                # Cumulative regret curve data
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)

                    # Header column; 'episode,cumulative_regret'
                    _ = next(reader)

                    curve = np.array([[int(ep_num), int(regret)] for ep_num, regret in reader])

                    # Aggregate cumulative regret curve data
                    strat_combi = (strat_fb, strat_gn, strat_as)
                    if strat_combi in results_cumulReg:
                        stats_agg = results_cumulReg[strat_combi]
                        for ep_num, regret in curve:
                            if ep_num in stats_agg:
                                stats_agg[ep_num].append(regret)
                            else:
                                stats_agg[ep_num] = [regret]
                    else:
                        results_cumulReg[strat_combi] = {
                            ep_num: [regret] for ep_num, regret in curve
                        }

            elif data_type == "outputs":
                # Test question-answer pairs, from which learning curves and confusion
                # matrices can be plotted
                num_exs = int(name_parse[-1])
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)

                    # Header column; 'episode,ground_truth,answer'
                    _ = next(reader)

                    # Summarize into confusion matrix, then reduce into accuracy
                    confusion_matrix = defaultdict(lambda: defaultdict(int))
                    for _, gt_conc, ans_conc in reader:
                        confusion_matrix[gt_conc][ans_conc] += 1
                        confusion_matrix[gt_conc]["total"] += 1

                    all_concs = list(confusion_matrix)      # Collect and assign order

                    confusion_matrix = {
                        gt_conc: {
                            conc: answers[conc] / answers["total"] if conc in answers else 0
                            for conc in all_concs
                        }
                        for gt_conc, answers in confusion_matrix.items()
                    }
                    accuracy = sum(
                        answers[gt_conc] for gt_conc, answers in confusion_matrix.items()
                    ) / len(confusion_matrix)           # Mean accuracy across ground truths

                    # Aggregate learning curve data
                    strat_combi = (strat_fb, strat_gn, strat_as)
                    results_learningCurve[strat_combi][num_exs].append(accuracy)

                    # Aggregate confusion matrix data
                    for gt_conc, answers in confusion_matrix.items():
                        for ans_conc, ratio in answers.items():
                            matrix_slot = results_confMat[strat_combi][gt_conc][ans_conc]
                            matrix_slot[num_exs].append(ratio)

            else:
                continue

    # Pre-defined ordering for listing legends
    config_ord = [
        # "semOnly_minHelp_doNotLearn", "semOnly_minHelp_alwaysLearn",
        "semOnly_medHelp_doNotLearn", "semOnly_medHelp_alwaysLearn",
        "semOnly_maxHelpNoexpl_doNotLearn", "semOnly_maxHelpNoexpl_alwaysLearn",
        "semOnly_maxHelpExpl_doNotLearn", "semOnly_maxHelpExpl_alwaysLearn",
        "semOnly_maxHelpExpl2_doNotLearn", "semOnly_maxHelpExpl2_alwaysLearn",
        # "semNeg_maxHelp_doNotLearn", "semNeg_maxHelp_alwaysLearn",
        # "semNegScal_maxHelp_doNotLearn", "semNegScal_maxHelp_alwaysLearn",
    ]
    config_aliases = {
        # "semOnly_minHelp_doNotLearn": "minHelp_doNotLearn",
        # "semOnly_minHelp_alwaysLearn": "minHelp_alwaysLearn",

        "semOnly_medHelp_doNotLearn": "Vision",
        "semOnly_medHelp_alwaysLearn": "Vision",

        "semOnly_maxHelpNoexpl_doNotLearn": "Vision+Generic",
        "semOnly_maxHelpNoexpl_alwaysLearn": "Vision+Generic",

        "semOnly_maxHelpExpl_doNotLearn": "Vision+Generic+ExplSuff",
        "semOnly_maxHelpExpl_alwaysLearn": "Vision+Generic+ExplSuff",

        "semOnly_maxHelpExpl2_doNotLearn": "Vision+Generic+ExplSuffCtfl",
        "semOnly_maxHelpExpl2_alwaysLearn": "Vision+Generic+ExplSuffCtfl",

        # "semNeg_maxHelp_doNotLearn": "maxHelp_semNeg_doNotLearn",
        # "semNeg_maxHelp_alwaysLearn": "maxHelp_semNeg_alwaysLearn",
        
        # "semNegScal_maxHelp_doNotLearn": "maxHelp_semNegScal_doNotLearn",
        # "semNegScal_maxHelp_alwaysLearn": "maxHelp_semNegScal_alwaysLearn",
    }   # To be actually displayed in legend
    config_colors = {
        # "semOnly_minHelp_doNotLearn": "tab:red",
        # "semOnly_minHelp_alwaysLearn": "tab:red",

        "semOnly_medHelp_doNotLearn": "tab:orange",
        "semOnly_medHelp_alwaysLearn": "tab:orange",

        "semOnly_maxHelpNoexpl_doNotLearn": "tab:green",
        "semOnly_maxHelpNoexpl_alwaysLearn": "tab:green",

        "semOnly_maxHelpExpl_doNotLearn": "tab:blue",
        "semOnly_maxHelpExpl_alwaysLearn": "tab:blue",

        "semOnly_maxHelpExpl2_doNotLearn": "tab:purple",
        "semOnly_maxHelpExpl2_alwaysLearn": "tab:purple",

        # "semNeg_maxHelp_doNotLearn": "tab:blue",
        # "semNeg_maxHelp_alwaysLearn": "tab:blue",

        # "semNegScal_maxHelp_doNotLearn": "tab:purple",
        # "semNegScal_maxHelp_alwaysLearn": "tab:purple",
    }
    config_lineStyles = {
        # "semOnly_minHelp_doNotLearn": "--",
        # "semOnly_minHelp_alwaysLearn": "-",

        "semOnly_medHelp_doNotLearn": "-",
        "semOnly_medHelp_alwaysLearn": "-",

        "semOnly_maxHelpNoexpl_doNotLearn": "-",
        "semOnly_maxHelpNoexpl_alwaysLearn": "-",

        "semOnly_maxHelpExpl_doNotLearn": "-",
        "semOnly_maxHelpExpl_alwaysLearn": "-",

        "semOnly_maxHelpExpl2_doNotLearn": "-",
        "semOnly_maxHelpExpl2_alwaysLearn": "-",

        # "semNeg_maxHelp_doNotLearn": "--",
        # "semNeg_maxHelp_alwaysLearn": "-",

        # "semNegScal_maxHelp_doNotLearn": "--",
        # "semNegScal_maxHelp_alwaysLearn": "-",
    }

    # Aggregate and visualize: cumulative regret curve
    if len(results_cumulReg) > 0:
        _, ax = plt.subplots(figsize=(8, 6), dpi=80)
        ymax = 0

        for (strat_fb, strat_gn, strat_as), data in results_cumulReg.items():
            stats = [
                (i, np.mean(rgs), 1.96 * np.std(rgs)/np.sqrt(len(rgs)))
                for i, rgs in data.items()
            ]
            ymax = max(ymax, max(mrg+cl for _, mrg, cl in stats))

            # Plot mean curve
            ax.plot(
                [i+1 for i, _, _ in stats],
                [mrg for _, mrg, _ in stats],
                label=f"{strat_gn}_{strat_fb}_{strat_as}",
                color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"],
                linestyle=config_lineStyles[f"{strat_gn}_{strat_fb}_{strat_as}"]
            )
            # Plot confidence intervals
            ax.fill_between(
                [i+1 for i, _, _ in stats],
                [mrg-cl for _, mrg, cl in stats],
                [mrg+cl for _, mrg, cl in stats],
                color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], alpha=0.2
            )

        # Plot curve
        ax.set_xlabel("# training episodes")
        ax.set_ylabel("cumulative regret")
        ax.set_ylim(0, ymax * 1.1)
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
        
        ax.set_title(f"Cumulative regret curve (N={len(all_seeds)} per config)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"cumulReg.png"))

    # Aggregate and visualize: learning curve
    if len(results_learningCurve) > 0:
        _, ax = plt.subplots(figsize=(8, 6), dpi=80)

        for (strat_fb, strat_gn, strat_as), data in results_learningCurve.items():
            data = sorted([entry for entry in data.items()], key=lambda x: x[0])
            stats = [
                (num_exs, np.mean(accs), 1.96 * np.std(accs)/np.sqrt(len(accs)))
                for num_exs, accs in data
            ]

            # Plot mean curve
            ax.plot(
                [num_exs for num_exs, _, _ in stats],
                [mmAP for _, mmAP, _ in stats],
                label=f"{strat_gn}_{strat_fb}_{strat_as}",
                color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"],
                linestyle=config_lineStyles[f"{strat_gn}_{strat_fb}_{strat_as}"]
            )
            ax.plot(
                [0, stats[0][0]], [0, stats[0][1]],
                color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], linestyle="dashed"
            )
            # Plot confidence intervals
            ax.fill_between(
                [0]+[num_exs for num_exs, _, _ in stats],
                [0]+[mmAP-cl for _, mmAP, cl in stats],
                [0]+[mmAP+cl for _, mmAP, cl in stats],
                color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], alpha=0.2
            )

        # Plot curve
        ax.set_xlabel("# training examples")
        ax.set_ylabel("mAP score")
        ax.set_xlim(0, stats[-1][0])
        ax.set_ylim(0, 1)
        ax.grid()

        # Ordering legends according to the prespecified ordering above
        handles, labels = ax.get_legend_handles_labels()
        hls_sorted = sorted(
            [(h, l) for h, l in zip(handles, labels)],
            key=lambda x: config_ord.index(x[1])
        )
        handles = [hl[0] for hl in hls_sorted]
        labels = [config_aliases.get(hl[1], hl[1]) for hl in hls_sorted]
        plt.legend(handles, labels)
        
        plt.title(f"Learning curve (N={len(all_seeds)} per config)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"learningCurve.png"))

    # Aggregate and visualize: confusion matrices
    if len(results_confMat) > 0:
        fig = plt.figure(figsize=(8, 6), dpi=80)
        gs = fig.add_gridspec(len(all_concs), len(all_concs), hspace=0, wspace=0)
        axs = gs.subplots(sharex='col', sharey='row')

        for (strat_fb, strat_gn, strat_as), data in results_confMat.items():
            # Draw a confusion matrix, with curve plots as matrix entries (instead of single
            # numbers at the last)
            for gt_conc, per_gt_conc in data.items():
                for ans_conc, stats in per_gt_conc.items():
                    # Slot to fill with curve
                    i = all_concs.index(gt_conc)
                    j = all_concs.index(ans_conc)
                    ax = axs[i][j]

                    # Set tick parameters, limits, etc.
                    ax.tick_params(left=False, right=True, labelleft=False, labelright=True)
                    ax.set_xlabel(ans_conc)
                    ax.set_ylabel(gt_conc)
                    ax.set_ylim(0, 1)
                    ax.xaxis.set_label_position('top')
                    ax.set_yticks([0.25, 0.5, 0.75, 1])

                    # Collect stats
                    stats = sorted([entry for entry in stats.items()], key=lambda x: x[0])
                    stats = [
                        (num_exs, np.mean(ratios), 1.96 * np.std(ratios)/np.sqrt(len(ratios)))
                        for num_exs, ratios in stats
                    ]

                    # Plot mean curve
                    ax.plot(
                        [num_exs for num_exs, _, _ in stats],
                        [mmAP for _, mmAP, _ in stats],
                        label=f"{strat_gn}_{strat_fb}_{strat_as}",
                        color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"],
                        linestyle=config_lineStyles[f"{strat_gn}_{strat_fb}_{strat_as}"]
                    )
                    ax.plot(
                        [0, stats[0][0]], [0, stats[0][1]],
                        color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], linestyle="dashed"
                    )
                    # Plot confidence intervals
                    ax.fill_between(
                        [0]+[num_exs for num_exs, _, _ in stats],
                        [0]+[mmAP-cl for _, mmAP, cl in stats],
                        [0]+[mmAP+cl for _, mmAP, cl in stats],
                        color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], alpha=0.2
                    )

        for ax in fig.get_axes():
            ax.label_outer()
        
        fig.suptitle(f"Confusion plot matrix for (N={len(all_seeds)} per config)", fontsize=16)
        fig.supxlabel("# training examples")
        fig.supylabel("Response rate")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"confMat.png"))

if __name__ == "__main__":
    main()
