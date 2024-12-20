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
TAB = "\t"


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
    results_reasonTypes = defaultdict(lambda: defaultdict(int))

    all_seeds = set()

    # Collect results
    for res_dir in tqdm.tqdm(dirs_with_results, total=len(dirs_with_results)):
        group_dir, run_dir = res_dir
        res_dir = os.path.join(outputs_root_dir, *res_dir, "results")
        dir_contents = list(os.listdir(res_dir))

        for data in dir_contents:
            name_parse = data.split(".")[0].split("_")
            data_type = name_parse[0]
            strat_fb, strat_gn, strat_as = name_parse[1], name_parse[2], name_parse[3]
            seed = name_parse[4]

            if strat_fb == "maxHelpExpl2": continue

            # # Uncomment for controlling prior part model quality
            # if strat_fb != "medHelp" and not group_dir.startswith("pr"): continue

            all_seeds.add(seed)

            if data_type == "cumulReg":
                # Cumulative regret curve & explanation type data
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)

                    # Header column; 'episode,cumulative_regret'
                    _ = next(reader)

                    row = [row_data for row_data in reader]
                    curve = np.array([
                        [int(ep_num), int(regret)] for ep_num, regret, _ in row
                    ])
                    reason_types = [
                        reason_type for _, _, reason_type in row if reason_type != "na"
                    ]

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

                    # Aggregate explanation type stats, disregarding "na" (correct answers,
                    # no explanation expected)
                    for rt in set(reason_types):
                        results_reasonTypes[strat_combi][rt] += reason_types.count(rt)

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
        # "semOnly_minHelp_doNotLearn", "semOnly_minHelp_threshold",
        "semOnly_medHelp_doNotLearn", "semOnly_medHelp_threshold",
        "semOnly_maxHelpNoexpl_doNotLearn", "semOnly_maxHelpNoexpl_threshold",
        "semOnly_maxHelpExpl_doNotLearn", "semOnly_maxHelpExpl_threshold",
        "semOnly_maxHelpExpl2_doNotLearn", "semOnly_maxHelpExpl2_threshold",
        # "semNeg_maxHelp_doNotLearn", "semNeg_maxHelp_threshold",
        # "semNegScal_maxHelp_doNotLearn", "semNegScal_maxHelp_threshold",
    ]
    config_aliases = {
        # "semOnly_minHelp_doNotLearn": "minHelp_doNotLearn",
        # "semOnly_minHelp_threshold": "minHelp_threshold",

        "semOnly_medHelp_doNotLearn": "Vis-Only",
        "semOnly_medHelp_threshold": "Vis-Only",

        "semOnly_maxHelpNoexpl_doNotLearn": "Vis+Genr",
        "semOnly_maxHelpNoexpl_threshold": "Vis+Genr",

        "semOnly_maxHelpExpl_doNotLearn": "Vis+Genr+Expl",
        "semOnly_maxHelpExpl_threshold": "Vis+Genr+Expl",
        # "semOnly_maxHelpExpl_doNotLearn": "Vis+Genr+ExplSuff",
        # "semOnly_maxHelpExpl_threshold": "Vis+Genr+ExplSuff",

        "semOnly_maxHelpExpl2_doNotLearn": "Vis+Genr+ExplSuffCtfl",
        "semOnly_maxHelpExpl2_threshold": "Vis+Genr+ExplSuffCtfl",

        # "semNeg_maxHelp_doNotLearn": "maxHelp_semNeg_doNotLearn",
        # "semNeg_maxHelp_threshold": "maxHelp_semNeg_threshold",
        
        # "semNegScal_maxHelp_doNotLearn": "maxHelp_semNegScal_doNotLearn",
        # "semNegScal_maxHelp_threshold": "maxHelp_semNegScal_threshold",
    }   # To be actually displayed in legend
    config_colors = {
        # "semOnly_minHelp_doNotLearn": "tab:red",
        # "semOnly_minHelp_threshold": "tab:red",

        "semOnly_medHelp_doNotLearn": "tab:orange",
        "semOnly_medHelp_threshold": "tab:orange",

        "semOnly_maxHelpNoexpl_doNotLearn": "tab:green",
        "semOnly_maxHelpNoexpl_threshold": "tab:green",

        "semOnly_maxHelpExpl_doNotLearn": "tab:blue",
        "semOnly_maxHelpExpl_threshold": "tab:blue",

        "semOnly_maxHelpExpl2_doNotLearn": "tab:purple",
        "semOnly_maxHelpExpl2_threshold": "tab:purple",

        # "semNeg_maxHelp_doNotLearn": "tab:blue",
        # "semNeg_maxHelp_threshold": "tab:blue",

        # "semNegScal_maxHelp_doNotLearn": "tab:purple",
        # "semNegScal_maxHelp_threshold": "tab:purple",
    }
    config_lineStyles = {
        # "semOnly_minHelp_doNotLearn": "--",
        # "semOnly_minHelp_threshold": "-",

        "semOnly_medHelp_doNotLearn": "-",
        "semOnly_medHelp_threshold": "-",

        "semOnly_maxHelpNoexpl_doNotLearn": "-",
        "semOnly_maxHelpNoexpl_threshold": "-",

        "semOnly_maxHelpExpl_doNotLearn": "-",
        "semOnly_maxHelpExpl_threshold": "-",

        "semOnly_maxHelpExpl2_doNotLearn": "-",
        "semOnly_maxHelpExpl2_threshold": "-",

        # "semNeg_maxHelp_doNotLearn": "--",
        # "semNeg_maxHelp_threshold": "-",

        # "semNegScal_maxHelp_doNotLearn": "--",
        # "semNegScal_maxHelp_threshold": "-",
    }

    plt.rcParams.update({'font.size': 22})

    # Aggregate and visualize: cumulative regret curve
    if len(results_cumulReg) > 0:
        _, ax = plt.subplots(figsize=(8, 6.5), dpi=80)
        ymax = 0

        print("")
        print(f"Endpoint cumulative regret CIs:")
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

            # Report final values on stdout
            print(f"{TAB}{strat_fb}: {stats[-1][1]:.2f} \xB1 {stats[-1][2]:.2f}")

        # Plot curve
        ax.set_xlabel("# training episodes")
        ax.set_xticks([30, 60, 90, 120])
        ax.set_ylabel("cumulative regret")
        ax.set_ylim(0, (ymax+1) * 1.1)
        ax.grid()

        # Ordering legends according to the prespecified ordering above
        handles, labels = ax.get_legend_handles_labels()
        hls_sorted = sorted(
            [(h, l) for h, l in zip(handles, labels)],
            key=lambda x: config_ord.index(x[1])
        )
        handles = [hl[0] for hl in hls_sorted]
        labels = [config_aliases.get(hl[1], hl[1]) for hl in hls_sorted]
        ax.legend(handles, labels, loc='lower right')
        
        ax.set_title(f"Cumulative regret ({len(all_seeds)} seeds)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"cumulReg.png"))

    # Aggregate and visualize: learning curve
    if len(results_learningCurve) > 0:
        _, ax = plt.subplots(figsize=(8, 6.5), dpi=80)

        print("")
        print(f"Endpoint accuracy CIs:")
        for (strat_fb, strat_gn, strat_as), data in results_learningCurve.items():
            data = sorted([entry for entry in data.items()], key=lambda x: x[0])
            stats = [
                (num_exs, np.mean(accs), 1.96 * np.std(accs)/np.sqrt(len(accs)))
                for num_exs, accs in data
            ]

            # Plot mean curve
            ax.plot(
                [num_exs for num_exs, _, _ in stats],
                [acc for _, acc, _ in stats],
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
                [0]+[acc-cl for _, acc, cl in stats],
                [0]+[acc+cl for _, acc, cl in stats],
                color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], alpha=0.2
            )

            # Report final values on stdout
            print(f"{TAB}{strat_fb}: {stats[-1][1]:.2%} \xB1 {stats[-1][2]:.2%}")

        # Plot curve
        ax.set_xlabel("# training examples")
        ax.set_xlim(0, stats[-1][0])
        ax.set_xticks([30, 60, 90, 120])
        ax.set_ylabel("mean accuracy")
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
        plt.legend(handles, labels, loc='lower right')
        
        plt.title(f"Learning curve ({len(all_seeds)} seeds)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"learningCurve.png"))

    # Aggregate and visualize: confusion matrices
    if len(results_confMat) > 0:
        fig = plt.figure(figsize=(10, 7.5), dpi=80)
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
                    ax.set_xticks([120])
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
                        [acc for _, acc, _ in stats],
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
                        [0]+[acc-cl for _, acc, cl in stats],
                        [0]+[acc+cl for _, acc, cl in stats],
                        color=config_colors[f"{strat_gn}_{strat_fb}_{strat_as}"], alpha=0.2
                    )

        for ax in fig.get_axes():
            ax.label_outer()
        
        fig.suptitle(f"Confusion plot matrix for ({len(all_seeds)} seeds)", fontsize=16)
        fig.supxlabel("# training examples")
        fig.supylabel("Response rate")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"confMat.png"))

    # Report reason type statistics
    print("")
    for strat_combi, data in results_reasonTypes.items():
        print(f"Reason type statistics for {strat_combi}:")

        total = sum(data.values())
        print(f"{TAB}Total: {total}")

        for rt, count in data.items():
            print(f"{TAB}{rt}: {count} ({count/total:.2%})")

        print("")

if __name__ == "__main__":
    main()
