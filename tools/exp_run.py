"""
Script for running interactive symbol grounding experiments. Each experiment run
consists of the initial probing question from the (simulated) user and the ensuing
series of reactions that differ based on the agent's belief states and dialogue
participants' choice of dialogue strategies.

Can be run in 'training mode' or 'test mode'. Different statistics are recorded
for each mode. For the former, we track the cumulative regret curves across the
series of interaction episodes. The latter mode is for evaluation; it disables
the agent's learning capability, fixing the agent's knowledge across the evaluation
suite, measuring the agent's answers and ground truths for the 'test problems'.
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import uuid
import logging
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from collections import defaultdict

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from python.itl import ITLAgent
from tools.sim_user import SimulatedTeacher
from tools.message_side_channel import StringMsgChannel


logger = logging.getLogger(__name__)
TAB = "\t"

OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../python/itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Set seed
    pl.seed_everything(cfg.seed)

    # Experiment tag
    exp_tag = "_".join([
        cfg.exp.strat_feedback,
        cfg.agent.strat_generic,
        cfg.agent.strat_assent,
        str(cfg.seed)
    ])

    # Path to save result metrics
    results_path = os.path.join(cfg.paths.outputs_dir, "results")
    os.makedirs(results_path, exist_ok=True)

    # Path to save agent models during & after training
    if not cfg.agent.test_mode:
        ckpt_path = os.path.join(cfg.paths.outputs_dir, "agent_model")    
        os.makedirs(ckpt_path, exist_ok=True)

    # Tensorboard writer to log learning progress
    writer = SummaryWriter(f"{cfg.paths.outputs_dir}/tensorboard_logs")
    mistakes = defaultdict(lambda: [(0,0)])         # Accumulating regrets

    # Set up student & teacher
    student = ITLAgent(cfg)
    teacher = SimulatedTeacher(cfg)

    # Concept repertoire:
    #   "prior": Distinguishing whole trucks and parts; injecting the concepts as
    #       prior concepts to be held by agents learning fine-grained truck types
    #   "main": Distinguishing between fine-grained types of trucks, assuming
    #       prior knowledge of whole truck & load part visual concepts (i.e., those
    #       injected by "prior" repertoire)
    repertoire = cfg.exp.concept_set

    # Set up target concepts to teach in each episode as a list of tuples
    teacher.target_concept_sets = {
        # Each tuple contains concepts to be tested and taught in a single episode.
        # The first tuple entry instructs the Unity simulation environment the
        # initialization parameters for the episode. The second entry provides a list
        # of paths to the Unity gameObjects and string concept names.
        "prior_supertypes": [
            (
                (("cabin_type", 0), ("load_type", 0)),
                [(None, "truck"), (("cabin", "hemtt cabin"), "cabin"), (("load", "platform"), "load")]
            ),
            (
                (("cabin_type", 1), ("load_type", 0)),
                [(None, "truck"), (("cabin", "quad cabin"), "cabin"), (("load", "platform"), "load")]
            ),
            (
                (("cabin_type", 0), ("load_type", 1)),
                [(None, "truck"), (("cabin", "hemtt cabin"), "cabin"), (("load", "dumper"), "load")]
            ),
            (
                (("cabin_type", 1), ("load_type", 1)),
                [(None, "truck"), (("cabin", "quad cabin"), "cabin"), (("load", "dumper"), "load")]
            ),
            (
                (("cabin_type", 0), ("load_type", 2)),
                [(None, "truck"), (("cabin", "hemtt cabin"), "cabin"), (("load", "ladder"), "load")]
            ),
            (
                (("cabin_type", 1), ("load_type", 2)),
                [(None, "truck"), (("cabin", "quad cabin"), "cabin"), (("load", "ladder"), "load")]
            ),
            (
                (("cabin_type", 0), ("load_type", 3)),
                [(None, "truck"), (("cabin", "hemtt cabin"), "cabin"), (("load", "rocket launcher"), "load")]
            ),
            (
                (("cabin_type", 1), ("load_type", 3)),
                [(None, "truck"), (("cabin", "quad cabin"), "cabin"), (("load", "rocket launcher"), "load")]
            )
        ],
        "prior_parts": [
            (
                (("cabin_type", 0), ("load_type", 0)),
                [(("cabin", "hemtt cabin"), "hemtt cabin"), (("load", "platform"), "platform")]
            ),
            (
                (("cabin_type", 1), ("load_type", 0)),
                [(("cabin", "quad cabin"), "quad cabin"), (("load", "platform"), "platform")]
            ),
            (
                (("cabin_type", 0), ("load_type", 1)),
                [(("cabin", "hemtt cabin"), "hemtt cabin"), (("load", "dumper"), "dumper")]
            ),
            (
                (("cabin_type", 1), ("load_type", 1)),
                [(("cabin", "quad cabin"), "quad cabin"), (("load", "dumper"), "dumper")]
            ),
            (
                (("cabin_type", 0), ("load_type", 2)),
                [(("cabin", "hemtt cabin"), "hemtt cabin"), (("load", "ladder"), "ladder")]
            ),
            (
                (("cabin_type", 1), ("load_type", 2)),
                [(("cabin", "quad cabin"), "quad cabin"), (("load", "ladder"), "ladder")]
            ),
            (
                (("cabin_type", 0), ("load_type", 3)),
                [(("cabin", "hemtt cabin"), "hemtt cabin"), (("load", "rocket launcher"), "rocket launcher")]
            ),
            (
                (("cabin_type", 1), ("load_type", 3)),
                [(("cabin", "quad cabin"), "quad cabin"), (("load", "rocket launcher"), "rocket launcher")]
            )
        ],
        "single_fourway": [
            (
                (("load_type", 0),),
                [(None, "base truck")]
            ),
            (
                (("load_type", 1),),
                [(None, "dump truck")]
            ),
            (
                (("load_type", 2),),
                [(None, "fire truck")]
            ),
            (
                (("load_type", 3),),
                [(None, "missile truck")]
            )
        ],
        "double_fiveway": [
            (
                (("cabin_type", 1), ("load_type", 0)),
                [(None, "base truck")]
            ),
            (
                (("cabin_type", 1), ("load_type", 1)),
                [(None, "dump truck")]
            ),
            (
                (("cabin_type", 1), ("load_type", 2)),
                [(None, "fire truck")]
            ),
            (
                (("cabin_type", 1), ("load_type", 3)),
                [(None, "missile truck")]
            ),
            (
                (("cabin_type", 0), ("load_type", 1)),
                [(None, "container truck")]
            )
        ]
    }

    # Student/teacher-side string message communication side channels
    # (UUIDs generated by UUID4)
    student_channel = StringMsgChannel("a1a6b269-0dd3-442c-99c6-9c735ebe43e1")
    teacher_channel = StringMsgChannel("da85d4e0-1b60-4c8a-877d-03af30c446f2")

    # This channel communicates environment parameters for random initializations
    env_par_channel = EnvironmentParametersChannel()

    # Start communication with Unity
    env = UnityEnvironment(
        # Uncomment next line when running with Unity linux build
        f"{cfg.paths.build_dir}/truck_domain.x86_64",
        side_channels=[student_channel, teacher_channel, env_par_channel],
        timeout_wait=600, seed=cfg.seed
    )

    for i in range(cfg.exp.num_episodes):
        logger.info(f"Sys> Episode {i+1})")

        # Obtain random initialization of each episode
        shrink_domain = cfg.exp.domain_shift and i < cfg.exp.num_episodes / 2
        random_inits = teacher.setup_episode(repertoire, shrink_domain=shrink_domain)

        # Send randomly initialized parameters to Unity
        for field, value in random_inits.items():
            env_par_channel.set_float_parameter(field, value)

        # Request sending ground-truth mask info to teacher at the beginning
        teacher_channel.send_string("System", "GT mask request: cabin, load", {})

        # Send teacher's episode-initial output---thus user's episode-initial input
        # (Comment out when testing in Heuristics mode)
        opening_output = teacher.initiate_dialogue()
        teacher_channel.send_string(
            "Teacher", opening_output[0]["utterance"], opening_output[0]["pointing"]
        )
        logger.info(f"L> {TAB}{opening_output[0]['utterance']}")

        # Let the settings take effect and begin the episode
        env.reset()

        # New scene bool flag
        new_scene = True

        while True:
            # Keep running until either student or teacher terminates episode
            terminate = False

            for b_name, b_spec in env.behavior_specs.items():
                # Decision steps (agents requiring decisions) and terminal steps
                # (agents terminated)
                dec_steps, _ = env.get_steps(b_name)

                # Handle each decision request
                for di in dec_steps:
                    dec_step = dec_steps[di]

                    # Handle student's decision request
                    if b_name.startswith("StudentBehavior"):
                        # Dictionary containing input for next agent loop
                        agent_loop_input = {
                            "v_usr_in": None,
                            "l_usr_in": [],
                            "pointing": []
                        }

                        # Obtain agent's visual observation from camera sensor
                        vis_obs = dec_step.obs[0]
                        vis_obs = (vis_obs * 255).astype(np.uint8)
                        i_h, i_w, _ = vis_obs.shape
                        agent_loop_input["v_usr_in"] = Image.fromarray(vis_obs, mode="RGB")

                        # Read messages stored in string message channel buffer
                        incoming_msgs = student_channel.incoming_message_buffer

                        if len(incoming_msgs) > 0:
                            while len(incoming_msgs) > 0:
                                _, utterance, dem_refs = incoming_msgs.pop(0)
                                # 1D to 2D according to visual scene dimension
                                dem_refs = {
                                    crange: np.array(mask).reshape(i_h, i_w)
                                    for crange, mask in dem_refs.items()
                                }
                                agent_loop_input["l_usr_in"].append(utterance)
                                agent_loop_input["pointing"].append(dem_refs)

                        # ITL agent loop: process input and generate output (action)
                        act_out = student.loop(**agent_loop_input, new_scene=new_scene)

                        if len(act_out) > 0:
                            # Process action output accordingly by setting Unity MLAgent
                            # actions and sending string messages via side channel
                            action = b_spec.action_spec.empty_action(1)
                            for act_type, act_data in act_out:
                                if act_type == "generate":
                                    action.discrete[0][0] = 1       # 'Utter' action
                                    utterance = act_data[0]
                                    dem_refs = {
                                        crange: mask.reshape(-1).tolist()
                                        for crange, mask in act_data[1].items()
                                    }
                                    student_channel.send_string(
                                        "Student", utterance, dem_refs
                                    )
                                    logger.info(f"L> {TAB}{utterance}")

                            # Finally apply actions
                            env.set_action_for_agent(b_name, dec_step.agent_id, action)
                        else:
                            terminate = True

                        # Disable new scene flag after any agent loop
                        new_scene = False

                    # Handle teacher's decision request
                    if b_name.startswith("TeacherBehavior"):
                        agent_reactions = []

                        # Read messages stored in string message channel buffer
                        incoming_msgs = teacher_channel.incoming_message_buffer

                        if len(incoming_msgs) > 0:
                            while len(incoming_msgs) > 0:
                                spk, utterance, dem_refs = incoming_msgs.pop(0)
                                # 1D to 2D according to visual scene dimension
                                dem_refs = {
                                    crange: np.array(mask).reshape(i_h, i_w)
                                    for crange, mask in dem_refs.items()
                                }
                                if spk == "System" and utterance.startswith("GT mask response:"):
                                    # Retrieve and store requested GT mask info in teacher
                                    teacher.current_gt_masks = {
                                        utterance[crange[0]:crange[1]]: msk
                                        for crange, msk in dem_refs.items()
                                    }
                                else:
                                    agent_reactions.append((utterance, dem_refs))

                        # Simulated teacher (user) response
                        user_response = teacher.react(agent_reactions)

                        if len(user_response) > 0:
                            action = b_spec.action_spec.empty_action(1)
                            for act_data in user_response:
                                action.discrete[0][0] = 1       # 'Utter' action
                                utterance = act_data["utterance"]
                                dem_refs = act_data["pointing"]
                                teacher_channel.send_string(
                                    "Teacher", utterance, dem_refs
                                )
                                logger.info(f"T> {TAB}{utterance}")

                            # Finally apply actions
                            env.set_action_for_agent(b_name, dec_step.agent_id, action)
                        else:
                            terminate = True

            if terminate:
                # End of episode, push record to history and break
                teacher.episode_records.append(teacher.current_episode_record)
                break
            else:
                env.step()

        for gt_conc, ep_log in teacher.current_episode_record.items():
            ans_conc = ep_log["answer"]

            # Update metrics
            regrets_conc, total_conc = mistakes[gt_conc][-1]
            regrets_all, total_all = mistakes["__all__"][-1]

            new_regrets_conc = regrets_conc+1 if gt_conc != ans_conc else regrets_conc
            new_total_conc = total_conc+1
            new_regrets_all = regrets_all+1 if gt_conc != ans_conc else regrets_all
            new_total_all = total_all+1

            # Log progress to tensorboard
            writer.add_scalar(
                f"Cumul. regret: {gt_conc}", new_regrets_conc, global_step=new_total_conc
            )
            writer.add_scalar(
                f"Cumul. regret: *All concepts*", new_regrets_all, global_step=new_total_all
            )
            mistakes[gt_conc].append((new_regrets_conc, new_total_conc))
            mistakes["__all__"].append((new_regrets_all, new_total_all))

        # If not test mode (i.e., training mode), save current agent model checkpoint
        # to output dir every 25 episodes
        if not cfg.agent.test_mode and (i+1) % cfg.exp.checkpoint_interval == 0:
            student.save_model(f"{ckpt_path}/{exp_tag}_{i+1}.ckpt")

    # Close Unity environment & tensorboard writer
    env.close()
    writer.close()

    if cfg.agent.test_mode:
        # If test mode, save exam records to output dir for later summary
        out_csv_fname = cfg.agent.model_path.split("/")[-1].replace(".ckpt",".csv")
        out_csv_fname = f"outputs_{out_csv_fname}"

        with open(os.path.join(results_path, out_csv_fname), "w") as out_csv:
            out_csv.write("episode,ground_truth,answer\n")

            for i, record in enumerate(teacher.episode_records):
                for gt_conc, ep_log in record.items():
                    ans_conc = ep_log["answer"]
                    out_csv.write(f"{i+1},{gt_conc},{ans_conc}\n")

    else:
        # Otherwise (i.e., learning enabled), save cumulative regret curves to output dir
        out_csv_fname = f"cumulReg_{exp_tag}.csv"

        with open(os.path.join(results_path, out_csv_fname), "w") as out_csv:
            out_csv.write("episode,cumulative_regret,reason_type\n")

            training_log = zip(mistakes["__all__"], [{}] + teacher.episode_records)
            for (regrets, total), ep_rec in training_log:
                ep_log = list(ep_rec.values())[0] if len(ep_rec) > 0 else {}
                if "reason" in ep_log:
                    reason_type = ep_log["reason"]
                else:
                    reason_type = "na"

                out_csv.write(f"{total},{regrets},{reason_type}\n")


if __name__ == "__main__":
    main()
