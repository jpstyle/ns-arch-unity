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
import uuid
import logging
import warnings
warnings.filterwarnings("ignore")
import random
from PIL import Image
from collections import defaultdict

import hydra
import numpy as np
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
    random.seed(cfg.seed)

    # Tensorboard writer to log learning progress
    writer = SummaryWriter(f"{cfg.paths.outputs_dir}/tensorboard_logs")
    mistakes = defaultdict(lambda: (0,0))     # Accumulating regrets

    # Set up student & teacher
    student = ITLAgent(cfg)
    teacher = SimulatedTeacher(cfg)

    # (Temp?) Concept teaching modes:
    #   0) Distinguishing whole trucks and parts
    #   1) Distinguishing between types of trucks
    mode = 0

    # Set up target concepts to teach in each episode as a list of dicts, where
    # each index corresponds to a truck type
    teacher.target_concept_sets = [
        # Each dict contains concepts to be tested and taught in a single episode,
        # with string concept names as key and Unity GameObject string name handle
        # as value
        [
            { "truck": "/truck" },
            { "truck": "/truck", "dumper": "/truck/load/load_dumper" },
            { "truck": "/truck", "ladder": "/truck/load/load_ladder" },
            { "truck": "/truck", "rocket launcher": "/truck/load/load_rocketLauncher" },
        ],
        [
            { "base truck": "/truck" },
            { "dump truck": "/truck" },
            { "fire truck": "/truck" },
            { "missile truck": "/truck" }
        ]
    ]

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

    for i in range(100):
        logger.info(f"Sys> Episode {i+1})")

        # Obtain random initialization of each episode
        random_inits = teacher.setup_episode(mode)

        # Send randomly initialized parameters to Unity
        for field, value in random_inits.items():
            env_par_channel.set_float_parameter(field, value)

        # Send teacher's episode-initial output---thus user's episode-initial input
        # (Comment out when testing in Heuristics mode)
        opening_output = teacher.initiate_dialogue()
        teacher_channel.send_string(
            opening_output[0]["utterance"], opening_output[0]["pointing"]
        )

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
                            # Process action output accordingly by setting Unity MLAgent actions
                            # and sending string messages via side channel
                            action = b_spec.action_spec.empty_action(1)
                            for act_type, act_data in act_out:
                                if act_type == "generate":
                                    action.discrete[0][0] = 1       # 'Utter' action
                                    utterance = act_data[0]
                                    dem_refs = {
                                        crange: mask.reshape(-1).tolist()
                                        for crange, mask in act_data[1].items()
                                    }
                                    student_channel.send_string(utterance, dem_refs)
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
                                # Each message consists of two consecutive buffer items;
                                # speaker and string content
                                _, utterance, _ = incoming_msgs.pop(0)
                                agent_reactions.append(utterance)
                        
                        # Simulated teacher (user) response
                        user_response = teacher.react(agent_reactions)

                        if len(user_response) > 0:
                            action = b_spec.action_spec.empty_action(1)
                            for act_data in user_response:
                                action.discrete[0][0] = 1       # 'Utter' action
                                utterance = act_data["utterance"]
                                dem_refs = act_data["pointing"]
                                teacher_channel.send_string(utterance, dem_refs)
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

        # Log progress to tensorboard
        for gt_conc, ans_conc in teacher.current_episode_record.items():
            regrets, total = mistakes[gt_conc]

            new_regrets = regrets+1 if gt_conc != ans_conc else regrets
            new_total = total+1

            writer.add_scalar(f"Cumul. regret: {gt_conc}", new_regrets, new_total)
            mistakes[gt_conc] = (new_regrets, new_total)

    # Close Unity environment & tensorboard writer
    env.close()
    writer.close()

    # Save current agent model checkpoint to output dir
    ckpt_path = os.path.join(cfg.paths.outputs_dir, "agent_model")
    os.makedirs(ckpt_path, exist_ok=True)
    student.save_model(f"{ckpt_path}/injected_{cfg.seed}.ckpt")


if __name__ == "__main__":
    main()
