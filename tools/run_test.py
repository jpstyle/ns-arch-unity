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
import warnings
warnings.filterwarnings("ignore")
import random
from PIL import Image, ImageDraw

import hydra
import numpy as np
from omegaconf import OmegaConf
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# from python.itl import ITLAgent
from tools.message_side_channel import StringMsgChannel


# OmegaConf.register_new_resolver(
#     "randid", lambda: str(uuid.uuid4())[:6]
# )
# @hydra.main(config_path="../../itl/configs", config_name="config")
# def main(cfg):
#     print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    # main()

    # This channel communicates environment parameters for random initializations
    env_par_channel = EnvironmentParametersChannel()

    # Student/teacher-side string message communication side channels
    # (UUIDs generated by UUID4)
    student_channel = StringMsgChannel("a1a6b269-0dd3-442c-99c6-9c735ebe43e1")
    teacher_channel = StringMsgChannel("da85d4e0-1b60-4c8a-877d-03af30c446f2")

    # Start communication with Unity
    env = UnityEnvironment(
        # "unity/Builds/table_domain.x86_64",       # Uncomment when running with Unity linux build
        side_channels=[env_par_channel, student_channel, teacher_channel],
        timeout_wait=600, seed=42
    )

    # Temp: Arbitrary single-token names for truck types by load; this list is
    # kept in sync with Unity
    loadTypes = ["foo", "bar", "baz", "qux"]

    while True:
        # Random environment initialization before reset; currently, sample fine-grained
        # type of truck as distinguished by load type
        sampled_type = random.sample(range(len(loadTypes)), 1)[0]
        env_par_channel.set_float_parameter("load_type", sampled_type)

        # Server-side request to prompt teacher's episode-initial message
        teacher_channel.send_string("[EP_START]", {})

        env.reset()

        while True:
            # Keep running until either student or teacher terminates episode
            terminate = False

            for b_name, b_spec in env.behavior_specs.items():
                # Decision steps (agents requiring decisions) and terminal steps
                # (agents terminated)
                dec_steps, ter_steps = env.get_steps(b_name)

                # Handle each decision request
                for di in dec_steps:
                    dec_step = dec_steps[di]

                    # Handle student's decision request
                    if b_name.startswith("StudentBehavior"):
                        # Obtain agent's visual observation from camera sensor
                        vis_obs = dec_step.obs[0]
                        vis_obs = (vis_obs * 255).astype(np.uint8)
                        i_h, i_w, _ = vis_obs.shape
                        image = Image.fromarray(vis_obs, mode="RGB")

                        # Read messages stored in string message channel buffer
                        incoming_msgs = student_channel.incoming_message_buffer

                        if len(incoming_msgs) > 0:
                            while len(incoming_msgs) > 0:
                                # Each message consists of two consecutive buffer items;
                                # speaker and string content
                                speaker, utterance, demRefs = incoming_msgs.pop(0)

                                # Drawing any demonstrative references on vis obs image
                                drawer = ImageDraw.Draw(image)
                                for (start, end), bbox in demRefs.items():
                                    # Rectangle from box coordinates
                                    x, y, w, h = bbox
                                    drawer.rectangle([x*i_w, y*i_h, (x+w)*i_w, (y+h)*i_h])

                                    # Corresponding demonstrative pronoun
                                    drawer.text((x*i_w, y*i_h), utterance[start:end])
                                image.save("foo.jpg")

                                # Handling messages; echo utterance content & demRefs
                                student_channel.send_string(utterance, demRefs)

                            # Set agent action to 'utter'
                            action = b_spec.action_spec.empty_action(1)
                            action.discrete[0][0] = 1
                            env.set_action_for_agent(b_name, dec_step.agent_id, action)

                    # Handle teacher's decision request
                    if b_name.startswith("TeacherBehavior"):
                        # Read messages stored in string message channel buffer
                        incoming_msgs = teacher_channel.incoming_message_buffer

                        if len(incoming_msgs) > 0:
                            # Terminate as soon as teacher gets student's response
                            terminate = True
                            continue

                            while len(incoming_msgs) > 0:
                                # Each message consists of two consecutive buffer items;
                                # speaker and string content
                                speaker, utterance, demRefs = incoming_msgs.pop(0)

                                # Handling messages; echo utterance content & demRefs
                                teacher_channel.send_string(utterance, demRefs)

                            # Set agent action to 'utter'
                            action = b_spec.action_spec.empty_action(1)
                            action.discrete[0][0] = 1
                            env.set_action_for_agent(b_name, dec_step.agent_id, action)

            if terminate:
                break
            else:
                env.step()

    env.close()
