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
from PIL import Image

import hydra
import numpy as np
from omegaconf import OmegaConf
from mlagents_envs.environment import UnityEnvironment

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

    # Student/teacher-side string message communication side channels
    # (UUIDs generated by UUID4)
    student_channel = StringMsgChannel("a1a6b269-0dd3-442c-99c6-9c735ebe43e1")
    teacher_channel = StringMsgChannel("da85d4e0-1b60-4c8a-877d-03af30c446f2")

    # Start communication with Unity
    env = UnityEnvironment("unity/Builds/table_domain.x86_64", side_channels=[student_channel, teacher_channel])

    # Set teacher's initial message to prompt 'chain reactions'
    # teacher_channel.send_string("Foo")

    while True:
        env.step()

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
                    image = Image.fromarray(vis_obs, mode="RGB")
                    # image.show()

                    # Read messages stored in string message channel buffer
                    incoming_msgs = student_channel.incoming_message_buffer
                    assert len(incoming_msgs) % 2 == 0

                    if len(incoming_msgs) > 0:
                        while len(incoming_msgs) > 0:
                            # Each message consists of two consecutive buffer items;
                            # speaker and string content
                            speaker = incoming_msgs.pop(0)
                            utterance = incoming_msgs.pop(0)

                            # Handling messages; generate appropriate string responses
                            # and queue to side channel
                            if utterance == "Foo":
                                student_channel.send_string("Bar")
                            else:
                                student_channel.send_string("Baz")

                        # Set agent action to 'utter'
                        action = b_spec.action_spec.empty_action(1)
                        action.discrete[0][0] = 1
                        env.set_action_for_agent(b_name, dec_step.agent_id, action)

                # Handle teacher's decision request
                if b_name.startswith("TeacherBehavior"):
                    # Read messages stored in string message channel buffer
                    incoming_msgs = teacher_channel.incoming_message_buffer
                    assert len(incoming_msgs) % 2 == 0

                    if len(incoming_msgs) > 0:
                        while len(incoming_msgs) > 0:
                            # Each message consists of two consecutive buffer items;
                            # speaker and string content
                            speaker = incoming_msgs.pop(0)
                            utterance = incoming_msgs.pop(0)

                            # Handling messages; generate appropriate string responses
                            # and queue to side channel
                            if utterance == "Bar":
                                teacher_channel.send_string("Qux")
                            else:
                                teacher_channel.send_string("Foo")

                        # Set agent action to 'utter'
                        action = b_spec.action_spec.empty_action(1)
                        action.discrete[0][0] = 1
                        env.set_action_for_agent(b_name, dec_step.agent_id, action)
                    else:
                        # Episode-initial; set agent action to 'utter' still, as message
                        # strings will already have been sent
                        action = b_spec.action_spec.empty_action(1)
                        action.discrete[0][0] = 1
                        env.set_action_for_agent(b_name, dec_step.agent_id, action)

    env.close()
