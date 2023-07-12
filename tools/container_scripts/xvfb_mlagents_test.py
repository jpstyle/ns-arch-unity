""" Test script: Should work if Xvfb + Unity MLAgents combo works """
from mlagents_envs.registry import default_registry

env = default_registry['VisualHallway'].make()
env.reset()
behaviour_names = list(env.behavior_specs.keys())
decision_steps, terminal_steps = env.get_steps(behaviour_names[0])
print(decision_steps.obs[0].shape)
env.close()
