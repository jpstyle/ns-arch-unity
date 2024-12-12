# Learning Visually Grounded Domain Ontologies via Embodied Conversation and Explanation

Code repository for AAAI 2025 submission, accepted with title: Learning Visually Grounded Domain Ontologies via Embodied Conversation and Explanation). Contains python codebase for the interactive task learner and the simulated teacher, and Unity codebase for simulated environment of the toy truck domain. May need correct version of Unity editor to compile the build.

## Some important command-line arguments for scripting experiments

`tools/exp_run.py` runs an experiment with the provided parameters. `tools/exp_summ.py` outputs summarization (numeric and visual) of the set of experiment outputs stored in `outputs` directory.

(Arguments are configured with `hydra`; see `itl/configs` directory for how they are set up if you are familiar with `hydra`)
- `+agent.model_path={PATH_TO_MODEL_CKPT}`: path to the trained agent model to be loaded, if any
- `paths.build_dir`: Path to the (Linux) build of the Unity simulation environment, if not developing with Unity editor
- `exp.strat_feedback=[medHelp/maxHelpNoexpl/maxHelpExpl]`: Teacher's strategy for providing feedback upon student's incorrect answers to episode-initial probing questions
- `exp.concept_set=[single_fourway/double_fiveway]`: Set experiment difficulty
- `exp.num_episode`: Set total number of interaction episodes in the training sequence
- `exp.checkpoint_interval`: How frequently agent models should be saved?
- `seed={N}`: integer random seed
