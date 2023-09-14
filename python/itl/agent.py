"""
Outermost wrapper containing ITL agent API
"""
import os
import copy
import pickle
import logging
from collections import defaultdict

import torch
from pytorch_lightning.loggers import WandbLogger

from .vision import VisionModule
from .lang import LanguageModule
from .memory import LongTermMemoryModule
from .symbolic_reasoning import SymbolicReasonerModule
from .practical_reasoning import PracticalReasonerModule
from .comp_actions import CompositeActions
from .lpmln import Literal

logger = logging.getLogger(__name__)


WB_PREFIX = "wandb://"

class ITLAgent:

    def __init__(self, cfg):
        self.cfg = cfg

        # Initialize component modules
        self.vision = VisionModule(cfg)
        self.lang = LanguageModule(cfg)
        self.symbolic = SymbolicReasonerModule()
        self.practical = PracticalReasonerModule()
        self.lt_mem = LongTermMemoryModule()

        # Provide access to methods in comp_actions
        self.comp_actions = CompositeActions(self)

        # Load agent model from specified path
        if "model_path" in cfg.agent:
            self.load_model(cfg.agent.model_path)

        # Agent learning strategy params
        self.strat_generic = cfg.agent.strat_generic

        # Show visual UI and plots
        self.vis_ui_on = False

        # (Fields below would categorize as 'working memory' in conventional
        # cognitive architectures...)

        # Bookkeeping pairs of visual concepts that confused the agent, which
        # are resolved by asking 'concept-diff' questions to the user. Jusk ask
        # once to get answers as symbolic generic rules when the agent is aware
        # of the confusion for the first time, for each concept pair.
        # (In a sense, this kinda overlaps with the notion of 'common ground'?
        # May consider fulfilling this later...)
        self.confused_no_more = set()

        # Snapshot of KB, to be taken at the beginning of every training episode,
        # with which scalar implicatures will be computed. Won't be necessary
        # if we were to use more structured discourse representation...
        self.kb_snap = copy.deepcopy(self.lt_mem.kb)

        # (Temporary) Episodic memory; may be bumped up into proper long-term memory
        # component if KB maintenance with episodic memory works well
        self.episodic_memory = []

    def loop(self, v_usr_in=None, l_usr_in=None, pointing=None, new_scene=True):
        """
        Single agent activity loop. Provide usr_in for programmatic execution; otherwise,
        prompt user input on command line REPL
        """
        self._vis_inp(usr_in=v_usr_in, new_scene=new_scene)
        self._lang_inp(usr_in=l_usr_in)
        self._update_belief(pointing=pointing)
        act_out = self._act()

        return act_out

    def save_model(self, ckpt_path):
        """
        Save current snapshot of the agent's long-term knowledge as torch checkpoint;
        in the current scope of the research, by 'long-term knowledge' we are referring
        to the following information:

        - Vision model; feature extractor backbone most importantly, and concept-specific
            vectors
        - Knowledge stored in long-term memory module:
            - Symbolic knowledge base, including generalized symbolic knowledge represented
                as logic programming rules
            - Visual exemplar base, including positive/negative exemplars of visual concepts
                represented as internal feature vectors along with original image patches from
                which the vectors are obtained
            - Lexicon, including associations between words (linguistic form) and their
                denotations (linguistic meaning; here, visual concepts)
        """
        ckpt = {
            "vision": {
                "inventories": self.vision.inventories
            },
            "lt_mem": {
                "exemplars": vars(self.lt_mem.exemplars),
                "kb": vars(self.lt_mem.kb),
                "lexicon": vars(self.lt_mem.lexicon)
            }
        }
        if "fs_model" in self.cfg.vision.model:
            ckpt["vision"]["fs_model_path"] = self.cfg.vision.model.fs_model

        torch.save(ckpt, ckpt_path)
        logger.info(f"Saved current agent model at {ckpt_path}")

    def load_model(self, ckpt_path):
        """
        Load from a torch checkpoint to initialize the agent; the checkpoint may contain
        a snapshot of agent knowledge obtained as an output of self.save_model() evoked
        previously, or just pre-trained weights of the vision module only (likely generated
        as output of the vision module's training API)
        """
        # Resolve path to checkpoint
        if ckpt_path.startswith(WB_PREFIX):
            # Loading agent models stored in W&B; not implemented yet
            raise NotImplementedError

            # wb_entity = os.environ.get("WANDB_ENTITY")
            # wb_project = os.environ.get("WANDB_PROJECT")
            # wb_run_id = self.fs_model_path[len(WB_PREFIX):]

            # local_ckpt_path = WandbLogger.download_artifact(
            #     artifact=f"{wb_entity}/{wb_project}/model-{wb_run_id}:best_k",
            #     save_dir=os.path.join(
            #         self.cfg.paths.assets_dir, "vision_models", "wandb", wb_run_id
            #     )
            # )
            # local_ckpt_path = os.path.join(local_ckpt_path, "model.ckpt")
        else:
            assert os.path.exists(ckpt_path)
            local_ckpt_path = ckpt_path

        # Load agent model checkpoint file
        try:
            ckpt = torch.load(local_ckpt_path)
        except RuntimeError:
            with open(local_ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

        # Fill in module components with loaded data
        for module_name, module_data in ckpt.items():
            for module_component, component_data in module_data.items():
                if isinstance(component_data, dict):
                    for component_prop, prop_data in component_data.items():
                        component = getattr(getattr(self, module_name), module_component)
                        setattr(component, component_prop, prop_data)
                else:
                    module = getattr(self, module_name)
                    prev_component_data = getattr(module, module_component)
                    setattr(module, module_component, component_data)
                
                # Handling vision.fs_model_path data
                if module_name == "vision":
                    if module_component == "fs_model_path":
                        if (prev_component_data is not None and
                            prev_component_data != component_data):
                            logger.warn(
                                "Path to few-shot components in vision module is already provided "
                                "in config and is inconsistent with the pointer saved in the agent "
                                f"model specified (config: {prev_component_data} vs. agent_model: "
                                f"{component_data}). Agent vision module might exhibit unexpected "
                                "behaviors."
                            )
                        
                        self.vision.load_weights()

    def _vis_inp(self, usr_in, new_scene):
        """ Handle provided visual input """
        self.vision.new_input = new_scene
        if self.vision.new_input:
            self.vision.last_input = usr_in

    def _lang_inp(self, usr_in):
        """ Handle provided language input (from user) """
        if usr_in is None:
            usr_in = []
        elif not isinstance(usr_in, list):
            assert isinstance(usr_in, str)
            usr_in = [usr_in]

        if len(usr_in) == 0:
            self.lang.new_input = False
        else:
            self.lang.new_input = True

            parsed_input = None
            try:
                parsed_input = self.lang.semantic.nl_parse(usr_in)
            except IndexError as e:
                logger.info(str(e))
            else:
                self.lang.last_input = parsed_input

    def _update_belief(self, pointing):
        """ Form beliefs based on visual and/or language input """

        if not (self.vision.new_input or self.lang.new_input):
            # No information whatsoever to make any belief updates
            return

        # Lasting storage of pointing info
        if pointing is None:
            pointing = {}

        # For showing visual UI on only the first time
        vis_ui_on = self.vis_ui_on

        # Index of latest dialogue turn
        ti_last = len(self.lang.dialogue.record)

        # Set of new visual concepts (equivalently, neologisms) newly registered
        # during the loop
        novel_concepts = set()

        # Keep updating beliefs until there's no more immediately exploitable learning
        # opportunities
        xb_updated = False      # Whether learning happened at 'neural'-level (in exemplar base)
        kb_updated = False      # Whether learning happened at 'symbolic'-level (in knowledge base)
        while True:
            ###################################################################
            ##                  Processing perceived inputs                  ##
            ###################################################################

            if self.vision.new_input or xb_updated:
                # Prior to resetting visual context, store current one into the
                # episodic memory (visual perceptions & user language inputs),
                # in the form of LP^MLN program fragments
                if self.vision.new_input:
                    if (self.vision.scene is not None and
                        len(self.vision.scene) > 1 and
                        self.symbolic.concl_vis_lang is not None):

                        pprog, _, dprog = self.symbolic.concl_vis_lang[1]
                        self.episodic_memory.append((pprog, dprog))

                # Ground raw visual perception with scene graph generation module
                self.vision.predict(
                    self.vision.last_input, self.lt_mem.exemplars,
                    visualize=vis_ui_on, lexicon=self.lt_mem.lexicon
                )
                vis_ui_on = False

            if self.vision.new_input:
                # Inform the language module of the visual context
                self.lang.situate(self.vision.scene)
                self.symbolic.refresh()

                # Reset below on episode-basis
                self.kb_snap = copy.deepcopy(self.lt_mem.kb)

            # Understand the user input in the context of the dialogue
            if self.lang.new_input:
                self.lang.dialogue.record = self.lang.dialogue.record[:ti_last]
                self.lang.understand(self.lang.last_input, pointing=pointing)

            ents_updated = False
            if self.vision.scene is not None:
                # If a new entity is registered as a result of understanding the latest
                # input, re-run vision module to update with new predictions for it
                new_ents = set(self.lang.dialogue.referents["env"]) - set(self.vision.scene)
                if len(new_ents) > 0:
                    masks = {
                        ent: self.lang.dialogue.referents["env"][ent]["mask"]
                        for ent in new_ents
                    }

                    # Incrementally predict on the designated bbox
                    self.vision.predict(
                        None, self.lt_mem.exemplars, masks=masks, visualize=False
                    )

                    ents_updated = True     # Set flag for another round of sensemaking

            ###################################################################
            ##       Sensemaking via synthesis of perception+knowledge       ##
            ###################################################################

            dialogue_state = self.lang.dialogue.export_as_dict()

            if self.vision.new_input or ents_updated or xb_updated or kb_updated:
                # Sensemaking from vision input only
                exported_kb = self.lt_mem.kb.export_reasoning_program()
                self.symbolic.sensemake_vis(self.vision.scene, exported_kb)

            if self.lang.new_input:
                # Reference & word sense resolution to connect vision & discourse
                self.symbolic.resolve_symbol_semantics(
                    dialogue_state, self.lt_mem.lexicon
                )

                if self.vision.scene is not None:
                    # Sensemaking from vision & language input
                    self.symbolic.sensemake_vis_lang(dialogue_state)

            ###################################################################
            ##           Identify & exploit learning opportunities           ##
            ###################################################################

            # Disable learning when agent is in test mode
            if self.cfg.agent.test_mode: break

            # Resetting flags
            xb_updated = False
            kb_updated = False

            # Generic statements to be added to KB
            generics = []

            # Info needed (along with generics) for computing scalar implicatures
            pair_rules = defaultdict(list)

            # Translate dialogue record into processable format based on the result
            # of symbolic.resolve_symbol_semantics
            translated = self.symbolic.translate_dialogue_content(dialogue_state)

            # Collect previous factual statements made during this dialogue
            prev_facts = [
                (spk, rule)
                for spk, turn_clauses in translated
                for (rule, _), _ in turn_clauses
                if rule is not None and len(rule[0])==1 and rule[1] is None
            ]

            # Collect previous questions made during this dialogue
            prev_Qs = [
                (spk, ques, presup, raw)
                for spk, turn_clauses in translated
                for (presup, ques), raw in turn_clauses
                if ques is not None
            ]

            # Process translated dialogue record to do the following:
            #   - Identify recognition mismatch btw. user provided vs. agent
            #   - Identify visual concept confusion
            #   - Identify new generic rules to be integrated into KB
            for speaker, turn_clauses in translated:
                if speaker != "U": continue

                for (rule, _), raw in turn_clauses:
                    if rule is None: continue

                    # Identify learning opportunities; i.e., any deviations from the
                    # agent's estimated states of affairs, or generic rules delivered
                    # via NL generic statements
                    self.comp_actions.identify_mismatch(rule)
                    self.comp_actions.identify_confusion(rule, prev_facts, novel_concepts)
                    self.comp_actions.identify_generics(rule, raw, prev_Qs, generics, pair_rules)

            # Update knowledge base with obtained generic statements
            for rule, w_pr, provenance in generics:
                kb_updated |= self.lt_mem.kb.add(rule, w_pr, provenance)

            # Compute scalar implicature if required by agent's strategy
            if self.strat_generic == "semNegScal":
                self.comp_actions.add_scalar_implicatures(pair_rules)

            # Handle neologisms
            xb_updated |= self.comp_actions.handle_neologisms(
                novel_concepts, dialogue_state
            )

            # Terminate the loop when 'equilibrium' is reached
            if not (xb_updated or kb_updated):
                break

    def _act(self):
        """
        Just eagerly try to resolve each item in agenda as much as possible, generating
        and performing actions until no more agenda items can be resolved for now. I
        wonder if we'll ever need a more sophisticated mechanism than this simple, greedy
        method for a good while?
        """
        ## Generate agenda items from maintenance goals
        # Currently, the maintenance goals are not to leave:
        #   - any unaddressed neologism which is unresolvable
        #   - any unaddressed recognition inconsistency btw. agent and user
        #   - any unanswered question that is answerable

        # Ideally, this is to be accomplished declaratively by properly setting up formal
        # maintenance goals and then performing automated planning or something to come
        # up with right sequence of actions to be added to agenda. However, the ad-hoc code
        # below (+ plan library in practical/plans/library.py) will do for our purpose right
        # now; we will see later if we'll ever need to generalize and implement the said
        # procedure.)

        for ti, si in self.lang.dialogue.unanswered_Q:
            self.practical.agenda.append(("address_unanswered_Q", (ti, si)))
        for n in self.lang.unresolved_neologisms:
            self.practical.agenda.append(("address_neologism", n))
        for m in self.symbolic.mismatches:
            self.practical.agenda.append(("address_mismatch", m))
        for c in self.vision.confusions:
            self.practical.agenda.append(("address_confusion", c))

        return_val = []

        while True:
            resolved_items = []
            for i, todo in enumerate(self.practical.agenda):
                todo_state, todo_args = todo

                # Check if this item can be resolved at this stage and if so, obtain
                # appropriate plan (sequence of actions) for resolving the item
                plan = self.practical.obtain_plan(todo_state)

                if plan is not None:
                    # Perform plan actions
                    for action in plan:
                        act_method = action["action_method"].extract(self)
                        act_args = action["action_args_getter"](todo_args)
                        if type(act_args) == tuple:
                            act_args = tuple(arg.extract(self) for arg in act_args)
                        else:
                            act_args = (act_args.extract(self),)

                        act_out = act_method(*act_args)
                        if act_out is not None:
                            return_val += act_out

                    resolved_items.append(i)

            if len(resolved_items) == 0:
                # No resolvable agenda item any more
                if (len(return_val) == 0 and self.lang.new_input):
                    # Nothing to add, acknowledge any user input
                    self.practical.agenda.append(("acknowledge", None))
                else:
                    # Break with return vals
                    break
            else:
                # Check off resolved agenda item
                resolved_items.reverse()
                for i in resolved_items:
                    del self.practical.agenda[i]

        return return_val
