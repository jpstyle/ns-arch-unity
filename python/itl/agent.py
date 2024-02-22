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
        self.lt_mem = LongTermMemoryModule(cfg)

        # Provide access to methods in comp_actions
        self.comp_actions = CompositeActions(self)

        # Load agent model from specified path
        if "model_path" in cfg.agent:
            self.load_model(cfg.agent.model_path)

        # Agent learning strategy params
        self.strat_generic = cfg.agent.strat_generic
        self.strat_assent = cfg.agent.strat_assent

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
        Single agent activity loop, called with inputs of visual scene, natural
        language utterances from user and affiliated gestures (demonstrative
        references by pointing in our case). new_scene flag indicates whether
        the visual scene input is new.
        """
        self._vis_inp(usr_in=v_usr_in, new_scene=new_scene)
        self._lang_inp(usr_in=l_usr_in, pointing=pointing)
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
        self.vision.new_input_provided = new_scene
        if self.vision.new_input_provided:
            self.vision.previous_input = self.vision.latest_input
            self.vision.latest_input = usr_in

    def _lang_inp(self, usr_in, pointing):
        """ Handle provided language input (from user) """
        if usr_in is None:
            usr_in = []
        elif not isinstance(usr_in, list):
            assert isinstance(usr_in, str)
            usr_in = [usr_in]

        assert len(usr_in) == len(pointing)

        if len(usr_in) == 0:
            self.lang.new_input_provided = False
        else:
            self.lang.new_input_provided = True

            # Patchwork handling of connectives, acknowledgements, interjections, etc.
            # that don't really need principled treatment and can be shaved off without
            # consequences. Treat these properly later if they ever become important...
            usr_in_new = []
            for utt_string, dem_ref in zip(usr_in, pointing):
                # 'Sanitization' process
                if utt_string.startswith("But ") or utt_string.startswith("And "):
                    pf_len = 4
                    utt_string = utt_string[pf_len:].capitalize()
                    dem_ref_items = list(dem_ref.items())
                    for k, v in dem_ref_items: del dem_ref[k]
                    for k, v in dem_ref_items:
                        dem_ref[(k[0]-pf_len, k[1]-pf_len)] = v
                if utt_string.startswith("It's true "):
                    pf_len = 10
                    utt_string = utt_string[pf_len:].capitalize()
                    dem_ref_items = list(dem_ref.items())
                    for k, v in dem_ref_items: del dem_ref[k]
                    for k, v in dem_ref_items:
                        dem_ref[(k[0]-pf_len, k[1]-pf_len)] = v
                if utt_string.endswith(", too."):
                    sf_len = 5
                    utt_string = utt_string[:-sf_len-1] + "."

                # Add final processed utterance string
                usr_in_new.append(utt_string)
            usr_in = usr_in_new

            parsed_input = None
            try:
                parsed_input = self.lang.semantic.nl_parse(usr_in)
            except IndexError as e:
                logger.info(str(e))
            else:
                self.lang.latest_input = parsed_input

    def _update_belief(self, pointing):
        """ Form beliefs based on visual and/or language input """

        if not (self.vision.new_input_provided or self.lang.new_input_provided):
            # No information whatsoever to make any belief updates
            return

        # Lasting storage of pointing info
        if pointing is None:
            pointing = {}

        # For showing visual UI on only the first time
        vis_ui_on = self.vis_ui_on

        # Translated dialogue record and visual context from currently stored values
        # (scene may or may not change)
        prev_translated = self.symbolic.translate_dialogue_content(self.lang.dialogue)
        prev_vis_scene = self.vision.scene
        prev_pr_prog = self.symbolic.concl_vis[1][1] if self.symbolic.concl_vis else None
        prev_kb = self.kb_snap
        prev_context = (prev_vis_scene, prev_pr_prog, prev_kb)

        # Some cleaning steps needed whenever visual context changes
        if self.vision.new_input_provided:
            # # Prior to resetting visual context, store current one into the episodic
            # # memory (visual perceptions & user language inputs), in the form of
            # # LP^MLN program fragments
            # if (self.vision.scene is not None and
            #     len(self.vision.scene) > 1 and
            #     self.symbolic.concl_vis_lang is not None):

            #     pr_prog, _, dl_prog = self.symbolic.concl_vis_lang[1]
            #     self.episodic_memory.append((pr_prog, dl_prog))
            
            # Refresh dialogue manager & symbolic reasoning module states
            self.lang.dialogue.refresh()
            self.symbolic.refresh()

            # Update KB snapshot on episode-basis
            self.kb_snap = copy.deepcopy(self.lt_mem.kb)

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

            if self.vision.new_input_provided:
                # Ground raw visual perception with scene graph generation module
                self.vision.predict(
                    self.vision.latest_input, self.lt_mem.exemplars,
                    visualize=vis_ui_on, lexicon=self.lt_mem.lexicon
                )
                vis_ui_on = False

                # Inform the language module of the new visual context
                self.lang.situate(self.vision.scene)

            elif xb_updated:
                # Concept exemplar base updated, need reclassification while keeping
                # the discovered objects and embeddings intact
                self.vision.predict(
                    None, self.lt_mem.exemplars, reclassify=True, visualize=False
                )

            if self.lang.new_input_provided:
                # Revert to pre-update dialogue state at the start of each loop iteration
                self.lang.dialogue.record = self.lang.dialogue.record[:ti_last]

                # Understand the user input in the context of the dialogue
                self.lang.understand(self.lang.latest_input, pointing=pointing)

            ents_updated = False
            if self.vision.scene is not None:
                # If a new entity is registered as a result of understanding the latest
                # input, re-run vision module to update with new predictions for it
                new_ents = set(self.lang.dialogue.referents["env"]) - set(self.vision.scene)
                new_ents.remove("_self")
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

            if self.vision.new_input_provided or ents_updated or xb_updated or kb_updated:
                # Sensemaking from vision input only
                exported_kb = self.lt_mem.kb.export_reasoning_program()
                visual_evidence = self.lt_mem.kb.visual_evidence_from_scene(self.vision.scene)
                self.symbolic.sensemake_vis(exported_kb, visual_evidence)
                self.lang.dialogue.sensemaking_v_snaps[ti_last] = self.symbolic.concl_vis

            if self.lang.new_input_provided:
                # Reference & word sense resolution to connect vision & discourse
                self.symbolic.resolve_symbol_semantics(
                    self.lang.dialogue, self.lt_mem.lexicon
                )

                # if self.vision.scene is not None:
                #     # Sensemaking from vision & language input
                #     self.symbolic.sensemake_vis_lang(self.lang.dialogue)
                #     self.lang.dialogue.sensemaking_vl_snaps[ti_last] = \
                #         self.symbolic.concl_vis_lang

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

            # Collect previous factual statements and questions made during this dialogue
            prev_statements = []; prev_Qs = []
            for ti, (spk, turn_clauses) in enumerate(prev_translated):
                for ci, ((rule, ques), raw) in enumerate(turn_clauses):
                    # Factual statement
                    if rule is not None and len(rule[0])==1 and rule[1] is None:
                        prev_statements.append(((ti, ci), (spk, rule)))

                    # Question
                    if ques is not None:
                        # Here, `rule` represents presuppositions included in `ques`
                        prev_Qs.append(((ti, ci), (spk, ques, rule, raw)))

            # Translate dialogue record into processable format based on the result
            # of symbolic.resolve_symbol_semantics
            translated = self.symbolic.translate_dialogue_content(self.lang.dialogue)

            # Process translated dialogue record to do the following:
            #   - Identify recognition mismatch btw. user provided vs. agent
            #   - Identify visual concept confusion
            #   - Identify new generic rules to be integrated into KB
            for ti, (speaker, turn_clauses) in enumerate(translated):
                if speaker != "U": continue

                for ci, ((rule, _), raw) in enumerate(turn_clauses):
                    if rule is None: continue

                    # Disregard clause if it is not domain-describing or is in irrealis mood
                    clause_info = self.lang.dialogue.clause_info[f"t{ti}c{ci}"]
                    if not clause_info["domain_describing"]:
                        continue
                    if clause_info["irrealis"]:
                        continue

                    # Identify learning opportunities; i.e., any deviations from the agent's
                    # estimated states of affairs, generic rules delivered via NL generic
                    # statements, or acknowledgements (positive or lack of negative)
                    self.comp_actions.identify_mismatch(rule)
                    self.comp_actions.identify_confusion(
                        rule, prev_statements, novel_concepts
                    )
                    self.comp_actions.identify_acknowledgement(
                        rule, prev_statements, prev_context
                    )
                    self.comp_actions.identify_generics(
                        rule, raw, prev_Qs, generics, pair_rules
                    )

            # By default, treat lack of any negative acknowledgements to an agent's statement
            # as positive acknowledgement
            prev_or_curr = "prev" if self.vision.new_input_provided else "curr"
            for (ti, ci), (speaker, (statement, _)) in prev_statements:
                if speaker != "A": continue         # Not interested

                stm_ind = (prev_or_curr, ti, ci)
                if stm_ind not in self.lang.dialogue.acknowledged_stms:
                    acknowledgement_data = (statement, True, prev_context)
                    self.lang.dialogue.acknowledged_stms[stm_ind] = acknowledgement_data

            # Update knowledge base with obtained generic statements
            for rule, w_pr, knowledge_source, knowledge_type in generics:
                kb_updated |= self.lt_mem.kb.add(
                    rule, w_pr, knowledge_source, knowledge_type
                )

            # Compute scalar implicature if required by agent's strategy
            if self.strat_generic == "semNegScal":
                self.comp_actions.add_scalar_implicature(pair_rules)

            # Handle neologisms
            xb_updated |= self.comp_actions.handle_neologism(
                novel_concepts, self.lang.dialogue
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
        #
        # Ideally, this is to be accomplished declaratively by properly setting up formal
        # maintenance goals and then performing automated planning or something to come
        # up with right sequence of actions to be added to agenda. However, the ad-hoc
        # code below (+ plan library in practical_reasoning/plans/library.py) will do
        # for our purpose right now; we will see later if we'll ever need to generalize
        # and implement the said procedure.)

        # This ordering ensures any knowledge updates (that doesn't require interaction)
        # happen first, addressing & answering questions happen next, finally asking
        # any questions afterwards
        for m in self.symbolic.mismatches:
            self.practical.agenda.append(("address_mismatch", m))
        for a in self.lang.dialogue.acknowledged_stms.items():
            self.practical.agenda.append(("address_acknowledgement", a))
        for ti, ci in self.lang.dialogue.unanswered_Qs:
            self.practical.agenda.append(("address_unanswered_Q", (ti, ci)))
        for n in self.lang.unresolved_neologisms:
            self.practical.agenda.append(("address_neologism", n))
        for c in self.vision.confusions:
            self.practical.agenda.append(("address_confusion", c))

        return_val = []

        num_resolved_items = 0; unresolved_items = []
        while True:
            # Loop through agenda stack items from the top; new agenda items may
            # be pushed to the top by certain plans
            while len(self.practical.agenda) > 0:
                # Pop the top agenda item from the stack
                todo_state, todo_args = self.practical.agenda.pop(0)

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

                            # Note) We don't need to consider any sort of plan failures
                            # right now, but when that happens (should be identifiable
                            # from act_out value), in principle, will need to break
                            # from plan execution and add to unresolved item list
                            if False:       # Plan failure check not implemented
                                unresolved_items.append((todo_state, todo_args))
                                break
                    else:
                        num_resolved_items += 1
                else:
                    # Plan not found, agenda item unresolved
                    unresolved_items.append((todo_state, todo_args))

            # Any unresolved items back to agenda stack
            self.practical.agenda = unresolved_items

            if num_resolved_items == 0 or len(self.practical.agenda) == 0:
                # No resolvable agenda item any more, or stack clear
                if len(return_val) == 0 and self.lang.new_input_provided:
                    # Nothing to utter, acknowledge any user input
                    self.practical.agenda.append(("acknowledge", None))
                else:
                    # Break loop with return vals
                    break

        return return_val
