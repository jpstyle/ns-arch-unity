"""
Simulated user which takes part in dialogue with rule-based pattern matching
-- no cognitive architecture ongoing within the user
"""
import os
import re
import copy
import random

import yaml
import inflect
import numpy as np

from python.itl.vision.utils import mask_iou


singularize = inflect.engine().singular_noun
pluralize = inflect.engine().plural

MATCH_THRES = 0.9

class SimulatedTeacher:
    
    def __init__(self, cfg):
        # History of ITL episode records
        self.episode_records = []

        # Pieces of generic constrastive knowledge taught across episodes
        self.taught_diffs = set()

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.strat_feedback = cfg.exp.strat_feedback

        # Whether the agent is running in test mode
        self.agent_test_mode = cfg.agent.test_mode

        # Load appropriate domain knowledge stored as yamls in assets dir
        self.domain_knowledge = {}
        yml_path = f"{cfg.paths.assets_dir}/domain_knowledge/{cfg.exp.concept_set}.yaml"
        if os.path.exists(yml_path):
            with open(yml_path) as yml_f:
                self.domain_knowledge.update(yaml.safe_load(yml_f))
        # Convert lists to sets for order invariance
        for info in self.domain_knowledge.values():
            if info["part_attributes"] is None:
                info["part_attributes"] = {}
            else:
                info["part_attributes"] = {
                    part: set(attrs)
                    for part, attrs in info["part_attributes"].items()
                }

        # Extract taxonomy knowledge implied by the domain knowledge, as a map from
        # part subtype to its supertype
        self.taxonomy_knowledge = {}
        for info in self.domain_knowledge.values():
            for part_type, part_subtype in info["parts"].items():
                self.taxonomy_knowledge[part_subtype] = part_type

    def setup_episode(self, concept_set, shrink_domain=False):
        """
        Preparation of a new interaction episode, comprising random initialization
        of the task for the episode and queueing of target concepts to teach
        """
        target_concepts = self.target_concept_sets[concept_set]
        self.concept_set = concept_set

        # Random environment initialization before reset; currently, sample fine-grained
        # type of truck as distinguished by load type
        if shrink_domain:
            # Sample from target concept set except the last one
            sampled_type = random.sample(range(len(target_concepts)-1), 1)[0]
        else:
            # Default: sample from whole target concept set
            sampled_type = random.sample(range(len(target_concepts)), 1)[0]

        # Initialize target concept queue and episode record
        self.current_queue = copy.deepcopy(target_concepts[sampled_type][1])
        self.current_episode_record = {}

        # Return environment parameters to pass
        random_inits = {
            part_type: part_subtype_ind
            for part_type, part_subtype_ind in target_concepts[sampled_type][0]
        }
        return random_inits

    def initiate_dialogue(self):
        """
        Prepare an opening line for starting a new thread of dialogue, based on
        the current queue of target concepts (unprompted dialogue starter, as
        opposed to self.react method below, which is invoked only when prompted
        by an agent reaction)
        """
        # Dequeue from current episode's target concept queue
        self.current_target_concept = self.current_queue.pop(0)
        gameObject_path, _ = self.current_target_concept

        if gameObject_path is None:
            gameObject_handle = "/truck"
            part_type = None
        else:
            part_type, part_subtype = gameObject_path
            part_subtype_cc = "".join(
                tok.capitalize() if i>0 else tok
                for i, tok in enumerate(part_subtype.split(" "))
            )           # camelCase
            gameObject_handle = f"/truck/{part_type}/{part_type}_{part_subtype_cc}"

        if self.concept_set == "prior_supertypes":
            return [{
                "utterance": "What is this?",
                "pointing": { (8, 12): gameObject_handle }
            }]
        elif self.concept_set == "prior_parts":
            return [{
                "utterance": f"What kind of {part_type} is this?",
                "pointing": { (17+len(part_type), 17+len(part_type)+4): gameObject_handle }
            }]
        elif self.concept_set == "single_fourway" or self.concept_set == "double_fiveway":
            return [{
                "utterance": "What kind of truck is this?",
                "pointing": { (22, 26): gameObject_handle }
            }]
        else:
            raise ValueError

    def react(self, agent_reactions):
        """ Rule-based pattern matching for handling agent responses """
        gameObject_path, string_name = self.current_target_concept
        if gameObject_path is None:
            gameObject_handle = "/truck"
        else:
            part_type, part_subtype = gameObject_path
            part_subtype_cc = "".join(
                tok.capitalize() if i>0 else tok
                for i, tok in enumerate(part_subtype.split(" "))
            )           # camelCase
            gameObject_handle = f"/truck/{part_type}/{part_type}_{part_subtype_cc}"

        response = []
        for utt, dem_refs in agent_reactions:
            if utt == "I am not sure.":
                # Agent answered it doesn't have any clue what the concept instance
                # is; provide correct label, even if taking minimalist strategy (after
                # all, learning cannot take place if we don't provide any)
                self.current_episode_record[string_name] = { "answer": None }

                if self.agent_test_mode:
                    # No feedback needed if agent running in test mode
                    if len(self.current_queue) > 0:
                        # Remaining target concepts to test and teach
                        response += self.initiate_dialogue()
                    else:
                        # No further interaction needed
                        pass
                else:
                    response.append({
                        "utterance": f"This is a {string_name}.",
                        "pointing": { (0, 4): gameObject_handle }
                    })

            elif utt.startswith("This is"):
                # Agent provided an answer what the instance is
                answer_content = re.findall(r"This is a (.*)\.$", utt)[0]
                self.current_episode_record[string_name] = { "answer": answer_content }

                if self.agent_test_mode:
                    # No feedback needed if agent running in test mode
                    if len(self.current_queue) > 0:
                        # Remaining target concepts to test and teach
                        response += self.initiate_dialogue()
                    else:
                        # No further interaction needed
                        pass
                else:
                    if string_name == answer_content:
                        # Correct answer

                        # # Teacher would acknowledge by saying "Correct" in the previous
                        # # project. but I think we can skip that for simplicity
                        # responses.append({
                        #     "utterances": ["Correct."],
                        #     "pointing": [{}]
                        # })

                        if len(self.current_queue) > 0:
                            # Remaining target concepts to test and teach
                            response += self.initiate_dialogue()
                    else:
                        # Incorrect answer; reaction branches here depending on teacher's
                        # strategy

                        # At all feedback level, let the agent know the answer is incorrect
                        response.append({
                            "utterance": f"This is not a {answer_content}.",
                            "pointing": { (0, 4): gameObject_handle }
                        })

                        # Correct label additionally provided if teacher strategy is 'greater'
                        # than [minimal feedback] or the concept hasn't ever been taught
                        taught_concepts = set(
                            conc for epi in self.episode_records for conc in epi
                        )
                        is_novel_concept = string_name not in taught_concepts
                        if self.strat_feedback != "minHelp" or is_novel_concept:
                            response.append({
                                "utterance": f"This is a {string_name}.",
                                "pointing": { (0, 4): gameObject_handle }
                            })

                        # Ask for an explanation why the agent have the incorrect answer,
                        # if the strategy is to take interest
                        if self.strat_feedback.startswith("maxHelpExpl"):
                            response.append({
                                "utterance": f"Why did you think this is a {answer_content}?",
                                "pointing": { (18, 22): gameObject_handle }
                            })

            elif utt == "I cannot explain.":
                # Agent couldn't provide any verbal explanations for its previous answer;
                # always provide generic rules about concept differences between agent
                # answer vs. ground truth
                assert self.strat_feedback.startswith("maxHelpExpl")

                # Log (lack of) cited reasons
                self.current_episode_record[string_name]["reason"] = "noExpl"

                conc_gt = string_name
                conc_ans = self.current_episode_record[string_name]["answer"]
                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc_gt, conc_ans
                )
                response += _properties_to_nl(conc_diffs)

            elif utt.startswith("Because I thought "):
                # Agent provided explanation for its previous answer; expecting the agent's
                # recognition of truck parts (or failure thereof), which it deemed relevant
                # to its reasoning
                reasons = re.findall(r"^Because I thought (.*).$", utt)[0].split(", and ")
                reasons = reasons[0].split(", ") + reasons[1:]
                reasons = [_parse_nl_reason(r) for r in reasons]

                # Log types of cited reasons
                self.current_episode_record[string_name]["reason"] = \
                    "|".join({reason_type for _, reason_type in reasons})

                dem_refs = sorted(dem_refs.items())

                # Prepare concept differences, needed for checking whether the agent's
                # knowledge (indirectly revealed through its explanations) about the
                # concepts is incomplete
                conc_gt = string_name
                conc_ans = self.current_episode_record[string_name]["answer"]
                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc_gt, conc_ans
                )

                # Part domain knowledge info dicts for ground truth and agent answer
                gt_parts_info = self.domain_knowledge[conc_gt]["parts"]
                ans_parts_info = self.domain_knowledge[conc_ans]["parts"]

                # Part types that actually play role for distinguishing conc_gt vs. conc_ans
                distinguishing_part_types = {p for _, p in conc_diffs["parts"]}

                has_diff_knowledge = False
                for (part, reason_type), (_, reference) in zip(reasons, dem_refs):
                    # For each part statement given as explanation, provide feedback
                    # on whether the agent's judgement was correct (will be most likely
                    # wrong... I suppose for now)

                    # Variable storing currently active pending connective
                    pending_connective = None

                    # Checking if mentioned part subtype is a property of the ground
                    # truth concept & the agent answer concept
                    part_type = self.taxonomy_knowledge[part]
                    part_gt_prop = part_type in gt_parts_info and \
                        gt_parts_info[part_type] == part
                    part_ans_prop = part_type in ans_parts_info and \
                        ans_parts_info[part_type] == part

                    # Current experiment design is such that all parts of the ground
                    # truth concept have their masks visible in the scene, not occluded
                    # or anything
                    assert part_type in self.current_gt_masks

                    if part in distinguishing_part_types:
                        # The part type cited in the reason bears relevance as to
                        # distinguishing the agent answer vs. ground truth concepts.
                        has_diff_knowledge = True

                    if isinstance(reference, str):
                        # Reference by Unity GameObject string name; test by the object
                        # name whether the agent's judgement was correct
                        # (Will reach here if the region localized by the agent's vision
                        # module has sufficiently high box-IoU with one of the existing
                        # real object... Will it ever, though?)
                        raise NotImplementedError

                    else:
                        # Reference by raw mask bitmap
                        assert isinstance(reference, np.ndarray)

                        if reason_type == "existence":
                            # Agent made an explicit part claim as sufficient explanation,
                            # which may or may not be true

                            # Sanity check: agent must be believing the mentioned part
                            # is a (distinguishing) property of the concept it provided
                            # as answer
                            assert part_ans_prop

                            if part_gt_prop:
                                # Mentioned part actually exists somewhere, obtain IoU
                                gt_mask = self.current_gt_masks[part_type]
                                match_score = mask_iou([gt_mask], [reference])[0][0]
                            else:
                                # Mentioned part absent, consider match score zero
                                match_score = 0

                            if match_score > MATCH_THRES:
                                # The part claim itself is true
                                _add_after_redundancy_check(response, {
                                    "utterance": f"It's true this is a {part}.",
                                    "pointing": { (10, 14): reference.reshape(-1).tolist() }
                                })
                                pending_connective = "but"
                            else:
                                # The part claim is false, need correction
                                _add_after_redundancy_check(response, {
                                    "utterance": f"This is not a {part}.",
                                    "pointing": { (0, 4): reference.reshape(-1).tolist() }
                                })
                                pending_connective = "and"

                            # If the mentioned part is also a property of the ground truth,
                            # and thus doesn't serve as distinguishing property, need to
                            # correct that
                            if part_gt_prop:
                                conc_gt_pl = pluralize(conc_gt).capitalize()
                                part_pl = pluralize(part)
                                _add_after_redundancy_check(response, _prepend_with_connective({
                                    "utterance": f"{conc_gt_pl} have {part_pl}, too.",
                                    "pointing": {}
                                }, pending_connective))
                                pending_connective = None

                        elif reason_type == "uncertain" or reason_type == "absence":
                            # Agent made a counterfactual excuse

                            # Sanity check: agent must be believing the mentioned part
                            # is a (distinguishing) property of the ground truth concept
                            assert part_gt_prop

                            if reason_type == "uncertain":
                                # Agent exhibited uncertainty on part identity as counterfactual
                                # excuse; the referenced entity may or may not be an instance of
                                # the part subtype

                                # Mentioned part always exists, obtain IoU
                                gt_mask = self.current_gt_masks[part_type]
                                match_score = mask_iou([gt_mask], [reference])[0][0]

                                if match_score > MATCH_THRES:
                                    # Sufficient overlap, match good enough; endorse the
                                    # proposed mask reference as correct
                                    _add_after_redundancy_check(response, {
                                        "utterance": f"This is a {part}.",
                                        "pointing": { (0, 4): reference.reshape(-1).tolist() }
                                    })
                                    pending_connective = "but"
                                else:
                                    # Not enough overlap, bad match; reject the suspected
                                    # part reference and provide correct mask
                                    _add_after_redundancy_check(response, {
                                        "utterance": f"This is not a {part}.",
                                        "pointing": { (0, 4): reference.reshape(-1).tolist() }
                                    })
                                    _add_after_redundancy_check(response, {
                                        "utterance": f"This is a {part}.",
                                        "pointing": { (0, 4): gt_mask.reshape(-1).tolist() }
                                    })
                                    pending_connective = "and"

                            elif reason_type == "absence":
                                # Agent made an explicit absence claim as counterfactual excuse,
                                # which may or may not be true
                                gt_mask = self.current_gt_masks[part_type]

                                # The absence claim is false; point to the ground-truth
                                _add_after_redundancy_check(response, {
                                    "utterance": f"This is a {part}.",
                                    "pointing": { (0, 4): gt_mask.reshape(-1).tolist() }
                                })
                                pending_connective = "and"

                            # If the mentioned part is also a property of the incorrect agent
                            # answer, and thus doesn't serve as distinguishing property, need
                            # to correct that
                            if part_ans_prop:
                                conc_ans_pl = pluralize(conc_ans).capitalize()
                                part_pl = pluralize(part)
                                _add_after_redundancy_check(response, _prepend_with_connective({
                                    "utterance": f"{conc_ans_pl} have {part_pl}, too.",
                                    "pointing": {}
                                }, pending_connective))
                                pending_connective = None

                        else:
                            # Don't know how to handle other reason types
                            raise NotImplementedError

                # If imperfect agent knowledge is revealed, provide appropriate
                # feedback regarding generic differences between conc_gt vs. conc_ans
                if not has_diff_knowledge:
                    conc_diff_feedback = [
                        _prepend_with_connective(fb, pending_connective) if i==0 else fb
                        for i, fb in enumerate(_properties_to_nl(conc_diffs))
                    ]
                    response += conc_diff_feedback
                    pending_connective = None

            elif utt.startswith("How are") and utt.endswith("different?"):
                # Agent requested generic differences between two similar concepts
                assert self.strat_feedback.startswith("maxHelp")

                # if contrast_concepts in self.taught_diffs:
                #     # Concept diffs requested again; do something? This would 'annoy'
                #     # the user if keeps happening
                #     ...

                # Extract two concepts being confused, then compute & select generic
                # characterizations that best describe how the two are different
                ques_content = re.findall(r"How are (.*) and (.*) different\?$", utt)[0]
                conc1, conc2 = singularize(ques_content[0]), singularize(ques_content[1])
                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc1, conc2
                )
                response += _properties_to_nl(conc_diffs)

                self.taught_diffs.add(frozenset([conc1, conc2]))

            elif utt == "OK.":
                if len(self.current_queue) > 0:
                    # Remaining target concepts to test and teach
                    response += self.initiate_dialogue()
                else:
                    # No further interaction needed
                    pass

            else:
                raise NotImplementedError

        return response


def _compute_concept_differences(domain_knowledge, conc1, conc2):
    """
    Compute differences of concepts to teach based on the domain knowledge provided.
    Compare 'properties' of the concepts by the following order:

    0) Return empty dict if entirely identical (by the domain knowledge)
    1) If supertypes are different, teach the supertype difference
    2) Else-if belonging part sets are different, teach the part set difference
    3) Else-if attributes of any parts are different, teach the part attribute difference
    """
    assert conc1 in domain_knowledge and conc2 in domain_knowledge
    conc1_props = domain_knowledge[conc1]
    conc2_props = domain_knowledge[conc2]

    conc_diffs = {}

    # Compare supertype
    if conc1_props["supertype"] != conc2_props["supertype"]:
        conc_diffs["supertype"] = [
            (conc1, conc1_props["supertype"]),
            (conc2, conc1_props["supertype"]),
        ]

        raise NotImplementedError

    elif conc1_props["parts"] != conc2_props["parts"]:
        conc_diffs["parts"] = []

        # Symmetric set difference on part sets
        part_type_union = set(conc1_props["parts"]) | set(conc2_props["parts"])
        for part_type in part_type_union:
            if conc1_props["parts"].get(part_type) == conc2_props["parts"].get(part_type):
                # No difference on part type info
                continue

            if part_type in conc1_props["parts"]:
                conc_diffs["parts"].append((conc1, conc1_props["parts"][part_type]))
            if part_type in conc2_props["parts"]:
                conc_diffs["parts"].append((conc2, conc2_props["parts"][part_type]))

    elif conc1_props["part_attributes"] != conc2_props["part_attributes"]:
        conc_diffs["part_attributes"] = []

        raise NotImplementedError

        # Symmetric set difference on part attributes for each corresponding part
        for part in conc1_props["parts"]:
            conc1_attrs = conc1_props["parts"][part]
            conc2_attrs = conc2_props["parts"][part]

            if len(conc1_attrs-conc2_attrs) > 0:
                conc_diffs["part_attributes"].append(
                    (conc1, part, list(conc1_attrs-conc2_attrs))
                )
            if len(conc2_attrs-conc1_attrs) > 0:
                conc_diffs["part_attributes"].append(
                    (conc2, part, list(conc2_attrs-conc1_attrs))
                )

    return conc_diffs


def _parse_nl_reason(nl_string):
    """
    Recognize which claim was made by the agent by means of the NL string. Return
    a tuple ()
    """
    # Case 1: Agent made 'positive statement' that the mask refers to a part type
    re_test = re.findall(r"^this is a (.*)$", nl_string)
    if len(re_test) > 0:
        return (re_test[0], "existence")

    # Case 2: Agent is not sure if the provided mask refers to a part type
    re_test = re.findall(r"^this might not be a (.*)$", nl_string)
    if len(re_test) > 0:
        return (re_test[0], "uncertain")

    # Case 3: Agent made 'negative statement' that it failed to find any occurrence
    # of a part type
    re_test = re.findall(r"^this doesn't have a (.*)$", nl_string)
    if len(re_test) > 0:
        return (re_test[0], "absence")


def _properties_to_nl(conc_props):
    """
    Helper method factored out for realizing some collection of concept properties
    to natural language feedback to agent
    """
    # Prepare user feedback based on the selected concept difference
    feedback = []

    if "supertype" in conc_props:
        # "Xs are Ys"
        for conc, super_conc in conc_props["supertype"]:
            conc_str = pluralize(conc).capitalize()
            super_conc_str = pluralize(super_conc)
            feedback.append({
                "utterance": f"{conc_str} are {super_conc_str}.",
                "pointing": {}
            })
        raise NotImplementedError       # Remove after sanity check

    if "parts" in conc_props:
        # "Xs have Ys"
        for conc, part in conc_props["parts"]:
            conc_str = pluralize(conc).capitalize()
            part_str = pluralize(part)
            feedback.append({
                "utterance": f"{conc_str} have {part_str}.",
                "pointing": {}
            })

    if "part_attributes" in conc_props:
        # "Xs have Z1, Z2, ... and Zn Ys"
        for conc, part, attrs in conc_props["part_attributes"]:
            conc_str = pluralize(conc).capitalize()
            part_str = pluralize(part)
            attrs_str = " and ".join([", ".join(attrs[:-1])] + attrs[-1:]) \
                if len(attrs) > 1 else attrs[0]
            feedback.append({
                "utterance": f"{conc_str} have {attrs_str} {part_str}.",
                "pointing": {}
            })
        raise NotImplementedError       # Remove after sanity check

    return feedback


def _prepend_with_connective(response, connective):
    """
    Helper method factored out for prepending a connective to an agent utterance
    (if and only if connective is not None).
    """
    if connective is None: return response

    utterance = response["utterance"]
    response_new = {
        "utterance": f"{connective.capitalize()} {utterance[0].lower() + utterance[1:]}",
        "pointing": response["pointing"]
    }
    return response_new


def _add_after_redundancy_check(outgoing_buffer, feedback):
    """ Add to buffer of outgoing feedback messages only if not already included """
    if feedback in outgoing_buffer: return
    outgoing_buffer.append(feedback)
