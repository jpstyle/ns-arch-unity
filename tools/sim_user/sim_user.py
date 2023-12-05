"""
Simulated user which takes part in dialogue with rule-based pattern matching
-- no cognitive architecture ongoing within the user
"""
import re
import glob
import copy
import random

import yaml
import inflect
import numpy as np


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

        # Load any domain knowledge stored as yamls in assets dir
        self.domain_knowledge = {}
        for yml_path in glob.glob(f"{cfg.paths.assets_dir}/domain_knowledge/*.yaml"):
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
                self.current_episode_record[string_name] = None

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
                self.current_episode_record[string_name] = answer_content

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
                        if self.strat_feedback == "maxHelpExpl":
                            response.append({
                                "utterance": f"Why did you think this is a {answer_content}?",
                                "pointing": { (18, 22): gameObject_handle }
                            })

            elif utt == "I cannot explain.":
                # Agent couldn't provide any verbal explanations for its previous answer;
                # do nothing for this utterance
                pass

            elif utt.startswith("Because"):
                # Agent provided explanation for its previous answer; expecting the agent's
                # recognition of truck parts, which it deemed relevant to its reasoning
                reasons = utt.strip("Because ").strip(".").split(", and ")
                reasons = reasons[0].split(",") + reasons[1:]
                reasons = [re.findall(r"this is a (.*)", r)[0] for r in reasons]

                dem_refs = sorted(dem_refs.items())

                for part, (_, reference) in zip(reasons, dem_refs):
                    # For each part statement given as explanation, provide feedback
                    # on whether the agent's judgement was correct (will be most likely
                    # wrong... I suppose for now)

                    if isinstance(reference, str):
                        # Reference by Unity GameObject string name; test by the object
                        # name whether the agent's judgement was correct
                        # (Will reach here if the region localized by the agent's vision
                        # module has sufficiently high box-IoU with one of the existing
                        # real object... Will it ever, though?)
                        raise NotImplementedError

                    else:
                        # Reference by raw mask bitmap; the referenced entity is a 'bogus'
                        # not corresponding to any real GameObject, always incorrect
                        assert isinstance(reference, np.ndarray)

                        # Let the agent know the part recognition is incorrect
                        response.append({
                            "utterance": f"This is not a {part}.",
                            "pointing": { (0, 4): reference.reshape(-1).tolist() }
                        })

            elif utt.startswith("How are") and utt.endswith("different?"):
                # Agent requested generic differences between two similar concepts
                assert self.strat_feedback.startswith("maxHelp")

                # if contrast_concepts in self.taught_diffs:
                #     # Concept diffs requested again; do something? This would 'annoy'
                #     # the user if keeps happening
                #     ...

                singularize = inflect.engine().singular_noun
                pluralize = inflect.engine().plural

                # Extract two concepts being confused, then compute & select generic
                # characterizations that best describe how the two are different
                ques_content = re.findall(r"How are (.*) and (.*) different\?$", utt)[0]
                conc1, conc2 = singularize(ques_content[0]), singularize(ques_content[1])

                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc1, conc2
                )

                # Prepare user feedback based on the selected concept difference
                if "supertype" in conc_diffs:
                    # "Xs are Ys"
                    for conc, super_conc in conc_diffs["supertype"]:
                        conc_str = pluralize(conc).capitalize()
                        super_conc_str = pluralize(super_conc)
                        response.append({
                            "utterance": f"{conc_str} are {super_conc_str}.",
                            "pointing": {}
                        })
                    raise NotImplementedError       # Remove after sanity check

                if "parts" in conc_diffs:
                    # "Xs have Ys"
                    for conc, part in conc_diffs["parts"]:
                        conc_str = pluralize(conc).capitalize()
                        part_str = pluralize(part)
                        response.append({
                            "utterance": f"{conc_str} have {part_str}.",
                            "pointing": {}
                        })

                if "part_attributes" in conc_diffs:
                    # "Xs have Z1, Z2, ... and Zn Ys"
                    for conc, part, attrs in conc_diffs["part_attributes"]:
                        conc_str = pluralize(conc).capitalize()
                        part_str = pluralize(part)
                        attrs_str = " and ".join([", ".join(attrs[:-1])] + attrs[-1:]) \
                            if len(attrs) > 1 else attrs[0]
                        response.append({
                            "utterance": f"{conc_str} have {attrs_str} {part_str}.",
                            "pointing": {}
                        })
                    raise NotImplementedError       # Remove after sanity check

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
