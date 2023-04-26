"""
Simulated user which takes part in dialogue with rule-based pattern matching
-- no cognitive architecture ongoing within the user
"""
import os
import re
import json
import random
from collections import defaultdict

import inflect
import torch
from torchvision.ops import box_convert


class SimulatedTeacher:
    
    def __init__(self, cfg):
        # History of ITL episode records
        self.episode_records = []

        # Pieces of generic constrastive knowledge taught across episodes
        self.taught_diffs = set()

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.strat_feedback = cfg.exp1.strat_feedback

        self.ep_counter = 0

    def initiate_episode(self):
        """
        Preparation of a new interaction episode, comprising random initialization
        of the task for the episode and generation of appropriate teacher input
        """
        self.ep_counter += 1

        # Random environment initialization before reset; currently, sample fine-grained
        # type of truck as distinguished by load type
        sampled_type = random.sample(range(len(self.target_concepts)), 1)[0]

        random_inits = {
            "load_type": sampled_type
        }

        opening_output = {
            "utterances": ["What is this?"],
            "pointing": [{ (8, 12): f"truck_ep{self.ep_counter}" }]
        }

        # Initialize episode record
        self.current_record = {
            "target_concept": self.target_concepts[sampled_type],
            "answered_concept": None,
            "answer_correct": None,
            "number_of_examples": 0        # Exemplars used for learning
        }

        return random_inits, opening_output

    def react(self, agent_reactions):
        """ Rule-based pattern matching for handling agent responses """
        responses = []      # Return value containing response utterances

        target_concept = self.current_record["target_concept"]

        if "I am not sure." in agent_reactions:
            # Agent answered it doesn't have any clue what the concept instance is;
            # provide correct label, even if taking minimalist strategy (after all,
            # learning cannot take place if we don't provide any)
            self.current_record["answered_concept"] = "N/A"
            self.current_record["answer_correct"] = False
            self.current_record["number_of_examples"] += 1

            responses.append({
                "utterances": [f"This is a {target_concept}."],
                "pointing": [{ (0, 4): f"truck_ep{self.ep_counter}" }]
            })

        elif any(utt.startswith("This is") for utt in agent_reactions):
            # Agent provided an answer what the instance is
            answer_utt = [
                utt for utt in agent_reactions if utt.startswith("This is")
            ][0]
            answer_content = re.findall(r"This is a (.*)\.$", answer_utt)[0]

            self.current_record["answered_concept"] = answer_content

            if target_concept == answer_content:
                # Correct answer
                self.current_record["answer_correct"] = True
                self.current_record["number_of_examples"] += 0

                responses.append({
                    "utterances": ["Correct."],
                    "pointing": [{}]
                })
            else:
                # Incorrect answer; reaction branches here depending on teacher's strategy
                self.current_record["answer_correct"] = False
                self.current_record["number_of_examples"] += 1

                # Minimal feedback; only let the agent know the answer is incorrect
                result_response = {
                    "utterances": [f"This is not a {answer_content}."],
                    "pointing": [{ (0, 4): f"truck_ep{self.ep_counter}" }]
                }

                # Correct label additionally provided if teacher strategy is 'greater' than
                # [minimal feedback] or the concept hasn't ever been taught
                taught_concepts = set(epi["target_concept"] for epi in self.episode_records)
                is_novel_concept = target_concept not in taught_concepts
                if self.strat_feedback != "minHelp" or is_novel_concept:
                    result_response["utterances"].append(f"This is a {target_concept}.")
                    result_response["pointing"].append({ (0, 4): f"truck_ep{self.ep_counter}" })

                responses.append(result_response)

        elif any(utt.startswith("How are") and utt.endswith("different?")
            for utt in agent_reactions):
            # Agent requested generic differences between two similar concepts
            assert self.strat_feedback == "maxHelp"

            # if contrast_concepts in self.taught_diffs:
            #     # Concept diffs requested again; do something? This would 'annoy'
            #     # the user if keeps happening
            #     ...

            singularize = inflect.engine().singular_noun
            pluralize = inflect.engine().plural

            ques_utt = [
                utt for utt in agent_reactions
                if utt.startswith("How are") and utt.endswith("different?")
            ][0]
            ques_content = re.findall(r"How are (.*) and (.*) different\?$", ques_utt)[0]
            conc1, conc2 = singularize(ques_content[0]), singularize(ques_content[1])

            conc1_props = self.domain_knowledge[conc1]["part_property"]
            conc1_props = {
                (part, prop) for part, props in conc1_props.items() for prop in props
            }
            conc2_props = self.domain_knowledge[conc2]["part_property"]
            conc2_props = {
                (part, prop) for part, props in conc2_props.items() for prop in props
            }

            answer_response = {
                "v_usr_in": "n",
                "l_usr_in": [],
                "pointing": None
            }

            # For each of two directions of relative differences, synthesize
            # appropriate constrastive generic explanations
            conc1_props_diff = defaultdict(set)
            conc1_subj = ques_content[0].capitalize()
            for part, prop in sorted(conc1_props - conc2_props):
                conc1_props_diff[part].add(prop)
            for part, props in conc1_props_diff.items():
                part_name = pluralize(part.split(".")[0])
                props = list(props)
                if len(props) <= 2:
                    part_descriptor = " and ".join(pr.split(".")[0] for pr in props)
                else:
                    part_descriptor = " and ".join(pr.split(".")[0] for pr in props[-2:])
                    part_descriptor = ", ".join(
                        [pr.split(".")[0] for pr in props[:-2]]+[part_descriptor]
                    )
                generic = f"{conc1_subj} have {part_descriptor} {part_name}."
                answer_response["l_usr_in"].append(generic)

            conc2_props_diff = defaultdict(set)
            conc2_subj = ques_content[1].capitalize()
            for part, prop in sorted(conc2_props - conc1_props):
                conc2_props_diff[part].add(prop)
            for part, props in conc2_props_diff.items():
                part_name = pluralize(part.split(".")[0])
                props = list(props)
                if len(props) <= 2:
                    part_descriptor = " and ".join(pr.split(".")[0] for pr in props)
                else:
                    part_descriptor = " and ".join(pr.split(".")[0] for pr in props[-2:])
                    part_descriptor = ", ".join(
                        [pr.split(".")[0] for pr in props[:-2]]+[part_descriptor]
                    )
                generic = f"{conc2_subj} have {part_descriptor} {part_name}."
                answer_response["l_usr_in"].append(generic)

            responses.append(answer_response)

            self.taught_diffs.add(frozenset([conc1, conc2]))

        elif len(agent_reactions)==1 and agent_reactions[0]=="OK.":
            # No further interaction needed
            pass

        else:
            raise NotImplementedError

        return responses
