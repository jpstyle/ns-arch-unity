"""
Agent actions API that implements and exposes composite actions. By 'composite',
it refers to actions requiring interplay between more than one agent modules.
By 'action', it may refer to internal (cognitive, epistemic) or external (physical,
environment-interactive) actions. Actions may be evoked by method name in plans
obtained from the practical reasoning module.
"""
from .dialogue import *
from .learn import *

class CompositeActions:
    def __init__(self, agent):
        # Register actor agent
        self.agent = agent

    # Dialogue actions
    def attempt_answer_Q(self, utt_pointer):
        return attempt_answer_Q(self.agent, utt_pointer)
    def prepare_answer_Q(self, utt_pointer):
        return prepare_answer_Q(self.agent, utt_pointer)

    # Learning actions
    def identify_mismatch(self, rule):
        return identify_mismatch(self.agent, rule)
    def handle_mismatch(self, mismatch):
        return handle_mismatch(self.agent, mismatch)
    def identify_confusion(self, rule, prev_statements, novel_concepts):
        return identify_confusion(self.agent, rule, prev_statements, novel_concepts)
    def handle_confusion(self, confusion):
        return handle_confusion(self.agent, confusion)
    def identify_generics(self, rule, provenance, prev_Qs, generics, pair_rules):
        return identify_generics(self.agent, rule, provenance, prev_Qs, generics, pair_rules)
    def identify_acknowledgement(self, rule, prev_statements, prev_context):
        return identify_acknowledgement(self.agent, rule, prev_statements, prev_context)
    def handle_acknowledgement(self, acknowledgement_info):
        return handle_acknowledgement(self.agent, acknowledgement_info)
    def add_scalar_implicature(self, pair_rules):
        return add_scalar_implicature(self.agent, pair_rules)
    def handle_neologism(self, novel_concepts, dialogue_state):
        return handle_neologism(self.agent, novel_concepts, dialogue_state)
