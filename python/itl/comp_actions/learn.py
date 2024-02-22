"""
Implements learning-related composite actions; by learning, we are primarily
referring to belief updates that modify long-term memory
"""
import re
import math
from itertools import permutations
from collections import defaultdict

import inflect

from ..lpmln import Literal, Polynomial
from ..lpmln.utils import flatten_cons_ante, wrap_args


EPS = 1e-10                  # Value used for numerical stabilization
SR_THRES = 0.8               # Mismatch surprisal threshold
U_IN_PR = 0.99               # How much the agent values information provided by the user
A_IM_PR = 0.99               # How much the agent values inferred implicature

# Recursive helper methods for checking whether rule cons/ante is grounded (variable-
# free), lifted (all variables), contains any predicate referent as argument, or uses
# a reserved (pred type *) predicate 
is_grounded = lambda cnjt: all(not is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_grounded(nc) for nc in cnjt)
is_lifted = lambda cnjt: all(is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_lifted(nc) for nc in cnjt)
has_pred_referent = \
    lambda cnjt: any(isinstance(a,str) and a[0].lower()=="p" for a, _ in cnjt.args) \
        if isinstance(cnjt, Literal) else any(has_pred_referent(nc) for nc in cnjt)
has_reserved_pred = \
    lambda cnjt: cnjt.name.startswith("*_") \
        if isinstance(cnjt, Literal) else any(has_reserved_pred(nc) for nc in cnjt)


def identify_mismatch(agent, rule):
    """
    Test against vision-only sensemaking result to identify any mismatch btw.
    agent's & user's perception of world state
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    # Skip handling duplicate cases in the context
    if rule in [r for r, _, _ in agent.symbolic.mismatches]: return

    if (rule_is_grounded and not rule_has_pred_referent and 
        agent.symbolic.concl_vis is not None):
        # Grounded event without constant predicate referents, only if vision-only
        # sensemaking result has been obtained

        # Make a yes/no query to obtain the likelihood of content
        reg_gr_v, _ = agent.symbolic.concl_vis
        q_response, _ = agent.symbolic.query(reg_gr_v, None, rule)
        ev_prob = q_response[()]

        surprisal = -math.log(ev_prob + EPS)
        if surprisal >= -math.log(SR_THRES):
            agent.symbolic.mismatches.append([rule, surprisal, False])


def identify_confusion(agent, rule, prev_statements, novel_concepts):
    """
    Test against vision module output to identify any 'concept overlap' -- i.e.
    whenever the agent confuses two concepts difficult to distinguish visually
    and mistakes one for another.
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if (
        agent.cfg.exp.strat_feedback.startswith("maxHelp") and
        rule_is_grounded and ante is None and not rule_has_pred_referent
    ):
        # Positive grounded fact with non-reserved predicates that don't have constant
        # predicate referent args; only if the user adopts maxHelp* strategy and provides
        # generic NL feedback

        # Fetch agent's last answer; this assumes the last factual statement
        # by agent is provided as answer to the last question from user. This
        # might change in the future when we adopt a more sophisticated formalism
        # for representing discourse to relax the assumption.
        prev_statements_A = [
            stm for _, (spk, stm) in prev_statements
            if spk=="A" and not any(has_reserved_pred(cnjt) for cnjt in stm[0])
        ]
        if len(prev_statements_A) == 0:
            # Hasn't given an answer (i.e., "I am not sure.")
            return

        agent_last_ans = prev_statements_A[-1][0][0]
        ans_conc_type, ans_conc_ind = agent_last_ans.name.split("_")
        ans_conc_ind = int(ans_conc_ind)

        for lit in cons:
            # Disregard negated conjunctions
            if not isinstance(lit, Literal): continue

            # (Temporary) Only consider 1-place predicates, so retrieve the single
            # and first entity from the arg list
            assert len(lit.args) == 1

            conc_type, conc_ind = lit.name.split("_")
            conc_ind = int(conc_ind)

            if (conc_ind, conc_type) not in novel_concepts:
                # Disregard if the correct answer concept is novel to the agent and
                # has to be newly registered in the visual concept inventory
                continue

            if (lit.args == agent_last_ans.args and conc_type == ans_conc_type
                and conc_ind != ans_conc_ind):
                # Potential confusion case, as unordered label pair
                confusion_pair = frozenset([conc_ind, ans_conc_ind])

                if (conc_type, confusion_pair) not in agent.confused_no_more:
                    # Agent's best guess disagrees with the user-provided
                    # information
                    agent.vision.confusions.add((conc_type, confusion_pair))


def identify_acknowledgement(agent, rule, prev_statements, prev_context):
    """
    Check whether the agent reported its estimate of something by its utterance
    and whether `rule` effectively acknowledges any of them positively or negatively.
    We may treat "Correct.", silence or explicit repetition as positive acknowledgement.
    Any conflicting statement from user would count as negative acknowledgement, as
    well as "Incorrect." or "No.".
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if rule_is_grounded and not rule_has_pred_referent:
        # Grounded event without constant predicate referents

        if agent.vision.new_input_provided:
            # Discussion about irrelevant contexts, can return early without doing anything
            return

        for (ti, ci), (speaker, (statement, _)) in prev_statements:
            # Only interested in whether an agent's statement is acknowledged
            if speaker != "A": continue

            # Entailment checks; cons vs. statement, and cons vs. ~statement
            pos_entail_check, _ = Literal.entailing_mapping_btw(cons, statement)
            neg_entail_check, _ = Literal.entailing_mapping_btw(cons, (list(statement),))
                # Recall that a list of literals stands for the negation of the conjunction;
                # wrap it in a tuple again to make it an iterable

            pos_ack = pos_entail_check is not None and pos_entail_check >= 0
            neg_ack = neg_entail_check is not None and neg_entail_check >= 0

            if not (pos_ack or neg_ack): continue       # Nothing to do here

            # Determine polarity of the acknowledgement, collect relevant visual embeddings
            # from prev_scene, and then record
            polarity = pos_ack
            acknowledgement_data = (statement, polarity, prev_context)
            agent.lang.dialogue.acknowledged_stms[("curr", ti, ci)] = acknowledgement_data
                # "curr" indicates the acknowledgement is relevant to a statement in the
                # ongoing dialogue record


def identify_generics(agent, rule, provenance, prev_Qs, generics, pair_rules):
    """
    For symbolic knowledge base expansion. Integrate the rule into KB by adding
    (for now we won't worry about intra-KB consistency, belief revision, etc.).
    Identified generics will be added to the provided `generics` list, optionally
    along with the appropriate negative implicatures as required by the agent's
    strategy. `pair_rules` will also be updated as needed for later computation
    of implicatures.
    """
    cons, ante = rule
    rule_is_lifted = (cons is None or is_lifted(cons)) and \
        (ante is None or is_lifted(ante))
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if rule_is_lifted:
        # Lifted generic rule statement, without any grounded term arguments

        # Assume default knowledge type here
        knowledge_type = "property"

        # First add the face-value semantics of the explicitly stated rule
        generics.append((rule, U_IN_PR, provenance, knowledge_type))

        if agent.strat_generic == "semNeg" or agent.strat_generic == "semNegScal":
            # Current rule cons conjunction & ante conjunction as list
            occurring_preds = {lit.name for lit in cons+ante}

            # Collect concept_diff questions made by the agent during this dialogue
            diff_Qs_args = []
            for _, (spk, (q_vars, (q_cons, _)), _, _) in prev_Qs:
                if spk!="A": continue
                diff_Qs_args += [
                    l.args for l in q_cons
                    if l.name=="*_diff" and l.args[2][0]==q_vars[0][0]
                ]

            if len(diff_Qs_args) > 0:
                # Fetch two concepts being compared in the latest
                # concept diff question
                c1, _ = diff_Qs_args[-1][0]
                c2, _ = diff_Qs_args[-1][1]
                # Note: more principled way to manage relevant (I)QAP pair utterances
                # would be to adopt a legitimate, established formalism for representing
                # discourse structure (e.g. SDRT)

                # (Ordered) Concept pair with which implicatures will be computed
                if c1 in occurring_preds and c2 not in occurring_preds:
                    rel_conc_pair = (c1, c2)
                elif c2 in occurring_preds and c1 not in occurring_preds:
                    rel_conc_pair = (c2, c1)
                else:
                    rel_conc_pair = None
            else:
                rel_conc_pair = None

            if rel_conc_pair is not None:
                # Compute appropriate implicatures for the concept pairs found
                c1, c2 = rel_conc_pair

                # Negative implicature; replace occurrence of c1 with c2 then negate
                # cons conjunction (i.e. move cons conj to ante)
                cons_repl = tuple(
                    l.substitute(preds={ c1: c2 }) for l in cons
                )
                ante_repl = tuple(
                    l.substitute(preds={ c1: c2 }) for l in ante
                )
                negImpl = ((list(cons_repl),), ante_repl)
                knowledge_source = f"{provenance} (Neg. Impl.)"
                generics.append(
                    (negImpl, A_IM_PR, knowledge_source, knowledge_type)
                )

                # Collect explicit generics provided for the concept pair and negative
                # implicature computed from the context, with which scalar implicatures
                # will be computed
                pair_rules[frozenset(rel_conc_pair)] += [
                    rule, negImpl
                ]

    if (rule_is_grounded and ante is None and not rule_has_pred_referent):
        # Grounded fact without constant predicate referents

        # For corrective feedback "This is Y" following the agent's incorrect answer
        # to the probing question "What kind of X is this?", extract 'Y entails X'
        # (e.g., What kind of truck is this? This is a fire truck => All fire trucks
        # are trucks). More of a universal statement rather than a generic one.

        # Collect concept entailment & constraints in questions made
        # by the user during this dialogue
        entail_consts = defaultdict(list)       # Map from pred var to set of pred consts
        instance_consts = {}                    # Map from ent const to pred var
        context_Qs = {}                         # Pointers to user's original question

        # Disregard all questions except the last one from user
        relevant_Qs = [
            (q_vars, q_cons, presup, raw)
            for _, (spk, (q_vars, (q_cons, _)), presup, raw) in prev_Qs if spk=="U"
        ][-1:]

        for q_vars, q_cons, presup, raw in relevant_Qs:

            if presup is None:
                p_cons = []
            else:
                p_cons, _ = presup

            for qv, is_pred in q_vars:
                # Consider only predicate variables
                if not is_pred: continue

                for ql in q_cons:
                    # Constraint: P should entail conjunction {p1 and p2 and ...}
                    if ql.name=="*_subtype" and ql.args[0][0]==qv:
                        entail_consts[qv] += [pl.name for pl in p_cons]

                    # Constraint: x should be an instance of P
                    if ql.name=="*_isinstance" and ql.args[0][0]==qv:
                        instance_consts[ql.args[1][0]] = qv
            
                context_Qs[qv] = raw

        # Synthesize into a rule encoding the appropriate entailment
        for ent, pred_var in instance_consts.items():
            if pred_var not in entail_consts: continue
            entailed_preds = entail_consts[pred_var]

            # (Temporary) Only consider 1-place predicates, so match the first and
            # only entity from the arg list. Disregard negated conjunctions.
            entailing_preds = tuple(
                lit.name for lit in cons
                if isinstance(lit, Literal) and len(lit.args)==1 and lit.args[0][0]==ent
            )
            if len(entailing_preds) == 0: continue

            entailment_rule = (
                tuple(Literal(pred, [("X", True)]) for pred in entailed_preds),
                tuple(Literal(pred, [("X", True)]) for pred in entailing_preds)
            )
            knowledge_source = f"{context_Qs[pred_var]} => {provenance}"
            knowledge_type = "taxonomy"
            generics.append(
                (entailment_rule, U_IN_PR, knowledge_source, knowledge_type)
            )


def handle_mismatch(agent, mismatch):
    """
    Handle cognition gap following some specified strategy. Note that we now
    assume the user (teacher) is an infallible oracle, and the agent doesn't
    question info provided from user.
    """
    rule, _, handled = mismatch

    if handled: return 

    objs_to_add = set(); pointers = defaultdict(set)
    for cons, ante in flatten_cons_ante(*rule):
        is_grounded = all(not is_var for l in cons+ante for _, is_var in l.args)

        if is_grounded and len(cons+ante)==1:
            if len(cons) == 1 and len(ante) == 0:
                # Positive grounded fact
                atom = cons[0]
                pol = "pos"
            else:
                # Negative grounded fact
                atom = ante[0]
                pol = "neg"

            conc_type, conc_ind = atom.name.split("_")
            conc_ind = int(conc_ind)
            args = [a for a, _ in atom.args]

            if conc_type == "cls" or conc_type == "att":
                if agent.vision.scene[args[0]]["exemplar_ind"] is None:
                    # New exemplar, mask & vector of the object should be added
                    objs_to_add.add(args[0])
                    pointers[(conc_type, conc_ind, pol)].add(args[0])
                else:
                    # Exemplar present in storage, only add pointer
                    ex_ind = agent.vision.scene[args[0]]["exemplar_ind"]
                    pointers[(conc_type, conc_ind, pol)].add(ex_ind)
            else:
                assert conc_type == "rel"
                raise NotImplementedError   # Step back for relation prediction...

    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    _add_scene_and_exemplars(
        objs_to_add, pointers,
        agent.vision.scene, agent.vision.latest_input, agent.lt_mem.exemplars
    )

    # Mark as handled
    mismatch[2] = True


def handle_confusion(agent, confusion):
    """
    Handle 'concept overlap' between two similar visual concepts. Two (fine-grained)
    concepts can be disambiguated by some symbolically represented generic rules,
    request such differences by generating an appropriate question. 
    """
    # This confusion is about to be handled
    agent.vision.confusions.remove(confusion)

    if agent.cfg.exp.strat_feedback.startswith("maxHelpExpl"):
        # When interacting with teachers with this strategy, generic KB rules are
        # elicited not by the difference questions
        return

    # New dialogue turn & clause index for the question to be asked
    ti_new = len(agent.lang.dialogue.record)
    ci_new = 0

    conc_type, conc_inds = confusion
    conc_inds = list(conc_inds)

    # For now we are only interested in disambiguating class (noun) concepts
    assert conc_type == "cls"

    # Prepare logical form of the concept-diff question to ask
    q_vars = ((f"X2t{ti_new}c{ci_new}", False),)
    q_rules = (
        (("diff", "*", tuple(f"{ri}t{ti_new}c{ci_new}" for ri in ["x0", "x1", "X2"]), False),),
        ()
    )
    ques_logical_form = (q_vars, q_rules)

    # Prepare surface form of the concept-diff question to ask
    pluralize = inflect.engine().plural
    conc_names = [
        agent.lt_mem.lexicon.d2s[(ci, conc_type)][0][0]
        for ci in conc_inds
    ]       # Fetch string name for concepts from the lexicon
    conc_names = [
        re.findall(r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", cn)
        for cn in conc_names
    ]       # Unpack camelCased names
    conc_names = [
        pluralize(" ".join(tok.lower() for tok in cn))
        for cn in conc_names
    ]       # Lowercase tokens and pluralize

    # Update cognitive state w.r.t. value assignment and word sense
    agent.symbolic.value_assignment.update({
        f"x0t{ti_new}c{ci_new}": f"{conc_type}_{conc_inds[0]}",
        f"x1t{ti_new}c{ci_new}": f"{conc_type}_{conc_inds[1]}"
    })

    ques_translated = f"How are {conc_names[0]} and {conc_names[1]} different?"

    agent.lang.dialogue.to_generate.append(
        ((None, ques_logical_form), ques_translated, {})
    )

    # No need to request concept differences again for this particular case
    # for the rest of the interaction episode sequence
    agent.confused_no_more.add(confusion)


def handle_acknowledgement(agent, acknowledgement_info):
    """
    Handle (positive) acknowledgements to an utterance by the agent reporting its
    estimation on some state of affairs. The learning process differs based on the
    agent's strategy regarding how to extract learning signals from user's assents.
    """
    if agent.strat_assent == "doNotLearn": return       # Nothing further to do here

    ack_ind, ack_data = acknowledgement_info
    if ack_data is None: return         # Marked 'already processed'

    statement, polarity, context = ack_data
    vis_scene, pr_prog, kb_snap = context
    if ack_ind[0] == "prev":
        vis_raw = agent.vision.previous_input
    else:
        vis_raw = agent.vision.latest_input

    if polarity != True:
        # Nothing to do with negative acknowledgements
        agent.lang.dialogue.acknowledged_stms[ack_ind] = None
        return

    # Vision-only (neural) estimations recognized
    lits_from_vis = {
        rule.body[0].as_atom(): r_pr[0] for rule, r_pr, _ in pr_prog.rules
        if len(rule.head)==0 and len(rule.body)==2 and rule.body[0].naf
    }

    # Find literals to be considered as confirmed via user's assent; always include
    # each literal in `statement`, and others if relevant via some rule in KB
    lits_to_learn = set()
    for main_lit in statement:
        if main_lit.name.startswith("*_"): continue

        # Find relevant KB entries (except taxonomy entries) containing the main literal
        relevant_kb_entries = [
            kb_snap.entries[ei][0]
            for ei in kb_snap.entries_by_pred[main_lit.name]
            if kb_snap.entries[ei][3] != "taxonomy"
        ]

        # Into groups of literals to be recognized as learning signals
        literal_groups = []
        for cons, ante in relevant_kb_entries:
            # Only consider rules with antecedents that don't have negated conjunction
            # as conjunct, when learning from assents
            if not all(isinstance(cnjt, Literal) for cnjt in ante): continue

            # Only consider unnegated conjuncts in consequent
            cons_pos = tuple(cnjt for cnjt in cons if isinstance(cnjt, Literal))

            for conjunct in ante:
                assert isinstance(conjunct, Literal)
                if conjunct.name==main_lit.name:
                    vc_map = {v: c for v, c in zip(conjunct.args, main_lit.args)}
                    literal_groups.append((cons_pos, ante, vc_map))

        for cons, ante, vc_map in literal_groups:
            # Substitute and identify the set of unique args occurring
            cons_subs = [l.substitute(terms=vc_map) for l in cons]
            ante_subs = [l.substitute(terms=vc_map) for l in ante]
            occurring_args = set.union(*[set(l.args) for l in cons_subs+ante_subs])

            # Consider all assignments of constants to variables & skolem functions
            # possible, and add viable cases to the set of acknowledged literals
            possible_assignments = [
                tuple(zip(occurring_args, prm))
                for prm in permutations(vis_scene, len(occurring_args))
            ]
            for prm in possible_assignments:
                # Skip this assignment if constant mismatch occurs
                const_mismatch = any(
                    arg_name!=cons and not (is_var or isinstance(arg_name, tuple))
                    for (arg_name, is_var), cons in prm
                )
                if const_mismatch: continue

                assig = {arg: (cons, False) for arg, cons in prm if arg[0]!=cons}
                cons_subs_full = {l.substitute(terms=assig) for l in cons_subs}
                ante_subs_full = {l.substitute(terms=assig) for l in ante_subs}

                # Union the fully substituted consequent to `lits_to_learn` only if
                # all the literals in the substituted antecedent can be found in visual
                # scene; apply some value threshold according to the strategy choice.
                if all(lit in lits_from_vis for lit in ante_subs_full):
                    for lit in cons_subs_full | ante_subs_full:
                        easy_positive = lit in lits_from_vis and lits_from_vis[lit] > 0.75
                        if agent.strat_assent == "threshold" and easy_positive:
                            # Adopting thresholding strategy, and estimated probability
                            # is already high enough; opt out of adding this as exemplar
                            continue

                        lits_to_learn.add(lit)

    # Add the instances represented by the literals as concept exemplars
    objs_to_add = set(); pointers = defaultdict(set)
    for lit in lits_to_learn:
        conc_type, conc_ind = lit.name.split("_")
        conc_ind = int(conc_ind)
        if conc_type == "rel": continue         # Relations are not neurally predicted

        pol = "pos" if not lit.naf else "neg"

        if vis_scene[lit.args[0][0]]["exemplar_ind"] is None:
            # New exemplar, mask & vector of the object should be added
            objs_to_add.add(lit.args[0][0])
            pointers[(conc_type, conc_ind, pol)].add(lit.args[0][0])
        else:
            # Exemplar present in storage, only add pointer
            ex_ind = agent.vision.scene[lit.args[0][0]]["exemplar_ind"]
            pointers[(conc_type, conc_ind, pol)].add(ex_ind)

    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    _add_scene_and_exemplars(
        objs_to_add, pointers,
        vis_scene, vis_raw, agent.lt_mem.exemplars
    )

    # Replace value with None to mark as 'already processed'
    agent.lang.dialogue.acknowledged_stms[ack_ind] = None


def add_scalar_implicature(agent, pair_rules):
    """
    For concepts c1 vs. c2 recorded in pair_rules, infer implicit concept similarities
    by copying properties for c1/c2 and replacing the predicates with c2/c1, unless
    the properties are denied by rules of "higher precedence level" (i.e. explicit
    statement or negative implicature by coherence requirement)
    """
    for (c1, c2), rules in pair_rules.items():
        scal_impls = []
        scal_impls += _compute_scalar_implicature(c1, c2, rules, agent.kb_snap)
        scal_impls += _compute_scalar_implicature(c2, c1, rules, agent.kb_snap)

        for cons, ante, knowledge_type in scal_impls:
            knowledge_source = f"[{c1} ~= {c2}] (Scal. Impl.)"
            agent.lt_mem.kb.add(
                (cons, ante), A_IM_PR, knowledge_source, knowledge_type
            )

    # Regular inspection of KB by weeding out defeasible rules inferred
    # from scalar implicatures, by comparison against episodic memory
    entries_from_scalImpl = [
        (ent_id, rule)
        for ent_id, (rule, _, provenances, _) in enumerate(agent.lt_mem.kb.entries)
        if all(prov[0].endswith("(Scal. Impl.)") for prov in provenances)
    ]
    entries_to_remove = {}
    for ent_id, rule in entries_from_scalImpl:
        # Mini-KB consisting of only this rule to be tested
        kb_type = type(agent.lt_mem.kb)
        mini_kb = kb_type()
        mini_kb.add(rule, 0.5, "For inspection")

        # Test for any deductive violations of the rule with cases in
        # the episodic memory
        mini_kb_prog, _ = mini_kb.export_reasoning_program()
        inspection_outputs = [
            (pr_prog+dl_prog+mini_kb_prog).compile()
            for pr_prog, dl_prog in agent.episodic_memory
        ]
        deduc_viol_cases = [
            reg_gr for reg_gr in inspection_outputs
            if any(atm.name=="deduc_viol_0" for atm in reg_gr.graph["atoms_map"])
        ]
        deduc_viol_probs = [
            {
                node: reg_gr.nodes[frozenset({node})]["beliefs"]
                for atm, node in reg_gr.graph["atoms_map"].items()
                if atm.name=="deduc_viol_0"
            }
            for reg_gr in deduc_viol_cases
        ]
        deduc_viol_probs = [
            [
                (
                    potentials[frozenset({node})],
                    sum(potentials.values(), Polynomial(float_val=0.0))
                )
                for node, potentials in per_rgr.items()
            ]
            for per_rgr in deduc_viol_probs
        ]
        deduc_viol_probs = [
            sum((unnorm / Z).at_limit() for unnorm, Z in per_rgr) / len(per_rgr)
            for per_rgr in deduc_viol_probs
        ]

        # Retract the defeasible inference if refuted by memory of
        # some episode with sufficiently high probability
        if len(deduc_viol_probs) > 0 and max(deduc_viol_probs) > 0.2:
            entries_to_remove[ent_id] = max(deduc_viol_probs)

    if len(entries_to_remove) > 0:
        # Remove the listed entries from KB
        agent.lt_mem.kb.remove_by_ids(list(entries_to_remove))


def handle_neologism(agent, novel_concepts, dialogue_state):
    """
    Identify neologisms (that the agent doesn't know which concepts they refer to)
    to be handled, attempt resolving from information available so far if possible,
    or record as unresolved neologisms for later addressing otherwise
    """
    # Return value, boolean flag whether there has been any change to the
    # exemplar base
    xb_updated = False

    neologisms = {
        tok: sym for tok, (sym, den) in agent.symbolic.word_senses.items()
        if den is None
    }

    if len(neologisms) == 0: return False       # Nothing to do

    objs_to_add = set(); pointers = defaultdict(set)
    for tok, sym in neologisms.items():
        neo_in_rule_cons = tok[2] == "rc"
        neos_in_same_rule_ante = [
            n for n in neologisms if tok[:3]==n[:3] and n[3].startswith("a")
        ]
        if neo_in_rule_cons and len(neos_in_same_rule_ante)==0:
            # Occurrence in rule cons implies either definition or exemplar is
            # provided by the utterance containing this token... Register new
            # visual concept, and perform few-shot learning if appropriate
            pos, name = sym
            if pos == "n":
                conc_type = "cls"
            elif pos == "a":
                conc_type = "att"
            else:
                assert pos == "v" or pos == "r"
                conc_type = "rel"

            # Expand corresponding visual concept inventory
            conc_ind = agent.vision.add_concept(conc_type)
            novel_concept = (conc_ind, conc_type)
            novel_concepts.add(novel_concept)

            # Acquire novel concept by updating lexicon
            agent.lt_mem.lexicon.add((name, pos), novel_concept)

            ti = int(tok[0].strip("t"))
            ci = int(tok[1].strip("c"))
            rule_cons, rule_ante = dialogue_state.record[ti][1][ci][0][0]

            if len(rule_ante) == 0:
                # Labelled exemplar provided; add new concept exemplars to
                # memory, as feature vectors at the penultimate layer right
                # before category prediction heads
                args = [
                    agent.symbolic.value_assignment[arg] for arg in rule_cons[0][2]
                ]

                if conc_type == "cls" or conc_type == "att":
                    if agent.vision.scene[args[0]]["exemplar_ind"] is None:
                        # New exemplar, mask & vector of the object should be added
                        objs_to_add.add(args[0])
                        pointers[(conc_type, conc_ind, "pos")].add(args[0])
                    else:
                        # Exemplar present in storage, only add pointer
                        ex_ind = agent.vision.scene[args[0]]["exemplar_ind"]
                        pointers[(conc_type, conc_ind, "pos")].add(ex_ind)
                else:
                    assert conc_type == "rel"
                    raise NotImplementedError   # Step back for relation prediction...

                # Register this instance as a handled mismatch, so that add_exs() won't
                # be called upon this one during this loop again by handle_mismatch()
                stm = ((Literal(f"{conc_type}_{conc_ind}", wrap_args(*args)),), None)
                agent.symbolic.mismatches.append([stm, None, True])

                # Set flag that XB is updated
                xb_updated |= True
        else:
            # Otherwise not immediately resolvable
            agent.lang.unresolved_neologisms.add((sym, tok))

    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    _add_scene_and_exemplars(
        objs_to_add, pointers,
        agent.vision.scene, agent.vision.latest_input, agent.lt_mem.exemplars
    )

    return xb_updated


def _add_scene_and_exemplars(
        objs_to_add, pointers, current_scene, current_raw_img, ex_mem
    ):
    """
    Helper method factored out for adding a scene, objects and/or concept exemplar
    pointers
    """
    # Check if this scene is already stored in memory and if so fetch the ID
    scene_ids = [
        obj_info["exemplar_ind"][0] for obj_info in current_scene.values()
        if obj_info["exemplar_ind"] is not None
    ]
    assert len(set(scene_ids)) <= 1         # All same or none
    scene_id = scene_ids[0] if len(scene_ids) > 0 else None

    # Add concept exemplars to memory
    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    scene_img = current_raw_img if scene_id is None else None
        # Need to pass the scene image if not already stored in memory
    exemplars = [
        {
            "scene_id": scene_id,
            "mask": current_scene[oi]["pred_mask"],
            "f_vec": current_scene[oi]["vis_emb"]
        }
        for oi in objs_to_add
    ]
    pointers = {
        conc_spec: {
            # Pair (whether object is newly added, index within objs_to_add if True,
            # (scene_id, obj_id) otherwise)
            (True, objs_to_add.index(oi)) if isinstance(oi, str) else (False, oi)
            for oi in objs
        }
        for conc_spec, objs in pointers.items()
    }
    added_inds = ex_mem.add_exs(
        scene_img=scene_img, exemplars=exemplars, pointers=pointers
    )

    for oi, xi in zip(objs_to_add, added_inds):
        # Record exemplar storage index for corresponding object in visual scene,
        # so that any potential redundant additions can be avoided
        current_scene[oi]["exemplar_ind"] = xi


def _compute_scalar_implicature(c1, c2, rules, kb_snap):
    """ Helper method factored out for symmetric applications """
    # Return value; list of inferred rules to add
    scal_impls = []

    # Recursive helper method for substituting predicates while preserving structure
    _substitute = lambda cnjt, ps: cnjt.substitute(preds=ps) \
        if isinstance(cnjt, Literal) else [_substitute(nc, ps) for nc in cnjt]

    # Existing properties of c1
    for i in kb_snap.entries_by_pred[c1]:
        # Fetch KB entry
        (cons, ante), _, _, knowledge_type = kb_snap.entries[i]

        # Replace occurrences of c1 with c2
        cons = tuple(_substitute(c, { c1: c2 }) for c in cons)
        ante = tuple(_substitute(a, { c1: c2 }) for a in ante)

        # Negation of the replaced copy
        if all(isinstance(c, Literal) for c in cons):
            # Positive conjunction cons
            cons_neg = (list(cons),)
        elif all(isinstance(c, list) for c in cons) and len(cons)==1:
            # Negated conjunction cons
            cons_neg = tuple(cons[0])
        else:
            # Cannot handle cases with rule cons that is mixture
            # of positive literals and negated conjunctions
            raise NotImplementedError
        
        # Test the negated copy against rules with higher precedence
        # (explicitly stated generics and their negative implicature
        # counterparts)
        defeated = False
        for r in rules:
            # Single-step entailment test; if the negation of inferred
            # rule is entailed by r, consider the scalar implicature
            # defeated
            r_cons, r_ante = r

            ent_dir_b, mapping_b = Literal.entailing_mapping_btw(
                r_ante, ante
            )
            if mapping_b is not None:
                ent_dir_h, mapping_h = Literal.entailing_mapping_btw(
                    r_cons, cons_neg, mapping_b
                )
                if mapping_h is not None and {ent_dir_h, ent_dir_b} != {1, -1}:
                    # Entailment relation detected
                    if ent_dir_h >= 0 and ent_dir_b <= 0:
                        defeated = True
                        break

        if not defeated:
            # Add the inferred generic that successfully survived
            # the test against the higher-precedence rules
            scal_impls.append((cons, ante, knowledge_type))
    
    return scal_impls
