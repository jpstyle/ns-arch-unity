"""
Implements learning-related composite actions; by learning, we are primarily
referring to belief updates that modify long-term memory
"""
import re
import math
from collections import defaultdict

import inflect
import numpy as np

from ..vision.utils import masks_bounding_boxes
from ..lpmln import Literal, Polynomial
from ..lpmln.utils import flatten_cons_ante


EPS = 1e-10                  # Value used for numerical stabilization
SR_THRES = 0.8               # Mismatch surprisal threshold
U_IN_PR = 1.00               # How much the agent values information provided by the user
A_IM_PR = 1.00               # How much the agent values inferred implicature

# Recursive helper methods for checking whether rule cons/ante is grounded (variable-
# free), lifted (all variables), or contains any predicate referent as argument
is_grounded = lambda cnjt: all(not is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_grounded(nc) for nc in cnjt)
is_lifted = lambda cnjt: all(is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_lifted(nc) for nc in cnjt)
has_pred_referent = \
    lambda cnjt: any(isinstance(a,str) and a[0].lower()=="p" for a, _ in cnjt.args) \
        if isinstance(cnjt, Literal) else any(has_pred_referent(nc) for nc in cnjt)

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

    if (rule_is_grounded and not rule_has_pred_referent and 
        agent.symbolic.concl_vis is not None):
        # Grounded event without constant predicate referents, only if vision-only
        # sensemaking result has been obtained

        # Make a yes/no query to obtain the likelihood of content
        bjt_v, _ = agent.symbolic.concl_vis
        q_response = agent.symbolic.query(bjt_v, None, rule)
        ev_prob = q_response[()]

        surprisal = -math.log(ev_prob + EPS)
        if surprisal >= -math.log(SR_THRES):
            m = (rule, surprisal)
            if m not in agent.symbolic.mismatches:
                agent.symbolic.mismatches.append(m)

def identify_confusion(agent, rule, prev_facts, novel_concepts):
    """
    Test against vision module output to identify  any 'concept overlap' -- i.e.
    whenever the agent confuses two concepts difficult to distinguish visually
    and mistakes one for another.
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if (rule_is_grounded and ante is None and not rule_has_pred_referent and
        agent.cfg.exp1.strat_feedback == "maxHelp"):
        # Grounded fact without constant predicate referents, only if the user
        # adopts maxHelp strategy and provides generic NL feedback

        # Fetch agent's last answer; this assumes the last factual statement
        # by agent is provided as answer to the last question from user. This
        # might change in the future when we adopt a more sophisticated formalism
        # for representing discourse to relax the assumption.
        prev_facts_A = [fct for spk, fct in prev_facts if spk=="A"]
        if len(prev_facts_A) == 0:
            # Hasn't given an answer (i.e., "I am not sure.")
            return

        agent_last_ans = prev_facts_A[-1][0][0]
        ans_conc_type, ans_conc_ind = agent_last_ans.name.split("_")
        ans_conc_ind = int(ans_conc_ind)

        for lit in cons:
            # Disregard negated conjunctions
            if not isinstance(lit, Literal): continue

            # (Temporary) Only consider 1-place predicates, so retrieve
            # the single and first entity from the arg list
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

        # First add the face-value semantics of the explicitly stated rule
        generics.append((rule, U_IN_PR, provenance))

        if agent.strat_generic == "semNeg" or agent.strat_generic == "semNegScal":
            # Current rule cons conjunction & ante conjunction as list
            occurring_preds = {lit.name for lit in cons+ante}

            # Collect concept_diff questions made by the agent during this dialogue
            diff_Qs_args = []
            for (spk, (q_vars, (q_cons, _)), _, raw) in prev_Qs:
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
                generics.append(
                    (negImpl, A_IM_PR, f"{provenance} (Neg. Impl.)")
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
        for (spk, (q_vars, (q_cons, _)), presup, raw) in prev_Qs:
            # Consider only questions from user
            if spk!="U": continue

            if presup is None:
                p_cons = []
            else:
                p_cons, _ = presup

            for qv, is_pred in q_vars:
                # Consider only predicate variables
                if not is_pred: continue

                for ql in q_cons:
                    # Constraint: P should entail conjunction {p1 and p2 and ...}
                    if ql.name=="*_entail" and ql.args[0][0]==qv:
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
                [Literal(pred, [("X", True)]) for pred in entailed_preds],
                [Literal(pred, [("X", True)]) for pred in entailing_preds]
            )
            generics.append(
                (entailment_rule, U_IN_PR, f"{context_Qs[pred_var]} => {provenance}")
            )

def handle_mismatch(agent, mismatch):
    """
    Handle cognition gap following some specified strategy. Note that we now
    assume the user (teacher) is an infallible oracle, and the agent doesn't
    question info provided from user.
    """
    # This mismatch is about to be handled, remove
    agent.symbolic.mismatches.remove(mismatch)

    rule, _ = mismatch
    for cons, ante in flatten_cons_ante(*rule):
        is_grounded = all(not is_var for l in cons+ante for _, is_var in l.args)

        if is_grounded and len(cons+ante)==1:
            if len(cons) == 1 and len(ante) == 0:
                # Positive grounded fact
                atom = cons[0]
                exm_pointer = ({0}, set())
            else:
                # Negative grounded fact
                atom = ante[0]
                exm_pointer = (set(), {0})

            conc_type, conc_ind = atom.name.split("_")
            conc_ind = int(conc_ind)
            args = [a for a, _ in atom.args]

            ex_bboxes = masks_bounding_boxes(
                [agent.lang.dialogue.referents["env"][arg]["mask"] for arg in args]
            )

            # Fetch current score for the asserted fact
            if conc_type == "cls" or conc_type == "att":
                f_vec = agent.vision.f_vecs[args[0]]
            else:
                assert conc_type == "rel"
                raise NotImplementedError   # Step back for relation prediction...

            # Add new concept exemplars to memory, as feature vectors at the
            # penultimate layer right before category prediction heads
            pointers_src = { 0: (0, tuple(ai for ai in range(len(args)))) }
            pointers_exm = { conc_ind: exm_pointer }

            agent.lt_mem.exemplars.add_exs(
                sources=[(np.asarray(agent.vision.last_input), ex_bboxes)],
                f_vecs={ conc_type: f_vec[None,:] },
                pointers_src={ conc_type: pointers_src },
                pointers_exm={ conc_type: pointers_exm }
            )

def handle_confusion(agent, confusion):
    """
    Handle 'concept overlap' between two similar visual concepts. Two (fine-grained)
    concepts can be disambiguated by some symbolically represented generic rules,
    request such differences by generating an appropriate question. 
    """
    # This confusion is about to be handled
    agent.vision.confusions.remove(confusion)

    # New dialogue turn & clause index for the question to be asked
    ti_new = len(agent.lang.dialogue.record)
    si_new = 0

    conc_type, conc_inds = confusion
    conc_inds = list(conc_inds)

    # For now we are only interested in disambiguating class (noun) concepts
    assert conc_type == "cls"

    # Prepare logical form of the concept-diff question to ask
    q_vars = ((f"X2t{ti_new}s{si_new}", False),)
    q_rules = (
        (("diff", "*", tuple(f"{ri}t{ti_new}s{si_new}" for ri in ["x0", "x1", "X2"]), False),),
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
        f"x0t{ti_new}s{si_new}": f"{conc_type}_{conc_inds[0]}",
        f"x1t{ti_new}s{si_new}": f"{conc_type}_{conc_inds[1]}"
    })

    ques_translated = f"How are {conc_names[0]} and {conc_names[1]} different?"

    agent.lang.dialogue.to_generate.append(
        ((None, ques_logical_form), ques_translated, {})
    )

    # No need to request concept differences again for this particular case
    # for the rest of the interaction episode sequence
    agent.confused_no_more.add(confusion)

def add_scalar_implicatures(agent, pair_rules):
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

        for cons, ante in scal_impls:
            agent.lt_mem.kb.add(
                (cons, ante), A_IM_PR, f"[{c1} ~= {c2}] (Scal. Impl.)"
            )

    # Regular inspection of KB by weeding out defeasible rules inferred
    # from scalar implicatures, by comparison against episodic memory
    entries_from_scalImpl = [
        (ent_id, rule)
        for ent_id, (rule, _, provenances) in enumerate(agent.lt_mem.kb.entries)
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
            (pprog+dprog+mini_kb_prog).compile()
            for pprog, dprog in agent.episodic_memory
        ]
        deduc_viol_cases = [
            bjt for bjt in inspection_outputs
            if any(atm.name=="deduc_viol_0" for atm in bjt.graph["atoms_map"])
        ]
        deduc_viol_probs = [
            {
                node: bjt.nodes[frozenset({node})]["output_beliefs"]
                for atm, node in bjt.graph["atoms_map"].items()
                if atm.name=="deduc_viol_0"
            }
            for bjt in deduc_viol_cases
        ]
        deduc_viol_probs = [
            [
                (
                    potentials[frozenset({node})],
                    sum(potentials.values(), Polynomial(float_val=0.0))
                )
                for node, potentials in per_bjt.items()
            ]
            for per_bjt in deduc_viol_probs
        ]
        deduc_viol_probs = [
            sum((unnorm / Z).at_limit() for unnorm, Z in per_bjt) / len(per_bjt)
            for per_bjt in deduc_viol_probs
        ]

        # Retract the defeasible inference if refuted by memory of
        # some episode with sufficiently high probability
        if len(deduc_viol_probs) > 0 and max(deduc_viol_probs) > 0.2:
            entries_to_remove[ent_id] = max(deduc_viol_probs)

    if len(entries_to_remove) > 0:
        # Remove the listed entries from KB
        agent.lt_mem.kb.remove_by_ids(list(entries_to_remove))

# Helper method factored out for symmetric applications
def _compute_scalar_implicature(c1, c2, rules, kb_snap):
    # Return value; list of inferred rules to add
    scal_impls = []

    # Recursive helper method for substituting predicates while preserving structure
    _substitute = lambda cnjt, ps: cnjt.substitute(preds=ps) \
        if isinstance(cnjt, Literal) else [_substitute(nc, ps) for nc in cnjt]

    # Existing properties of c1
    for i in kb_snap.entries_by_pred[c1]:
        # Fetch KB entry
        (cons, ante), *_ = kb_snap.entries[i]

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

            mapping_b, ent_dir_b = Literal.entailing_mapping_btw(
                r_ante, ante
            )
            if mapping_b is not None:
                mapping_h, ent_dir_h = Literal.entailing_mapping_btw(
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
            scal_impls.append((cons, ante))
    
    return scal_impls

def handle_neologisms(agent, novel_concepts, dialogue_state):
    """
    Identify neologisms (that the agent doesn't know which concepts they refer to)
    to be handled, attempt resolving from information available so far if possible,
    or record as unresolved neologisms for later addressing otherwise
    """
    xb_updated = False

    neologisms = {
        tok: sym for tok, (sym, den) in agent.symbolic.word_senses.items()
        if den is None
    }
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
            si = int(tok[1].strip("s"))
            rule_cons, rule_ante = dialogue_state["record"][ti][1][si][0][0]

            if len(rule_ante) == 0:
                # Labelled exemplar provided; add new concept exemplars to
                # memory, as feature vectors at the penultimate layer right
                # before category prediction heads
                args = [
                    agent.symbolic.value_assignment[arg] for arg in rule_cons[0][2]
                ]
                ex_bboxes = masks_bounding_boxes(
                    [dialogue_state["referents"]["env"][arg]["mask"] for arg in args]
                )

                if conc_type == "cls" or conc_type == "att":
                    f_vec = agent.vision.f_vecs[args[0]]
                else:
                    assert conc_type == "rel"
                    raise NotImplementedError   # Step back for relation prediction...
                
                pointers_src = { 0: (0, tuple(ai for ai in range(len(args)))) }
                pointers_exm = { conc_ind: ({0}, set()) }

                agent.lt_mem.exemplars.add_exs(
                    sources=[(np.asarray(agent.vision.last_input), ex_bboxes)],
                    f_vecs={ conc_type: f_vec[None,:] },
                    pointers_src={ conc_type: pointers_src },
                    pointers_exm={ conc_type: pointers_exm }
                )

                # Set flag that XB is updated
                xb_updated = True
        else:
            # Otherwise not immediately resolvable
            agent.lang.unresolved_neologisms.add((sym, tok))

    return xb_updated
