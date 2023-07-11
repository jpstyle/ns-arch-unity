"""
Implements dialogue-related composite actions
"""
import re
import random
from functools import reduce
from collections import defaultdict

import numpy as np

from ..lpmln import Literal
from ..lpmln.utils import flatten_cons_ante


def attempt_answer_Q(agent, utt_pointer):
    """
    Attempt to answer an unanswered question from user.
    
    If it turns out the question cannot be answered at all with the agent's
    current knowledge (e.g. question contains unresolved neologism), do nothing
    and wait for it to become answerable.

    If the agent can come up with an answer to the question, right or wrong,
    schedule to actually answer it by adding a new agenda item.
    """
    dialogue_state = agent.lang.dialogue.export_as_dict()
    translated = agent.symbolic.translate_dialogue_content(dialogue_state)

    ti, si = utt_pointer
    (_, question), _ = translated[ti][1][si]

    if question is None:
        # Question cannot be answered for some reason
        return
    else:
        # Schedule to answer the question
        agent.practical.agenda.append(("answer_Q", utt_pointer))
        return

def prepare_answer_Q(agent, utt_pointer):
    """
    Prepare an answer to a question that has been deemed answerable, by first
    computing raw ingredients from which answer candidates can be composed,
    picking out an answer among the candidates, then translating the answer
    into natural language form to be uttered
    """
    # The question is about to be answered
    agent.lang.dialogue.unanswered_Q.remove(utt_pointer)

    dialogue_state = agent.lang.dialogue.export_as_dict()
    translated = agent.symbolic.translate_dialogue_content(dialogue_state)

    ti, si = utt_pointer
    (presup, question), _ = translated[ti][1][si]
    assert question is not None

    q_vars, (cons, ante) = question
    bjt_v, _ = agent.symbolic.concl_vis
    # bjt_vl, _ = symbolic.concl_vis_lang

    # # New dialogue turn & clause index for the answer to be provided
    ti_new = len(agent.lang.dialogue.record)
    si_new = 0

    # Mapping from predicate variables to their associated entities
    pred_var_to_ent_ref = {
        ql.args[0][0]: ql.args[1][0] for ql in cons
        if ql.name == "*_isinstance"
    }

    qv_to_dis_ref = {
        qv: f"x{ri}t{ti_new}s{si_new}" for ri, (qv, _) in enumerate(q_vars)
    }
    conc_type_to_pos = { "cls": "n" }

    # Process any 'concept conjunctions' provided in the presupposition into a more
    # legible format, for easier processing right after
    if presup is None:
        restrictors = {}
    else:
        conc_conjs = defaultdict(set)
        for lit in presup[0]:
            conc_conjs[lit.args[0][0]].add(lit.name)
        # for lit in presup[0]:
        #     conc_conjs[lit.args[0][0]].add(tuple(lit.name.split("_")))
        # conc_conjs = {
        #     pred_ref: {(int(conc_ind), conc_type) for (conc_type, conc_ind) in concepts}
        #     for pred_ref, concepts in conc_conjs.items()
        # }

        # Extract any '*_entail' statements and cast into appropriate query restrictors
        restrictors = {
            lit.args[0][0]: agent.lt_mem.kb.find_entailer_concepts(conc_conjs[lit.args[1][0]])
            for lit in cons if lit.name=="*_entail"
        }
        # Remove the '*_entail' statements from cons now that they are processed
        cons = tuple(lit for lit in cons if lit.name!="*_entail")
        question = (q_vars, (cons, ante))

    # Ensure it has every ingredient available needed for making most informed judgements
    # on computing the best answer to the question. Specifically, scene graph outputs from
    # vision module may be omitting some entities, whose presence and properties may have
    # critical influence on the symbolic sensemaking process. Make sure such entities, if
    # actually present, are captured in scene graphs by performing visual search as needed.
    if len(agent.lt_mem.kb.entries) > 0:
        search_specs = _search_specs_from_kb(agent, question, bjt_v, restrictors)
        if len(search_specs) > 0:
            agent.vision.predict(
                None, agent.lt_mem.exemplars,
                specs=search_specs, visualize=False, lexicon=agent.lt_mem.lexicon
            )

            #  ... and another round of sensemaking
            exported_kb = agent.lt_mem.kb.export_reasoning_program()
            agent.symbolic.sensemake_vis(agent.vision.scene, exported_kb)
            agent.symbolic.resolve_symbol_semantics(dialogue_state, agent.lt_mem.lexicon)
            # symbolic.sensemake_vis_lang(dialogue_state)

            bjt_v, _ = agent.symbolic.concl_vis
            # bjt_vl, _ = symbolic.concl_vis_lang

    # Compute raw answer candidates by appropriately querying compiled BJT
    answers_raw = agent.symbolic.query(bjt_v, q_vars, (cons, ante), restrictors)

    # Pick out an answer to deliver; maximum confidence
    if len(answers_raw) > 0:
        max_score = max(answers_raw.values())
        answer_selected = random.choice([
            a for (a, s) in answers_raw.items() if s == max_score
        ])
        ev_prob = answers_raw[answer_selected]
    else:
        answer_selected = (None,) * len(q_vars)
        ev_prob = None

    # From the selected answer, prepare ASP-friendly logical form of the response to
    # generate, then translate into natural language
    # (Parse the original question utterance, manipulate, then generate back)
    if len(answer_selected) == 0:
        # Yes/no question
        raise NotImplementedError
        if ev_prob < SC_THRES:
            # Positive answer
            ...
        else:
            # Negative answer
            ...
    else:
        # Wh- question
        for (qv, is_pred), ans in zip(q_vars, answer_selected):
            # Referent index in the new answer utterance
            ri = qv_to_dis_ref[qv]

            # Value to replace the designated wh-quantified referent with
            if is_pred:
                # Predicate name; fetch from lexicon
                if ans is None:
                    # No answer predicate to "What is X" question; let's simply generate
                    # "I am not sure" as answer for these cases
                    agent.lang.dialogue.to_generate.append(
                        # Will just pass None as "logical form" for this...
                        (None, "I am not sure.", {})
                    )
                    return
                else:
                    ans = ans.split("_")
                    ans = (int(ans[1]), ans[0])

                    nl_val = agent.lt_mem.lexicon.d2s[ans][0][0]

                    # Update cognitive state w.r.t. value assignment and word sense
                    agent.symbolic.value_assignment[ri] = \
                        pred_var_to_ent_ref[qv]
                    tok_ind = (f"t{ti_new}", f"s{si_new}", "rc", "0")
                    agent.symbolic.word_senses[tok_ind] = \
                        ((conc_type_to_pos[ans[1]], nl_val), f"{ans[1]}_{ans[0]}")

                    answer_logical_form = (
                        ((nl_val, conc_type_to_pos[ans[1]], (ri,), False),), ()
                    )
            else:
                # Entity by their constant name handle
                nl_val = ans

                # TODO?: Logical form for this case
                raise NotImplementedError

        # Split camelCased predicate name
        splits = re.findall(r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", nl_val)
        splits = [w[0].lower()+w[1:] for w in splits]
        answer_translated = f"This is a {' '.join(splits)}."

    # Fetch segmentation mask for the demonstratively referenced entity
    dem_mask = dialogue_state["referents"]["env"][pred_var_to_ent_ref[qv]]["mask"]

    # Push the translated answer to buffer of utterances to generate
    agent.lang.dialogue.to_generate.append((
        (answer_logical_form, None), answer_translated, { (0, 4): dem_mask }
    ))

def _search_specs_from_kb(agent, question, ref_bjt, restrictors):
    """
    Factored helper method for extracting specifications for visual search,
    based on the agent's current knowledge-base entries and some sensemaking
    result provided as a compiled binary join tree (BJT)
    """
    q_vars, (cons, _) = question

    # Return value: Queries to feed into KB for fetching search specs. Represent
    # each query as a pair of predicates of interest & arg entities of interest
    kb_queries = set()

    # Inspecting literals in each q_rule for identifying search specs to feed into
    # visual search calls
    for q_lit in cons:
        if q_lit.name == "*_isinstance":
            # Literal whose predicate is question-marked (contained for questions
            # like "What (kind of X) is this?", etc.); the first argument term,
            # standing for the predicate variable, must be contained in q_vars
            assert q_lit.args[0] in q_vars

            # Assume we are only interested in cls concepts with "What is this?"
            # type of questions
            kb_query_preds = frozenset([
                pred for pred in agent.lt_mem.kb.entries_by_pred 
                if pred.startswith("cls")
            ])
            # Filter further by provided restrictors if applicable
            if q_lit.args[0][0] in restrictors:
                kb_query_preds = frozenset([
                    pred for pred in kb_query_preds
                    if pred in restrictors[q_lit.args[0][0]]
                ])
            kb_query_args = tuple(q_lit.args[1:])
        else:
            # Literal with fixed predicate, to which can narrow down the KB query
            kb_query_preds = frozenset([q_lit.name])
            kb_query_args = tuple(q_lit.args)
        
        kb_queries.add((kb_query_preds, kb_query_args))

    # Query the KB to collect search specs
    search_spec_cands = []
    for kb_query_preds, kb_query_args in kb_queries:

        for pred in kb_query_preds:
            # Relevant KB entries containing predicate of interest
            relevant_entries = agent.lt_mem.kb.entries_by_pred[pred]
            relevant_entries = [
                agent.lt_mem.kb.entries[entry_id]
                for entry_id in relevant_entries
            ]

            # Set of literals for each relevant KB entry
            relevant_literals = sum([
                flatten_cons_ante(*entry[0]) for entry in relevant_entries
            ], [])
            relevant_literals = [
                set(cons+ante) for cons, ante in relevant_literals
            ]
            # Depending on which literal (with matching predicate name) in literal
            # sets to use as 'anchor', there can be multiple choices of search specs
            relevant_literals = [
                { l: lits-{l} for l in lits if l.name==pred }
                for lits in relevant_literals
            ]

            # Collect search spec candidates. We will disregard attribute concepts as
            # search spec elements, noticing that it is usually sufficient and generalizable
            # to provide object class info only as specs for searching potentially relevant,
            # yet unrecognized entities in a scene. This is more of a heuristic for now --
            # maybe justify this on good grounds later...
            specs = [
                {
                    tgt_lit: (
                        {rl for rl in rel_lits if not rl.name.startswith("att_")},
                        {la: qa for la, qa in zip(tgt_lit.args, kb_query_args)}
                    )
                    for tgt_lit, rel_lits in lits.items()
                }
                for lits in relevant_literals
            ]
            specs = [
                {
                    tgt_lit.substitute(terms=term_map): frozenset({
                        rl.substitute(terms=term_map) for rl in rel_lits
                    })
                    for tgt_lit, (rel_lits, term_map) in spc.items()
                }
                for spc in specs
            ]
            search_spec_cands += specs

    # Merge and flatten down to a single layer dict
    def set_add_merge(d1, d2):
        for k, v in d2.items(): d1[k].add(v)
        return d1
    search_spec_cands = reduce(set_add_merge, [defaultdict(set)]+search_spec_cands)

    # Finalize set of search specs, excluding those which already have satisfying
    # entities in the current sensemaking output
    final_specs = []
    for lits_sets in search_spec_cands.values():
        for lits in lits_sets:
            # Lift any remaining function term args to non-function variable args
            all_fn_args = {
                arg for arg in set.union(*[set(l.args) for l in lits])
                if type(arg[0])==tuple
            }
            all_var_names = {
                t_val for t_val, t_is_var in set.union(*[l.nonfn_terms() for l in lits])
                if t_is_var
            }
            fn_lifting_map = {
                fa: (f"X{i+len(all_var_names)}", True)
                for i, fa in enumerate(all_fn_args)
            }

            search_vars = all_var_names | {vn for vn, _ in fn_lifting_map.values()}
            search_vars = tuple(search_vars)
            if len(search_vars) == 0:
                # Disregard if there's no variables in search spec (i.e. no search target
                # after all)
                continue

            lits = [l.substitute(terms=fn_lifting_map) for l in lits]
            lits = [l for l in lits if any(la_is_var for _, la_is_var in l.args)]

            # Disregard if there's already an isomorphic literal set
            has_isomorphic_spec = any(
                Literal.entailing_mapping_btw(lits, spc[1])[1] == 0
                for spc in final_specs
            )
            if has_isomorphic_spec:
                continue

            # Check if the agent is already (visually) aware of the potential search
            # targets; if so, disregard this one
            check_result = agent.symbolic.query(
                ref_bjt, tuple((v, False) for v in search_vars), (lits, None)
            )
            if len(check_result) > 0:
                continue

            final_specs.append((search_vars, lits))

    # Perform incremental visual search...
    O = len(agent.vision.scene)
    oi_offsets = np.cumsum([0]+[len(vars) for vars, _ in final_specs][:-1])
    final_specs = {
        tuple(f"o{offset+i+O}" for i in range(len(spc[0]))): spc
        for spc, offset in zip(final_specs, oi_offsets)
    }

    return final_specs
