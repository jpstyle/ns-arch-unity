"""
Implements dialogue-related composite actions
"""
import re
import random
from functools import reduce
from itertools import permutations
from collections import defaultdict

from ..lpmln import Literal
from ..lpmln.utils import flatten_cons_ante
from ..memory.kb import KnowledgeBase

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

    ti, ci = utt_pointer
    (_, question), _ = translated[ti][1][ci]

    if question is None:
        # Question cannot be answered for some reason
        return
    else:
        # Schedule to answer the question
        agent.practical.agenda.insert(0, ("answer_Q", utt_pointer))
        return

def prepare_answer_Q(agent, utt_pointer):
    """
    Prepare an answer to a question that has been deemed answerable, by first
    computing raw ingredients from which answer candidates can be composed,
    picking out an answer among the candidates, then translating the answer
    into natural language form to be uttered
    """
    # The question is about to be answered
    agent.lang.dialogue.unanswered_Qs.remove(utt_pointer)

    ti, ci = utt_pointer
    dialogue_state = agent.lang.dialogue.export_as_dict()
    translated = agent.symbolic.translate_dialogue_content(dialogue_state)

    if dialogue_state["clause_info"][f"t{ti}c{ci}"]["domain_describing"]:
        _answer_domain_Q(agent, utt_pointer, dialogue_state, translated)
    else:
        _answer_nondomain_Q(agent, utt_pointer, dialogue_state, translated)

def _answer_domain_Q(agent, utt_pointer, dialogue_state, translated):
    """
    Helper method factored out for computation of answers to an in-domain question;
    i.e. having to do with the current states of affairs of the task domain.
    """
    ti, ci = utt_pointer
    presup, question = translated[ti][1][ci][0]
    assert question is not None

    q_vars, (q_cons, q_ante) = question

    bjt_v, _ = agent.symbolic.concl_vis
    # bjt_vl, _ = symbolic.concl_vis_lang

    # New dialogue turn & clause index for the answer to be provided
    ti_new = len(agent.lang.dialogue.record)
    ci_new = 0

    # Mapping from predicate variables to their associated entities
    pred_var_to_ent_ref = {
        ql.args[0][0]: ql.args[1][0] for ql in q_cons
        if ql.name == "*_isinstance"
    }

    qv_to_dis_ref = {
        qv: f"x{ri}t{ti_new}c{ci_new}" for ri, (qv, _) in enumerate(q_vars)
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

        # Extract any '*_subtype' statements and cast into appropriate query restrictors
        restrictors = {
            lit.args[0][0]: agent.lt_mem.kb.find_entailer_concepts(conc_conjs[lit.args[1][0]])
            for lit in q_cons if lit.name=="*_subtype"
        }
        # Remove the '*_subtype' statements from q_cons now that they are processed
        q_cons = tuple(lit for lit in q_cons if lit.name!="*_subtype")
        question = (q_vars, (q_cons, q_ante))

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

            # If new entities is registered as a result of visual search, update env
            # referent list
            new_ents = set(agent.vision.scene) - set(agent.lang.dialogue.referents["env"])
            for ent in new_ents:
                mask = agent.vision.scene[ent]["pred_mask"]
                agent.lang.dialogue.referents["env"][ent] = {
                    "mask": mask,
                    "area": mask.sum().item()
                }
                agent.lang.dialogue.referent_names[ent] = ent

            #  ... and another round of sensemaking
            exported_kb = agent.lt_mem.kb.export_reasoning_program()
            visual_evidence = agent.lt_mem.kb.visual_evidence_from_scene(agent.vision.scene)
            agent.symbolic.sensemake_vis(exported_kb, visual_evidence)
            agent.lang.dialogue.sensemaking_v_snaps[ti_new] = agent.symbolic.concl_vis

            agent.symbolic.resolve_symbol_semantics(dialogue_state, agent.lt_mem.lexicon)
            # agent.symbolic.sensemake_vis_lang(dialogue_state)
            # agent.lang.dialogue.sensemaking_vl_snaps[ti_new] = agent.symbolic.concl_vis_lang

            bjt_v, _ = agent.symbolic.concl_vis
            # bjt_vl, _ = symbolic.concl_vis_lang

    # Compute raw answer candidates by appropriately querying compiled BJT
    answers_raw, _ = agent.symbolic.query(bjt_v, q_vars, (q_cons, q_ante), restrictors)

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

                    pred_name = agent.lt_mem.lexicon.d2s[ans][0][0]

                    # Update cognitive state w.r.t. value assignment and word sense
                    agent.symbolic.value_assignment[ri] = \
                        pred_var_to_ent_ref[qv]
                    tok_ind = (f"t{ti_new}", f"c{ci_new}", "rc", "0")
                    agent.symbolic.word_senses[tok_ind] = \
                        ((conc_type_to_pos[ans[1]], pred_name), f"{ans[1]}_{ans[0]}")

                    answer_logical_form = (
                        ((pred_name, conc_type_to_pos[ans[1]], (ri,), False),), ()
                    )

                    # Split camelCased predicate name
                    splits = re.findall(
                        r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                    )
                    splits = [w[0].lower()+w[1:] for w in splits]
                    answer_nl = f"This is a {' '.join(splits)}."
            else:
                # Need to give some referring expression as answer; TODO? implement
                raise NotImplementedError

    # Fetch segmentation mask for the demonstratively referenced entity
    dem_mask = dialogue_state["referents"]["env"][pred_var_to_ent_ref[qv]]["mask"]

    # Push the translated answer to buffer of utterances to generate
    agent.lang.dialogue.to_generate.append(
        ((answer_logical_form, None), answer_nl, { (0, 4): dem_mask })
    )

def _answer_nondomain_Q(agent, utt_pointer, dialogue_state, translated):
    """
    Helper method factored out for computation of answers to a question that needs
    procedures different from those for in-domain questions for obtaining. Currently
    includes why-questions.
    """
    ti, ci = utt_pointer
    _, question = translated[ti][1][ci][0]
    assert question is not None

    _, (q_cons, _) = question

    # New dialogue turn & clause index for the answer to be provided
    ti_new = len(agent.lang.dialogue.record)
    ci_new = 0

    conc_type_to_pos = { "cls": "n" }

    if any(lit.name=="*_expl" for lit in q_cons):
        # Answering why-question

        # Extract target event to explain from the question; fetch the explanandum
        # statement
        expl_lits = [lit for lit in q_cons if lit.name=="*_expl"]
        expd_utts = [
            re.search("^t(\d+)c(\d+)$", lit.args[1][0])
            for lit in expl_lits
        ]
        expd_utts = [
            (int(clause_ind.group(1)), int(clause_ind.group(2)))
            for clause_ind in expd_utts
        ]
        expd_contents = [
            translated[ti_exd][1][ci_exd][0]
            for ti_exd, ci_exd in expd_utts
        ]

        # Can treat "why ~" and "why did you think ~" as the same for our purpose
        self_think_lits = [
            [
                lit for lit in expd_cons[0]
                if lit.name=="*_think" and lit.args[0][0]=="_self"
            ]
            for expd_cons, _ in expd_contents
        ]
        st_expd_utts = [
            [
                re.search("^t(\d+)c(\d+)$", st_lit.args[1][0])
                for st_lit in per_expd
            ]
            for per_expd in self_think_lits
        ]
        st_expd_utts = [
            [
                (int(clause_ind.group(1)), int(clause_ind.group(2)))
                for clause_ind in per_expd
            ]
            for per_expd in st_expd_utts
        ]
        # Assume all explananda consist of consequent-only rules
        st_expd_contents = [
            translated[ti_exd][1][ci_exd][0][0][0]
            for per_expd in st_expd_utts
            for ti_exd, ci_exd in per_expd
        ]
        nst_expd_contents = [
            tuple(
                lit for lit in expd_cons[0]
                if not (lit.name=="*_think" and lit.args[0][0]=="_self")
            )
            for expd_cons, _ in expd_contents
        ]
        target_events = nst_expd_contents + st_expd_contents

        # Fetch the BJT based on which the explanandum (i.e. explanation target)
        # utterance have been made; that is, shouldn't use BJT compiled *after*
        # agent has uttered the explanandum statement
        latest_reasoning_ind = max(
            snap_ti for snap_ti in dialogue_state["sensemaking_v_snaps"]
            if snap_ti < ti
        )
        bjt_v, (kb_prog, _) = dialogue_state["sensemaking_v_snaps"][latest_reasoning_ind]
        analyzed_kb_prog = KnowledgeBase.analyze_exported_reasoning_program(kb_prog)
        scene_ents = {
            a[0] for atm in bjt_v.graph["atoms_map"]
            for a in atm.args if isinstance(a[0], str)
        }

        for tgt_ev in target_events:
            if len(tgt_ev) == 0: continue

            # Only considers singleton events right now
            assert len(tgt_ev) == 1

            # Select candidate explanans from the inference program used for answering
            # the question; start from rules containing the target event predicate and
            # continue spanning along (against?) the abductive direction until all
            # potential evidence atoms are identified
            evidence_atoms = set(); frontier = {tgt_ev[0]}
            while len(frontier) > 0:
                expd_atm = frontier.pop()       # Atom representing explanandum event

                for rule_info in analyzed_kb_prog.values():
                    # Disregard rules without abductive force
                    if not rule_info.get("abductive"): continue

                    # Check if rule is relevant; i.e. whether the popped explanandum
                    # event is included in rule antecedent
                    entail_dir, mapping = Literal.entailing_mapping_btw(
                        rule_info["ante"], [expd_atm]
                    )

                    if entail_dir is not None and entail_dir == 1:
                        # Rule relevant, target event may be abduced from (grounded)
                        # consequent literals
                        r_cons_subs = [
                            c_lit.substitute(**mapping) for c_lit in rule_info["cons"]
                        ]
                        
                        # All possible substitutions for the remaining variables; will
                        # automatically give empty singleton list if len(remaining_vars)
                        # is larger than len(remaining_ents)
                        remaining_vars = {
                            arg for c_lit in r_cons_subs for arg, is_var in c_lit.args
                            if is_var
                        }
                        remaining_vars = tuple(remaining_vars)      # Int indexing
                        remaining_ents = scene_ents - set(
                            e for e, _ in mapping["terms"].values()
                        )

                        possible_remaining_subs = permutations(
                            remaining_ents, len(remaining_vars)
                        )
                        for r_subs in possible_remaining_subs:
                            r_subs = {
                                (remaining_vars[i], True): (e, False)
                                for i, e in enumerate(r_subs)
                            }
                            for c_lit in r_cons_subs:
                                # Considering visual evidence for class & attributes
                                # concepts only, which are neurally predicted
                                if not (
                                    c_lit.name.startswith("cls") or
                                    c_lit.name.startswith("att")
                                ):
                                    continue

                                # Substitute, add event atom to frontier, evidence atom
                                # to the evidence atom set
                                c_lit = c_lit.substitute(terms=r_subs)
                                ev_lit = Literal(f"v_{c_lit.name}", c_lit.args)

                                frontier.add(c_lit)
                                evidence_atoms.add(ev_lit)

            # Cast as tuple to assign (some arbitrary) ordered indexing
            evidence_atoms = tuple(evidence_atoms)

            # Manually select 'competitor' atoms that could've been the answer
            # in place of the true one (not necessarily mutually exclusive);
            # this is a band-aid solution right now as we are manually providing
            # the truck subtype predicates, may need a more principled selection
            # logic (by referring to the question that invoked the answer, etc.)
            competing_evts = [
                atm for atm in bjt_v.graph["atoms_map"]
                if (
                    atm.name in [f"cls_{ci}" for ci in [4,5,6,7]] and
                    atm.name!=tgt_ev[0].name and atm.args==tgt_ev[0].args
                )
            ]
            # Determine the probability threshold to be provided into the causal
            # attribution procedure method
            ans_threshold = 0.0
            for ev_atm in competing_evts:
                ans, _ = agent.symbolic.query(bjt_v, None, ((ev_atm,), None), {})
                ans_threshold = max(ans_threshold, ans[()])

            # Obtain all sufficient explanations by causal attribution
            suff_expls = agent.symbolic.attribute(
                bjt_v, tgt_ev, evidence_atoms, threshold=ans_threshold
            )

            if len(suff_expls) > 0:
                # Found some sufficient explanations; report the first one as the
                # answer using the template "Because {}, {} and {}."
                answer_logical_form = []
                answer_nl = "Because "
                dem_refs = {}; dem_offset = len(answer_nl)

                for i, exps_lit in enumerate(suff_expls[0]):
                    # For each explanans literal, add the string "this is a X"
                    conc_pred = exps_lit.name.strip("v_")
                    conc_type, conc_ind = conc_pred.split("_")
                    pred_name = agent.lt_mem.lexicon.d2s[(int(conc_ind), conc_type)][0][0]

                    # Update cognitive state w.r.t. value assignment and word sense
                    ri = f"x{i}t{ti_new}c{ci_new}"
                    agent.symbolic.value_assignment[ri] = exps_lit.args[0][0]
                    tok_ind = (f"t{ti_new}", f"c{ci_new}", "rc", "0")
                    agent.symbolic.word_senses[tok_ind] = \
                        ((conc_type_to_pos[conc_type], pred_name), conc_pred)

                    answer_logical_form.append(
                        (pred_name, conc_type_to_pos[conc_type], (ri,), False)
                    )

                    # Split camelCased predicate name
                    splits = re.findall(
                        r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                    )
                    splits = [w[0].lower()+w[1:] for w in splits]
                    reason_nl = f"this is a {' '.join(splits)}"
                
                    # Append a suffix appropriate for the number of reasons
                    if i == len(suff_expls[0])-1:
                        # Last entry, period
                        reason_nl += "."
                    elif i == len(suff_expls[0])-2:
                        # Next to last entry, comma+and
                        reason_nl += ", and "
                    else:
                        # Otherwise, comma
                        reason_nl += ", "

                    answer_nl += reason_nl

                    # Fetch mask for demonstrative reference and shift offset
                    dem_refs[(dem_offset, dem_offset+4)] = \
                        dialogue_state["referents"]["env"][exps_lit.args[0][0]]["mask"]
                    dem_offset += len(reason_nl)

                # Wrapping logical form (consequent part, to be precise) as needed
                answer_logical_form = (tuple(answer_logical_form), ())

                # Push the translated answer to buffer of utterances to generate
                agent.lang.dialogue.to_generate.append(
                    ((answer_logical_form, None), answer_nl, dem_refs)
                )

                # Push another record for the rhetorical relation as well, namely the
                # fact that the provided answer clause explains the agent's previous
                # question. Not to be uttered explicitly (already uttered, as a matter
                # of fact), but for bookkeeping purpose.
                rrel_logical_form = (
                    "expl", "*", (f"t{ti_new}c{ci_new}", q_cons[0].args[1][0]), False
                )
                rrel_logical_form = ((rrel_logical_form,), ())
                agent.lang.dialogue.to_generate.append(
                    (
                        (rrel_logical_form, None),
                        "# rhetorical relation due to 'because'",
                        {}
                    )
                )

            else:
                # Couldn't find any sufficient explanations that could be provided
                # verbally; answer "I cannot explain."
                agent.lang.dialogue.to_generate.append(
                    # Will just pass None as "logical form" for this...
                    (None, "I cannot explain.", {})
                )

    else:
        # Don't know how to handle other non-domain questions
        raise NotImplementedError

def _search_specs_from_kb(agent, question, ref_bjt, restrictors):
    """
    Helper method factored out for extracting specifications for visual search,
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
                Literal.entailing_mapping_btw(lits, spc[1])[0] == 0
                for spc in final_specs
            )
            if has_isomorphic_spec:
                continue

            # Check if the agent is already (visually) aware of the potential search
            # targets; if so, disregard this one
            check_result, _ = agent.symbolic.query(
                ref_bjt, tuple((v, False) for v in search_vars), (lits, None)
            )
            if len(check_result) > 0:
                continue

            final_specs.append((search_vars, lits))

    return final_specs
