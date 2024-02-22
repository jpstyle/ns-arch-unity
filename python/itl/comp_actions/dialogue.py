"""
Implements dialogue-related composite actions
"""
import re
import random
from copy import deepcopy
from functools import reduce
from itertools import permutations, combinations, product
from collections import defaultdict

import numpy as np
import networkx as nx

from ..lpmln import Literal
from ..lpmln.utils import flatten_cons_ante
from ..memory.kb import KnowledgeBase
from ..symbolic_reasoning.query import query
from ..symbolic_reasoning.utils import rgr_extract_likelihood, rgr_replace_likelihood


# As it is simply impossible to enumerate 'every possible continuous value',
# we consider a discretized likelihood threshold as counterfactual alternative.
# The value represents an alternative visual observation that results in evidence
# that is 'reasonably' stronger that the corresponding literal actually being true.
# (cf. ..symbolic_reasoning.attribute)
HIGH = 0.85

# Recursive helper method for checking whether rule cons/ante uses a reserved (pred
# type *) predicate 
has_reserved_pred = \
    lambda cnjt: cnjt.name.startswith("*_") \
        if isinstance(cnjt, Literal) else any(has_reserved_pred(nc) for nc in cnjt)

def attempt_answer_Q(agent, utt_pointer):
    """
    Attempt to answer an unanswered question from user.
    
    If it turns out the question cannot be answered at all with the agent's
    current knowledge (e.g. question contains unresolved neologism), do nothing
    and wait for it to become answerable.

    If the agent can come up with an answer to the question, right or wrong,
    schedule to actually answer it by adding a new agenda item.
    """
    translated = agent.symbolic.translate_dialogue_content(agent.lang.dialogue)

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
    translated = agent.symbolic.translate_dialogue_content(agent.lang.dialogue)

    if agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["domain_describing"]:
        _answer_domain_Q(agent, utt_pointer, translated)
    else:
        _answer_nondomain_Q(agent, utt_pointer, translated)

def _answer_domain_Q(agent, utt_pointer, translated):
    """
    Helper method factored out for computation of answers to an in-domain question;
    i.e. having to do with the current states of affairs of the task domain.
    """
    ti, ci = utt_pointer
    presup, question = translated[ti][1][ci][0]
    assert question is not None

    q_vars, (q_cons, q_ante) = question

    reg_gr_v, _ = agent.symbolic.concl_vis
    # reg_gr_vl, _ = symbolic.concl_vis_lang

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
        search_specs = _search_specs_from_kb(agent, question, restrictors, reg_gr_v)
        if len(search_specs) > 0:
            agent.vision.predict(
                None, agent.lt_mem.exemplars, specs=search_specs,
                visualize=False, lexicon=agent.lt_mem.lexicon
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

            agent.symbolic.resolve_symbol_semantics(agent.lang.dialogue, agent.lt_mem.lexicon)
            # agent.symbolic.sensemake_vis_lang(agent.lang.dialogue)
            # agent.lang.dialogue.sensemaking_vl_snaps[ti_new] = agent.symbolic.concl_vis_lang

            reg_gr_v, _ = agent.symbolic.concl_vis
            # reg_gr_vl, _ = symbolic.concl_vis_lang

    # Compute raw answer candidates by appropriately querying compiled region graph
    answers_raw, _ = agent.symbolic.query(reg_gr_v, q_vars, (q_cons, q_ante), restrictors)

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
    dem_mask = agent.lang.dialogue.referents["env"][pred_var_to_ent_ref[qv]]["mask"]

    # Push the translated answer to buffer of utterances to generate
    agent.lang.dialogue.to_generate.append(
        ((answer_logical_form, None), answer_nl, { (0, 4): dem_mask })
    )

def _answer_nondomain_Q(agent, utt_pointer, translated):
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

        # Fetch the region graph based on which the explanandum (i.e. explanation
        # target) utterance have been made; that is, shouldn't use region graph
        # compiled *after* agent has uttered the explanandum statement
        latest_reasoning_ind = max(
            snap_ti for snap_ti in agent.lang.dialogue.sensemaking_v_snaps
            if snap_ti < ti
        )
        reg_gr_v, (kb_prog, _) = agent.lang.dialogue.sensemaking_v_snaps[latest_reasoning_ind]
        kb_prog_analyzed = KnowledgeBase.analyze_exported_reasoning_program(kb_prog)
        scene_ents = {
            a[0] for atm in reg_gr_v.graph["atoms_map"]
            for a in atm.args if isinstance(a[0], str)
        }

        # Collect previous factual statements and questions made during this dialogue
        prev_statements = []; prev_Qs = []
        for ti, (spk, turn_clauses) in enumerate(translated):
            for ci, ((rule, ques), _) in enumerate(turn_clauses):
                # Factual statement
                if rule is not None and len(rule[0])==1 and rule[1] is None:
                    prev_statements.append(((ti, ci), (spk, rule)))
                
                # Question
                if ques is not None:
                    # Here, `rule` represents presuppositions included in `ques`
                    prev_Qs.append(((ti, ci), (spk, ques, rule)))

        # Fetch teacher's last correction, which is the expected ground truth
        prev_statements_U = [
            stm for (ti, ci), (spk, stm) in prev_statements
            if spk=="U" and \
                agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["domain_describing"] and \
                not agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["irrealis"]
        ]
        expected_gt = prev_statements_U[-1][0][0]

        # Fetch teacher's last probing question, which restricts the type of the answer
        # predicate anticipated by taxonomy entailment
        prev_Qs_U = [
            (ques, presup) for _, (spk, ques, presup) in prev_Qs
            if spk=="U" and "*_isinstance" in {ql.name for ql in ques[1][0]}
        ]
        probing_Q, probing_presup = prev_Qs_U[-1]
        # Process any 'concept conjunctions' provided in the presupposition into a more
        # legible format, for easier processing right after
        conc_conjs = defaultdict(set)
        for lit in probing_presup[0]:
            conc_conjs[lit.args[0][0]].add(lit.name)
        # Extract any '*_subtype' statements and cast into appropriate query restrictors
        restrictors = {
            lit.args[0][0]: agent.lt_mem.kb.find_entailer_concepts(conc_conjs[lit.args[1][0]])
            for lit in probing_Q[1][0] if lit.name=="*_subtype"
        }

        # Find valid templates of potential explanantia for the expected ground-truth
        # answer based on the inference program
        exps_templates_gt = _find_explanans_templates(expected_gt, kb_prog_analyzed)

        for tgt_ev in target_events:
            if len(tgt_ev) == 0: continue

            # Only considers singleton events right now
            assert len(tgt_ev) == 1
            tgt_lit = tgt_ev[0]
            v_tgt_lit = Literal(f"v_{tgt_lit.name}", tgt_lit.args)

            # Find valid templates of potential explanantia for the agent answer based
            # on the inference program
            exps_templates_ans = _find_explanans_templates(tgt_lit, kb_prog_analyzed)

            # Asymmetric set difference for selecting distinguishing properties; based
            # on the notion that shared explanantia shouldn't be considered as valid
            # explanations (though they often are selected if not explicitly filtered...)
            shared_template_pairs = [
                (tpl1, tpl2)
                for tpl1, tpl2 in product(exps_templates_ans[1:], exps_templates_gt[1:])
                if Literal.entailing_mapping_btw(tpl1, tpl2)[0] == 0
            ]
            shared_templates_ans = {frozenset(tpl1) for tpl1, _ in shared_template_pairs}
            distinguishing_templates_ans = [
                tpl for tpl in exps_templates_ans[1:]
                if frozenset(tpl) not in shared_templates_ans
            ]
            selected_templates = exps_templates_ans[:1] + distinguishing_templates_ans

            # Obtain every possible instantiations of the discovered templates, then
            # flatten to a set of (grounded) potential evidence atoms
            exps_instances = _instantiate_templates(selected_templates, scene_ents)
            evidence_atoms = {
                Literal(f"v_{c_lit.name}", c_lit.args)
                for conjunction in exps_instances for c_lit in conjunction
                if c_lit.name.startswith("cls") or c_lit.name.startswith("att")
            }       # Considering visual evidence for class & attributes concepts only

            # Manually select 'competitor' events that could've been the answer
            # in place of the true one (not necessarily mutually exclusive)
            possible_answers = [
                atm for atm in reg_gr_v.graph["atoms_map"]
                if (
                    atm.name in restrictors[probing_Q[0][0][0]] and
                    atm.args==tgt_lit.args
                )
            ]
            competing_evts = [
                (atm,) for atm in possible_answers if atm.name!=tgt_lit.name
            ]

            # Obtain a sufficient explanations by causal attribution (greedy search);
            # veto dud explanations like 'Because it looked liked one'
            suff_expl = agent.symbolic.attribute(
                reg_gr_v, tgt_ev, evidence_atoms, competing_evts, vetos=[v_tgt_lit]
            )

            if suff_expl is not None:
                # Found some sufficient explanations; report the first one as the
                # answer using the template "Because {}, {} and {}."
                answer_logical_form = []
                answer_nl = "Because I thought "
                dem_refs = {}; dem_offset = len(answer_nl)

                for i, exps_lit in enumerate(suff_expl):
                    # For each explanans literal, add the string "this is a X"
                    conc_pred = exps_lit.name.strip("v_")
                    conc_type, conc_ind = conc_pred.split("_")
                    pred_name = agent.lt_mem.lexicon.d2s[(int(conc_ind), conc_type)][0][0]
                    # Split camelCased predicate name
                    splits = re.findall(
                        r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                    )
                    splits = [w[0].lower()+w[1:] for w in splits]
                    conc_nl = ' '.join(splits)

                    # Update cognitive state w.r.t. value assignment and word sense
                    ri = f"x{i}t{ti_new}c{ci_new}"
                    agent.symbolic.value_assignment[ri] = exps_lit.args[0][0]
                    tok_ind = (f"t{ti_new}", f"c{ci_new}", "rc", "0")
                    agent.symbolic.word_senses[tok_ind] = \
                        ((conc_type_to_pos[conc_type], pred_name), conc_pred)

                    answer_logical_form.append(
                        (pred_name, conc_type_to_pos[conc_type], (ri,), False)
                    )

                    # Realize the reason as natural language utterance
                    reason_prefix = f"this is a "

                    # Append a suffix appropriate for the number of reasons
                    if i == len(suff_expl)-1:
                        # Last entry, period
                        reason_suffix = "."
                    elif i == len(suff_expl)-2:
                        # Next to last entry, comma+and
                        reason_suffix = ", and "
                    else:
                        # Otherwise, comma
                        reason_suffix = ", "

                    reason_nl = f"{reason_prefix}{conc_nl}{reason_suffix}"
                    answer_nl += reason_nl

                    # Fetch mask for demonstrative reference and shift offset
                    dem_refs[(dem_offset, dem_offset+4)] = \
                        agent.lang.dialogue.referents["env"][exps_lit.args[0][0]]["mask"]
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
                # No 'meaningful' (i.e., 'Because it looked like one') sufficient
                # explanations found, try finding a good counterfactual explanation
                # that would've led to the expected ground truth answer
                selected_cf_expl = None

                if agent.cfg.exp.strat_feedback != "maxHelpExpl2":
                    # Short circuit, just give a dud explanation
                    agent.lang.dialogue.to_generate.append(
                        # Will just pass None as "logical form" for this...
                        (None, "I cannot explain.", {})
                    )
                    continue

                # GT side of the distinguishing properties
                shared_templates_gt = {frozenset(tpl2) for _, tpl2 in shared_template_pairs}
                distinguishing_templates_gt = [
                    tpl for tpl in exps_templates_gt[1:]
                    if frozenset(tpl) not in shared_templates_gt
                ]
                # Not including the dud explanation ('I would've said that if it looked like one')
                # unlike above
                selected_templates = distinguishing_templates_gt

                # len(selected_templates) > 0 iff there's any KB rule that can be leveraged
                # to abduce the expected ground-truth answer
                gt_inferrable_from_kb = len(selected_templates) > 0

                if gt_inferrable_from_kb:
                    # Try to find some meaningful counterfactual explanation that would
                    # increase the likelihood of the expected answer to a sufficiently
                    # high value. Basically an existence test; try every instantiation
                    # of every template discovered until some valid counterfactual is
                    # found. If found one, answer with that; otherwise, fall back to
                    # dud explanation.

                    # Test each template one-by-one
                    for template in selected_templates:
                        # Find every possible instantiation of the template
                        exps_instances = _instantiate_templates([template], scene_ents)

                        # Considering visual evidence for class & attributes concepts,
                        # as above
                        evidence_atoms = {
                            Literal(f"v_{c_lit.name}", c_lit.args)
                            for conjunction in exps_instances for c_lit in conjunction
                            if c_lit.name.startswith("cls") or c_lit.name.startswith("att")
                        }
                        evidence_atoms = {
                            evd_atm for evd_atm in evidence_atoms
                            if evd_atm in reg_gr_v.graph["atoms_map"]
                        }       # Can try replacement only if `evd_atm` is registered in graph

                        if len(evidence_atoms) > 0:
                            # Potential explanation exists in scene, albeit with a
                            # probability value not high enough. Try raising likelihood
                            # of relevant evidence atoms and querying the updated region
                            # graph for ground truth event probability.

                            # Obtain a modified graph where the likelihoods are raised to
                            # 'sufficiently high' values. (Note we are assuming evidence
                            # literals occur in rules with positive polarity only, which
                            # will suffice within our current scope.)
                            replacements = { evd_atm: HIGH for evd_atm in evidence_atoms }
                            backups = {
                                evd_atm: rgr_extract_likelihood(reg_gr_v, evd_atm)
                                for evd_atm in evidence_atoms
                            }           # For rolling back to original values
                            rgr_replace_likelihood(reg_gr_v, replacements)

                            # Query the updated graph for the event probabilities
                            max_prob_evt = (None, float("-inf"))
                            for atm in possible_answers:
                                evt = (atm,)
                                _, prob_scores = query(reg_gr_v, None, (evt, None), {})

                                # Update max probability event if applicable
                                evt_prob = [
                                    prob for prob, is_evt in prob_scores[()].values() if is_evt
                                ][0]
                                if evt_prob > max_prob_evt[1]:
                                    max_prob_evt = (evt, evt_prob)
                            rgr_replace_likelihood(reg_gr_v, backups)

                            assert max_prob_evt[0] is not None
                            if max_prob_evt[0] == (expected_gt,):
                                # The counterfactual case successfully subverts the ranking
                                # of answers, making the expected ground truth as the most
                                # likely event

                                # Provide the evidence atoms with current likelihood values
                                # as data needed for generating the counterfactual explanation.
                                selected_cf_expl = (backups, template)
                                break

                        else:
                            # Potential explanation doesn't exist in scene. Try adding
                            # hypothetical entities into the scene with appropriate
                            # likelihood values based on the info contained in template.
                            # Compile a new region graph then query for ground truth
                            # event probability.
                            occurring_vars = {
                                arg for t_lit in template for arg, is_var in t_lit.args
                                if is_var
                            }
                            hyp_ents = [f"h{i}" for i in range(len(occurring_vars))]
                            hyp_subs = {
                                (v, True): (e, False)
                                for v, e in zip(occurring_vars, hyp_ents)
                            }

                            # Template instance grounded with the hypothetical entities
                            hyp_instance = [
                                t_lit.substitute(terms=hyp_subs) for t_lit in template
                            ]

                            # Deepcopy vision.scene for counterfactual manipulation
                            scene_new = deepcopy(agent.vision.scene)
                            scene_new = {
                                **scene_new,
                                **{h: {} for h in hyp_ents}
                            }

                            # Add hypothetical likelihood values as designated by the
                            # template instance
                            for h_lit in hyp_instance:
                                conc_type, conc_ind = h_lit.name.split("_")
                                conc_ind = int(conc_ind)
                                field = f"pred_{conc_type}"
                                C = getattr(agent.vision.inventories, conc_type)

                                if conc_type == "cls" or conc_type == "att":
                                    arg1 = h_lit.args[0][0]
                                    if field not in scene_new[arg1]:
                                        scene_new[arg1][field] = np.zeros(C)
                                    scene_new[arg1][field][conc_ind] = HIGH

                                else:
                                    assert conc_type == "rel"
                                    assert len(h_lit.args) == 2

                                    arg1 = h_lit.args[0][0]; arg2 = h_lit.args[1][0]
                                    if field not in scene_new[arg1]:
                                        scene_new[arg1][field] = {}
                                    if arg2 not in scene_new[arg1][field]:
                                        scene_new[arg1][field][arg2] = np.zeros(C)
                                    scene_new[arg1][field][arg2][conc_ind] = HIGH

                            hyp_evidence = agent.lt_mem.kb.visual_evidence_from_scene(scene_new)
                            reg_gr_hyp = (kb_prog + hyp_evidence).compile()

                            # Query the hypothetical region graph for the event probabilities
                            max_prob_evt = (None, float("-inf"))
                            for atm in possible_answers:
                                evt = (atm,)
                                _, prob_scores = query(reg_gr_hyp, None, (evt, None), {})

                                # Update max probability event if applicable
                                evt_prob = [
                                    prob for prob, is_evt in prob_scores[()].values() if is_evt
                                ][0]
                                if evt_prob > max_prob_evt[1]:
                                    max_prob_evt = (evt, evt_prob)

                            assert max_prob_evt[0] is not None
                            if max_prob_evt[0] == (expected_gt,):
                                # The counterfactual case successfully subverts the ranking
                                # of answers, making the expected ground truth as the most
                                # likely event

                                # Provide the hypothetical evidence atoms (denoted with None
                                # as 'current' likelihood) as data needed for generating the
                                # counterfactual explanation.
                                evidence_likelihoods = {
                                    Literal(f"v_{h_lit.name}", h_lit.args): None
                                    for h_lit in hyp_instance
                                    if h_lit.name.startswith("cls") or h_lit.name.startswith("att")
                                }
                                selected_cf_expl = (evidence_likelihoods, template)
                                break

                # Generate appropriate agent response
                if selected_cf_expl is None:
                    # Agent couldn't find any meaningful explanations that could be
                    # provided verbally; answer "I cannot explain."
                    agent.lang.dialogue.to_generate.append(
                        # Will just pass None as "logical form" for this...
                        (None, "I cannot explain.", {})
                    )

                else:
                    # Found data needed for generating some counterfactual explanation,
                    # in the form of dict { [potential_evidence]: [current_likelihood] },
                    # and the template that yielded the explanans instance
                    evidence_likelihoods, template = selected_cf_expl

                    answer_nl = "Because I thought "
                    dem_refs = {}; dem_offset = len(answer_nl)

                    reasons = {}
                    for evd_atom, pr_val in evidence_likelihoods.items():
                        # For each counterfactual explanation, add appropriate string
                        conc_pred = evd_atom.name.strip("v_")
                        conc_type, conc_ind = conc_pred.split("_")
                        pred_name = agent.lt_mem.lexicon.d2s[(int(conc_ind), conc_type)][0][0]
                        # Split camelCased predicate name
                        splits = re.findall(
                            r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                        )
                        splits = [w[0].lower()+w[1:] for w in splits]
                        conc_nl = ' '.join(splits)

                        if pr_val is not None and pr_val >= 0.5:
                            # Had right reason, but wasn't confident enough
                            reason_prefix = "this might not be a "

                            # Demonstrative "this" refers to the (potential) part
                            dem_ref = evd_atom.args[0][0]

                        else:
                            # Wasn't aware of any instance of the potential explanans
                            # template in the scene

                            # This will need a more principled treatment if we ever get
                            # to handle relations other than "have"
                            reason_prefix = "this doesn't have a "

                            # Demonstrative "this" refers to the whole object
                            dem_ref = tgt_lit.args[0][0]

                            # Avoid redundancy
                            if conc_nl in reasons: continue

                        reasons[conc_nl] = (reason_prefix, dem_ref)

                    for i, (conc_nl, (reason_prefix, dem_ref)) in enumerate(reasons.items()):
                        # Compose NL explanation utterance string

                        # Append a suffix appropriate for the number of reasons
                        if i == len(reasons)-1:
                            # Last entry, period
                            reason_suffix = "."
                        elif i == len(reasons)-2:
                            # Next to last entry, comma+and
                            reason_suffix = ", and "
                        else:
                            # Otherwise, comma
                            reason_suffix = ", "

                        reason_nl = f"{reason_prefix}{conc_nl}{reason_suffix}"
                        answer_nl += reason_nl

                        # Add demonstrative reference and shift offset
                        dem_refs[(dem_offset, dem_offset+4)] = \
                                agent.lang.dialogue.referents["env"][dem_ref]["mask"]
                        dem_offset += len(reason_nl)

                    # Push the translated answer to buffer of utterances to generate; won't
                    # care for logical form, doesn't matter much now
                    agent.lang.dialogue.to_generate.append(
                        (None, answer_nl, dem_refs)
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
        # Don't know how to handle other non-domain questions
        raise NotImplementedError

def _search_specs_from_kb(agent, question, restrictors, ref_reg_gr):
    """
    Helper method factored out for extracting specifications for visual search,
    based on the agent's current knowledge-base entries and some sensemaking
    result provided as a compiled region graph
    """
    q_vars, (cons, _) = question

    # Prepare taxonomy graph extracted from KB taxonomy entries; used later for
    # checking any concepts groupable by closest common supertypes
    taxonomy_graph = nx.DiGraph()
    for (r_cons, r_ante), _, _, knowledge_type in agent.lt_mem.kb.entries:
        if knowledge_type == "taxonomy":
            taxonomy_graph.add_edge(r_cons[0].name, r_ante[0].name)

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
                if agent.lt_mem.kb.entries[entry_id][3] != "taxonomy"
                    # Not using taxonomy knowledge as leverage for visual reasoning
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

            final_specs.append((search_vars, lits, {}))

    # See if any of the search specs can be grouped and combined by closest common
    # supertypes; continue grouping pairs of specs with matching signatures and
    # shared supertypes as much as possible
    grouping_finished = False; disj_index = 0
    while not grouping_finished:
        for si, sj in combinations(range(len(final_specs)), 2):
            # If not isomorphic after replacing all cls predicates with the same
            # dummy predicate, the pair is not groupable
            is_cls_si = [
                lit.name.startswith("cls_") or lit.name.startswith("disj_")
                for lit in final_specs[si][1]
            ]
            is_cls_sj = [
                lit.name.startswith("cls_")  or lit.name.startswith("disj_")
                for lit in final_specs[sj][1]
            ]
            spec_subs_si = [
                lit.substitute(preds={ lit.name: "cls_dummy" }) if is_cls else lit
                for lit, is_cls in zip(final_specs[si][1], is_cls_si)
            ]
            spec_subs_sj = [
                lit.substitute(preds={ lit.name: "cls_dummy" }) if is_cls else lit
                for lit, is_cls in zip(final_specs[sj][1], is_cls_sj)
            ]
            entail_dir, mapping = Literal.entailing_mapping_btw(spec_subs_si, spec_subs_sj)
            if entail_dir != 0: continue

            # If there is no one-to-one correspondence between the cls predicates
            # such that the predicates in each pair belong to the same taxonomy
            # tree, the pair is not groupable
            assert sum(is_cls_si) == sum(is_cls_sj)
            cls_inds_si = [i for i, is_cls in enumerate(is_cls_si) if is_cls]
            cls_inds_sj = [i for i, is_cls in enumerate(is_cls_sj) if is_cls]

            valid_bijections = []
            for prm in permutations(cls_inds_sj):
                # Obtain bijection between cls literals in the first spec and the second
                bijection = [(cls_inds_si[i], i_sj) for i, i_sj in enumerate(prm)]
                matched_lits = [
                    (final_specs[si][1][i_si], final_specs[sj][1][i_sj])
                    for i_si, i_sj in bijection
                ]
                matched_lits = [
                    (lit_si, lit_sj.substitute(**mapping))
                    for lit_si, lit_sj in matched_lits
                ]

                # Reject bijection if any of the pairs do not have matching args
                if any(lit_si.args!=lit_sj.args for lit_si, lit_sj in matched_lits):
                    continue

                # Find closest common supertypes for each matched pair in bijection
                grouping_supertypes = []
                for lit_si, lit_sj in matched_lits:
                    # Predicate names won't be same; duplicate isomorphic specs are
                    # filtered out above
                    assert lit_si.name != lit_sj.name

                    cls_conc_si = lit_si.name if lit_si.name.startswith("cls_") \
                        else final_specs[si][2][lit_si.name][0]
                    cls_conc_sj = lit_sj.name if lit_sj.name.startswith("cls_") \
                        else final_specs[sj][2][lit_sj.name][0]
                    closest_common_supertype = nx.lowest_common_ancestor(
                        taxonomy_graph, cls_conc_si, cls_conc_sj
                    )

                    if closest_common_supertype is None:
                        # Not in the same taxonomy tree, cannot group
                        break
                    else:
                        # Common supertype identified, record relevant info
                        elem_concs_si = {lit_si.name} if lit_si.name.startswith("cls_") \
                            else final_specs[si][2][lit_si.name][1]
                        elem_concs_sj = {lit_sj.name} if lit_sj.name.startswith("cls_") \
                            else final_specs[sj][2][lit_sj.name][1]
                        grouping_supertypes.append(
                            (closest_common_supertype, elem_concs_si | elem_concs_sj)
                        )

                # Add to list of valid bijections if supertypes successfully identified
                # for all matched pairs
                if len(grouping_supertypes) == len(bijection):
                    valid_bijections.append((bijection, grouping_supertypes))

            if len(valid_bijections) == 0: continue

            # If reached here, update the spec list accordingly by replacing the two
            # specs being processed with new grouped specs (possibly multiple, in
            # principle -- though we won't see such cases in our scope)
            grouped_specs = []
            for bijection, grouping_supertypes in valid_bijections:
                # (Arbitrarily) Select the first spec to be the 'base' of the new grouped
                # spec to be appended
                search_vars, lits, _ = final_specs[si]

                # Dict describing which set of elementary concepts is referred to by
                # each disjunction predicate
                pred_glossary = {}

                # Bijection info reshaped for easier processing
                bijection = {
                    i_si: (i_sj, gr_info)
                    for (i_si, i_sj), gr_info in zip(bijection, grouping_supertypes)
                }

                # Prepare new literal set for search spec description, starting from
                # the base and appropriately replacing the predicate names
                lits_new = []
                for i_si, lit in enumerate(lits):
                    if i_si in bijection:
                        # Need to be processed before being added to the literal set
                        i_sj, grouping_info = bijection[i_si]

                        lit_si = final_specs[si][1][i_si]
                        lit_sj = final_specs[sj][1][i_sj]
                        is_elem_si = lit_si.name.startswith("cls_")
                        is_elem_sj = lit_sj.name.startswith("cls_")

                        if is_elem_si and is_elem_sj:
                            # Case 1: Both predicates elementary concepts, need to
                            # introduce a fresh disjunction predicate
                            disj_name = f"disj_{disj_index}"
                            disj_index += 1
                        elif is_elem_si and not is_elem_sj:
                            # Case 2: First predicate refers to elementary concept, while
                            # second refers to a disjunction; 'absorb' former to latter
                            disj_name = lit_sj.name
                        elif not is_elem_si and is_elem_sj:
                            # Symmetric case of the above Case 2, treat similarly
                            disj_name = lit_si.name
                        else:
                            # Case 3: Both predicates refer to disjunctions; (arbitrarily)
                            # select the first to absorb the second
                            disj_name = lit_si.name

                        pred_glossary[disj_name] = grouping_info
                        lits_new.append(Literal(disj_name, lit.args))

                    else:
                        # Nothing to do, add as-is
                        lits_new.append(lit)

                grouped_specs.append((search_vars, lits_new, pred_glossary))

            # Don't forget to take the remaining specs not being processed
            final_specs = [
                spc for i, spc in enumerate(final_specs) if i not in (si, sj)
            ]
            final_specs += grouped_specs

            # Then break to find any possible grouping, from the top with the updated list
            break

        else:
            # No more groupable spec pairs; terminate while loop
            grouping_finished = True

    # # Check if the agent is already (visually) aware of the potential search
    # # targets; if so, disregard this one
    # check_result, _ = agent.symbolic.query(
    #     ref_reg_gr, tuple((v, False) for v in search_vars), (lits, None)
    # )
    # if len(check_result) > 0:
    #     continue

    return final_specs


def _find_explanans_templates(tgt_lit, kb_prog_analyzed):
    """
    Helper method factored out for finding templates of potential explanantia
    (causal chains possibly not fully grounded, which could raise the possibility
    of the target explanandum when appropriately grounded), based on the inference
    program exported from some specific version of (exported) KB.

    Start from rules containing the target event predicate and continue spanning
    along (against?) the abductive direction until all potential evidence atoms
    are identified.
    """
    exps_templates = [[tgt_lit]]; frontier = {tgt_lit}
    while len(frontier) > 0:
        expd_atm = frontier.pop()       # Atom representing explanandum event

        for rule_info in kb_prog_analyzed.values():
            # Disregard rules without abductive force
            if not rule_info.get("abductive"): continue

            # Check if rule is relevant; i.e. whether the popped explanandum
            # event is included in rule antecedent
            entail_dir, mapping = Literal.entailing_mapping_btw(
                rule_info["ante"], [expd_atm]
            )

            if entail_dir is not None and entail_dir >= 0:
                # Rule relevant, target event may be abduced from consequent
                r_cons_subs = [
                    c_lit.substitute(**mapping) for c_lit in rule_info["cons"]
                ]

                # Add the whole substituted consequent to the list of valid
                # explanans templates
                exps_templates.append(r_cons_subs)

                # Add each literal in the substituted consequent to the search
                # frontier
                frontier |= set(r_cons_subs)

    return exps_templates

def _instantiate_templates(exps_templates, scene_ents):
    """
    Helper method factored out for enumerating every possible instantiations of
    the provided explanans templates with scene entities
    """
    exps_instances = []

    for conjunction in exps_templates:
        # Variables and constants occurring in the template conjunction
        occurring_consts = {
            arg for c_lit in conjunction for arg, is_var in c_lit.args
            if not is_var
        }
        occurring_vars = {
            arg for c_lit in conjunction for arg, is_var in c_lit.args
            if is_var
        }
        occurring_vars = tuple(occurring_vars)      # Int indexing
        remaining_ents = scene_ents - occurring_consts

        # All possible substitutions for the remaining variables; permutations()
        # will give empty list if len(occurring_vars) is larger than len(remaining_ents)
        possible_remaining_subs = permutations(remaining_ents, len(occurring_vars))
        for subs in possible_remaining_subs:
            subs = { (occurring_vars[i], True): (e, False) for i, e in enumerate(subs) }
            instance = [c_lit.substitute(terms=subs) for c_lit in conjunction]
            exps_instances.append(instance)

    return exps_instances
