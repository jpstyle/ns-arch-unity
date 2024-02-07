"""
symbolic reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base.

Implements 'sensemaking' process; that is, process of integrating and re-organizing
perceived information from various modalities to establish a set of judgements structured
in such a manner that they can later be exploited for other symbolic reasoning tasks --
in light of the existing general knowledge held by the perceiving agent.

(I borrow the term 'sensemaking' from the discipline of symbolic science & psychology.
According to Klein (2006), sensemaking is "the process of creating situational awareness
and understanding in situations of high complexity or uncertainty in order to make decisions".)

Here, we resort to declarative programming to encode individual sensemaking problems into
logic programs (written in the language of weighted ASP), which are solved with a belief
propagation method.
"""
from .query import query
from .attribute import attribute
from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import wrap_args, flatten_cons_ante


TAB = "\t"              # For use in format strings

EPS = 1e-10             # Value used for numerical stabilization
U_IN_PR = 0.99          # How much the agent values information provided by the user

class SymbolicReasonerModule:

    def __init__(self):
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = {}

        self.value_assignment = {}    # Store best assignments (tentative) obtained by reasoning
        self.word_senses = {}         # Store current estimate of symbol denotations

        self.mismatches = []

    def refresh(self):
        self.__init__()

    def sensemake_vis(self, exported_kb, visual_evidence):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) with existing knowledge to make final verdicts on the state of
        affairs, 'all things considered'.

        Args:
            exported_kb: Output from KnowledgeBase().export_reasoning_program()
            visual_evidence: Output from KnowledgeBase().visual_evidence_from_scene()
        """
        # Solve to find the best models of the program
        prog = exported_kb + visual_evidence
        reg_gr_v = prog.compile()

        # Store sensemaking result as module state
        self.concl_vis = reg_gr_v, (exported_kb, visual_evidence)
    
    def resolve_symbol_semantics(self, dialogue_state, lexicon):
        """
        Find a fully specified mapping from symbols in discourse record to corresponding
        entities and concepts; in other words, perform reference resolution and word
        sense disambiguation.

        Args:
            dialogue_state: Current dialogue information state exported from the dialogue
                manager
            lexicon: Agent's lexicon, required for matching between environment entities
                vs. discourse referents for variable assignment
        """
        # Find the best estimate of referent value assignment
        sm_prog = Program()

        # Environmental referents
        occurring_atoms = set()
        for ent in dialogue_state.referents["env"]:
            if ent not in occurring_atoms:
                sm_prog.add_absolute_rule(Rule(head=Literal("env", wrap_args(ent))))
                occurring_atoms.add(ent)

        # Discourse referents
        for rf, v in dialogue_state.referents["dis"].items():
            # No need to assign if universally quantified or wh-quantified
            if v["univ_quantified"] or v["wh_quantified"]: continue
            # No need to assign if not an entity referent
            if not rf.startswith("x"): continue

            sm_prog.add_absolute_rule(Rule(head=Literal("dis", wrap_args(rf))))
            if v["referential"]:
                sm_prog.add_absolute_rule(Rule(head=Literal("referential", wrap_args(rf))))

        # Hard assignments by pointing, etc.
        for ref, env in dialogue_state.assignment_hard.items():
            sm_prog.add_absolute_rule(
                Rule(body=[Literal("assign", [(ref, False), (env, False)], naf=True)])
            )

        # Add priming effect by recognized visual concepts
        if self.concl_vis is not None:
            reg_gr_v, _ = self.concl_vis
            ## TODO: Update to comply with the recent changes
            # marginals_v = reg_gr_v.compute_marginals()

            # if marginals_v is not None:
            #     vis_concepts = defaultdict(float)
            #     for atom, pr in marginals_v.items():
            #         is_cls = atom.name.startswith("cls")
            #         is_att = atom.name.startswith("att")
            #         is_rel = atom.name.startswith("rel")
            #         if is_cls or is_att or is_rel:
            #             # Collect 'priming intensity' by recognized concepts
            #             vis_concepts[atom.name] += pr[0]

            #     for vc, score in vis_concepts.items():
            #         sm_prog.add_absolute_rule(Rule(
            #             head=Literal("vis_prime", wrap_args(vc, int(score * 100)))
            #         ))

        # Recursive helper methods for extracting predicates and args & flattening
        # nested lists with arbitrary depths into a single list (along with pointer
        # to source location)
        extract_preds = lambda cnjt: cnjt[:2] if isinstance(cnjt, tuple) \
            else [extract_preds(nc) for nc in cnjt]
        extract_args = lambda cnjt: cnjt[2] if isinstance(cnjt, tuple) \
            else [extract_args(nc) for nc in cnjt]
        def flatten(ls):
            for ind, x in enumerate(ls):
                if isinstance(x, list):
                    yield from ((inds+(ind,), x2) for inds, x2 in flatten(x))
                else:
                    yield (ind,), x

        # Understood dialogue record contents
        occurring_preds = set()
        for ti, (speaker, turn_clauses) in enumerate(dialogue_state.record):
            # Nothing particular to do with agent's own utterances
            if speaker == "A": continue

            for ci, ((rule, question), _) in enumerate(turn_clauses):
                if rule is not None:
                    cons, ante = rule

                    cons_preds = [extract_preds(c) for c in cons]
                    ante_preds = [extract_preds(a) for a in ante]

                    # Symbol token occurrence locations
                    for c, preds in [("c", cons_preds), ("a", ante_preds)]:
                        for src, p in flatten(preds):
                            # Skip special reserved predicates
                            if p[1] == "*": continue

                            occurring_preds.add(p)

                            # Serialized source location
                            src_loc = "_".join(str(i) for i in src)

                            sym = f"{p[1]}_{p[0].split('/')[0]}"
                            tok_loc = f"t{ti}_c{ci}_r{c}_{src_loc}"
                            sm_prog.add_absolute_rule(
                                Rule(head=Literal("pred_token", wrap_args(tok_loc, sym)))
                            )

                    cons_args = [extract_args(c) for c in cons]
                    ante_args = [extract_args(a) for a in ante]
                    occurring_ent_refs = sum([a for _, a in flatten(cons_args+ante_args)], ())
                    occurring_ent_refs = sum([
                        (ref,) if isinstance(ref, str) else ref[1]
                        for ref in occurring_ent_refs
                    ], ())
                    # Only referents starting with "x" refer to environment entities
                    occurring_ent_refs = {
                        ref for ref in occurring_ent_refs if ent.startswith("x")
                    }

                    if all(arg in dialogue_state.assignment_hard for arg in occurring_ent_refs):
                        # Below not required if all occurring args are hard-assigned to some entity
                        continue

                    # If reg_gr_v is present and rule is grounded, add bias in favor of
                    # assignments which would satisfy the rule
                    is_grounded = all(not arg[0].isupper() for arg in occurring_ent_refs)
                    if self.concl_vis is not None and is_grounded:
                        # TODO: Update to comply with the recent changes... When coreference
                        # resolution becomes a serious issue to address
                        pass

                        # # Rule instances by possible word sense selections
                        # wsd_cands = [lexicon.s2d[sym[:2]] for sym in head_preds+body_preds]

                        # for vcs in product(*wsd_cands):
                        #     if head is not None:
                        #         head_vcs = vcs[:len(head_preds)]
                        #         rule_head = [
                        #             Literal(f"{vc[1]}_{vc[0]}", wrap_args(*h[2]), naf=h[3])
                        #             for h, vc in zip(head, head_vcs)
                        #         ]
                        #     else:
                        #         rule_head = None

                        #     if body is not None:
                        #         body_vcs = vcs[len(head_preds):]
                        #         rule_body = [
                        #             Literal(f"{vc[1]}_{vc[0]}", wrap_args(*b[2]), naf=b[3])
                        #             for b, vc in zip(body, body_vcs)
                        #         ]
                        #     else:
                        #         rule_body = None

                        #     # Question to query models_v with
                        #     q_rule = Rule(head=rule_head, body=rule_body)
                        #     q_vars = tuple((a, False) for a in occurring_ent_refs)
                        #     query_result = models_v.query(q_vars, q_rule)

                        #     for ans, (_, score) in query_result.items():
                        #         c_head = Literal("cons", wrap_args(f"r{ri}", int(score*100)))
                        #         c_body = [
                        #             Literal("assign", wrap_args(x[0], e)) for x, e in zip(q_vars, ans)
                        #         ]
                        #         if head is not None:
                        #             c_body += [
                        #                 Literal("denote", wrap_args(f"u{ui}_r{ri}_h{hi}", f"{d[1]}_{d[0]}"))
                        #                 for hi, d in enumerate(head_vcs)
                        #             ]
                        #         if body is not None:
                        #             c_body += [
                        #                 Literal("denote", wrap_args(f"u{ui}_r{ri}_b{bi}", f"{d[1]}_{d[0]}"))
                        #                 for bi, d in enumerate(body_vcs)
                        #             ]

                        #         sm_prog.add_absolute_rule(Rule(head=c_head, body=c_body))
                
                if question is not None:
                    _, (cons, ante) = question

                    cons_preds = [extract_preds(c) for c in cons]
                    ante_preds = [extract_preds(a) for a in ante]

                    # Symbol token occurrence locations
                    for c, preds in [("c", cons_preds), ("a", ante_preds)]:
                        for src, p in flatten(preds):
                            # Skip special reserved predicates
                            if p[1] == "*": continue

                            occurring_preds.add(p)

                            # Serialized source location
                            src_loc = "_".join(str(i) for i in src)

                            sym = f"{p[1]}_{p[0].split('/')[0]}"
                            tok_loc = f"t{ti}_c{ci}_q{c}_{src_loc}"
                            sm_prog.add_absolute_rule(
                                Rule(head=Literal("pred_token", wrap_args(tok_loc, sym)))
                            )

        # Predicate info needed for word sense selection
        for p in occurring_preds:
            # Skip special reserved predicates
            if p[1] == "*": continue

            sym = f"{p[1]}_{p[0].split('/')[0]}"

            # Consult lexicon to list denotation candidates
            if p in lexicon.s2d:
                for vc in lexicon.s2d[p]:
                    pos_match = (p[1], vc[1]) == ("n", "cls") \
                        or (p[1], vc[1]) == ("a", "att") \
                        or (p[1], vc[1]) == ("r", "rel") \
                        or (p[1], vc[1]) == ("v", "rel")
                    if not pos_match: continue

                    den = f"{vc[1]}_{vc[0]}"

                    sm_prog.add_absolute_rule(
                        Rule(head=Literal("may_denote", wrap_args(sym, den)))
                    )
                    sm_prog.add_absolute_rule(
                        Rule(head=Literal("d_freq", wrap_args(den, lexicon.d_freq[vc])))
                    )
            else:
                # Predicate symbol not found in lexicon: unresolved neologism
                sm_prog.add_absolute_rule(
                    Rule(head=Literal("may_denote", wrap_args(sym, "_neo")))
                )

        ## Assignment program rules

        # 1 { assign(X,E) : env(E) } 1 :- dis(X), referential(X).
        sm_prog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"),
                conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"))
            ],
            lb=1, ub=1
        ))

        # { assign(X,E) : env(E) } 1 :- dis(X), not referential(X).
        sm_prog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"),
                conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"), naf=True)
            ],
            ub=1
        ))

        # 1 { denote(T,D) : may_denote(S,D) } 1 :- pred_token(T,S).
        sm_prog.add_rule(Rule(
            head=Literal(
                "denote", wrap_args("T", "D"),
                conds=[Literal("may_denote", wrap_args("S", "D"))]
            ),
            body=[
                Literal("pred_token", wrap_args("T", "S"))
            ],
            lb=1, ub=1
        ))

        # 'Base cost' for cases where no assignments are any better than others
        sm_prog.add_absolute_rule(Rule(head=Literal("zero_p", [])))

        # By querying for the optimal assignment, essentially we are giving the user a 'benefit
        # of doubt', such that any statements made by the user are considered as True, and the
        # agent will try to find the 'best' assignment to make it so.
        # (Note: this is not a probabilistic inference, and the confidence scores provided as 
        # arguments are better understood as properties of the env. entities & disc. referents.)
        opt_models = sm_prog.optimize([
            # (Note: Earlier statements receive higher optimization priority)
            # Prioritize assignments that agree with given statements
            ("maximize", [
                ([Literal("zero_p", [])], "0", []),
                ([Literal("cons", wrap_args("RI", "S"))], "S", ["RI"])
            ]),
            # Prioritize word senses that occur in visual scene (if any): 'priming effect'
            ("maximize", [
                ([
                    Literal("denote", wrap_args("T", "D")),
                    Literal("vis_prime", wrap_args("D", "S"))
                ], "S", ["T"])
            ]),
            # Prioritize word senses with higher frequency
            ("maximize", [
                ([
                    Literal("denote", wrap_args("T", "D")),
                    Literal("d_freq", wrap_args("D", "F"))
                ], "F", ["T"])
            ])
        ])

        best_assignment = [atom.args for atom in opt_models[0] if atom.name == "assign"]
        best_assignment = {args[0][0]: args[1][0] for args in best_assignment}

        tok2sym_map = [atom.args[:2] for atom in opt_models[0] if atom.name == "pred_token"]
        tok2sym_map = {
            tuple(token[0].split("_")): tuple(symbol[0].split("_"))
            for token, symbol in tok2sym_map
        }

        word_senses = [atom.args[:2] for atom in opt_models[0] if atom.name == "denote"]
        word_senses = {
            tuple(token[0].split("_")): denotation[0]
            for token, denotation in word_senses
        }
        word_senses = {
            token: (tok2sym_map[token], denotation)
                if denotation != "_neo" else (tok2sym_map[token], None)
            for token, denotation in word_senses.items()
        }

        self.value_assignment.update(best_assignment)
        self.word_senses.update(word_senses)

    def translate_dialogue_content(self, dialogue_state):
        """
        Translate logical content of dialogue record (which should be already
        ASP- compatible) based on current estimate of value assignment and word
        sense selection. Dismiss (replace with None) any utterances containing
        unresolved neologisms.
        """
        a_map = lambda args: [self.value_assignment.get(arg, arg) for arg in args]

        # Recursive helper methods for encoding pre-translation tuples representing
        # literals into actual Literal objects
        encode_lits = lambda cnjt, ti, ci, rqca, inds: Literal(
                self.word_senses.get(
                    (f"t{ti}",f"c{ci}",rqca)+tuple(str(i) for i in inds),
                    # If not found (likely reserved predicate), fall back to cnjt's pred
                    (None, "_".join(cnjt[1::-1]))
                )[1],
                args=wrap_args(*a_map(cnjt[2])), naf=cnjt[3]
            ) \
            if isinstance(cnjt, tuple) \
            else [encode_lits(nc, ti, ci, rqca, inds+(i,)) for i, nc in enumerate(cnjt)]

        record_translated = []
        for ti, (speaker, turn_clauses) in enumerate(dialogue_state.record):
            turn_translated = []
            for ci, ((rule, question), raw) in enumerate(turn_clauses):
                # If the utterance contains an unresolved neologism, give up translation
                # for the time being
                contains_unresolved_neologism = any([
                    den is None for tok, (_, den) in self.word_senses.items()
                    if tok[:2]==(f"t{ti}", f"c{ci}")
                ])
                if contains_unresolved_neologism:
                    turn_translated.append(((None, None), raw))
                    continue

                # Translate rules
                if rule is not None:
                    cons, ante = rule

                    if len(cons) > 0:
                        tr_cons = tuple(
                            encode_lits(c_lit, ti, ci, "rc", (rci,))
                            for rci, c_lit in enumerate(cons)
                        )
                    else:
                        tr_cons = None

                    if len(ante) > 0:
                        tr_ante = tuple(
                            encode_lits(a_lit, ti, ci, "ra", (rai,))
                            for rai, a_lit in enumerate(ante)
                        )
                    else:
                        tr_ante = None

                    translated_rule = (tr_cons, tr_ante)
                else:
                    translated_rule = None
                
                # Translate question
                if question is not None:
                    q_vars, (cons, ante) = question

                    if len(cons) > 0:
                        tr_cons = tuple(
                            encode_lits(c_lit, ti, ci, "qc", (qci,))
                            for qci, c_lit in enumerate(cons)
                        )
                    else:
                        tr_cons = None

                    if len(ante) > 0:
                        tr_ante = tuple(
                            encode_lits(a_lit, ti, ci, "qa", (qai,))
                            for qai, a_lit in enumerate(ante)
                        )
                    else:
                        tr_ante = None

                    translated_question = q_vars, (tr_cons, tr_ante)
                else:
                    translated_question = None

                turn_translated.append(((translated_rule, translated_question), raw))

            record_translated.append((speaker, turn_translated))

        return record_translated

    def sensemake_vis_lang(self, dialogue_state):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) and the current dialogue information state with existing knowledge
        to make final verdicts on the state of affairs, 'all things considered'.

        Args:
            dialogue_state: Current dialogue information state exported from the dialogue
                manager
        """
        dl_prog = Program()
        reg_gr_v, (pr_prog, kb_prog) = self.concl_vis

        # Incorporate additional information provided by the user in language for updated
        # sensemaking
        translated = self.translate_dialogue_content(dialogue_state)
        for ti, (speaker, turn_clauses) in enumerate(translated):
            if speaker != "U": continue

            for ci, ((rule, _), _) in enumerate(turn_clauses):
                # Disregard clause if it is not domain-describing or is in irrealis mood
                clause_info = dialogue_state.clause_info[f"t{ti}c{ci}"]
                if not clause_info["domain_describing"]: continue
                if clause_info["irrealis"]: continue

                if rule is not None:
                    for cons, ante in flatten_cons_ante(*rule):
                        # Skip any non-grounded content
                        cons_has_var = len(cons) > 0 and any([
                            any(is_var for _, is_var in c.args) for c in cons
                        ])
                        ante_has_var = len(ante) > 0 and any([
                            any(is_var for _, is_var in a.args) for a in ante
                        ])
                        if cons_has_var or ante_has_var: continue

                        # Skip any rules with non-entity referents
                        cons_has_pred_ent = len(cons) > 0 and any([
                            any(arg.startswith("p") for arg, _ in c.args) for c in cons
                        ])
                        ante_has_pred_ent = len(ante) > 0 and any([
                            any(arg.startswith("p") for arg, _ in a.args) for a in ante
                        ])
                        if cons_has_pred_ent or ante_has_pred_ent: continue

                        if len(cons) > 0:
                            # One ASP rule per cons
                            for cl in cons:
                                dl_prog.add_rule(Rule(head=cl, body=ante), U_IN_PR)
                        else:
                            # Cons-less; single constraint
                            dl_prog.add_rule(Rule(body=ante), U_IN_PR)

        # Finally, reasoning with all visual+language info
        if len(dl_prog) > 0:
            prog = pr_prog + kb_prog + dl_prog
            reg_gr_vl = prog.compile()
        else:
            reg_gr_vl = reg_gr_v

        # Store sensemaking result as module state
        self.concl_vis_lang = reg_gr_vl, (pr_prog, kb_prog, dl_prog)

    @staticmethod
    def query(reg_gr, q_vars, event, restrictors=None):
        return query(reg_gr, q_vars, event, restrictors or {})

    @staticmethod
    def attribute(reg_gr, target_event, evidence, competing_evts, vetos=None):
        return attribute(reg_gr, target_event, evidence, competing_evts, vetos)
