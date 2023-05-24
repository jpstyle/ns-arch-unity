import re
from collections import defaultdict

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import wrap_args, flatten_cons_ante


P_C = 0.01          # Catchall hypothesis probability

class KnowledgeBase:
    """
    Knowledge base containing pieces of knowledge in the form of LP^MLN (weighted
    ASP) rules
    """
    def __init__(self):
        # Knowledge base entries stored as collections of antecedent-consequent pairs
        # containing generically quantified variables; each collection corresponds to
        # a batch (i.e. conjunction) of generic rules extracted from the same provenance
        # (utterances from teacher, etc.), associated with a shared probability weight
        # value between 0 ~ 1.
        self.entries = []

        # Indexing entries by contained predicates
        self.entries_by_pred = defaultdict(set)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"KnowledgeBase(len={len(self)})"

    def __contains__(self, item):
        """ Test if an entry isomorphic to item exists """
        cons, ante = item

        for (ent_cons, ent_ante), _ in self.entries:
            # Don't even bother with different set sizes
            if len(cons) != len(ent_cons): continue
            if len(ante) != len(ent_ante): continue

            cons_isomorphic = Literal.isomorphic_conj_pair(cons, ent_cons)
            ante_isomorphic = Literal.isomorphic_conj_pair(ante, ent_ante)

            if cons_isomorphic and ante_isomorphic:
                return True

        return False

    def add(self, rule, weight, source):
        """ Returns whether KB is expanded or not """
        cons, ante = rule

        # Neutralizing variable & function names by stripping off turn/clause
        # indices, etc.
        rename_var = {
            a for a, _ in _flatten(_extract_terms(cons+ante))
            if isinstance(a, str)
        }
        rename_var = {
            (vn, True): (re.match("(.+?)(t\d+c\d+)?$", vn).group(1), True)
            for vn in rename_var
        }
        rename_fn = {
            a[0] for a, _ in _flatten(_extract_terms(cons+ante))
            if isinstance(a, tuple)
        }
        rename_fn = {
            fn: re.match("(.+?)(_.+)?$", fn).group(1)+f"_{i}"
            for i, fn in enumerate(rename_fn)
        }
        cons = tuple(_substitute(c, rename_var, rename_fn) for c in cons)
        ante = tuple(_substitute(a, rename_var, rename_fn) for a in ante)

        preds_cons = set(_flatten(_extract_preds(cons)))
        preds_ante = set(_flatten(_extract_preds(ante)))

        # Check if the input knowledge can be logically entailed by some existing
        # KB entry (or is already contained in the KB). For now, just check
        # simple one-step entailments (by A,B |- A).

        # Initial filtering of irrelevant entries without any overlapping for
        # both cons and ante
        entries_with_overlap = set.union(*[
            self.entries_by_pred.get(pred, set()) for pred in preds_cons
        ]) & set.union(*[
            self.entries_by_pred.get(pred, set()) for pred in preds_ante
        ])

        entries_entailed = set()       # KB entries entailed by input
        entries_entailing = set()      # KB entries that entail input
        for ent_id in entries_with_overlap:
            (ent_cons, ent_ante), ent_weight, _ = self.entries[ent_id]

            # Find (partial) term mapping between the KB entry and input with
            # which they can unify
            mapping_a, ent_dir_a = Literal.entailing_mapping_btw(
                ante, ent_ante
            )
            if mapping_a is not None:
                mapping_c, ent_dir_c = Literal.entailing_mapping_btw(
                    cons, ent_cons, mapping_a
                )
                if mapping_c is not None and {ent_dir_c, ent_dir_a} != {1, -1}:
                    # Entailment relation detected
                    if ent_dir_c >= 0 and ent_dir_a <= 0 and weight >= ent_weight:
                        entries_entailed.add(ent_id)
                    if ent_dir_c <= 0 and ent_dir_a >= 0 and weight <= ent_weight:
                        entries_entailing.add(ent_id)

        if len(entries_entailing) == len(entries_entailed) == 0:
            # Add the input as a whole new entry along with the weight & source
            # and index it by occurring predicates
            self.entries.append(((cons, ante), weight, [(source, weight)]))
            for pred in preds_cons | preds_ante:
                self.entries_by_pred[pred].add(len(self.entries)-1)
            
            kb_updated = True

        else:
            # Due to the invariant condition that there's no two KB entries such
            # that one is strictly stronger than the other, the input shouldn't be
            # entailed by some entries and entail others at the same time -- except
            # for the case of exact match.
            if len(entries_entailing) > 0 and len(entries_entailed) > 0:
                assert entries_entailing == entries_entailed
                # Equivalent entry exists; just add to provenance list
                for ent_id in entries_entailing:
                    self.entries[ent_id][2].append((source, weight))

                kb_updated = False

            else:
                if len(entries_entailed) > 0:
                    # Stronger input entails some KB entries and render them
                    # 'obsolete'; the entailed entries may be removed and merged
                    # into the newly added entry
                    self.remove_by_ids(entries_entailed)

                    # Add the stronger input as new entry
                    self.entries.append(
                        ((cons, ante), weight, [(source, weight)])
                    )
                    for pred in preds_cons | preds_ante:
                        self.entries_by_pred[pred].add(len(self.entries)-1)

                    kb_updated = True
                else:
                    assert len(entries_entailing) > 0
                    # Entry entailing the given input exists; just add to provenance list
                    for ent_id in entries_entailing:
                        self.entries[ent_id][2].append((source, weight))

                    kb_updated = False

        return kb_updated

    def remove_by_ids(self, ent_ids):
        """
        Update KB by removing entries designated by the list of entry ids provided
        """
        # First find the mapping from previous set of indices to new set of indices,
        # as indices will change as entries shift their positions to fill in blank
        # positions
        ind_map = {}; ni = 0
        for ent_id in range(len(self.entries)):
            if ent_id in ent_ids:
                ni += 1
            else:
                ind_map[ent_id] = ent_id - ni

        # Cull the specified entries and update entry indexing by predicate
        # (self.entries_by_pred) according to the mapping found above
        self.entries = [
            entry for ent_id, entry in enumerate(self.entries)
            if ent_id in ind_map
        ]
        self.entries_by_pred = defaultdict(set, {
            pred: {ind_map[ei] for ei in ent_ids if ei in ind_map}
            for pred, ent_ids in self.entries_by_pred.items()
        })

    def find_entailer_concepts(self, concepts):
        """
        Given a set of concepts, return a set of concepts that entail the conjunction
        of the provided concepts.
        
        (The current version is a hack implementation that assumes all concerned concepts
        are unary, and the query doesn't require more than single step entailments. If
        we want to implement a proper querying mechanism later, we may employ a logical
        reasoning package like clingo...)
        """
        assert isinstance(concepts, set)

        if len(self.entries) > 0:
            entailment_rules = [
                (cons, ante) for (cons, ante), _, _ in self.entries
                if any(lit.name in concepts for lit in cons)
            ]

            candidate_preds = defaultdict(set)
            for cons, ante in entailment_rules:
                # For now will just consider single concepts as candidates: i.e., those
                # which can entail the target concepts in its own right without requiring
                # extra concepts
                if len(ante) > 1: continue
                candidate_preds[ante[0].name] |= {lit.name for lit in cons}

            # Filter to finally leave concepts that fully entail the target concepts and
            # return
            entailer_concepts = {
                pred for pred, entailed in candidate_preds.items()
                if entailed == concepts
            }
            return entailer_concepts

        else:
            # Empty KB, no entailing concepts
            return set()

    def export_reasoning_program(self):
        """
        Returns an ASP program that implements deductive & abductive reasonings by
        virtue of the listed entries
        """
        inference_prog = Program()

        # Add rules implementing deductive inference
        entries_by_cons, intermediate_outputs = \
            self._add_deductive_inference_rules(inference_prog)

        # Add rules implementing abductive inference
        self._add_abductive_inference_rules(
            inference_prog, entries_by_cons, intermediate_outputs
        )

        # Set of predicates that warrant consideration as possibility even with score
        # below the threshold because they're mentioned in KB
        preds_in_kb = set(self.entries_by_pred)
        preds_in_kb = {
            conc_type: {
                int(name.strip(f"{conc_type}_")) for name in preds_in_kb
                if name.startswith(conc_type)
            }
            for conc_type in ["cls", "att", "rel"]
        }

        return inference_prog, preds_in_kb

    def _add_deductive_inference_rules(self, inference_prog):
        """
        As self.export_reasoning_program() was getting too long, refactored code for
        deductive inference program synthesis from KB
        """
        # For collecting entries by same cons, so that abductive inference rules can
        # be implemented for each collection
        entries_by_cons = defaultdict(list)

        # For caching intermediate outputs assembled during the first (deductive) part
        # and reusing in the second (abductive) part
        intermediate_outputs = []

        # Process each entry
        for i, (rule, weight, _) in enumerate(self.entries):
            cons, ante = flatten_cons_ante(*rule)

            # Keep track of variable names used to avoid accidentally using
            # overlapping names for 'lifting' variables (see below)
            all_var_names = {
                v for v, _ in _flatten(_extract_terms(cons+ante))
                if isinstance(v, str)
            }

            # All function term args used in this rule
            all_fn_args = {
                fa for fa in _flatten(_extract_terms(cons+ante))
                if isinstance(fa[0], tuple)
            }

            # Attach unique identifier suffixes to function names, so that functions
            # from different KB entries can be distinguished; names are shared across
            # within entry
            all_fn_names = {fa[0][0] for fa in all_fn_args}
            fn_name_map = { fn: f"{fn}_{i}" for fn in all_fn_names }

            # Map for lifting function term to new variable arg term
            fn_lifting_map = {
                ((fn_name_map[fa[0][0]], fa[0][1]), fa[1]):
                    (f"X{i+len(all_var_names)}", True)
                for i, fa in enumerate(all_fn_args)
            }

            rule_fn_subs = {
                "cons": [c.substitute(functions=fn_name_map) for c in cons],
                "ante": [a.substitute(functions=fn_name_map) for a in ante]
            }
            rule_lifted = {
                "cons": [c.substitute(terms=fn_lifting_map) for c in rule_fn_subs["cons"]],
                "ante": [a.substitute(terms=fn_lifting_map) for a in rule_fn_subs["ante"]]
            }

            # List of unique non-function variable arguments in 1) rule cons and 2) rule ante
            # (effectively whole rule) in the order of occurrence
            c_var_signature = []; a_var_signature = []
            for hl in rule_fn_subs["cons"]:
                for v_val, _ in hl.nonfn_terms():
                    if v_val not in c_var_signature: c_var_signature.append(v_val)
            for bl in rule_fn_subs["ante"]:
                for v_val, _ in bl.nonfn_terms():
                    if v_val not in a_var_signature: a_var_signature.append(v_val)

            # Rule cons/ante satisfaction flags literals
            c_sat_lit = Literal(f"cons_sat_{i}", wrap_args(*c_var_signature))
            a_sat_lit = Literal(f"ante_sat_{i}", wrap_args(*a_var_signature))

            # Flag literal is derived when cons/ante is satisfied; in the meantime, lift
            # occurrences of function terms and add appropriate function value assignment
            # literals
            c_sat_conds_pure = [        # Conditions having only 'pure' non-function args
                lit for lit in rule_lifted["cons"]
                if all(arg[0] in c_var_signature for arg in lit.args)
            ]
            c_fn_terms = set.union(*[
                {arg for arg in l.args if type(arg[0])==tuple} for l in rule_fn_subs["cons"]
            ]) if len(rule_fn_subs["cons"]) > 0 else set()
            c_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in c_fn_terms
            ]

            a_sat_conds_pure = [
                lit for lit in rule_lifted["ante"]
                if all(arg[0] in a_var_signature for arg in lit.args)
            ]
            a_fn_terms = set.union(*[
                {arg for arg in l.args if type(arg[0])==tuple} for l in rule_fn_subs["ante"]
            ])
            a_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in a_fn_terms
            ]

            if len(c_sat_conds_pure+c_fn_assign) > 0:
                # Skip cons-less rules
                inference_prog.add_absolute_rule(
                    Rule(head=c_sat_lit, body=c_sat_conds_pure+c_fn_assign)
                )
            inference_prog.add_absolute_rule(
                Rule(head=a_sat_lit, body=a_sat_conds_pure+a_fn_assign)
            )

            # Indexing & storing the entry by cons for later abductive rule
            # translation (thus, no need to consider cons-less constraints)
            if len(cons) > 0:
                for c_lits in entries_by_cons:
                    ism, ent_dir = Literal.entailing_mapping_btw(cons, c_lits)
                    if ent_dir == 0:
                        entries_by_cons[c_lits].append((i, ism))
                        break
                else:
                    entries_by_cons[frozenset(cons)].append((i, None))

            # Choice rule for function value assignments
            def add_assignment_choices(fn_terms, sat_conds):
                for ft in fn_terms:
                    # Function arguments and function term lifted
                    ft_lifted = fn_lifting_map[ft]

                    # Filter relevant conditions for filtering options worth considering
                    rel_conds = [cl for cl in sat_conds if ft_lifted in cl.args]
                    inference_prog.add_absolute_rule(
                        Rule(
                            head=Literal(
                                f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[ft_lifted]
                            ),
                            body=rel_conds
                        )
                    )
            add_assignment_choices(c_fn_terms, rule_lifted["cons"])
            add_assignment_choices(a_fn_terms, rule_lifted["ante"])

            # Rule violation flag
            r_unsat_lit = Literal(f"deduc_viol_{i}", wrap_args(*a_var_signature))
            inference_prog.add_absolute_rule(Rule(
                head=r_unsat_lit,
                body=[a_sat_lit] + ([c_sat_lit.flip()] if len(cons) > 0 else [])
            ))
            
            # Add appropriately weighted rule for applying 'probabilistic pressure'
            # against deductive rule violation
            inference_prog.add_rule(Rule(body=r_unsat_lit), weight)

            # Store intermediate outputs for later reuse
            intermediate_outputs.append((
                c_sat_lit, a_sat_lit, c_var_signature, a_var_signature
            ))

        return entries_by_cons, intermediate_outputs

    def _add_abductive_inference_rules(
        self, inference_prog, entries_by_cons, intermediate_outputs
    ):
        """
        As self.export_reasoning_program() was getting too long, refactored code for
        abductive inference program synthesis from KB
        """
        for i, entry_collection in enumerate(entries_by_cons.values()):
            # (If there are more than one entries in collection) Standardize names
            # to comply with the first entry in collection, using the discovered
            # isomorphic mappings (which should not be None)
            standardized_outputs = []
            for ei, ism in entry_collection:
                c_sat_lit, a_sat_lit, c_var_signature, a_var_signature \
                    = intermediate_outputs[ei]

                if ism is not None:
                    c_sat_lit = c_sat_lit.substitute(**ism)
                    a_sat_lit = a_sat_lit.substitute(**ism)

                    c_var_signature = [ism["terms"][(v, True)][0] for v in c_var_signature]
                    a_var_signature = [ism["terms"][(v, True)][0] for v in a_var_signature]

                standardized_outputs.append((
                    c_sat_lit, a_sat_lit, c_var_signature, a_var_signature
                ))

            coll_c_var_signature = standardized_outputs[0][2]

            # Index-neutral flag holding when any (and all) of the explanandum (cons(s))
            # in the collection holds
            coll_c_sat_lit = Literal(
                f"coll_cons_sat_{i}", wrap_args(*coll_c_var_signature)
            )

            for s_out in standardized_outputs:
                # coll_c_sat_lit holds when any (and all) of the cons hold
                inference_prog.add_absolute_rule(
                    Rule(head=coll_c_sat_lit, body=s_out[0])
                )

            # Flag holding when the explanandum (cons) is not explained by any of
            # the explanantia (bodies), and thus evoke 'catchall' hypothesis
            coll_c_catchall_lit = Literal(
                f"abduc_catchall_{i}", wrap_args(*coll_c_var_signature)
            )

            # r_catchall_lit holds when coll_c_sat_lit holds but none of the
            # explanantia (bodies) hold
            unexpl_lits = [s_out[1].flip() for s_out in standardized_outputs]
            inference_prog.add_absolute_rule(Rule(
                head=coll_c_catchall_lit, body=[coll_c_sat_lit]+unexpl_lits
            ))

            # Add appropriately weighted rule for applying 'probabilistic pressure'
            # against resorting to catchall hypothesis due to absence of abductive
            # explanation of cons
            inference_prog.add_rule(Rule(body=coll_c_catchall_lit), 1-P_C)


# Recursive helper methods for fetching predicate terms and names, substituting
# variables and function names while preserving structure, and flattening nested
# lists with arbitrary depths into a single list
_extract_terms = lambda cnjt: cnjt.args if isinstance(cnjt, Literal) \
    else [_extract_terms(nc) for nc in cnjt]
_extract_preds = lambda cnjt: cnjt.name if isinstance(cnjt, Literal) \
    else [_extract_preds(nc) for nc in cnjt]
_substitute = lambda cnjt, ts, fs: cnjt.substitute(terms=ts, functions=fs) \
    if isinstance(cnjt, Literal) else [_substitute(nc, ts, fs) for nc in cnjt]
def _flatten(ls):
    for x in ls:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x
