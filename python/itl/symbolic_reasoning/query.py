""" Inference query to region graphs factored out """
from itertools import product

from ..lpmln import Rule, Polynomial
from ..lpmln.utils import flatten_cons_ante
from ..lpmln.program.compile import rg_query


def query(reg_gr, q_vars, event, restrictors):
    """
    Query a region graph compiled from LP^MLN program to estimate the likelihood of
    each possible answer to the provided question, represented as tuple of entities
    (empty tuple for y/n questions). For each entity tuple that have some possible
    models satisfying the provided event specification, compute and return the
    marginal probability.

    If q_vars is None we have a yes/no (polar) question, where having a non-empty
    tuple as q_vars indicates we have a wh-question.

    The dict 'restrictors' contains constraints (if any) for predicate variables
    such that their values can only be picked among them.
    """
    if isinstance(event, tuple):
        # Accept single-rule (tuple) event and wrap in a set
        event = [event]
    else:
        # Treat as list
        event = list(event)

    event = sum([
        flatten_cons_ante(ev_cons, ev_ante) for ev_cons, ev_ante in event
    ], [])
    event = set(sum([
        [Rule(head=l) for l in ev_cons] if len(ev_cons) > 0 else [Rule(body=ev_ante)]
        for ev_cons, ev_ante in event
    ], []))
    assert all(
        ev_rule.is_fact() or ev_rule.is_single_body_constraint()
        for ev_rule in event
    ), "Currently only supports facts or one-body constraints as query"

    if q_vars is None:
        # Empty tuple representing no wh-quantified variables to test. () in the
        # returned dicts may as well be interpreted as the "Yes" answer.
        q_vars = ()

        # Event is already grounded
        ev_instances = { (): list(event) }
    else:
        # Assign some arbitrary order among the variables
        assert type(q_vars) == tuple, "Provide q_vars as tuple"

        # Set of atoms that appear in models covered by this Models instance
        atoms_covered = set(reg_gr.graph["atoms_map"])

        # Set of entities and predicates (along w/ arity info - values defined only for
        # predicate q_var) occurring in atoms_covered
        if len(atoms_covered) > 0:
            ents = set.union(*[{arg[0] for arg in atm.args} for atm in atoms_covered])
        else:
            ents = set()
        preds = set((atm.name, len(atm.args)) for atm in atoms_covered)

        # Arities of the occurring preds for preliminary filtering of possible
        # grounded event instances
        pred_var_arities = {
            l for l in set.union(*[ev_rule.literals() for ev_rule in event])
            if l.name=="*_isinstance"
        }
        pred_var_arities = { l.args[0][0]: len(l.args)-1 for l in pred_var_arities }

        # All possible variable substitutions based on ents & preds (& restrictors)
        subs_options = []       # List index matching with q_vars
        for qv, is_pred in q_vars:
            if is_pred:
                preds_filtered = {
                    p[0] for p in preds
                    if pred_var_arities[qv]==p[1] and
                        (qv not in restrictors or p[0] in restrictors[qv])
                    # Filter to leave predicates that have matching arities and are
                    # allowed by restrictors; if qv is not in restrictors, no constaints
                    # placed for qv
                }
                preds_filtered = {
                    p for p in preds_filtered if not p.startswith("v_")
                }   # These predicates shouldn't be used in query answers
                subs_options.append(preds_filtered)
            else:
                subs_options.append(ents)

        # All combinatorially possible grounded instances of event, obtained from
        # the substitutions
        ev_instances = {
            s_opt: [
                ev_rule.substitute(
                    preds={
                        qv[0]: o for qv, o in zip(q_vars, s_opt)
                        if qv[0].startswith("P")
                    },
                    terms={
                        (qv[0], True): (o, False) for qv, o in zip(q_vars, s_opt)
                        if not qv[0].startswith("P")
                    }
                )
                for ev_rule in event
            ]
            for s_opt in product(*subs_options)
        }

        # Initial pruning of q_vars assignments that are not worth considering; may
        # disregard assignments yielding any ante-less rules (i.e. facts) whose cons
        # atom(s) does not appear in atoms_covered
        ev_instances = {
            assig: ev_ins for assig, ev_ins in ev_instances.items()
            if not any(
                len(r.body)==0 and not any(h in atoms_covered for h in r.head)
                for r in ev_ins
            )
        }

    # Appropriate query key to region graph
    query_keys = {
        assig: _ev_ins_to_query_key(reg_gr, ev_ins)
        for assig, ev_ins in ev_instances.items()
    }

    # Obtain unnormalized potential table
    unnorm_potentials = {
        assig: _query_wrapper(reg_gr, q_key) for assig, q_key in query_keys.items()
    }

    # Compute normalized marginals, then fetch probability scores for the event
    prob_scores = {
        assig: (
            pt_table,
            sum(pt_table.values(), Polynomial(float_val=0.0))
        )
        for assig, pt_table in unnorm_potentials.items()
    }
    prob_scores = {
        assig: {
            ev: ((unnorm / Z).at_limit(), ev==query_keys[assig])
            for ev, unnorm in pt_table.items()
        }
        for assig, (pt_table, Z) in prob_scores.items()
    }
    answers = {
        assig: prob_table[query_keys[assig]][0]
        for assig, prob_table in prob_scores.items()
    }

    # Return both (answers, probability scores) and full score tables for reference;
    # `answers` is basically an abridged version of `prob_scores` that only lists
    # probabilities for the queried event only
    return answers, prob_scores


def _ev_ins_to_query_key(reg_gr, ev_ins):
    """
    Subroutine for converting grounded event instance into appropriate query key,
    in the form of (frozen)set of signed atom integer indices.
    """
    query_key = set()
    for ei in ev_ins:
        if ei.is_fact():
            if ei.head[0].as_atom() not in reg_gr.graph["atoms_map"]:
                # Grounded atom doesn't exist in region graph
                if ei.head[0].naf:
                    # ev_ins trivially satisfiable, doesn't need inclusion in key
                    pass
                else:
                    # ev_ins never satisfiable, query has to give zero potential
                    return None
            else:
                atm_id = reg_gr.graph["atoms_map"][ei.head[0].as_atom()]
                sign = 1 if ei.head[0].naf==False else -1
                query_key.add(atm_id * sign)
        else:
            assert ei.is_single_body_constraint()
            if ei.body[0].as_atom() not in reg_gr.graph["atoms_map"]:
                # Grounded atom doesn't exist in graph
                if ei.body[0].naf:
                    # ev_ins never satisfiable, query has to give zero potential
                    return None
                else:
                    # ev_ins trivially satisfiable, doesn't need inclusion in key
                    pass
            else:
                atm_id = reg_gr.graph["atoms_map"][ei.body[0].as_atom()]
                sign = 1 if ei.body[0].naf==True else -1
                query_key.add(atm_id * sign)
    
    return frozenset(query_key)


def _query_wrapper(reg_gr, q_key):
    """
    Wrapper subroutine for obtaining an appropriate table of unnormalized potential
    values for the provided query keys. Handle unsatisfiable & trivially satisfiable
    queries, or invoke the 'proper' query method.
    """
    if q_key is None:
        # Unsatisfiable query
        return { None: Polynomial(float_val=0.0), frozenset(): Polynomial(float_val=1.0) }

    if len(q_key) == 0:
        # Trivially satisfiable query
        return { q_key: Polynomial(float_val=1.0) }

    return rg_query(reg_gr, q_key)
