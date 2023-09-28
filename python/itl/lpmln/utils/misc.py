"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import numpy as np


def wrap_args(*args):
    """
    Wrap list of arguments, adding whether each arg is variable or not by looking at
    if the first letter is uppercased
    """
    wrapped = []
    for arg in args:
        if type(arg) == str:
            # Non-function term
            wrapped.append((arg, arg[0].isupper()))
        elif type(arg) == tuple:
            # Function term
            _, f_args = arg
            wrapped.append((arg, all(fa[0].isupper() for fa in f_args)))
        elif type(arg) == int or type(arg) == float:
            # Number, definitely not a variable
            wrapped.append((arg, False))
        else:
            raise NotImplementedError

    return wrapped

def logit(p, large=float("inf")):
    """ Compute logit of the probability value p """
    if p == 1:
        return large
    elif p == 0:
        if type(large)==str:
            return f"-{large}"
        else:
            assert type(large)==float
            return -large
    else:
        return float(np.log(p/(1-p)))

def sigmoid(l):
    """ Compute probability of the logit value l """
    if l == float("inf") or l == "a":
        return 1
    elif l == float("-inf") or l == "-a":
        return 0
    else:
        return float(1 / (1 + np.exp(-l)))

def flatten_cons_ante(cons, ante):
    """
    Rearrange into (possibly multiple) flattened rules until any all of the nested
    conjunctions are all properly fished up, such that the rule can be translated
    into appropriate ASP clauses.

    Example:
        A consequent-antecedent pair "p, not q <= r" would yield a list of flattened
        cons-ante pairs of ["p <= r", "<= q, r"].
    """
    from .. import Literal

    cons = list(cons) if cons is not None else []
    ante = list(ante) if ante is not None else []

    flattened = [(cons, ante)]
    while any(
        not isinstance(conjunct, Literal)
        for cons, ante in flattened for conjunct in cons+ante
    ):
        flattened_new = []

        for cons, ante in flattened:
            # Positive & negative conjuncts in consequent
            # (Reminder for self: a list of conjuncts stands for negation of
            # the conjunction in current implementation)
            cons_cnjts_p = [c for c in cons if isinstance(c, Literal)]
            cons_cnjts_n = [c for c in cons if not isinstance(c, Literal)]

            if any(not isinstance(a, Literal) for a in ante):
                # Introduce auxiliary literals that are derived when each conjunction
                # in ante is satisfied (Not needed, not implemented yet :p)
                # (Mind differences of semantics of strong vs. weak negation...)
                raise NotImplementedError

            if len(cons_cnjts_p) > 0:
                # Positive conjuncts in consequent <= antecedent
                flattened_new.append((cons_cnjts_p, ante))

            for neg_cnjt in cons_cnjts_n:
                # Migrate each negated conjunction to ante, creating a new (cons, ante)
                # pair to be processed and included
                flattened_new.append(([], ante+neg_cnjt))

        flattened = flattened_new
    
    return flattened

def unify_mappings(mappings):
    """
    Find the unification of a sequence of mappings; return None if not unifiable
    """
    unified = {}
    for mapping in mappings:
        for k, v in mapping.items():
            if k in unified:
                if unified[k] != v:
                    return None
            else:
                unified[k] = v

    return unified
