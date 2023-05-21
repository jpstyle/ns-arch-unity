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
    Rearrange until any nested conjunctions are all properly flattened out,
    so that rule can be translated into appropriate ASP clause
    """
    from .. import Literal

    cons = list(cons) if cons is not None else []
    ante = list(ante) if ante is not None else []
    conjuncts = cons + ante
    while any(not isinstance(c, Literal) for c in conjuncts):
        # Migrate any negated conjunctions in cons to ante
        conjuncts_p = [c for c in cons if isinstance(c, Literal)]
        conjuncts_n = [c for c in cons if not isinstance(c, Literal)]

        cons = conjuncts_p
        ante = ante + sum(conjuncts_n, [])

        if any(not isinstance(a, Literal) for a in ante):
            # Introduce auxiliary literals that are derived when
            # each conjunction in ante is satisfied
            # (Not needed, not implemented yet :p)
            raise NotImplementedError

        conjuncts = cons + ante
    
    return cons, ante

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
