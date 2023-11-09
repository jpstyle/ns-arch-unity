""" Explanation query to BJT factored out """
from copy import deepcopy
from itertools import product

import numpy as np
import networkx as nx

from .query import query
from ..lpmln import Literal, Polynomial


# As it is simply impossible to enumerate 'every possible continuous value',
# we consider at most two discretized likelihoods as counterfactual alternatives,
# 0.25 and 0.75. Each value represents an alternative visual observation that
# results in 'reasonably' weaker/stronge evidence that the corresponding literal
# actually being true.
LOW = 0.25
HIGH = 0.75

def attribute(bjt, target_event, evidence_atoms, threshold=0.0):
    """
    Query a BJT compiled from LP^MLN program to attribute why the specified target
    event holds, in terms of specified evidence atoms. The provided BJT might or
    might not have run the belief propagation already, but the belief propagation
    outcome must be consistent with the provided target event.

    Implements the causal attribution mechanism for Bayesian networks proposed in
    the paper by Koopman and Renooij (2021), appropriately modified for probabilistic
    reasoning with LP^MLN. Return the collection of sufficient explanations, where
    each sufficient explanation is a list of facts that are jointly 

    `target_event` and `evidence_atoms` are each a collection of literals that need
    to be explained, and those that can serve as explanation (i.e. explanandum and
    explanans) respectively.

    If `threshold` is none, the criterion for sufficient explanations would be modes
    of the marginal distributions (as in the original paper). Otherwise, `threshold`
    is a value in [0,1] to be used as threshold for the target event probability,
    such that probabilities lower than the value would be treated as non-occurrence
    of the target event.
    """
    if len(evidence_atoms) == 0:
        # Can return empty in case of empty potential explanans atoms
        return []

    # 'Explanation lattice' as per Koopman and Renooij. Implemented as a dict, where
    # each key corresponds to a subset of evidence and its corresponding value is
    # the label annotation (l_S). We are only interested in sufficient explanations
    # right now, so we won't track counterfactual modes (M_S). We also store the
    # incrementally updated BJT's obtained for each possible counterfactuals, so as
    # to minimize redundant computations

    # Compute mode and label for the top element
    _, prob_scores = query(bjt, None, (target_event, None), {})

    # Sanity check; corresponding probability value should be the highest among
    # possible outcomes and should be above the threshold
    tgt_prob = [prob for prob, is_tgt in prob_scores[()].values() if is_tgt][0]
    mode_prob = max(prob for prob, _ in prob_scores[()].values())
    assert tgt_prob == mode_prob and tgt_prob > threshold

    # Initialize lattice with the top element, namely the full evidence. Index
    # each lattice node by (frozen)set of indices of evidence atoms whose values
    # differ from the original ones.
    expl_lattice = { frozenset(): ("true", { (): bjt }) }

    # For each evidence, find the corresponding likelihood value from the
    # original BJT, then find possible alternative likelihood values to explore.
    # Will be used for enumerating all possible counterfactual cases below.
    #
    # Our heuristics for choice of the likelihoods: If a virtual evidence is not
    # 'strong enough' for either direction, i.e. the corresponding likelihood
    # value lies in between [LOW+0.1, HIGH-0.1], we consider both extremes.
    # Otherwise, we only consider the extreme towards the 'opposite' direction
    # (e.g. 0.8 -> LOW), as we deem it is more certain that the 'tipping point'
    # would be crossed by exploring the opposite extreme, but possibly not the
    # other extreme, if at all.
    alternative_likelihoods = {}
    for evd_atom in evidence_atoms:
        state_atom = Literal(evd_atom.name.strip("v_"), evd_atom.args)

        evd_ai = bjt.graph["atoms_map"][evd_atom]
        state_ai = bjt.graph["atoms_map"][state_atom]

        # (Wow this bit must be really cryptic for strangers to understand...)
        orig_vev_likelihood = [
            bjt.nodes[n]["input_potential"][n[0]][(frozenset(), n[0])]
            for n in bjt.nodes
            if n[0] == frozenset([evd_ai, state_ai])
        ][0]
        orig_vev_likelihood = float(np.exp(orig_vev_likelihood.primitivize()))

        if orig_vev_likelihood > HIGH-0.1:
            alternative_likelihoods[evd_atom] = [LOW]
        elif orig_vev_likelihood < LOW+0.1:
            alternative_likelihoods[evd_atom] = [HIGH]
        else:
            alternative_likelihoods[evd_atom] = [LOW, HIGH]

    # Initialize breadth-first search queue with the children of the top element.
    search_queue = list(_lattice_children(frozenset(), len(evidence_atoms)))

    # Potential sufficient explanation candidates; initialize with top element
    potential_suff_expls = set()
    potential_suff_expls.add(frozenset())

    while len(search_queue) > 0:
        node = search_queue.pop(0)
        ev_atoms_ordered = sorted(node)     # Sorted list for indexing

        # New lattice element entry
        expl_lattice[node] = ("oth", {})

        # List of parent lattice nodes
        parents = list(_lattice_parents(node))

        # No need to process further if any of the parents have non-true label
        if any(expl_lattice[pn][0] != "true" for pn in parents):
            continue

        label_true = True

        # Each possible combination of values correspond to one counterfactual case,
        # compute modes and labels for each
        counterfactual_cases = product(*[
            alternative_likelihoods[evidence_atoms[ei]]
            for ei in ev_atoms_ordered
        ])
        for cf_case in counterfactual_cases:
            # Incremental reasoning with BJT; taking one of the parent's BJT with
            # matching counterfactual values cached in lattice, replacing target
            # likelihood values and marking and all nodes with 'outdated' indicators
            # (which are needed for another round of incremental belief propagation)

            # Find parent with matching counterfactual value signature for shared
            # evidence atoms
            alt_vals_for_shared = [
                tuple(
                    alt_val for ei, alt_val in zip(ev_atoms_ordered, cf_case)
                    if ei in pn
                )
                for pn in parents
            ]
            parent_selected, bjt_parent = [
                (pn, expl_lattice[pn][1][alt_vals])
                for pn, alt_vals in zip(parents, alt_vals_for_shared)
            ][0]
            bjt_new = deepcopy(bjt_parent)
            bjt_undirected = bjt_new.to_undirected()    # For tree traversal

            # Modify the copied BJT for incremental reasoning
            for ei, alt_val in zip(ev_atoms_ordered, cf_case):
                if ei in parent_selected: continue      # Already reflected

                alt_val_poly = Polynomial.from_primitive(float(np.log(alt_val)))
                alt_val_1m_poly = Polynomial.from_primitive(float(np.log(1-alt_val)))

                evd_atom = evidence_atoms[ei]
                state_atom = Literal(evd_atom.name.strip("v_"), evd_atom.args)

                evd_ai = bjt_new.graph["atoms_map"][evd_atom]
                state_ai = bjt_new.graph["atoms_map"][state_atom]

                for n in bjt_new.nodes:
                    # Process below only if at appropriate node
                    if n[0] != frozenset([evd_ai, state_ai]): continue

                    # Apply likelihood value changes
                    occ_case = frozenset([evd_ai, state_ai])    # == n[0]
                    occ_selector = (frozenset(), n[0])
                    nonocc_case = frozenset([evd_ai, -state_ai])
                    nonocc_selector = (frozenset([state_ai]), n[0])

                    inp_pt = bjt_new.nodes[n]["input_potential"]
                    inp_pt[occ_case][occ_selector] = alt_val_poly
                    inp_pt[nonocc_case][nonocc_selector] = alt_val_1m_poly

                    # Tree traversal with the node as root, setting all outward-edges
                    # and nodes along the ways as 'stale' (i.e. marking in need of
                    # recomputation for output beliefs & messages)
                    bjt_new.nodes[n]["update_needed"] = True
                    for u, v in nx.dfs_tree(bjt_undirected, source=n).edges():
                        bjt_new.nodes[v]["update_needed"] = True
                        bjt_new.edges[(u, v)]["update_needed"] = True

                    # Making sure likelihood change is reflected at most once
                    break

            # Query the updated BJT for the target event likelihood, indirectly
            # invoking incremental belief propagation as needed
            _, prob_scores = query(bjt_new, None, (target_event, None), {})

            # Mode check and labeling
            tgt_prob = [prob for prob, is_tgt in prob_scores[()].values() if is_tgt][0]
            mode_prob = max(prob for prob, _ in prob_scores[()].values())

            # Record the updated BJT to lattice along with the counterfactual case
            expl_lattice[node][1][cf_case] = bjt_new

            if not (tgt_prob == mode_prob and tgt_prob > threshold):
                # No further processing necessary for our purpose, can break
                label_true = False
                break

        if label_true:
            # Label value true, add to potential sufficient explanation list
            # and enqueue children
            expl_lattice[node] = ("true", expl_lattice[node][1])
            potential_suff_expls.add(node)
            for child in _lattice_children(node, len(evidence_atoms)):
                search_queue.append(child)

    # Search terminated: find and return sufficient explanations
    suff_expls = [
        {evidence_atoms[ei] for ei in set(range(len(evidence_atoms)))-node}
        for node in potential_suff_expls
        if all(
            expl_lattice[chd][0] != "true"
            for chd in _lattice_children(node, len(evidence_atoms))
        )
    ]

    return suff_expls


# Helper methods for obtaining parents/children of a lattice element (given the
# size of the set of evidence atoms considered for the latter)
def _lattice_parents(lattice_node):
    for ei in lattice_node:
        yield lattice_node - {ei}
def _lattice_children(lattice_node, ev_size):
    for ei in range(ev_size):
        if ei not in lattice_node:
            yield lattice_node | {ei}
