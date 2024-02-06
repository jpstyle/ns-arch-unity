""" Explanation query to regiongraphs graphs factored out """
from itertools import product

from .query import query
from ..symbolic_reasoning.utils import rgr_extract_likelihood, rgr_replace_likelihood


# As it is simply impossible to enumerate 'every possible continuous value',
# we consider a discretized likelihood threshold as counterfactual alternative.
# The value represents an alternative visual observation that results in evidence
# (cf. ..comp_actions.dialogue)
# that is 'reasonably' weaker that the corresponding literal actually being true.
LOW = 0.15

def attribute(reg_gr, target_event, evidence_atoms, competing_evts, vetos):
    """
    Query a region graph compiled from LP^MLN program to attribute why the specified
    target event holds, in terms of specified evidence atoms.

    Implements the causal attribution mechanism for Bayesian networks proposed in
    the paper by Koopman and Renooij (2021), appropriately modified for probabilistic
    reasoning with LP^MLN. Return the collection of sufficient explanations, where
    each sufficient explanation is a list of facts that are jointly 

    `target_event` and `evidence_atoms` are each a collection of literals that need
    to be explained, and those that can serve as explanation (i.e. explanandum and
    explanans) respectively.

    If `competing_evts` is none, the criterion for sufficient explanations would be
    modes of the marginal distributions (as in the original paper). Otherwise, the
    expected values for `competing_evts` is a collection of event atoms such that
    their probabilities are considered as criteria against which the target event
    probability will be compared.

    `vetos` (optional) lists literals that are deemed not useful as explanations,
    disqualifying them as part of valid sufficient explanations (hence omitted).
    """
    if len(evidence_atoms) == 0:
        # Can return None in case we don't have any potential explanans atoms
        return

    if vetos is None: vetos = []

    # 'Explanation lattice' as per Koopman and Renooij. Implemented as a dict, where
    # each key corresponds to a subset of evidence and its corresponding value is
    # the label annotation (l_S). We are only interested in sufficient explanations
    # right now, so we won't track counterfactual modes (M_S).

    # For each evidence, find the corresponding likelihood value from the original
    # region graph, then find possible alternative likelihood values to explore.
    # Will be used for enumerating all possible counterfactual cases below.
    #
    # Our heuristics for choice of the likelihoods: We consider alternatives of
    # the likelihood values higher than 0.5 discounted down to some moderately
    # low values (e.g. 0.8 -> LOW). In effect, this considers counterfactual cases
    # in which some 'positive' observations were 'negative' instead, but not the
    # other direction. This reflects the general intuition that it makes (more)
    # sense to give occurrences of properties as sufficient explanation, whilst
    # non-occurrences of properties are less appropriate (where non-occurrences
    # are the 'norm', or the 'default expectation').
    alt_likelihoods = []
    for evd_atom in evidence_atoms:
        if evd_atom not in reg_gr.graph["atoms_map"]:
            # Visual observation not registered due to very low probability score
            continue

        orig_vev_likelihood = rgr_extract_likelihood(reg_gr, evd_atom)
        if orig_vev_likelihood > 0.5:
            # Some ordering is automatically imposed among the atoms by list index
            alt_likelihoods.append((evd_atom, [LOW]))

    if len(alt_likelihoods) == 0:
        # Can return None in case we don't have any alternative likelihoods worth
        # considering
        return

    # Initialize lattice with the top element, namely the full evidence. Index
    # each lattice node by (frozen)set of indices of evidence atoms whose values
    # differ from the original ones.
    expl_lattice = { frozenset(): "true" }

    # Initialize breadth-first search queue with the children of the top element.
    search_queue = list(_lattice_children(frozenset(), len(alt_likelihoods)))

    # Potential sufficient explanation candidates; initialize with top element
    potential_suff_expls = set()
    potential_suff_expls.add(frozenset())

    while len(search_queue) > 0:
        node = search_queue.pop(0)
        ev_atoms_ordered = sorted(node)     # Sorted list for indexing

        # New lattice element entry
        expl_lattice[node] = "oth"

        # List of parent lattice nodes
        parents = list(_lattice_parents(node))

        # No need to process further if any of the parents have non-true label
        if any(expl_lattice[pn] != "true" for pn in parents):
            continue

        label_true = True

        backups = {
            alt_likelihoods[ei][0]: rgr_extract_likelihood(reg_gr, alt_likelihoods[ei][0])
            for ei in ev_atoms_ordered
        }           # For rolling back to original values

        # Each possible combination of values correspond to one counterfactual case,
        # compute probabilities and labels for each
        counterfactual_cases = product(*[
            alt_vals for _, alt_vals in alt_likelihoods
        ])
        for cf_case in counterfactual_cases:

            # Modify the region graph for reasoning
            replacements = {
                alt_likelihoods[ei][0]: alt_val
                for ei, alt_val in zip(ev_atoms_ordered, cf_case)
            }
            rgr_replace_likelihood(reg_gr, replacements)

            # Query the updated graph for the event probabilities, indirectly invoking
            # incremental belief propagation as needed
            max_prob_evt = (None, float("-inf"))
            for evt in [target_event] + competing_evts:
                _, prob_scores = query(reg_gr, None, (evt, None), {})

                # Update max probability event if applicable
                evt_prob = [prob for prob, is_evt in prob_scores[()].values() if is_evt][0]
                if evt_prob > max_prob_evt[1]:
                    max_prob_evt = (evt, evt_prob)
            rgr_replace_likelihood(reg_gr, backups)

            assert max_prob_evt[0] is not None
            if max_prob_evt[0] != target_event:
                # Target event no longer the most favorable answer. No further
                # processing necessary for our purpose, can break
                label_true = False
                break

        if label_true:
            # Label value true, add to potential sufficient explanation list
            # and enqueue children
            expl_lattice[node] = "true"
            potential_suff_expls.add(node)
            for child in _lattice_children(node, len(alt_likelihoods)):
                search_queue.append(child)

        # At the end of each while loop, test every potential explanation to see if
        # any of them qualify as a valid sufficient explanation. As soon as one is
        # found, return it, making this a greedy search.
        potential_suff_expls_new = set()
        for expl in potential_suff_expls:
            # Check if any of its children have 'true' label
            all_non_true = True; cannot_judge = False

            for chd in _lattice_children(expl, len(alt_likelihoods)):
                if chd not in expl_lattice:
                    # Doesn't have info in lattice yet, cannot judge
                    cannot_judge = True
                    break

                if expl_lattice[chd] == "true":
                    # Has a child with 'true' label, disqualify as potential explanation
                    all_non_true = False
                    break

            if cannot_judge:
                # Defer judgement on this one and add to the (new) set, moving on to next
                potential_suff_expls_new.add(expl)
                continue

            if all_non_true:
                # Found a qualifying candidate, see if non-empty after filtering out
                # the vetoed literals
                expl = {
                    alt_likelihoods[ei][0]
                    for ei in set(range(len(alt_likelihoods)))-expl
                }
                expl = {lit for lit in expl if lit not in vetos}

                # Return if expl is nonempty after filtering out vetoed literals
                if len(expl) > 0:
                    return expl

        # Switcheroo
        potential_suff_expls = potential_suff_expls_new

    # If reached here, no valid sufficient explanation found; return None
    return


# Helper methods for obtaining parents/children of a lattice element (given the
# size of the set of evidence atoms considered for the latter)
def _lattice_parents(lattice_node):
    for ei in lattice_node:
        yield lattice_node - {ei}
def _lattice_children(lattice_node, ev_size):
    for ei in range(ev_size):
        if ei not in lattice_node:
            yield lattice_node | {ei}
