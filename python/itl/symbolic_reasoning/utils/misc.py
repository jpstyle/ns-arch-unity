"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import numpy as np

from ...lpmln import Literal, Polynomial


def rgr_extract_likelihood(reg_gr, evd_atom):
    """
    Find and fetch the current likelihood value for the virtual evidence of the
    specified target event literal stored in the provided region graph
    """
    # Event state atom corresponding to the given virtual evidence atom
    state_atom = Literal(evd_atom.name.strip("v_"), evd_atom.args)

    # Evidence & corresponding event atom index in the graph
    evd_ai = reg_gr.graph["atoms_map"][evd_atom]
    state_ai = reg_gr.graph["atoms_map"][state_atom]

    node_atoms, node_factors = [
        n for n in reg_gr.nodes
        if n[0] == frozenset([evd_ai, state_ai])
    ][0]
    factor_pt = reg_gr.graph["factor_potentials"][list(node_factors)[0]]
    vev_likelihood = factor_pt[node_atoms][(frozenset(), node_atoms)]
    vev_likelihood = vev_likelihood.at_limit()

    return vev_likelihood


def rgr_replace_likelihood(reg_gr, replacements):
    """
    Updating likelihood values for the virtual evidence of the specified target
    event literals stored in the provided region graph
    """
    # Modify the graph, replacing appropriate likelihood value one by one
    for evd_atom, alt_val in replacements.items():
        
        alt_val_poly = Polynomial.from_primitive(float(np.log(alt_val)))
        alt_val_1m_poly = Polynomial.from_primitive(float(np.log(1-alt_val)))

        state_atom = Literal(evd_atom.name.strip("v_"), evd_atom.args)

        evd_ai = reg_gr.graph["atoms_map"][evd_atom]
        state_ai = reg_gr.graph["atoms_map"][state_atom]

        for n in reg_gr.nodes:
            node_atoms, node_factors = n

            # Process below only if at appropriate node
            if node_atoms != frozenset([evd_ai, state_ai]): continue

            # Apply likelihood value changes
            occ_case = frozenset([evd_ai, state_ai])    # == node_atoms
            occ_selector = (frozenset(), node_atoms)
            nonocc_case = frozenset([evd_ai, -state_ai])
            nonocc_selector = (frozenset([state_ai]), node_atoms)

            factor_pt = reg_gr.graph["factor_potentials"][list(node_factors)[0]]
            factor_pt[occ_case][occ_selector] = alt_val_poly
            factor_pt[nonocc_case][nonocc_selector] = alt_val_1m_poly

            # Making sure likelihood change is reflected at most once
            break

    # Clean up existing messages and reset convergence flag
    for e in reg_gr.edges:
        reg_gr.edges[e]["message"] = (None, set())
    reg_gr.graph["converged"] = False
