"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import numpy as np
import networkx as nx

from ...lpmln import Literal, Polynomial


def bjt_extract_likelihood(bjt, evd_atom):
    """
    Find and fetch the current likelihood value for the virtual evidence of the
    specified target event literal stored in the provided BJT

    (This bit must be really cryptic for new readers to understand...)
    """
    # Event state atom corresponding to the given virtual evidence atom
    state_atom = Literal(evd_atom.name.strip("v_"), evd_atom.args)

    # Evidence & corresponding event atom index in the BJT
    evd_ai = bjt.graph["atoms_map"][evd_atom]
    state_ai = bjt.graph["atoms_map"][state_atom]

    vev_likelihood = [
        bjt.nodes[n]["input_potential"][n[0]][(frozenset(), n[0])]
        for n in bjt.nodes
        if n[0] == frozenset([evd_ai, state_ai])
    ][0]
    vev_likelihood = float(np.exp(vev_likelihood.primitivize()))

    return vev_likelihood


def bjt_replace_likelihood(bjt, replacements):
    """
    Updating likelihood values for the virtual evidence of the specified target
    event literals stored in the provided BJT
    """
    bjt_undirected = bjt.to_undirected()    # For tree traversal

    # Modify the BJT, replacing appropriate likelihood value one by one
    for evd_atom, alt_val in replacements.items():
        
        alt_val_poly = Polynomial.from_primitive(float(np.log(alt_val)))
        alt_val_1m_poly = Polynomial.from_primitive(float(np.log(1-alt_val)))

        state_atom = Literal(evd_atom.name.strip("v_"), evd_atom.args)

        evd_ai = bjt.graph["atoms_map"][evd_atom]
        state_ai = bjt.graph["atoms_map"][state_atom]

        for n in bjt.nodes:
            # Process below only if at appropriate node
            if n[0] != frozenset([evd_ai, state_ai]): continue

            # Apply likelihood value changes
            occ_case = frozenset([evd_ai, state_ai])    # == n[0]
            occ_selector = (frozenset(), n[0])
            nonocc_case = frozenset([evd_ai, -state_ai])
            nonocc_selector = (frozenset([state_ai]), n[0])

            inp_pt = bjt.nodes[n]["input_potential"]
            inp_pt[occ_case][occ_selector] = alt_val_poly
            inp_pt[nonocc_case][nonocc_selector] = alt_val_1m_poly

            # Tree traversal with the node as root, setting all outward-edges
            # and nodes along the ways as 'stale' (i.e. marking in need of
            # recomputation for output beliefs & messages)
            bjt.nodes[n]["update_needed"] = True
            for u, v in nx.dfs_tree(bjt_undirected, source=n).edges():
                bjt.nodes[v]["update_needed"] = True
                bjt.edges[(u, v)]["update_needed"] = True

            # Making sure likelihood change is reflected at most once
            break
