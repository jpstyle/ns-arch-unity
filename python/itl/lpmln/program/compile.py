""" Program().compile() subroutine factored out """
import operator
from functools import reduce
from itertools import product, combinations
from collections import defaultdict

import clingo
import numpy as np
import networkx as nx

from ..literal import Literal
from ..rule import Rule
from ..polynomial import Polynomial
from ..utils import logit


def compile(prog):
    """
    Compiles program into a binary join tree from equivalent directed graph, which
    would contain data needed to answer probabilistic inference queries. Achieved
    by running the following operations:

    1) Ground the program with clingo to find set of actually grounded atoms
    2) Compile the factor graph into binary join tree, following the procedure
        presented in Shenoy, 1997 (modified to comply with semantics of logic
        programs). Include singleton node sets for each grounded atom, so that
        the compiled join tree will be prepared to answer marginal queries for
        atoms afterwards.
    3) Identify factor potentials associated with valid sets of nodes. Each
        program rule corresponds to a piece of information contributing to the
        final factor potential for each node set. Essentially, this amounts to
        finding a factor graph corresponding to the provided program.
    4) Run modified Shafer-Shenoy belief propagation algorithm to fill in values
        needed for answering probabilistic inference queries later.
    
    Returns a binary join tree populated with the required values resulting from belief
    propagation.
    """
    bjt = nx.DiGraph()

    if _grounded_facts_only([r for r, _, _ in prog.rules]):
        # Can take simpler shortcut for constructing binary join tree if program
        # consists only of grounded facts
        bjt.graph["atoms_map"] = {}
        bjt.graph["atoms_map_inv"] = {}

        for i, (rule, r_pr, weighting) in enumerate(prog.rules):
            # Integer indexing should start from 1, to represent negated atoms as
            # negative ints
            i += 1

            # Mapping between atoms and their integer indices
            bjt.graph["atoms_map"][rule.head[0]] = i
            bjt.graph["atoms_map_inv"][i] = rule.head[0]

            fact_input_potential = _rule_to_potential(
                rule, r_pr, weighting, { rule.head[0]: i }
            )

            # Add singleton atom set node for the atom, with an appropriate input
            # potential from the rule weight
            bjt.add_node((frozenset({i}), 0), input_potential=fact_input_potential)

    else:
        grounded_rules_by_atms, atoms_map, atoms_map_inv = _ground_and_index(prog)

        # Mapping between atoms and their integer indices
        bjt.graph["atoms_map"] = atoms_map
        bjt.graph["atoms_map_inv"] = atoms_map_inv

        # Identifying duplicate variable subsets in multisets by indexing
        ss_inds = defaultdict(int)

        if len(atoms_map) == 0 or len(grounded_rules_by_atms) == 0:
            # Happens to have no atoms and rules grounded; nothing to do here
            pass
        else:
            # Binary join tree construction (cf. Shenoy, 1997)

            # Set of variables to be processed; Psi_u in Shenoy
            atoms_unpr = set(atoms_map_inv)

            # 'Multiset' of variables to be processed; Phi_u in Shenoy. Initialized
            # from 1) subset of variables for which we have rules (hence potentials),
            # and 2) subsets for which we need marginals --- all singletons in our case
            # (Multiset implemented with dicts, with variable subsets as key and
            # sets of indices defined per subset as values)
            atomsets_unpr = set()
            atomsets_unpr |= {(atms, 0) for atms in grounded_rules_by_atms}
            atomsets_unpr |= {(frozenset({atm}), 0) for atm in atoms_map_inv}

            while len(atomsets_unpr) > 1:
                # Pick a variable with a heuristic, namely one that would lead to smallest
                # union of relevant nodes in factors
                atomset_unions = {
                    atm: frozenset.union(*{atms for atms, _ in atomsets_unpr if atm in atms})
                    for atm in atoms_unpr
                }
                # Variable picked for the loop by the heuristic; Y in Shenoy
                picked = sorted(atoms_unpr, key=lambda atm: len(atomset_unions[atm]))[0]

                # Set of variable subsets; Phi_Y in Shenoy
                atomsets_relevant = {
                    (atms, a_i) for atms, a_i in atomsets_unpr if picked in atms
                }

                while len(atomsets_relevant) > 1:
                    # Pick a node set pair that would give smallest union
                    atomset_pair_unions = {
                        (n1, n2): n1[0] | n2[0]
                        for n1, n2 in combinations(atomsets_relevant, 2)
                    }
                    n1, n2 = sorted(
                        atomset_pair_unions,
                        key=lambda atmsp: len(atomset_pair_unions[atmsp])
                    )[0]                                # r_1 and r_2 in Shenoy
                    smallest_union = n1[0] | n2[0]      # s_k in Shenoy

                    # Distinct subset (possibly duplicate with n1 or n2) with fresh index
                    ss_inds[smallest_union] += 1
                    nu = (smallest_union, ss_inds[smallest_union])

                    # Add new nodes and edges; note that the node set (`N` in Shenoy)
                    # is a multiset as long as we don't distinguish among the possibly
                    # duplicate variable subsets. We implement N as a set by indexing
                    # each entry of each variable subset.
                    bjt.add_node(n1); bjt.add_node(n2); bjt.add_node(nu)

                    # Both directions for the edges, for storing messages of two opposite
                    # directions
                    bjt.add_edge(n1, nu); bjt.add_edge(nu, n1)
                    bjt.add_edge(n2, nu); bjt.add_edge(nu, n2)

                    # Update relevant subset set; set size reduced exactly by one
                    atomsets_relevant -= {n1, n2}
                    atomsets_relevant |= {nu}

                if len(atoms_unpr) >= 1:
                    # Sole subset in Phi_Y, which must be a singleton set here; r in Shenoy
                    n = list(atomsets_relevant)[0]
                    atms, a_i = n
                    # s_k := r - {Y} in Shenoy
                    atms_cmpl_picked = atms - {picked}
                    # Note that the two subsets just assigned are different by definition,
                    # as atms must include the picked atom and atms_cmpl_picked must not

                    n = (atms, a_i)

                    # Add corresponding nodes and edges
                    bjt.add_node(n)
                    if len(atms_cmpl_picked) > 0:
                        # Should process atms_cmpl_picked only if non-empty; empty ones
                        # usually signals isolated nodes without any neighbors
                        ss_inds[atms_cmpl_picked] += 1
                        nc = (atms_cmpl_picked, ss_inds[atms_cmpl_picked])

                        bjt.add_node(nc)
                        bjt.add_edge(n, nc); bjt.add_edge(nc, n)

                        # Updating atomsets_unpr with a new element (marked with the
                        # fresh index)
                        atomsets_unpr.add(nc)

                    # Updating atoms_unpr & atomsets_unpr by complement
                    atoms_unpr -= {picked}
                    atomsets_unpr = {
                        (atms, a_i) for (atms, a_i) in atomsets_unpr
                        if picked not in atms
                    }

            if len(atomsets_unpr) > 0:
                # atomsets_unpr must be a singleton set if entered
                assert len(atomsets_unpr) == 1
                atms_last = list(atomsets_unpr)[0]
                bjt.add_node((atms_last, ss_inds[atms_last]))

        # Condense the compiled BJT; merge nodes sharing the same variable subset,
        # as long as the merging wouldn't breach the binary constraint (i.e., keeps
        # neighbor counts no larger than 3)
        while True:
            undirected_edges = bjt.to_undirected().edges
            for n1, n2 in undirected_edges:
                # Not mergeable if nodes have different var subsets
                if n1[0] != n2[0]: continue
                # Not mergeable if merging would result in neighbor count larger than 3
                if (bjt.in_degree[n1]-1)+(bjt.in_degree[n2]-1) > 3: continue

                # Sort to leave lower index --- just my OCD
                n1, n2 = sorted([n1, n2])

                # Merge by removing n2 and connecting n2's other neighbors to n1
                n2_neighbors = [nb for nb, _ in bjt.in_edges(n2) if nb != n1]
                bjt.remove_node(n2)
                for nb in n2_neighbors:
                    bjt.add_edge(n1, nb); bjt.add_edge(nb, n1)
                
                # End this loop and start anew
                break

            else:
                # No more nodes to merge, BJT fully condensed
                break

        # Associate each ground rule with an appropriate BJT node that covers all and
        # only atoms occurring in the rule, then populate the BJT node with appropriate
        # input potential
        for atms, gr_rules in grounded_rules_by_atms.items():
            # Convert each rule to matching potential, and combine into single potential
            input_potentials = [
                _rule_to_potential(gr_rule, r_pr, weighting, atoms_map)
                for gr_rule, (r_pr, weighting), _ in gr_rules
            ]

            if any(inp is None for inp in input_potentials):
                # Falsity included, program has no answer set
                return None

            if len(atms) > 0:
                inps_combined = _combine_factors_outer(input_potentials)
                target_node = [n for n in bjt.nodes if atms==n[0]][0]
                    # There can be multiple candidates; pick one and only one
                bjt.nodes[target_node]["input_potential"] = inps_combined
        
        # Null-potentials for BJT nodes without any input potentials
        for node in bjt.nodes:
            if "input_potential" not in bjt.nodes[node]:
                atms = node[0]
                cases = {frozenset(case) for case in product(*[(ai, -ai) for ai in atms])}
                bjt.nodes[node]["input_potential"] = {
                    case: { (frozenset(), frozenset()): Polynomial(float_val=1.0) }
                    for case in cases
                }

    # Mark 'not-up-to-date' to all nodes & edges (i.e., values nonexistent or
    # potentially incorrect b/c input changed)
    for n in bjt.nodes:
        bjt.nodes[n]["update_needed"] = True
        for e in bjt.in_edges(n):
            bjt.edges[e]["update_needed"] = True

    return bjt


def bjt_query(bjt, q_key):
    """ Query a BJT for (unnormalized) belief table """
    relevant_nodes = frozenset({abs(n) for n in q_key})
    relevant_cliques = [n for n in bjt.nodes if relevant_nodes <= n[0]]

    if len(relevant_cliques) > 0:
        # In-clique query; find the BJT node with the smallest node set that
        # comply with the key
        smallest_node = sorted(relevant_cliques, key=lambda x: len(x[0]))[0]

        if bjt.nodes[smallest_node]["update_needed"]:
            belief_propagation(bjt, smallest_node[0])
            assert not bjt.nodes[smallest_node]["update_needed"]
        beliefs = bjt.nodes[smallest_node]["output_beliefs"]

        # Marginalize and return
        return _marginalize_simple(beliefs, relevant_nodes)

    else:
        # Not really used for now. Make sure later, when needed, this works correctly;
        # in particular, ensure it works alright with the newly introduced incremental
        # belief propagation feature
        raise NotImplementedError

        # Out-clique query; first divide query key node set by belonging components
        # in the BJT
        components = {
            frozenset.union(*comp): bjt.subgraph(comp)
            for comp in nx.connected_components(bjt.to_undirected())
        }

        divided_keys_and_subtrees = {
            frozenset({l for l in q_key if abs(l) in comp_nodes}): sub_bjt
            for comp_nodes, sub_bjt in components.items()
            if len(comp_nodes & relevant_nodes) > 0
        }

        if len(divided_keys_and_subtrees) == 1:
            # All query key nodes in the same component; variable elimination needed
            raise NotImplementedError
        else:
            # Recursively query each subtree with corresponding 'subkey'
            query_results = {
                subkey: bjt_query(sub_bjt, subkey)
                for subkey, sub_bjt in divided_keys_and_subtrees.items()
            }

            # Combine independent query results
            return _combine_factors_simple(list(query_results.values()))


class _Observer:
    """ For tracking added grounded rules """
    def __init__(self):
        self.rules = []
    def rule(self, choice, head, body):
        self.rules.append((head, body, choice))


def _grounded_facts_only(rules):
    """ Test if set of rules consists only of grounded facts """
    return all(r.is_fact() and r.is_grounded() for r in rules)


def _ground_and_index(prog):
    """
    Ground program, construct a directed graph reflecting dependency between grounded
    atoms (by program rules), index grounded_rules by occurring atoms. Return the
    indexed grounded rules & mappings between grounded atoms and their integer indices.
    """
    # Feed compiled program string to clingo.Control object and ground program
    rules_obs = _Observer()
    ctl = clingo.Control(["--warn=none"])
    ctl.register_observer(rules_obs)
    ctl.add("base", [], prog._pure_ASP_str())
    ctl.ground([("base", [])])

    # All grounded atoms that are worth considering
    atoms_map = {
        Literal.from_clingo_symbol(atom.symbol): atom.literal
        for atom in ctl.symbolic_atoms
    }
    atoms_map_inv = {v: k for k, v in atoms_map.items()}

    # All grounded atoms that each occurring atom can instantiate (grounded atom
    # can instantiate only itself)
    instantiable_atoms = {
        ra: {
            (ma, tuple((rarg[0], marg[0]) for rarg, marg in zip(ra.args, ma.args)))
            for ma in atoms_map
            if ra.name == ma.name and len(ra.args) == len(ma.args) and all([
                rarg[1] == True or rarg[0] == marg[0]
                for rarg, marg in zip(ra.args, ma.args)
            ])
        }
        for ra in prog._rules_by_atom
    }

    # Iterate over the grounded rules for the following processes:
    #   1) Track which rule in the original program could have instantiated each
    #        grounded rule (wish clingo python API supported such feature...)
    #   2) Index the grounded rule by occurring atoms, positive or negative
    grounded_rules_by_atms = defaultdict(set)

    for ri, (rule, r_pr, weighting) in enumerate(prog.rules):
        # All possible grounded rules that may originate from this rule
        gr_head_insts = [instantiable_atoms[hl.as_atom()] for hl in rule.head]
        gr_head_insts = [
            # Make sure literals (weak-)negated in the original rule are
            # properly flipped
            {(ghl[0].flip(), ghl[1]) if hl.naf else ghl for ghl in ghls}
            for ghls, hl in zip(gr_head_insts, rule.head)
        ]
        gr_body_insts = [instantiable_atoms[bl.as_atom()] for bl in rule.body]
        gr_body_insts = [
            {(gbl[0].flip(), gbl[1]) if bl.naf else gbl for gbl in gbls}
            for gbls, bl in zip(gr_body_insts, rule.body)
        ]

        # Possible mappings from variables to constants worth considering
        if len(gr_head_insts+gr_body_insts) > 0:
            possible_substs = set.union(*[
                set.union(*[set(gl[1]) for gl in gls]) if len(gls)>0 else set()
                for gls in gr_head_insts+gr_body_insts
            ])          # All var-cons pair witnessed
            possible_substs = {
                t1_1: {t2_2 for t1_2, t2_2 in possible_substs if t1_1==t1_2}
                for t1_1, _ in possible_substs
            }           # Collect by variable
            possible_substs = [
                # t1!=t2 ensures t1 is a variable
                {t1: t2 for t1, t2 in zip(possible_substs, cs) if t1!=t2}
                for cs in product(*possible_substs.values())
            ]           # Flatten products into list of all possible groundings
        else:
            possible_substs = [{}]
        
        for subst in possible_substs:
            # For each possible grounding of this rule
            subst = { (v, True): (c, False) for v, c in subst.items() }
            gr_rule = rule.substitute(terms=subst)

            gr_body_pos = [gbl for gbl in gr_rule.body if gbl.naf==False]
            gr_body_neg = [gbl for gbl in gr_rule.body if gbl.naf==True]

            # Check whether this grounded rule would turn out to be never satisfiable
            # because there exists ungroundable positive body atom; in such cases,
            # unsat will never fire, and we can dismiss the rule
            if any(gbl not in atoms_map for gbl in gr_body_pos):
                continue

            # Negative rule body after dismissing ungroundable atoms; ungroundable
            # atoms in negative body can be ignored as they are trivially satisfied
            # (always reduced as all models will not have occurrence of the atom)
            gr_body_neg_filtered = [
                gbl for gbl in gr_body_neg if gbl.as_atom() in atoms_map
            ]
            gr_body_filtered = gr_body_pos + gr_body_neg_filtered

            # Index and add this grounded rule with r_pr and index
            gr_rule = Rule(head=gr_rule.head, body=gr_body_filtered)
            occurring_atoms = frozenset([
                atoms_map[lit.as_atom()] for lit in gr_rule.head+gr_rule.body
            ])
            grounded_rules_by_atms[occurring_atoms].add((gr_rule, (r_pr, weighting), ri))

    return grounded_rules_by_atms, atoms_map, atoms_map_inv


def _rule_to_potential(rule, r_pr, weighting, atoms_map):
    """
    Subroutine for converting a (grounded) rule weighted with probability r_pr into
    an appropriate input potential
    """
    # Edge case; body-less integrity constraint with absolute probability: falsity
    if len(rule.head+rule.body) == 0 and r_pr is None:
        return None

    # Use integer indices of the atoms
    atoms_by_ind = frozenset({atoms_map[lit.as_atom()] for lit in rule.head+rule.body})
    rh_by_ind = frozenset([
        atoms_map[hl] if hl.naf==False else -atoms_map[hl.as_atom()]
        for hl in rule.head
    ])
    rb_by_ind = frozenset([
        atoms_map[bl] if bl.naf==False else -atoms_map[bl.as_atom()]
        for bl in rule.body
    ])

    # We won't consider idiosyncratic cases with negative rule head literals
    assert all(hl>0 for hl in rh_by_ind)

    cases = {
        frozenset(case) for case in product(*[(ai, -ai) for ai in atoms_by_ind])
    }

    # Requirements of external support (for positive atoms), computed as follows:
    # 1) Any atoms positive in the case that are not the rule head...
    # 2) ... but head atom is exempt if body is true (i.e. the rule licenses head
    #    if body is true)
    pos_requirements = {
        case: frozenset({cl for cl in case if cl>0}) - \
            (rh_by_ind if rb_by_ind <= case else frozenset())
        for case in cases
    }

    # In fact, what we need for combination for each case is the **complement**
    # of positive atom requirements w.r.t. the domain; i.e. set of atoms whose
    # requirements for external support is cleared (either not present from the
    # beginning or requirement satisfied during message passing)
    pos_clearances = { 
        # (Clearance of positive requirements, full domain of atoms as reference)
        case: (atoms_by_ind - pos_requirements[case], atoms_by_ind)
        for case in cases
    }

    if r_pr is not None:
        # Convert probability to logit/log-weight, depending on weighting scheme
        if weighting == "logit":
            r_weight = logit(r_pr[0], large="a")
        else:
            assert weighting == "log"
            r_weight = float(np.log(r_pr[0])) if r_pr[0] > 0 else "-a"

        potential = {
            # Singleton dict (with pos_non_req as key) as value
            frozenset(case): {
                # Rule weight of exp(w) missed in case of deductive violations (i.e.
                # when body is true but head is not)
                pos_clearances[case]: Polynomial(float_val=1.0)
                    if (rb_by_ind <= case) and (len(rh_by_ind)==0 or not rh_by_ind <= case)
                    else Polynomial.from_primitive(r_weight)
            }
            for case in cases
        }
    else:
        # r_pr of None signals 'absolute' rule (i.e. even stronger than hard-weighted
        # rule) that eliminates possibility of deductive violation (body true, head
        # false), yet doesn't affect model weights
        potential = {
            frozenset(case): { pos_clearances[case]: Polynomial(float_val=1.0) }
            for case in cases
            # Zero potential for deductive violations
            if not ((rb_by_ind <= case) and (len(rh_by_ind)==0 or not rh_by_ind <= case))
        }

    return potential


def belief_propagation(bjt, atom_subset):
    """
    (Modified) Shafer-Shenoy belief propagation on binary join trees, whose
    input potential storage registers are properly filled in. Populate output
    belief storage registers as demanded by `atom_subset`.

    Lazy, incremental belief propagation is achieved by use of "updated_needed"
    fields on nodes and edges. On nodes, true "updated_needed" indicates its
    "output_beliefs" value is nonexistent or incorrect due to changes in some
    input potentials. Similarly, true "updated_needed" on edges indicates its
    message for the edge direction is nonexistent or incorrect.
    """
    # Corresponding node in BJT; there can be multiple, pick any
    bjt_node = [n for n in bjt.nodes if atom_subset==n[0]][0]

    # Fetch input potentials for the BJT node
    input_potential = bjt.nodes[bjt_node]["input_potential"]

    # Fetch incoming messages for the BJT node, if any
    incoming_messages = []
    for from_node, _, msg in bjt.in_edges(bjt_node, data="message"):
        # Message will be computed if nonexistent or outdated; otherwise,
        # cached value will be used
        _compute_message(bjt, from_node, bjt_node)
        msg = bjt.edges[(from_node, bjt_node)]["message"]

        incoming_messages.append(msg)

    # Combine incoming messages; combine entries by multiplying 'fully-alive'
    # weights and 'half-alive' weights respectively
    if len(incoming_messages) > 0:
        msgs_combined = _combine_factors_outer(incoming_messages)

        # Final binary combination of (combined) input potentials & (combined)
        # incoming messages
        inps_msgs_combined = _combine_factors_outer(
            [input_potential, msgs_combined]
        )
    else:
        # Empty messages; just consider input potential
        inps_msgs_combined = input_potential

    # (Partial) marginalization down to domain for the node set for this BJT node
    output_beliefs = _marginalize_outer(inps_msgs_combined, bjt_node[0])

    # Weed out any subcases that still have lingering positive atom requirements
    # (i.e. non-complete positive clearances), then fully marginalize per case
    output_beliefs = {
        case: sum(
            [
                val for subcase, val in inner.items()
                if len(subcase[0])==len(subcase[1])
            ],
            Polynomial(float_val=0.0)
        )
        for case, inner in output_beliefs.items()
    }

    # Populate the output belief storage register for the BJT node, and mark
    # node as up-to-date
    bjt.nodes[bjt_node]["output_beliefs"] = output_beliefs
    bjt.nodes[bjt_node]["update_needed"] = False


def _compute_message(bjt, from_node, to_node):
    """
    Recursive subroutine called during belief propagation for computing outgoing
    message from one BJT node to another; populates the corresponding directed edge's
    message storage register
    """
    if not bjt.edges[(from_node, to_node)]["update_needed"]:
        # Can use existing message value, saving computation
        assert "message" in bjt.edges[(from_node, to_node)]
        return

    # Fetch input potentials for the BJT node
    input_potential = bjt.nodes[from_node]["input_potential"]

    # Fetch incoming messages for the BJT node from the neighbors, except the
    # message recipient
    incoming_messages = []
    for neighbor_node, _, msg in bjt.in_edges(from_node, data="message"):
        if neighbor_node == to_node: continue    # Disregard to_node

        # Message will be computed if nonexistent or outdated; otherwise,
        # cached value will be used
        _compute_message(bjt, neighbor_node, from_node)
        msg = bjt.edges[(neighbor_node, from_node)]["message"]

        incoming_messages.append(msg)
    
    # Combine incoming messages; combine entries by multiplying 'fully-alive'
    # weights and 'half-alive' weights respectively
    if len(incoming_messages) > 0:
        msgs_combined = _combine_factors_outer(incoming_messages)

        # Final binary combination of (combined) input potentials & (combined)
        # incoming messages
        inps_msgs_combined = _combine_factors_outer(
            [input_potential, msgs_combined]
        )
    else:
        inps_msgs_combined = input_potential

    # (Partial) Marginalization down to domain for the intersection of from_node
    # and to_node
    common_atoms = from_node[0] & to_node[0]
    outgoing_msg = _marginalize_outer(inps_msgs_combined, common_atoms)

    # Dismissable nodes; if some nodes occur in the message sender but not in the
    # message recipient, such nodes will never appear again at any point of the
    # onward message path (due to running intersection property of join trees).
    # The implication, that we exploit here for computational efficiency, is that
    # we can now stop caring about clearances of such dismissable nodes, and if
    # needed, safely eliminating subcases that still have lingering uncleared
    # requirements of dismissable nodes.
    dismissables = from_node[0] - to_node[0]
    if len(dismissables) > 0:
        outgoing_msg = {
            case: {
                (subcase[0]-dismissables, subcase[1]-dismissables): val
                for subcase, val in inner.items()
                if len(dismissables & (subcase[1]-subcase[0])) == 0
            }
            for case, inner in outgoing_msg.items()
        }

    # Populate the outgoing message storage register for the BJT edge, and mark
    # edge as up-to-date
    bjt.edges[(from_node, to_node)]["message"] = outgoing_msg
    bjt.edges[(from_node, to_node)]["update_needed"] = False


def _combine_factors_outer(factors):
    """
    Subroutine for combining a set of input factors at the outer-layer; entries
    are combined by calling _combine_factors_inner()
    """
    assert len(factors) > 0
    # Each factors are specifications of cases sharing the same atom set signature,
    # differing only whether elements are positive or negative
    assert all(
        len({frozenset([abs(atm) for atm in ff]) for ff in f})==1
        for f in factors
    )

    # Valid unions of cases
    valid_cases = _consistent_unions(factors)

    # Compute entry values for possible cases considered
    combined_factor = {}
    for case_common, case_specifics in valid_cases.items():
        for case_sp in case_specifics:
            # Corresponding entry fetched from the factor
            entries_per_factor = [
                factors[i][frozenset.union(c, case_common)]
                for i, c in enumerate(case_sp)
            ]

            case_union = frozenset.union(case_common, *case_sp)

            # Outer-layer combination where entries can be considered 'mini-factors'
            # defined per postive atom clearances
            combined_factor[case_union] = _combine_factors_inner(entries_per_factor)

    return combined_factor


def _combine_factors_inner(factors):
    """
    Subroutine for combining a set of input factors at the inner-layer; entries
    (which are Polynomials) are multiplied then marginalized by case union
    """
    assert len(factors) > 0

    # Compute entry values for possible cases considered
    combined_factor = defaultdict(lambda: Polynomial(float_val=0.0))
    for case in product(*factors):
        # Union of case specification
        case_union = (
            frozenset.union(*[c[0] for c in case]),
            frozenset.union(*[c[1] for c in case]),
        )

        # Corresponding entry fetched from the factor
        entries_per_factor = [factors[i][c] for i, c in enumerate(case)]

        combined_factor[case_union] += reduce(operator.mul, entries_per_factor)

    return dict(combined_factor)


def _combine_factors_simple(factors):
    """
    Subroutine for combining a set of 'simple' factors without layers (likely those
    stored in 'output_beliefs' register for a BJT node); entries are combined by
    multiplication
    """
    assert len(factors) > 0

    # Compute entry values for possible cases considered
    combined_factor = {}
    for case in product(*factors):
        # Union of case specification
        case_union = frozenset.union(*case)

        # Incompatible, cannot combine
        if not _literal_set_is_consistent(case_union): continue

        # Corresponding entry fetched from the factor
        entries_per_factor = [factors[i][c] for i, c in enumerate(case)]

        # Outer-layer combination where entries can be considered 'mini-factors'
        # defined per postive atom clearances
        combined_factor[case_union] = reduce(operator.mul, entries_per_factor)

    return combined_factor


def _marginalize_outer(factor, atom_subset):
    """
    Subroutine for (partially) marginalizing a factor at the outer-layer (while
    maintaining subdivision by positive requirement clearance) down to some domain
    specified by atom_subset
    """
    marginalized_factor = defaultdict(
        lambda: defaultdict(lambda: Polynomial(float_val=0.0))
    )
    atomset_pn = frozenset(sum([(atm, -atm) for atm in atom_subset], ()))

    for f_case, f_inner in factor.items():
        matching_case = atomset_pn & f_case

        for pos_clrs, val in f_inner.items():
            marginalized_factor[matching_case][pos_clrs] += val

    marginalized_factor = {
        case: dict(outer_marginals)
        for case, outer_marginals in marginalized_factor.items()
    }

    return marginalized_factor


def _marginalize_simple(factor, atom_subset):
    """
    Subroutine for simple marginalization of factor without layers (likely those
    stored in 'output_beliefs' register for a BJT node) down to some domain specified
    by atom_subset
    """
    marginalized_factor = defaultdict(lambda: Polynomial(float_val=0.0))
    atomset_pn = frozenset(sum([(atm, -atm) for atm in atom_subset], ()))

    for f_case, f_val in factor.items():
        matching_case = atomset_pn & f_case
        marginalized_factor[matching_case] += f_val

    return dict(marginalized_factor)


def _literal_set_is_consistent(lit_set):
    """
    Subroutine for checking if a set of literals (represented with signed integer
    indices) is consistent; i.e. doesn't contain a literal and its negation at the
    same time
    """
    atm_set = {abs(lit) for lit in lit_set}

    # Inconsistent if and only if lit_set contained both atm & -atm for some atm,
    # which would be reduced to atm in atm_set
    return len(atm_set) == len(lit_set)


def _consistent_unions(factors):
    """
    Efficient factor combination by exploiting factorization with common atom
    signatures and respective complements
    """
    # Find atoms common to all signatures of the factors, then partition each
    # case in the factors according to the partition
    factor_signatures = [
        frozenset(abs(atm) for atm in list(f)[0]) for f in factors
    ]
    signature_common = frozenset.intersection(*factor_signatures)
    signature_common_pn = frozenset(
        sum([[atm, -atm] for atm in signature_common], [])
    )
    signature_diffs_pn = [
        frozenset(
            sum([[atm, -atm] for atm in f_sig-signature_common], [])
        )
        for f_sig in factor_signatures
    ]
    factors_partitioned = [
        {(ff&signature_common_pn, ff&sig_diff) for ff in f}
        for f, sig_diff in zip(factors, signature_diffs_pn)
    ]

    # Collect factor-specific cases by common cases
    factored_cases = defaultdict(lambda: defaultdict(set))
    for fi, per_factor in enumerate(factors_partitioned):
        for f_common, f_uniq in per_factor:
            factored_cases[fi][f_common].add(f_uniq)

    # Combine and expand each product of factor-specific cases
    valid_cases_common = set.intersection(*[
        set(cases_common) for cases_common in factored_cases.values()
    ])
    valid_cases_by_common = defaultdict(list)
    for case_common in valid_cases_common:
        for fi, f_common in factored_cases.items():
            valid_cases_by_common[case_common].append(
                frozenset(f_common[case_common])
            )
    valid_cases = {
        case_common: reduce(_pairwise_consistent_unions, [[]]+case_specifics)
        for case_common, case_specifics in valid_cases_by_common.items()
    }

    return valid_cases


def _pairwise_consistent_unions(cumul, next_cases):
    """
    Helper method to be provided as argument for functool.reduce(); collect
    unions of cases that are consistent (i.e. doesn't contain some positive
    literal atm and negative literal -atm at the same time), across some
    sequence of set of cases. Intended to be used with sequences whose common
    atoms in signatures are first factored out.
    """
    # Reduction of each choice of cases into component unions
    if len(cumul) > 0:
        cases_cumul = [frozenset.union(*c) for c in cumul]
    else:
        cases_cumul = [frozenset()]

    # Common atom signatures between cumulation so far and next_cases
    signature_common = {abs(atm) for atm in list(cases_cumul)[0]} & \
        {abs(atm) for atm in list(next_cases)[0]}
    signature_common_pn = set(sum([(-atm, atm) for atm in signature_common], ()))

    cumul_new = []
    for i, lits1 in enumerate(cases_cumul):
        cm_case1 = lits1 & signature_common_pn

        for lits2 in next_cases:
            cm_case2 = lits2 & signature_common_pn

            # Add to new cumulation list if not inconsistent
            if cm_case1 == cm_case2:
                if len(cumul) > 0:
                    cumul_new.append(cumul[i] + (lits2,))
                else:
                    cumul_new.append((lits2,))

    return cumul_new
