""" Program().compile() subroutine factored out """
import operator
from functools import reduce
from itertools import product
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
    Compiles program (or the equivalent directed graph thereof) into a region
    graph, which would contain data needed to answer probabilistic inference
    queries after running sufficient number of (generalized?) belief propagation
    message passing iterations. Achieved by running the following procedure:

    1) Ground the program with clingo to find set of actually grounded atoms.
    2) Identify factor potentials associated with valid sets of nodes. Each
        program rule corresponds to a piece of information contributing to the
        final factor potential for each node set. Essentially, this amounts to
        finding a factor graph corresponding to the provided program.
    3) Compile the factor graph into a region graph, which represents the (G)BP
        (G)BP algorithm instance corresponding to the graph (cf. Yedidia et al.,
        2004). Currently uses Bethe's method of approximating Gibbs free energy,
        ultimately corresponding to the standard loopy BP algorithm on factor
        graphs. (May try to employ a more accurate approximation mechanism later?)
        Note the message passing procedure is modified to comply with semantics
        of normal logic programs.
    4) Run modified belief propagation algorithm until convergence to fill in
        values needed for answering probabilistic inference queries later.
    
    Returns a region graph populated with the required values resulting from belief
    propagation.
    """
    # Region graph returned. Each node is indexed with a pair (frozenset of atoms,
    # frozenset of factors) and labeled with a counting number. Each (directed) edge
    # connects a region to its subregion and stores a message from the source to the
    # target. Also holds a graph-level storage of a mapping between grounded atoms
    # and their integer indices, and also input potential tables for each factor (as
    # factors can appear in different region nodes and their potentials are shared).
    reg_gr = nx.DiGraph()

    if _grounded_facts_only([r for r, _, _ in prog.rules]):
        # Can take simpler shortcut for constructing region graph if program consists
        # only of grounded facts
        reg_gr.graph["atoms_map"] = {}
        reg_gr.graph["atoms_map_inv"] = {}
        reg_gr.graph["factor_potentials"] = {}

        # Integer indexing should start from 1, to represent negated atoms as
        # negative ints
        ai = 1
        for i, (rule, r_pr, weighting) in enumerate(prog.rules):
            # Mapping between atoms and their integer indices
            reg_gr.graph["atoms_map"][rule.head[0]] = ai
            reg_gr.graph["atoms_map_inv"][ai] = rule.head[0]

            # Add an appropriate input potential from the rule weight
            reg_gr.graph["factor_potentials"][i] = _rule_to_potential(
                rule, r_pr, weighting, { rule.head[0]: ai }
            )

            # Add a large region ~ small region node pair
            large_label = (frozenset([ai]), frozenset([i]))
            small_label = (frozenset([ai]), frozenset())
            reg_gr.add_node(large_label)
            reg_gr.add_node(small_label)
            reg_gr.add_edge(large_label, small_label)
            reg_gr.edges[(large_label, small_label)]["message"] = (None, set())

            ai += 1

    else:
        grounded_rules_by_atms, atoms_map, atoms_map_inv = _ground_and_index(prog)

        # Mapping between atoms and their integer indices
        reg_gr.graph["atoms_map"] = atoms_map
        reg_gr.graph["atoms_map_inv"] = atoms_map_inv

        # Maintain factor potentials in graph-level (instead of node-level) storage
        reg_gr.graph["factor_potentials"] = {}

        # Implements Bethe's approximation method, enumerating over each collection of
        # grounded rules (combined & considered as a factor with the atoms as argument)
        for i, (atms, gr_rules) in enumerate(grounded_rules_by_atms.items()):
            # Skip invalid rule collections
            if len(atms) == 0: continue

            # Convert each rule to matching potential
            input_potentials = [
                _rule_to_potential(gr_rule, r_pr, weighting, atoms_map)
                for gr_rule, (r_pr, weighting), _ in gr_rules
            ]

            if any(inp is None for inp in input_potentials):
                # Falsity included, program has no answer set
                return None

            # Combine the converted input potentials into a single factor potential
            # then store
            reg_gr.graph["factor_potentials"][i] = _combine_factors_outer(input_potentials)

            # A node for each 'large region' containing the index of the (combined)
            # factor and the argument atoms
            large_label = (atms, frozenset([i]))
            reg_gr.add_node(large_label)

            for ai in atms:
                # A node for each 'small region' containing exactly one atom
                small_label = (frozenset([ai]), frozenset())
                reg_gr.add_node(small_label)

                # An edge from the 'large region' node added above to each 'small
                # 'region' node
                reg_gr.add_edge(large_label, small_label)
                reg_gr.edges[(large_label, small_label)]["message"] = (None, set())

    # Annotate each region node with its counting number
    all_annotated = False
    while not all_annotated:
        for r in reg_gr.nodes:
            parents = [p for p, _ in reg_gr.in_edges(r)]
            if all("counting_num" in reg_gr.nodes[p] for p in parents):
                reg_gr.nodes[r]["counting_num"] = 1 - sum(
                    reg_gr.nodes[p]["counting_num"] for p in parents
                )

        all_annotated = all("counting_num" in reg_gr.nodes[r] for r in reg_gr.nodes)

    reg_gr.graph["converged"] = False

    return reg_gr


def rg_query(reg_gr, q_key):
    """ Query a region graph for (unnormalized) belief table """
    relevant_nodes = frozenset({abs(n) for n in q_key})
    relevant_cliques = [
        (atms, fcts) for atms, fcts in reg_gr.nodes if relevant_nodes <= atms
    ]

    if len(relevant_cliques) > 0:
        # In-clique query; find the region node with the smallest node set that
        # comply with the key
        smallest_node = sorted(
            relevant_cliques, key=lambda x: len(list(x[0])+list(x[1]))
        )[0]

        if not reg_gr.graph["converged"]:
            belief_propagation(reg_gr)
        beliefs = reg_gr.nodes[smallest_node]["beliefs"]

        # Marginalize and return
        return _marginalize_simple(beliefs, relevant_nodes)

    else:
        # Not really used for now. Make sure later, when needed, this works correctly
        raise NotImplementedError


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
        for ra in prog.rules_by_atom
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


def belief_propagation(reg_gr):
    """
    Generalized belief propagation algorithm, the parent-to-child flavor (cf. Yedidia
    et al., 2004). Iterate on the set of (products of) messages until convergence,
    so that the marginalized beliefs obtained afterwards are 'sufficiently' accurate.
    (Note that we cannot run the standard recursive BP for exact inference since the
    provided region graph might have loops and thus no terminal nodes.) If the provided
    region graph is obtained by Bethe approximation, this amounts to loopy BP in effect,
    resulting in the same output values.
    """
    # Extracting invariant information that will be used throughout the iteration
    # from the provided region graph

    # Dict containing all parent-to-child message identifiers; edge source & target
    # are dict keys, and values are '3-layer partitioning' of the region graph for
    # the specified edge (that define N(P,R) and D(P,R) in the paper by Yedidia et al.)
    p2r_edges = {
        (p, r): (
            set(reg_gr.nodes) - ({p} | nx.descendants(reg_gr, p)),
            ({p} | nx.descendants(reg_gr, p) - ({r} | nx.descendants(reg_gr, r))),
            ({r} | nx.descendants(reg_gr, r))
        )
        for p, r in reg_gr.edges
    }

    for _ in range(10):
        # Continue iterating over the parent-to-child messages to update their values,
        # until convergence
        # (Caveat: Not explicitly testing for convergence currently, max 15 iterations)

        # Collect old message values
        msgs_old = { e: reg_gr.edges[e].get("message") for e in reg_gr.edges }

        for (p, r), partition in p2r_edges.items():
            # Compute messages (products) obtained by equating marginalized parent belief
            # to child belief
            _update_message(reg_gr, p, r, msgs_old, partition)

        # Compute beliefs at current iteration (at variable nodes only, for now)
        msgs_now = { e: reg_gr.edges[e].get("message") for e in reg_gr.edges }
        for r in reg_gr.nodes():
            atoms, factors = r
            full_atom = reg_gr.graph["atoms_map_inv"][list(atoms)[0]]
            if len(atoms) > 1 or len(factors) > 0: continue       # Temporary...?
            if not full_atom.name.startswith("cls_"): continue    # Temporary...?

            # Factor potential products
            if len(factors) > 0:
                factors_combined = _combine_factors_outer([
                    reg_gr.graph["factor_potentials"][fi] for fi in factors
                ])
            else:
                factors_combined = None

            # Message potential products
            D_r = nx.descendants(reg_gr, r); E_r = {r} | D_r
            messages_needed = {
                e for e in reg_gr.in_edges(r)
            } | {
                (pd, d) for d in D_r for pd, _ in reg_gr.in_edges(d)
                if pd not in E_r
            }           # Parents to the node, other non-descendant parents of descendants

            # See if the collection of the needed messages can be cleanly assembled from
            # the available message products. If not possible, division has to happen...
            available_msg_products = {
                frozenset({e} | remainder): e for e, (_, remainder) in msgs_now.items()
            }           # Product key to edge handle value
            products_by_size = sorted(available_msg_products, key=len, reverse=True)
            # Greedy heuristic: First use as many larger chunks as possible, then fill in
            # any remainder using message singletons (with necessary divisions marked)
            recipe = []
            for prd in products_by_size:
                if prd <= messages_needed:
                    recipe.append((available_msg_products[prd], None))  # (edge handle, divisor)
                    messages_needed -= prd
            if len(messages_needed) > 0:
                recipe += [(e, frozenset(msgs_now[e][1])) for e in messages_needed]

            # Then combine with the messages as needed
            all_msg_potentials = [
                [
                    msgs_now[e][0],
                    _exp_m1(msgs_now[available_msg_products[divisors]][0]) \
                        if divisors is not None else None
                ]
                for e, divisors in recipe
            ]
            all_msg_potentials = [
                pot for potentials in all_msg_potentials for pot in potentials
                if pot is not None
            ]
            if len(all_msg_potentials) > 0:
                messages_combined = _combine_factors_outer(all_msg_potentials)
                if factors_combined is None:
                    all_potentials_combined = messages_combined
                else:
                    all_potentials_combined = _combine_factors_outer(
                        [factors_combined, messages_combined]
                    )
            else:
                all_potentials_combined = factors_combined

            # (Partial) marginalization down to domain for the node set for this node
            current_beliefs = _marginalize_outer(all_potentials_combined, atoms)

            # Weed out any subcases that still have lingering positive atom requirements
            # (i.e. non-complete positive clearances), then fully marginalize per case
            current_beliefs = {
                case: sum(
                    [
                        val for subcase, val in inner.items()
                        if len(subcase[0])==len(subcase[1])
                    ],
                    Polynomial(float_val=0.0)
                )
                for case, inner in current_beliefs.items()
            }

            # Update the output belief storage register for the node
            reg_gr.nodes[r]["beliefs"] = current_beliefs
    
    reg_gr.graph["converged"] = True


def _update_message(reg_gr, from_node, to_node, msgs_old, partition):
    """
    Subroutine called during belief propagation for computing outgoing message from
    one region graph node to another; updates the corresponding directed edge's message
    storage register. As we are dealing with region graphs, update rules are obtained
    by equating marginalized beliefs of `from_node` to beliefs of `to_node`, utilizing
    the 3-layer partitioning of the region graph.
    """
    # Variables (atoms) and factor sets for the two nodes
    atms_from, factors_from = from_node
    atms_to, factors_to = to_node

    # Top, middle, bottom 3-layered partitions
    top_pt, mid_pt, btm_pt = partition

    # Identify the old messages needed from all existing edges originating from
    # nodes in the top partition and ending up in the middle partition (N(P,R))
    messages_old_needed = {(p, r) for p, r in reg_gr.edges if p in top_pt and r in mid_pt}

    # Identify the new messages to be updated from all existing edges originating
    # from nodes in the middle partition and ending up in the bottom partition
    # (D(P,R)); always includes the message from `from_node` to `to_node`
    messages_to_update = {(p, r) for p, r in reg_gr.edges if p in mid_pt and r in btm_pt}
    assert (from_node, to_node) in messages_to_update

    # Fetch and combine input potentials, if any, for factors in `from_node` but
    #not in `to_node`
    factors_needed = factors_from - factors_to
    if len(factors_needed) > 0:
        factors_combined = _combine_factors_outer([
            reg_gr.graph["factor_potentials"][fi] for fi in factors_needed
        ])
    else:
        factors_combined = None

    # See if the collection of the needed messages can be cleanly assembled from
    # the available message products. If not possible, division has to happen...
    available_msg_products = {
        frozenset({e} | remainder): e for e, (_, remainder) in msgs_old.items()
    }           # Product key to edge handle value
    products_by_size = sorted(available_msg_products, key=len, reverse=True)
    # Greedy heuristic: First use as many larger chunks as possible, then fill in
    # any remainder using message singletons (with necessary divisions marked)
    recipe = []
    for prd in products_by_size:
        if prd <= messages_old_needed:
            recipe.append((available_msg_products[prd], None))  # (edge handle, divisor)
            messages_old_needed -= prd
    if len(messages_old_needed) > 0:
        recipe += [(e, frozenset(msgs_old[e][1])) for e in messages_old_needed]

    # Then combine with the messages as needed
    all_msg_potentials = [
        [
            msgs_old[e][0],
            _exp_m1(msgs_old[available_msg_products[divisors]][0]) \
                if divisors is not None else None
        ]
        for e, divisors in recipe
    ]
    all_msg_potentials = [
        pot for potentials in all_msg_potentials for pot in potentials
        if pot is not None
    ]
    if len(all_msg_potentials) > 0:
        messages_old_combined = _combine_factors_outer(all_msg_potentials)
        if factors_combined is None:
            all_potentials_combined = messages_old_combined
        else:
            all_potentials_combined = _combine_factors_outer(
                [factors_combined, messages_old_combined]
            )
    else:
        all_potentials_combined = factors_combined

    # (Partial) Marginalization down to domain for the common atoms, now equal to
    # the 'right-hand side' of the marginalization constraint equation
    righthand_side = _marginalize_outer(all_potentials_combined, atms_from & atms_to)

    # Dismissable nodes; if some nodes occur in `from_node` but not in `to_node`,
    # such nodes will never appear again at any point of the onward message path
    # due to the definitional characteristic of region graphs. The implication,
    # that we exploit here for computational efficiency, is that we can now stop
    # caring about clearances of such dismissable nodes, and if needed, safely
    # eliminating subcases that still have lingering uncleared requirements of
    # dismissable nodes.
    dismissables = atms_from - atms_to
    if len(dismissables) > 0:
        righthand_side = {
            case: {
                (subcase[0]-dismissables, subcase[1]-dismissables): val
                for subcase, val in inner.items()
                if len(dismissables & (subcase[1]-subcase[0])) == 0
            }
            for case, inner in righthand_side.items()
        }

    righthand_side = _normalize_potential(righthand_side)

    # Populate the message storage register for the edge with the pair (RHS value,
    # remainder of the message product if any)
    old_potential = reg_gr.edges[(from_node, to_node)]["message"]
    new_potential = _mix_factors_outer(old_potential[0], righthand_side)
    new_potential = _normalize_potential(new_potential)

    reg_gr.edges[(from_node, to_node)]["message"] = (
        new_potential, messages_to_update - {(from_node, to_node)}
    )


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
    stored in 'beliefs' register for a region node) down to some domain specified
    by atom_subset
    """
    marginalized_factor = defaultdict(lambda: Polynomial(float_val=0.0))
    atomset_pn = frozenset(sum([(atm, -atm) for atm in atom_subset], ()))

    for f_case, f_val in factor.items():
        matching_case = atomset_pn & f_case
        marginalized_factor[matching_case] += f_val

    return dict(marginalized_factor)


def _mix_factors_outer(old_factor, update_factor, momentum=0.1):
    """
    Subroutine for mixing potential values of previous iteration with a newer
    one with a certain mixing ratio (momentum), at the outer layer
    """
    assert momentum > 0 and momentum <= 1
    mmt_1m_poly = Polynomial(float_val=1-momentum)

    if old_factor is None:
        # Old is empty
        new_factor = {
            case: {
                subcase: val * mmt_1m_poly
                for subcase, val in inner.items()
            }
            for case, inner in update_factor.items()
        }
    else:
        # Valid unions of cases
        valid_cases = _consistent_unions([old_factor, update_factor])

        # Compute entry values for possible cases considered
        new_factor = {}
        for case_common, case_specifics in valid_cases.items():
            for case_sp in case_specifics:
                # Corresponding entry fetched from the factor
                old_entry = old_factor[frozenset.union(case_sp[0], case_common)]
                update_entry = update_factor[frozenset.union(case_sp[1], case_common)]

                case_union = frozenset.union(case_common, *case_sp)

                # Outer-layer combination where entries can be considered 'mini-factors'
                # defined per postive atom clearances
                new_factor[case_union] = _mix_factors_inner(
                    old_entry, update_entry, momentum
                )

    return new_factor


def _mix_factors_inner(old_factor, update_factor, momentum):
    """
    Subroutine for mixing two factors at the inner-layer, using the provided
    momentum as mixing ratio 
    """
    mmt_poly = Polynomial(float_val=momentum)
    mmt_1m_poly = Polynomial(float_val=1-momentum)

    # Compute entry values for possible cases considered
    mixed_factor = {}
    for case in set(old_factor) | set(update_factor):
        # Corresponding entry fetched from the factor, with default value of 0
        old_entry = old_factor.get(case, Polynomial(float_val=0.0))
        update_entry = update_factor.get(case, Polynomial(float_val=0.0))

        # Linear mixing with momentum ratio
        new_entry = (old_entry * mmt_poly) + (update_entry * mmt_1m_poly)
        mixed_factor[case] = new_entry

    return mixed_factor


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


def _normalize_potential(potential):
    """ Normalize the unnormalized potential entries """
    Z = sum(
        [val for inner in potential.values() for val in inner.values()],
        Polynomial(float_val=0)
    )
    rescaled_potential = {
        case: {
            subcase: val / Z
            for subcase, val in inner.items()
        }
        for case, inner in potential.items()
    }

    return rescaled_potential


# Power of -1 to all potential table entries
_exp_m1 = lambda pot: {
    case: {
        subcase: Polynomial(float_val=1.0) / val
        for subcase, val in inner.items()
    }
    for case, inner in pot.items()
}
