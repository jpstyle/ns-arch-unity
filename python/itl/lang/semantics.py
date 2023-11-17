import os
import string
from itertools import product
from collections import defaultdict

import inflect
from delphin import ace, predicate


class SemanticParser:
    """
    Semantic parser that first processes free-form language inputs into MRS, and then
    translates them into ASP-friendly custom formats
    """
    def __init__(self, grammar, ace_bin):
        self.grammar = grammar
        self.ace_bin = f"{ace_bin}/ace"

        # For redirecting ACE's stderr (so that messages don't print on console)
        self.null_sink = open(os.devnull, "w")
    
    def __del__(self):
        """ For closing the null sink before destruction """
        self.null_sink.close()

    def nl_parse(self, usr_in):
        assert isinstance(usr_in, list)

        parses = []         # Return value
        for utt_string in usr_in:
            parse = {
                "relations": {
                    "by_args": defaultdict(lambda: []),
                    "by_id": {}
                },
                "utt_type": {},
                "clause_source": {},
                "raw": utt_string
            }

            # For now use the top result
            mrs_out = ace.parse(
                self.grammar, utt_string, executable=self.ace_bin, stderr=self.null_sink
            )
            mrs_out = mrs_out.result(0).mrs()

            # Map from higher-scoped handles to lower-scoped handles in parser output;
            # chiefly used for handling complement clause args
            hcons_map = { hc.hi: hc.lo for hc in mrs_out.hcons if hc.relation=="qeq" }
            h2id_map = { ep.label: ep.id for ep in mrs_out.rels }

            # Extract essential parse, assuming flat structure (no significant semantic
            # scoping or quantifier ambiguity)
            for ep in mrs_out.rels:

                args = {
                    arg: ep.args[arg] for arg in ep.args
                    if arg.startswith("ARG") and len(arg)>3
                }
                args = [ep.args[f"ARG{i}"] for i in range(len(args))]

                # Handling complement clause arguments; handle labels to higher-to-lower
                # scoped handles, then to matching referent ids (assuming single-chain
                # hcons right now)
                args = [
                    h2id_map[hcons_map[a]] if a in hcons_map else a
                    for a in args
                ]

                # Cast into tuples (hashable) so that they can be used as dict keys as well
                args = tuple(args)

                rel = {
                    "args": args,
                    "lexical": ep.predicate.startswith("_"),
                    "id": ep.id,
                    "handle": ep.label,
                    "crange": (ep.cfrom, ep.cto)
                }

                if "ARG" in ep.args: rel["arg_udef"] = ep.args["ARG"]

                if ep.predicate.startswith("_"):
                    # 'Surface' (in ERG parlance) predicates with lexical symbols
                    lemma, pos, sense = predicate.split(ep.predicate)

                    if lemma == "and" and pos == "c": continue

                    if sense == "unknown":
                        # Predicate not covered by parser lexicon
                        lemma, pos = lemma.split("/")

                        # Multi-clause input occasionally includes '.' after unknown nouns
                        lemma = lemma.strip(".")

                        # Translate the tag obtained from the POS tagger to corresponding
                        # MRS POS code
                        if pos.startswith("n"):
                            pos = "n"

                            # Ensure singularized lemma
                            singularized = inflect.engine().singular_noun(lemma)
                            if singularized:
                                # singular_noun() returns False if already singular. Weird
                                # behavior, but welp
                                lemma = singularized
                        elif pos.startswith("j"):
                            # In our use case, unknown predicate with jj POS are mostly
                            # fragments of compound noun tokens (e.g. "pinot"). Let's
                            # intervene and manipulate the parse result as if relations
                            # came from compound nouns
                            pos = "n"

                            # Instance about which this 'jj' originally predicates
                            predicated_inst = rel["args"][1]

                            # Need new instance var index
                            max_x = max(
                                int(ep.iv.strip("x")) for ep in mrs_out.rels
                                if ep.iv.startswith("x")
                            )
                            rel["id"] = f"x{max_x+1}"
                            rel["args"] = (f"x{max_x+1}",)

                            # Need new event var index
                            max_e = max(
                                int(ep.iv.strip("e")) for ep in mrs_out.rels
                                if ep.iv.startswith("e")
                            )

                            # Need new handle indices
                            max_h = max(int(ep.label.strip("h")) for ep in mrs_out.rels)
                            rel["handle"] = f"h{max_h+1}"

                            # Add two additional relations: udef and compound
                            new_udef_rel = {
                                "args": rel["args"],
                                "lexical": False,
                                "id": f"q{max_x+1}",
                                "handle": f"h{max_h+2}",
                                "crange": rel["crange"],
                                "predicate": "udef",
                                "pos": "q"
                            }
                            parse["relations"]["by_args"][rel["args"]].append(new_udef_rel)
                            parse["relations"]["by_id"][f"q{max_x+1}"] = new_udef_rel

                            cmpd_args = (
                                f"e{max_e+1}", predicated_inst, f"x{max_x+1}"
                            )
                            cmpd_rel = [
                                ep for ep in mrs_out.rels if ep.id==predicated_inst
                            ][0]        # Label of instance being predicated
                            new_cmpd_rel = {
                                "args": cmpd_args,
                                "lexical": False,
                                "id": f"e{max_e+1}",
                                "handle": cmpd_rel.label,
                                "crange": (rel["crange"][0], cmpd_rel.cto),
                                "predicate": "compound",
                                "pos": None
                            }
                            parse["relations"]["by_args"][cmpd_args].append(new_cmpd_rel)
                            parse["relations"]["by_id"][f"e{max_e+1}"] = new_cmpd_rel
                            
                        elif pos.startswith("v"):
                            pos = "v"
                        elif pos.startswith("fw"):
                            # Foreign token (e.g. 'noir')
                            pos = "n"
                        else:
                            raise ValueError(f"I don't know how to grammatically process the token '{lemma}'")

                    # (R)MRS pos codes to WordNet synset pos tags
                    if pos == "p": pos = "r"

                    rel.update({
                        "predicate": lemma,
                        "pos": pos,
                        "sense": sense
                    })

                else:
                    # Reserved, 'abstract' (in ERG parlance) predicates
                    if ep.is_quantifier():
                        # Abstract quantifier predicate
                        rel.update({
                            "predicate": ep.predicate.strip("_q"),
                            "pos": "q",
                            "lexical": False
                        })

                    else:
                        # Abstract non-quantifier predicate
                        rel.update({
                            "predicate": ep.predicate,
                            "pos": None
                        })

                        if ep.predicate == "named":
                            rel["predicate"] = ep.carg.lower()

                        if ep.predicate == "neg":
                            negated = hcons_map[ep.args["ARG1"]]
                            negated = [ep for ep in mrs_out.rels if ep.label == negated][0]

                            rel["neg"] = negated.label
                    
                parse["relations"]["by_args"][rel["args"]].append(rel)
                parse["relations"]["by_id"][rel["id"]] = rel
            
            parse["relations"]["by_args"] = dict(parse["relations"]["by_args"])
            
            # SF supposedly stands for 'sentential force' -- handle the utterance types
            # here, along with conjunctions
            def index_clauses_with_SF(rel_id, parent_id):
                index_rel = parse["relations"]["by_id"][rel_id]
                if index_rel["predicate"] == "implicit_conj":
                    # Recursively process conjuncts
                    index_clauses_with_SF(index_rel["args"][1], rel_id)
                    index_clauses_with_SF(index_rel["args"][2], rel_id)
                else:
                    parse["utt_type"][rel_id] = mrs_out.variables[rel_id]["SF"]

                    # Recover part of original input corresponding to this clause by
                    # iteratively expanding set of args covered by this clause, then
                    # obtaining min/max of the cranges
                    args_covered = {rel_id}; fixpoint_reached = False
                    while not fixpoint_reached:
                        new_coverage = set()
                        for id, rel in parse["relations"]["by_id"].items():
                            if parent_id in rel["args"]: continue

                            covered = any(arg in args_covered for arg in rel["args"])
                            if covered:
                                new_coverage |= set(rel["args"]+(id,))

                        fixpoint_reached = len(new_coverage - args_covered) == 0
                        args_covered |= new_coverage

                    covered_cranges = [
                        parse["relations"]["by_id"][arg]["crange"]
                        for arg in args_covered if arg in parse["relations"]["by_id"]
                    ]
                    cj_start = min(cfrom for cfrom, _ in covered_cranges)
                    cj_end = max(cto for _, cto in covered_cranges)
                    parse["clause_source"][rel_id] = parse["raw"][cj_start:cj_end]

            index_clauses_with_SF(mrs_out.index, None)

            # Record index (top variable)
            parse["index"] = mrs_out.index

            parses.append(parse)

        return parses

    @staticmethod
    def asp_translate(parses):
        """ Additional translation of MRS parser outputs into ASP-friendly format """
        translations = []
        ref_maps = []
        for parse in parses:
            # First find chains of compound NPs and then appropriately merge them
            cmpd_rels = [
                rel for rel in parse["relations"]["by_id"].values()
                if rel["lexical"] == False and rel["predicate"]=="compound"
            ]
            while len(cmpd_rels) > 0:
                # Pick a compound rel arg pair that is immediately mergeable; i.e.
                # one without any smaller crange
                mergeable_rels = [
                    rel for rel in cmpd_rels
                    if not any(
                        rel2["crange"][0] >= rel["crange"][0] and
                            rel2["crange"][1] <= rel["crange"][1] and
                            rel2["id"] != rel["id"]
                        for rel2 in cmpd_rels
                    )
                ]
                rel_to_merge = mergeable_rels[0]

                # Fetch predicate rels; notice how arg order flips (1-2 vs. 2-1),
                # due to how MRS handles compound rels
                p1 = parse["relations"]["by_id"][rel_to_merge["args"][2]]
                p2 = parse["relations"]["by_id"][rel_to_merge["args"][1]]

                # Merge & camelCase predicate names, update p2 predicate & crange
                merged = p1["predicate"] + p2["predicate"][:1].upper() + p2["predicate"][1:]
                p2["predicate"] = merged
                p2["crange"] = (p1["crange"][0], p2["crange"][1])

                # Delete any traces of p1; it's now merged into p2
                parse["relations"]["by_args"] = {
                    args: rels for args, rels in parse["relations"]["by_args"].items()
                    if p1["id"] not in args
                }
                parse["relations"]["by_id"] = {
                    r["id"]: r
                    for rels in parse["relations"]["by_args"].values() for r in rels
                }

                # Update list of mergeable relations
                cmpd_rels = [
                    rel for rel in cmpd_rels if rel["id"] != rel_to_merge["id"]
                ]

            # Equivalence classes mapping referents to rule variables, while tracking whether
            # they come from referential expressions or not
            ref_map = {}
            ref_map[parse["index"]] = {
                "provenance": parse["relations"]["by_id"][parse["index"]]["crange"],
                "irrealis": False,
                "domain_describing": True
            }       # Filling with the index event referent info

            # Negated relations by handle
            negs = [rel["neg"] for rel in parse["relations"]["by_id"].values()
                if rel["predicate"] == "neg"]

            # Traverse the semantic dependency tree represented by parse to collect appropriately
            # structured set of ASP literals
            translation = _traverse_dt(parse, parse["index"], ref_map, set(), negs)

            # Tag instance referents ('x*') with the index event ids of their source
            # clauses (complement clauses, 'conjunct' of implicit_conj, etc.)
            for index_id, (topic_msgs, focus_msgs) in translation.items():
                for msg in topic_msgs + focus_msgs:
                    if isinstance(msg, tuple):
                        # Positive message, as single tuple entry
                        args = msg[2]
                        for arg in args:
                            if not (isinstance(arg, tuple) or arg.startswith("x")):
                                continue
                            ref_map[arg]["source_evt"] = index_id
                    else:
                        # Negative message, consisting of negated msgs
                        assert isinstance(msg, list)
                        for nmsg in msg:
                            args = nmsg[2]
                            for arg in args:
                                if not (isinstance(arg, tuple) or arg.startswith("x")):
                                    continue
                                ref_map[arg]["source_evt"] = index_id

            # We assume bare NPs (underspecified quant.) have universal reading when they are arg1
            # of index (and existential reading otherwise)
            is_bare = lambda rf: (rf,) in parse["relations"]["by_args"] and any([
                r["pos"] == "q" and r["predicate"] == "udef"
                for r in parse["relations"]["by_args"][(rf,)]
            ])
            for ev_id in translation:
                ev_arg1 = parse["relations"]["by_id"][ev_id]["args"][1]
                if is_bare(ev_arg1):
                    ref_map[ev_arg1]["univ_quantified"] = True
            for ref in ref_map:
                # Handle function terms as well
                if isinstance(ref, tuple):
                    # If all function term args are universally quantified
                    if all(ref_map[fa]["univ_quantified"] for fa in ref[1]):
                        ref_map[ref]["univ_quantified"] = True
            
            translations.append(translation)
            ref_maps.append(dict(ref_map))

        return translations, ref_maps


def _traverse_dt(parse, rel_id, ref_map, covered, negs):
    """
    Recursive pre-order traversal of semantic dependency tree of referents (obtained
    from MRS)
    """
    translated = {}         # Return value

    # Can short-circuit for referents
    # Handle conjunction
    if rel_id in parse["relations"]["by_id"]:
        if parse["relations"]["by_id"][rel_id]["predicate"] == "implicit_conj":
            # Conjunction arg1 & arg2 id
            conj_args = parse["relations"]["by_id"][rel_id]["args"]

            # Make sure parts of arg1/arg2 are each processed only by the first/second
            # pass
            ref_map[conj_args[1]] = None
            ref_map[conj_args[2]] = None
            covered.add(rel_id)
            
            a1_out = _traverse_dt(parse, conj_args[1], ref_map, covered, negs)
            a2_out = _traverse_dt(parse, conj_args[2], ref_map, covered, negs)

            return {**a1_out, **a2_out}

    # Helper methods that check whether a discourse referent is introduced by a
    # referential expression | universally quantified | wh-quantified
    is_referential = lambda rf: (rf,) in parse["relations"]["by_args"] and any([
        (r["pos"] == "q" and r["predicate"] == "the") \
            or (r["pos"] == "q" and r["predicate"] == "pronoun") \
            or (r["pos"] is None and r["predicate"] == "named") \
            or (r["pos"] == "q" and "sense" in r and r["sense"] == "dem")
        for r in parse["relations"]["by_args"][(rf,)]
    ])
    is_univ_quantified = lambda rf: (rf,) in parse["relations"]["by_args"] and any([
        r["pos"] == "q" and r["predicate"] in {"all", "every"}
        for r in parse["relations"]["by_args"][(rf,)]
    ])
    is_wh_quantified = lambda rf: (rf,) in parse["relations"]["by_args"] and any([
        r["pos"] == "q" and r["predicate"] == "which"
        for r in parse["relations"]["by_args"][(rf,)]
    ])

    # Collect 'relevant' relations that have rel_id as arg
    rel_rels = {args: rels for args, rels in parse["relations"]["by_args"].items()
        if rel_id in args}

    rel_rels_to_cover = []

    # This node is connected to nodes with ids not registered in ref_map yet
    daughters = set()

    for args, rels in rel_rels.items():
        for arg in args:
            if arg not in ref_map and arg != rel_id: daughters.add(arg)

        for rel in rels:
            if rel["id"] not in covered:
                rel_rels_to_cover.append(rel)
                covered.add(rel["id"])

    # Then register the remaining referents unless already registered
    for id in daughters:
        if id.startswith("x"):
            max_map_id = max(
                [v.get("map_id", -1) for v in ref_map.values() if v is not None],
                default=-1
            )
            ref_map[id] = {
                "map_id": max_map_id+1,
                "provenance": parse["relations"]["by_id"][id]["crange"],
                    # Referent's source expression in original utterance, represented
                    # as char range
                "referential": is_referential(id),
                "univ_quantified": is_univ_quantified(id),
                "wh_quantified": is_wh_quantified(id),
                "is_pred": False
                    # Whether it is a predicate (False; init default) or individual (True)
            }
        elif id.startswith("e"):
            # Event referents, give them a dict (TODO?: handle stative verb predicates?)
            if "neg" in parse["relations"]["by_id"][id]["predicate"]:
                # We won't treat negation of an event as a standalone event, though
                continue

            ref_map[id] = {
                "provenance": parse["relations"]["by_id"][id]["crange"],
                "irrealis": False,          # False by default, may be overridden later
                "domain_describing": True   # True by default, may be overridden later
                # Any other info to fill in?
            }
        else:
            # For referents we are not interested about
            ref_map[id] = None

    # Recursive calls to traversal function for collecting predicates specifying each
    # daughter
    dt_translations = {
        id: _traverse_dt(parse, id, ref_map, covered, negs) for id in daughters
    }

    # Need the returned translations flattened for the daughters in most times
    dt_translations_flattened = {
        id: sum([tp+fc for tp, fc in outs.values()], [])
        for id, outs in dt_translations.items()
    }

    # Return empty list if key not found; no further predicates found (i.e. leaf node)
    dt_translations = defaultdict(lambda: ([], []), dt_translations)
    dt_translations_flattened = defaultdict(lambda: [], dt_translations_flattened)

    # Build return values; resp. topic & focus
    topic_msgs = []; focus_msgs = []

    for rel in rel_rels_to_cover:
        # Procedure for processing each relevant relation factored out for brevity's
        # sake (see the refactored method below)
        topic_msgs_rel, focus_msgs_rel, separable_clauses = _process_rel(
            parse, rel, rel_id, ref_map, negs,
            dt_translations, dt_translations_flattened
        )
        topic_msgs += topic_msgs_rel
        focus_msgs += focus_msgs_rel
        translated.update(separable_clauses)

    # Recognize any why-question-patterns (i.e. for+which_reason) and re-format the
    # translation appropriately
    all_msgs = topic_msgs + focus_msgs
    for_msgs = [msg for msg in all_msgs if msg[0]=="for"]
    wh_reason_msgs = [
        msg for msg in all_msgs
        if msg[0]=="reason" and ref_map[msg[2][0]]["wh_quantified"]
    ]
    valid_why_patterns = [
        (for_msg, wh_rs_msg) for for_msg, wh_rs_msg in product(for_msgs, wh_reason_msgs)
        if for_msg[2][1] == wh_rs_msg[2][0]
    ]

    # Remove the pattern fragments from the main topic/focus messages to be returned
    # and add a new eventuality standing for 'explanation rhetorical relation'
    for for_msg, wh_rs_msg in valid_why_patterns:
        if for_msg in topic_msgs: topic_msgs.remove(for_msg)
        if for_msg in focus_msgs: focus_msgs.remove(for_msg)
        if wh_rs_msg in topic_msgs: topic_msgs.remove(wh_rs_msg)
        if wh_rs_msg in focus_msgs: focus_msgs.remove(wh_rs_msg)

        expl_ev_id = [
            id for id, rel in parse["relations"]["by_id"].items()
            if rel["predicate"]=="for" and rel["args"][1:3]==for_msg[2]
        ][0]
        expl_lit = ("expl", "*", (for_msg[2][1], for_msg[2][0]))
            # More intuitive arg order: `event` for `reason` => `reason` explains `event`
        translated.update({ expl_ev_id: ([], [expl_lit]) })

        # Channel the interrogative SF to the newly introduced eventuality, while
        # converting that of the explanation target to 'prop'
        parse["utt_type"][expl_ev_id] = "ques"
        parse["utt_type"][for_msg[2][0]] = "prop"

        # Also annotating the source of the rhetorical relation
        clause_source = f"# rhetorical relation due to 'why'"
        parse["clause_source"][expl_ev_id] = clause_source

        # Finally, mark this eventuality is not describing the task domain
        ref_map[expl_ev_id]["domain_describing"] = False

    # Update return value with the translation
    translated.update({ rel_id: (topic_msgs, focus_msgs) })

    return translated


def _process_rel(
        parse, rel, rel_id, ref_map, negs,
        dt_translations, dt_translations_flattened
    ):
    """ 
    Factoring out relation processing procedure, as _traverse_dt method was getting
    too bulky
    """
    # Each list entry in topic_msgs and focus_msgs is a literal, which is a tuple:
    #   1) predicate
    #   2) part-of-speech
    #   3) list of arg variables (args may be function terms if skolemized)
    # or a list of such literals, for representing a negated scope over them.
    # Basically equivalent to conjunction normal form with haphazard formalism...
    # I will have to incorporate some static typing later, if I want to clean up
    # this messy code
    topic_msgs = []; focus_msgs = []

    # Translations of any 'events' that need to be treated as standalone clauses;
    # may have indicative or irrealis mood (at least in principle)
    separable_clauses = {}

    # Setting up arg1 (& arg2) variables
    rel_args = rel["args"]
    if len(rel_args) > 1 and rel["pos"]=="n" and rel_args[1].startswith("i"):
        # Let's deal with i-referents later... Just ignore it for now
        rel_args = rel_args[:1]

    if len(rel_args) > 1:
        if len(rel_args) > 2 and rel_args[2].startswith("i"):
            # Let's deal with i-referents later... Just ignore it for now
            rel_args = rel_args[:2]

        if len(rel_args) > 2:
            if rel["predicate"] == "be" and ref_map[rel_args[2]].get("wh_quantified"):
                # MRS flips the arg order when subject is quantified with 'which',
                # presumably seeing it as wh-movement? Re-flip, except when the
                # wh-word is 'what' (represented as 'which thing' in MRS)
                wh_ref_pred = parse["relations"]["by_id"][rel_args[2]]["predicate"]
                if wh_ref_pred == "thing":
                    arg1 = rel_args[1]
                    arg2 = rel_args[2]
                # ... or when the wh-phrase is 'what kind/type of X'
                elif wh_ref_pred == "kind" or wh_ref_pred == "type":
                    arg1 = rel_args[1]
                    arg2 = rel_args[2]
                else:
                    arg1 = rel_args[2]
                    arg2 = rel_args[1]

            elif rel["pos"]=="v" and rel_args[1].startswith("i"):
                # Strong indication of passive participial adjective (e.g. flared)
                arg1 = rel_args[2]
                arg2 = None
                rel["pos"] = "a"
                rel["predicate"] = parse["raw"][rel["crange"][0]:rel["crange"][1]]
                rel["predicate"] = rel["predicate"].translate(
                    str.maketrans("", "", string.punctuation)
                )       # There may be trailing punctuations, remove them

            else:
                # Default case
                arg1 = rel_args[1]
                arg2 = rel_args[2]

        else:
            # (These won't & shouldn't be referred)
            arg1 = rel_args[1]
            arg2 = None

        # We assume predicates describing arg1 are topic of the covered constituents
        # (This will do for most cases... May later need to refine depending on voice,
        # discourse context, etc.?)
        topic_msgs += dt_translations_flattened[arg1]
    else:
        # (These won't & shouldn't be referred)
        arg1 = None
        arg2 = None

    negate_focus = rel["handle"] in negs

    # For negative polar (yes/no) question we don't negate focus message
    if negate_focus and rel_id==parse["index"] and parse["utt_type"]=="ques":
        is_polar_q = not any(
            v["wh_quantified"] for v in ref_map.values() if v is not None
        )
        if is_polar_q:
            negate_focus = False

    # Handle predicates accordingly
    if rel["pos"] == "a":
        # Adjective predicates with args ('event', referent)

        # (Many) Adjectives require consideration of the nominal 'class' of the
        # object they modify for proper interpretation of their semantics (c.f.
        # small elephant vs. big flea). For now, we will attach such nominal
        # predicate names after (all) adjectival predicate names; practically,
        # this allows the agent to search for exemplars within its exemplar-base
        # with added info.
        pred_name = rel["predicate"]
        modified_ent = parse["relations"]["by_id"][arg1]
        if modified_ent["lexical"]:
            pred_name += "/" + modified_ent["predicate"]
        rel_lit = (pred_name, "a", (arg1,))

        if negate_focus:
            focus_msgs.append([rel_lit])
        else:
            focus_msgs.append(rel_lit)

    elif rel["pos"] == "n":
        # Noun predicates with args (referent[, more optional referents])

        if rel["predicate"]=="kind" or rel["predicate"]=="type":
            # Kind/type of X, specifying some predicate as a subtype of X
            rel_lit = ("subtype", "*", (rel_id, arg1))
            if negate_focus:
                # Not sure yet what should happen in this case...
                raise NotImplementedError
            else:
                focus_msgs.append(rel_lit)

            # Mark arg1 refers to a predicate, not an entity
            ref_map[arg1]["is_pred"] = True

            # Note to self: Something of an abuse of notation is happening here...
            # 'predicate(*arg)', where arg is a singleton list, has different meaning
            # depending on whether the argument refers to an entity (is_pred is False)
            # or a predicate (is_pred is True). The meaning of the former is rather
            # straightforward, namely that the arg referent is an instance of the
            # predicate. For the latter, it means the arg referent is a 'conjunctive
            # concept', where the predicate is one of the concept pieces that make up
            # the conjunction.

        else:
            # General cases
            rel_lit = (rel["predicate"], "n", (rel_id,))

            if negate_focus:
                focus_msgs.append([rel_lit])
            else:
                focus_msgs.append(rel_lit)
    
    elif rel["predicate"] == "be":
        # "be" has different semantics depending on the construction of the clause:
        #   1) Predicational: Non-referential complement
        #   2) Equative: Referential subject, referential complement
        #   3) Specificational: Non-referential subject, referential complement
        # where referential NP is of type <e>, and non-referential NP is of type <e,t>.
        #
        # (* Technically speaking, the dominant taxonomy in the field for copula
        # semantics seems that predicational clauses need referential subjects, and
        # quantificational expressions are allowed as subject as well. But testing
        # these conditions is a bit hairy, and for now we will go with a simplified
        # condition of having non-referential complement only.)

        if not ref_map[arg2]["referential"]:
            a2_pred = parse["relations"]["by_id"][rel_args[2]]["predicate"]

            if (rel_id in parse["utt_type"] and parse["utt_type"][rel_id]=="ques" and 
                (a2_pred=="thing" or a2_pred=="kind" or a2_pred=="type")):
                # "What (kind/type of Y) is X?" type of question; attach a special
                # reserved predicate to focus_msgs, which signals that which predicate
                # arg2 belongs to is under question
                rel_lit = ("isinstance", "*", (arg2, arg1))
                if negate_focus:
                    focus_msgs.append([rel_lit])
                else:
                    focus_msgs.append(rel_lit)

                # Any restrictor of the wh-quantified predicate, denoted as the
                # reserved subtype check requirement predicates, included here
                focus_msgs += dt_translations_flattened[arg2]

                # Mark arg2 refers to a predicate, not an entity
                ref_map[arg2]["is_pred"] = True

            else:
                # Predicational; provide predicates as additional info about subject
                if negate_focus:
                    focus_msgs.append(dt_translations_flattened[arg2])
                else:
                    focus_msgs += dt_translations_flattened[arg2]

                # arg2 is the same entity as arg1
                ref_map[arg2] = ref_map[arg1]

        else:
            if ref_map[arg1].get("referential"):
                # Equative; claim referent identity btw. subject and complement
                topic_msgs += dt_translations_flattened[arg2]

                rel_lit = ("=", "*", (arg1, arg2))

                if negate_focus:
                    focus_msgs.append([rel_lit])
                else:
                    focus_msgs.append(rel_lit)

            else:
                # We are not really interested in 3) (or others?), at least for now
                raise ValueError("Weird sentence with copula")

    elif arg2 is not None:
        # Two-place predicates have different semantics depending on the nature
        # of arg2
        if arg2.startswith("e"):
            # arg2 is an event referent, likely a complement clause argument for
            # verbs like 'think', 'say', etc. Pull out the translated clause content
            # from the daughter's translated content set, and add to the list of
            # utterances (segments).
            separable_clauses.update(dt_translations[arg2])
            clause_source = f"# complement clause of '{rel['predicate']}'"
            parse["utt_type"][arg2] = "prop"
            parse["clause_source"][arg2] = clause_source

            # Generally these 'reporting' verbs do not describe the task domain itself
            ref_map[rel_id]["domain_describing"] = False

            # Mark in ref_map[arg2] that the clause has an irrealis mood, in such
            # cases (E.g., 'think', 'believe')
            if rel["predicate"] == "think":
                ref_map[arg2]["irrealis"] = True

            # Then add the literal for the verb, with reserved predicate marker "*"
            # as it's not a first-order statement about domain instances
            rel_lit = (rel["predicate"], "*", (arg1, arg2))

            if negate_focus:
                focus_msgs.append([rel_lit])
            else:
                focus_msgs.append(rel_lit)

        else:
            # arg2 is an instance referent; arguably more common...
            a2_referential = ref_map[arg2].get("referential")
            a2_wh_quantified = ref_map[arg2].get("wh_quantified")

            if a2_referential or a2_wh_quantified:
                # Simpler case; use constant for arg2, and provided predicates
                # concerning arg2 are considered topic message
                topic_msgs += dt_translations_flattened[arg2]

                rel_lit = (rel["predicate"], rel["pos"], (arg1, arg2))

                if negate_focus:
                    focus_msgs.append([rel_lit])
                else:
                    focus_msgs.append(rel_lit)

            else:
                # arg2 is existentially quantified; skolemize the referent (ensuring
                # ASP program safety) and yield conjunction of literals. Note that
                # we are assuming any quantifier scope ambiguity is resolved such
                # that existentials are placed inside: i.e. surface scope reading.
                arg2_sk = (f"f_{arg2}", (arg1,))
                ref_map[arg2_sk] = ref_map[arg2]
                del ref_map[arg2]

                lits = [(rel["predicate"], rel["pos"], (arg1, arg2_sk))]
                for a2_lit in dt_translations_flattened[arg2]:
                    # Replace occurrences of arg2 with the skolem term
                    args_sk = [arg2_sk if arg2==arg else arg for arg in a2_lit[2]]
                    a2_lit_sk = a2_lit[:2] + (args_sk,)
                    lits.append(a2_lit_sk)
                
                if negate_focus:
                    focus_msgs.append(lits)
                else:
                    focus_msgs += lits

    elif rel["predicate"] == "pron":
        # Reserved MRS predicate 'pron', introduced by parsing of, well, pronouns.
        # Annotate the relation with the pronoun info. No need to add as an independent
        # message. Note that rel["pos"] is None as the predicate is not lexical.

        # Extract the pronoun from the raw input
        pron_predicate = parse["raw"][rel["crange"][0]:rel["crange"][1]]
        rel["pronoun"] = pron_predicate

    elif rel["predicate"] == "reason":
        # Reserved MRS predicate 'reason', introduced by parsing of "why" token
        # (along with 'for' and 'which'... So "Why" <=> "For which reason" in MRS).
        # Simple treatment like that for nouns would suffice. Note that rel["pos"]
        # is None as the predicate is not lexical.
        rel_lit = (rel["predicate"], "n", (rel_id,))

        if negate_focus:
            focus_msgs.append([rel_lit])
        else:
            focus_msgs.append(rel_lit)

    else:
        # Other general cases; currently, the reserved MRS predicate 'thing' funnels
        # all the way down to here since pos is None
        pass

    return topic_msgs, focus_msgs, separable_clauses
