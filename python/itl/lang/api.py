"""
Language processing module API that exposes only the high-level functionalities
required by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
from collections import defaultdict

from .semantics import SemanticParser
from .dialogue import DialogueManager


class LanguageModule:

    def __init__(self, cfg):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.semantic = SemanticParser(
            cfg.lang.paths.grammar_image, cfg.lang.paths.ace_binary
        )
        self.dialogue = DialogueManager()

        self.unresolved_neologisms = set()

    def situate(self, vis_scene):
        """
        Put entities in the physical environment into domain of discourse
        """
        # No-op if no new visual input
        if vis_scene is None:
            return

        # Incorporate parsed scene graph into dialogue context
        for oi, obj in vis_scene.items():
            mask = obj["pred_mask"]
            self.dialogue.referents["env"][oi] = {
                "mask": mask,
                "area": mask.sum().item()
            }
            self.dialogue.referent_names[oi] = oi
        
        # Register these indices as names, for starters
        self.dialogue.referent_names = {i: i for i in self.dialogue.referents["env"]}

    def understand(self, parses, pointing=None):
        """
        Parse language input into MRS, process into ASP-compatible form, and then
        update dialogue state.

        pointing (optional) is a dict summarizing the 'gesture' made along with the
        utterance, indicating the reference (represented as mask) made by the n'th
        occurrence of linguistic token. Mostly for programmed experiments.
        """
        ti = len(self.dialogue.record)      # New dialogue turn index
        new_record = []                     # New dialogue record for the turn

        # Processing natural language into appropriate logical form
        translations, ref_maps = self.semantic.asp_translate(parses)

        # For indexing clauses in dialogue turn
        se2ci = defaultdict(lambda: len(se2ci))

        per_sentence = enumerate(zip(parses, translations, ref_maps))
        for si, (parse, translation, ref_map) in per_sentence:
            # For indexing referents within individual clauses
            reind_per_src = {
                ei: defaultdict(lambda: len(reind_per_src[ei]))
                for ei in translation
            }
            r2i = {
                rf: reind_per_src[v["source_evt"]][v["map_id"]]
                for rf, v in ref_map.items()
                if v is not None and (isinstance(rf, tuple) or rf.startswith("x"))
            }
            # For indexing individual clauses
            r2i.update({
                rf: f"t{ti}c{se2ci[(si,rf)]}" for rf in translation
            })

            # Add to the list of discourse referents
            for rf, v in ref_map.items():
                if v is not None:
                    if type(rf) == tuple:
                        # Annotate instance referents (function terms)
                        term_char = "p" if ref_map[rf]["is_pred"] else "x"
                        turn_clause_tag = f"t{ti}c{se2ci[(si,v['source_evt'])]}"

                        f_args = tuple(
                            f"{term_char.upper()}{r2i[arg]}{turn_clause_tag}"
                                if ref_map[arg]["univ_quantified"] or ref_map[arg]["wh_quantified"]
                                else f"{term_char}{r2i[arg]}{turn_clause_tag}"
                            for arg in rf[1]
                        )
                        rf = (rf[0], f_args)

                        self.dialogue.referents["dis"][rf] = {
                            "provenance": v["provenance"],
                            "referential": v["referential"],
                            "univ_quantified": v["univ_quantified"],
                            "wh_quantified": v["wh_quantified"]
                        }
                    else:
                        assert type(rf) == str
                        if rf.startswith("x"):
                            # Annotate instance referents (const terms)
                            term_char = "p" if ref_map[rf]["is_pred"] else "x"
                            turn_clause_tag = f"t{ti}c{se2ci[(si,v['source_evt'])]}"

                            if v["univ_quantified"] or v["wh_quantified"]:
                                rf_i = f"{term_char.upper()}{r2i[rf]}{turn_clause_tag}"
                            else:
                                rf_i = f"{term_char}{r2i[rf]}{turn_clause_tag}"

                            self.dialogue.referents["dis"][rf_i] = {
                                "provenance": v["provenance"],
                                "referential": v["referential"],
                                "univ_quantified": v["univ_quantified"],
                                "wh_quantified": v["wh_quantified"]
                            }
                        elif rf.startswith("e"):
                            # Annotate eventuality referents (i.e. clauses)
                            self.dialogue.clause_info[r2i[rf]] = {
                                "provenance": v["provenance"],
                                "irrealis": v["irrealis"],
                                "domain_describing": v["domain_describing"]
                            }

            # Fetch arg1 of index (i.e. sentence 'subject' referent)
            index = parse["relations"]["by_id"][parse["index"]]
            if not len(index["args"]) > 1:
                raise ValueError("Input is not a sentence")

            # Handle certain hard assignments
            for rel in parse["relations"]["by_id"].values():

                # Hard assignments handling is for instance referents only
                if rel["args"][0].startswith("x") and rel["args"][0] in r2i:
                    ri = r2i[rel["args"][0]]
                    clause_tag = se2ci[(si, ref_map[rel["args"][0]]["source_evt"])]
                else:
                    continue

                if rel["pos"] == "q":
                    # Demonstratives need pointing
                    if "sense" in rel and rel["sense"] == "dem":
                        matching_masks = [
                            msk for crange, msk in pointing[si].items()
                            if crange[0]>=rel["crange"][0] and crange[1]<=rel["crange"][1]
                            # ERG parser includes punctuations in crange, test inclusion
                        ] if pointing is not None else []

                        if len(matching_masks) > 0:
                            dem_mask = matching_masks[0].astype(bool)
                        else:
                            dem_mask = None

                        pointed = self.dialogue.dem_point(dem_mask)
                        self.dialogue.assignment_hard[f"x{ri}t{ti}c{clause_tag}"] = pointed

                if rel["predicate"] == "named":
                    # Provided symbolic name
                    if rel["carg"] not in self.dialogue.referent_names:
                        # Couldn't resolve the name; explicitly ask again for name resolution
                        # TODO: Update to comply with recent changes
                        raise NotImplementedError

                    self.dialogue.assignment_hard[f"x{ri}t{ti}c{clause_tag}"] = \
                        self.dialogue.referent_names[rel["carg"]]

                if rel["predicate"] == "pron":
                    # Handle resolvable pronouns ('you' only, for now)
                    if rel["pronoun"] == "you":
                        self.dialogue.assignment_hard[f"x{ri}t{ti}c{clause_tag}"] = "_self"

            # ASP-compatible translation
            for ev_id, (topic_msgs, focus_msgs) in translation.items():
                cons = []; ante = []

                # Process topic messages
                for m in topic_msgs:
                    if type(m) == tuple:
                        # Non-negated message
                        occurring_args = sum([arg[1] if type(arg)==tuple else (arg,) for arg in m[2]], ())
                        occurring_args = tuple(set(occurring_args))

                        var_free = not any([ref_map[arg]["univ_quantified"] for arg in occurring_args])

                        if var_free:
                            # Add the grounded literal to cons
                            cons.append(m[:2]+(tuple(m[2]),False))
                        else:
                            # Add the non-grounded literal to ante
                            ante.append(m[:2]+(tuple(m[2]),False))
                    else:
                        # Negation of conjunction
                        occurring_args = sum([
                            sum([arg[1] if type(arg)==tuple else (arg,) for arg in l[2]], ()) for l in m
                        ], ())
                        occurring_args = tuple(set(occurring_args))

                        var_free = not any([ref_map[arg]["univ_quantified"] for arg in occurring_args])

                        conj = [l[:2]+(tuple(l[2]),False) for l in m]
                        conj = list(set(conj))      # Remove duplicate literals

                        if var_free:
                            # Add the non-grounded conjunction to cons
                            cons.append(conj)
                        else:
                            # Add the non-grounded conjunction to ante
                            ante.append(conj)

                # Process focus messages
                for m in focus_msgs:
                    if type(m) == tuple:
                        # Non-negated message
                        cons.append(m[:2] + (tuple(m[2]), False))
                    else:
                        # Negation of conjunction
                        conj = [l[:2]+(tuple(l[2]),False) for l in m]
                        conj = list(set(conj))      # Remove duplicate literals
                        cons.append(conj)

                if parse["utt_type"][ev_id] == "prop":
                    # Indicatives
                    prop = _map_and_format((cons, ante), ref_map, ti, si, r2i, se2ci)

                    new_record.append(((prop, None), parse["clause_source"][ev_id]))

                elif parse["utt_type"][ev_id] == "ques":
                    # Interrogatives

                    # Determine type of question: Y/N or wh-
                    wh_refs = {
                        rf for rf, v in ref_map.items()
                        if rf.startswith("x") and v is not None and v["wh_quantified"]
                    }

                    if len(wh_refs) == 0:
                        # Y/N question with no wh-quantified entities
                        q_vars = None
                        raise NotImplemented
                    else:
                        # wh- question with wh-quantified entities
                        
                        # Pull out any literals containing wh-quantified referents
                        # and build into question. Remaining var-free literals are
                        # considered as presupposed statements (all grounded facts;
                        # currently I cannot think of any universally quantified
                        # presuppositions that can be conveyed via questions -- at
                        # least we don't have such cases in our use scenarios).
                        presup = [
                            l for l in cons if len(wh_refs & set(l[2])) == 0
                        ] + [
                            l for l in ante if len(wh_refs & set(l[2])) == 0
                        ]       # presup should be extracted first before cons & ante update
                        cons = [l for l in cons if len(wh_refs & set(l[2])) > 0]
                        ante = [l for l in ante if len(wh_refs & set(l[2])) > 0]

                        q_vars = wh_refs & set.union(*[set(l[2]) for l in cons+ante])

                    question = (q_vars, (cons, ante))

                    if len(presup) > 0:
                        presup = _map_and_format((presup, []), ref_map, ti, si, r2i, se2ci)
                    else:
                        presup = None
                    question = _map_and_format(question, ref_map, ti, si, r2i, se2ci)

                    new_record.append(((presup, question), parse["clause_source"][ev_id]))

                    # Record turn-clause index of unanswered question
                    self.dialogue.unanswered_Qs.add((ti, se2ci[(si, ev_id)]))

                elif parse["utt_type"][ev_id] == "comm":
                    # Imperatives
                    raise NotImplementedError
                
                else:
                    # Ambiguous SF
                    raise NotImplementedError
        
        self.dialogue.record.append(("U", new_record))    # Add new record

    def acknowledge(self):
        """ Push an acknowledging utterance to generation buffer """
        self.dialogue.to_generate.append((None, "OK.", {}))

    def generate(self):
        """ Flush the buffer of utterances prepared """
        if len(self.dialogue.to_generate) > 0:
            return_val = []

            new_record = []
            for logical_forms, surface_form, dem_refs in self.dialogue.to_generate:
                if logical_forms is None:
                    logical_forms = (None, None)

                new_record.append((logical_forms, surface_form))

                # NL utterance to log/print (as long as not prefixed by a special
                # symbol '#')
                if not surface_form.startswith("#"):
                    return_val.append(("generate", (surface_form, dem_refs)))

            self.dialogue.record.append(("A", new_record))

            self.dialogue.to_generate = []

            return return_val
        else:
            return


def _map_and_format(data, ref_map, ti, si, r2i, se2ci):
    # Map MRS referents to ASP terms and format

    def fmt(rf):
        if type(rf) == tuple:
            return (rf[0], tuple(fmt(arg) for arg in rf[1]))
        else:
            assert type(rf) == str

            if rf.startswith("x"):
                # Instance referents, may be variable or constant for now
                is_var = ref_map[rf]['univ_quantified'] or ref_map[rf]['wh_quantified']

                term_char = "p" if ref_map[rf]["is_pred"] else "x"
                term_char = term_char.upper() if is_var else term_char
                turn_clause_tag = f"t{ti}c{se2ci[(si, ref_map[rf]['source_evt'])]}"

                return f"{term_char}{r2i[rf]}{turn_clause_tag}"
            else:
                # Eventuality referents, always constant, referring to clauses that
                # already took place and are part of dialogue records
                assert rf.startswith("e")
                return r2i[rf]

    process_conjuncts = lambda conjuncts: tuple(
        (cnjt[0], cnjt[1], tuple(fmt(arg) for arg in cnjt[2]), cnjt[3])
            if isinstance(cnjt, tuple) else list(process_conjuncts(cnjt))
        for cnjt in conjuncts
    )

    if isinstance(data[0], list):
        # Proposition
        cons, ante = data
        return (process_conjuncts(cons), process_conjuncts(ante))

    elif isinstance(data[0], set):
        # Question
        q_vars, (cons, ante) = data

        # Annotate whether the variable is zeroth-order (entity) or first-order
        # (predicate)
        new_q_vars = tuple(
            (fmt(e), ref_map[e]["is_pred"]) for e in q_vars
        ) if q_vars is not None else None

        return (new_q_vars, (process_conjuncts(cons), process_conjuncts(ante)))

    else:
        raise ValueError("Can't _map_and_format this")
