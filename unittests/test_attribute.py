"""
Test suite for probabilistic reasoning explanation within our implementation of
probabilistic reasoning faculty (according to LP^MLN + our custom choice of
architecture)
"""
import unittest

import numpy as np

from python.itl.lpmln import Literal
from python.itl.lpmln.program.compile import bjt_query
from python.itl.memory.kb import KnowledgeBase
from python.itl.symbolic_reasoning.attribute import attribute


class TestCausalAttribution(unittest.TestCase):

    def test_singleton_suff_expl_1(self):
        """ One sufficient cause, consisting of one necessary condition atom (1) """
        kb = KnowledgeBase()
        kb.add(
            (
                (Literal("att_0", [("O", True)]),),
                (Literal("cls_0", [("O", True)]),),
            ),
            1.0,
            "att_0(O) <= cls_0(O)"
        )
        kb_prog = kb.export_reasoning_program()

        cls0_score = 0.65; att0_score = 0.9
        scene = {
            "o0": {
                "pred_cls": np.array([cls0_score]),
                "pred_att": np.array([att0_score])
            }
        }
        ev_prog = kb.visual_evidence_from_scene(scene)

        prog = kb_prog + ev_prog
        bjt = prog.compile()

        cls0_lit = Literal("cls_0", [("o0", False)])
        cls0_ind = bjt.graph["atoms_map"][cls0_lit]
        cls0_singleton = frozenset({cls0_ind})
        bjt_query(bjt, cls0_singleton)       # Run belief propagation

        v_cls0_lit = Literal("v_cls_0", [("o0", False)])
        v_att0_lit = Literal("v_att_0", [("o0", False)])
        suff_expls = attribute(
            bjt, (cls0_lit,), (v_cls0_lit, v_att0_lit), threshold=0.7
        )
        self.assertEqual(len(suff_expls), 1)
        self.assertEqual(suff_expls[0], {v_att0_lit})

    def test_singleton_suff_expl_2(self):
        """ One sufficient cause, consisting of one necessary condition atom (2) """
        kb = KnowledgeBase()
        kb.add(
            (
                (Literal("att_0", [("O", True)]),),
                (Literal("cls_0", [("O", True)]),),
            ),
            1.0,
            "att_0(O) <= cls_0(O)"
        )
        kb_prog = kb.export_reasoning_program()

        cls0_score = 0.9; att0_score = 0.65
        scene = {
            "o0": {
                "pred_cls": np.array([cls0_score]),
                "pred_att": np.array([att0_score])
            }
        }
        ev_prog = kb.visual_evidence_from_scene(scene)

        prog = kb_prog + ev_prog
        bjt = prog.compile()

        cls0_lit = Literal("cls_0", [("o0", False)])
        cls0_ind = bjt.graph["atoms_map"][cls0_lit]
        cls0_singleton = frozenset({cls0_ind})
        bjt_query(bjt, cls0_singleton)       # Run belief propagation

        v_cls0_lit = Literal("v_cls_0", [("o0", False)])
        v_att0_lit = Literal("v_att_0", [("o0", False)])
        suff_expls = attribute(
            bjt, (cls0_lit,), (v_cls0_lit, v_att0_lit), threshold=0.7
        )
        self.assertEqual(len(suff_expls), 1)
        self.assertEqual(suff_expls[0], {v_cls0_lit})

    def test_jointly_suff_expl(self):
        """ One sufficient cause, consisting of two necessary condition atoms """
        kb = KnowledgeBase()
        kb.add(
            (
                (Literal("att_0", [("O", True)]),),
                (Literal("cls_0", [("O", True)]),),
            ),
            1.0,
            "att_0(O) <= cls_0(O)"
        )
        kb_prog = kb.export_reasoning_program()

        cls0_score = 0.65; att0_score = 0.65
        scene = {
            "o0": {
                "pred_cls": np.array([cls0_score]),
                "pred_att": np.array([att0_score])
            }
        }
        ev_prog = kb.visual_evidence_from_scene(scene)

        prog = kb_prog + ev_prog
        bjt = prog.compile()

        cls0_lit = Literal("cls_0", [("o0", False)])
        cls0_ind = bjt.graph["atoms_map"][cls0_lit]
        cls0_singleton = frozenset({cls0_ind})
        bjt_query(bjt, cls0_singleton)       # Run belief propagation

        v_cls0_lit = Literal("v_cls_0", [("o0", False)])
        v_att0_lit = Literal("v_att_0", [("o0", False)])
        suff_expls = attribute(
            bjt, (cls0_lit,), (v_cls0_lit, v_att0_lit), threshold=0.7
        )
        self.assertEqual(len(suff_expls), 1)
        self.assertEqual(suff_expls[0], {v_cls0_lit, v_att0_lit})

    def test_multiple_suff_expls(self):
        """ Two sufficient causes, each consisting of one necessary condition """
        kb = KnowledgeBase()
        kb.add(
            (
                (Literal("att_0", [("O", True)]),),
                (Literal("cls_0", [("O", True)]),),
            ),
            1.0,
            "att_0(O) <= cls_0(O)"
        )
        kb_prog = kb.export_reasoning_program()

        cls0_score = 0.9; att0_score = 0.9
        scene = {
            "o0": {
                "pred_cls": np.array([cls0_score]),
                "pred_att": np.array([att0_score])
            }
        }
        ev_prog = kb.visual_evidence_from_scene(scene)

        prog = kb_prog + ev_prog
        bjt = prog.compile()

        cls0_lit = Literal("cls_0", [("o0", False)])
        cls0_ind = bjt.graph["atoms_map"][cls0_lit]
        cls0_singleton = frozenset({cls0_ind})
        bjt_query(bjt, cls0_singleton)       # Run belief propagation

        v_cls0_lit = Literal("v_cls_0", [("o0", False)])
        v_att0_lit = Literal("v_att_0", [("o0", False)])
        suff_expls = attribute(
            bjt, (cls0_lit,), (v_cls0_lit, v_att0_lit), threshold=0.7
        )
        self.assertEqual(len(suff_expls), 2)
        self.assertEqual(
            suff_expls[0]=={v_cls0_lit} or suff_expls[1]=={v_cls0_lit}, True
        )
        self.assertEqual(
            suff_expls[0]=={v_att0_lit} or suff_expls[1]=={v_att0_lit}, True
        )

if __name__ == '__main__':
    unittest.main()
