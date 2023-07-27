#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for incoutputs.py.
"""

import unittest

import numpy as np

from inceval.incoutputs import IncOutputs
from inceval.aux import GOLD

EMPTY = np.inf


class TestIncOutputs(unittest.TestCase):
    """Tests for methods in IncOutputs."""
    def setUp(self):
        recomputations = np.array([False, False, True, True, True,
                                   False, True, True, True, True])
        outputs = IncOutputs(10, recomputations=recomputations)
        outputs.add_prefix(0, [1])
        outputs.add_prefix(1, [1, 2])
        outputs.add_prefix(2, [1, 1, 3])
        outputs.add_prefix(3, [1, 1, 3, 2])
        outputs.add_prefix(4, [1, 2, 1, 1, 3])
        outputs.add_prefix(5, [1, 2, 1, 1, 3, 2])
        outputs.add_prefix(6, [1, 2, 2, 1, 3, 1, 1])
        outputs.add_prefix(7, [1, 2, 1, 1, 3, 1, 1, 3])
        outputs.add_prefix(8, [1, 2, 1, 1, 3, 1, 1, 2, 1])
        outputs.add_prefix(9, [1, 2, 1, 1, 3, 2, 2, 3, 1, 1])
        self.outputs_1 = outputs

        outputs = IncOutputs(5, [1, 2, 1, 2, 3], eval_mode=GOLD)
        outputs.add_all_prefixes(np.array([
            [1, np.inf, np.inf, np.inf, np.inf],
            [3, 2,      np.inf, np.inf, np.inf],
            [1, 2,      1,      np.inf, np.inf],
            [2, 2,      3,      1,      np.inf],
            [2, 2,      3,      1,      3],
        ]))
        self.outputs_2 = outputs

    def test_n_tokens(self):
        self.assertEqual(self.outputs_1.n_tokens, 10)
        self.assertEqual(self.outputs_2.n_tokens, 5)

    def test_final_accuracy(self):
        self.assertEqual(self.outputs_1.final_accuracy, 1.)
        self.assertEqual(self.outputs_2.final_accuracy, 2/5)

    def test_get_accuracy_at_t(self):
        self.assertEqual(self.outputs_1.get_accuracy_at_t(0), 1.)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(1), 1.)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(2), 1/3)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(3), 1/4)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(4), 1.)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(5), 1.)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(6), 4/7)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(7), 6/8)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(8), 6/9)
        self.assertEqual(self.outputs_1.get_accuracy_at_t(9), 1.)
        self.assertEqual(self.outputs_2.get_accuracy_at_t(0), 1.)
        self.assertEqual(self.outputs_2.get_accuracy_at_t(1), 1/2)
        self.assertEqual(self.outputs_2.get_accuracy_at_t(2), 1.)
        self.assertEqual(self.outputs_2.get_accuracy_at_t(3), 1/4)
        self.assertEqual(self.outputs_2.get_accuracy_at_t(4), 2/5)

    def test_accuracy_by_turn(self):
        accs = np.array([1., 1., 1/3, 1/4, 1., 1., 4/7, 6/8, 6/9, 1.])
        np.testing.assert_array_equal(self.outputs_1.accuracy_by_turn, accs)
        accs = np.array([1., 1/2, 1., 1/4, 2/5])
        np.testing.assert_array_equal(self.outputs_2.accuracy_by_turn, accs)

    def test_revision_timesteps(self):
        self.assertEqual(self.outputs_1.revision_timesteps, [2, 4, 6, 7, 8, 9])
        self.assertEqual(self.outputs_2.revision_timesteps, [1, 2, 3])

    def test_effective_revision_timesteps(self):
        self.assertEqual(self.outputs_1.effective_revision_timesteps, [4, 7, 9])
        self.assertEqual(self.outputs_2.effective_revision_timesteps, [2])

    def test_write_timesteps(self):
        self.assertEqual(self.outputs_1.write_timesteps, [0, 1, 3, 5])
        self.assertEqual(self.outputs_2.write_timesteps, [0, 4])

    def test_correct_prefixes(self):
        self.assertEqual(self.outputs_1.correct_prefixes, [0, 1, 4, 5, 9])
        self.assertEqual(self.outputs_2.correct_prefixes, [0, 2])

    def test_incorrect_prefixes(self):
        self.assertEqual(self.outputs_1.incorrect_prefixes, [2, 3, 6, 7, 8])
        self.assertEqual(self.outputs_2.incorrect_prefixes, [1, 3, 4])

    def test_n_correct_prefixes(self):
        self.assertEqual(self.outputs_1.n_correct_prefixes, 5)
        self.assertEqual(self.outputs_2.n_correct_prefixes, 2)

    def test_n_correct_acted_prefixes(self):
        self.assertEqual(self.outputs_1.n_correct_acted_prefixes, 5)
        self.assertEqual(self.outputs_2.n_correct_acted_prefixes, 3)

    def test_perc_correct_prefixes(self):
        self.assertEqual(self.outputs_1.perc_correct_prefixes, 100*5/10)
        self.assertEqual(self.outputs_2.perc_correct_prefixes, 100*2/5)

    def test_n_incorrect_prefixes(self):
        self.assertEqual(self.outputs_1.n_incorrect_prefixes, 5)
        self.assertEqual(self.outputs_2.n_incorrect_prefixes, 3)

    def test_n_incorrect_acted_prefixes(self):
        self.assertEqual(self.outputs_1.n_incorrect_acted_prefixes, 5)
        self.assertEqual(self.outputs_2.n_incorrect_acted_prefixes, 2)

    def test_perc_incorrect_prefixes(self):
        self.assertEqual(self.outputs_1.perc_incorrect_prefixes, 100*5/10)
        self.assertEqual(self.outputs_2.perc_incorrect_prefixes, 100*3/5)

    def test_n_revisions(self):
        self.assertEqual(self.outputs_1.n_revisions, 6)
        self.assertEqual(self.outputs_2.n_revisions, 3)

    def test_perc_revisions(self):
        self.assertEqual(self.outputs_1.perc_revisions, 100*6/10)
        self.assertEqual(self.outputs_2.perc_revisions, 100*3/5)

    def test_n_effective_revisions(self):
        self.assertEqual(self.outputs_1.n_effective_revisions, 3)
        self.assertEqual(self.outputs_2.n_effective_revisions, 1)

    def test_perc_effective_revisions(self):
        self.assertEqual(self.outputs_1.perc_effective_revisions, 100*3/10)
        self.assertEqual(self.outputs_2.perc_effective_revisions, 100*1/5)

    def test_n_recomputations(self):
        self.assertEqual(self.outputs_1.n_recomputations, 7)
        self.assertTrue(np.isnan(self.outputs_2.n_recomputations))

    def test_perc_recomputations(self):
        self.assertEqual(self.outputs_1.perc_recomputations, 100*7/10)
        self.assertTrue(np.isnan(self.outputs_2.perc_recomputations))

    def test_recomputation_timesteps(self):
        steps = [2, 3, 4, 6, 7, 8, 9]
        self.assertEqual(self.outputs_1.recomputation_timesteps, steps)
        self.assertTrue(np.isnan(self.outputs_2.recomputation_timesteps))

    def test_n_active_recomputations(self):
        self.assertEqual(self.outputs_1.n_active_recomputations, 6)
        self.assertTrue(np.isnan(self.outputs_2.n_active_recomputations))

    def test_perc_active_recomputations(self):
        self.assertEqual(self.outputs_1.perc_active_recomputations, 100*6/7)
        self.assertTrue(np.isnan(self.outputs_2.perc_active_recomputations))

    def test_n_inactive_recomputations(self):
        self.assertEqual(self.outputs_1.n_inactive_recomputations, 1)
        self.assertTrue(np.isnan(self.outputs_2.n_inactive_recomputations))

    def test_perc_inactive_recomputations(self):
        self.assertEqual(self.outputs_1.perc_inactive_recomputations, 100*1/7)
        self.assertTrue(np.isnan(self.outputs_2.perc_inactive_recomputations))

    def test_n_writes(self):
        self.assertEqual(self.outputs_1.n_writes, 4)
        self.assertEqual(self.outputs_2.n_writes, 2)

    def test_perc_writes(self):
        self.assertEqual(self.outputs_1.perc_writes, 100*4/10)
        self.assertEqual(self.outputs_2.perc_writes, 100*2/5)

    def test_n_revision_and_correct_prefix(self):
        self.assertEqual(self.outputs_1.n_revision_and_correct_prefix, 2)
        self.assertEqual(self.outputs_2.n_revision_and_correct_prefix, 2)

    def test_n_revision_and_incorrect_prefix(self):
        self.assertEqual(self.outputs_1.n_revision_and_incorrect_prefix, 4)
        self.assertEqual(self.outputs_2.n_revision_and_incorrect_prefix, 1)

    def test_n_effective_revision_and_correct_prefix(self):
        self.assertEqual(self.outputs_1.n_effective_revision_and_correct_prefix, 0)
        self.assertEqual(self.outputs_2.n_effective_revision_and_correct_prefix, 0)

    def test_n_effective_revision_and_incorrect_prefix(self):
        self.assertEqual(self.outputs_1.n_effective_revision_and_incorrect_prefix, 3)
        self.assertEqual(self.outputs_2.n_effective_revision_and_incorrect_prefix, 1)

    def test_n_write_and_correct_prefix(self):
        self.assertEqual(self.outputs_1.n_write_and_correct_prefix, 3)
        self.assertEqual(self.outputs_2.n_write_and_correct_prefix, 1)

    def test_n_write_and_incorrect_prefix(self):
        self.assertEqual(self.outputs_1.n_write_and_incorrect_prefix, 1)
        self.assertEqual(self.outputs_2.n_write_and_incorrect_prefix, 1)

    def test_r_pertinence(self):
        self.assertEqual(self.outputs_1.r_pertinence, 4/6)
        self.assertEqual(self.outputs_2.r_pertinence, 1/3)

    def test_r_pertinence_complement(self):
        self.assertEqual(self.outputs_1.r_pertinence_complement, 2/6)
        self.assertEqual(self.outputs_2.r_pertinence_complement, 2/3)

    def test_r_effective_pertinence(self):
        self.assertEqual(self.outputs_1.r_effective_pertinence, 3/6)
        self.assertEqual(self.outputs_2.r_effective_pertinence, 1/3)

    def test_a_pertinence(self):
        self.assertEqual(self.outputs_1.a_pertinence, 3/4)
        self.assertEqual(self.outputs_2.a_pertinence, 1/2)

    def test_a_pertinence_complement(self):
        self.assertEqual(self.outputs_1.a_pertinence_complement, 1/4)
        self.assertEqual(self.outputs_2.a_pertinence_complement, 1/2)

    def test_r_appropriateness(self):
        self.assertEqual(self.outputs_1.r_appropriateness, 4/5)
        self.assertEqual(self.outputs_2.r_appropriateness, 1/2)

    def test_r_appropriateness_complement(self):
        self.assertEqual(self.outputs_1.r_appropriateness_complement, 1/5)
        self.assertEqual(self.outputs_2.r_appropriateness_complement, 1/2)

    def test_r_effective_appropriateness(self):
        self.assertEqual(self.outputs_1.r_effective_appropriateness, 3/5)
        self.assertEqual(self.outputs_2.r_effective_appropriateness, 1/2)

    def test_a_appropriateness(self):
        self.assertEqual(self.outputs_1.a_appropriateness, 3/5)
        self.assertEqual(self.outputs_2.a_appropriateness, 1/3)

    def test_a_appropriateness_complement(self):
        self.assertEqual(self.outputs_1.a_appropriateness_complement, 2/5)
        self.assertEqual(self.outputs_2.a_appropriateness_complement, 2/3)

    def test_edit_overhead(self):
        self.assertEqual(self.outputs_1.edit_overhead, 11 / (11 + 10))
        self.assertEqual(self.outputs_2.edit_overhead, 4 / (4 + 5))

    def test_delayed_edit_overhead(self):
        d1_eo_1 = self.outputs_1.delayed_edit_overhead(1)
        self.assertEqual(d1_eo_1, 7 / (7 + 9))
        d1_eo_2 = self.outputs_2.delayed_edit_overhead(1)
        self.assertEqual(d1_eo_2, 2 / (2 + 4))

        d2_eo_1 = self.outputs_1.delayed_edit_overhead(2)
        self.assertEqual(d2_eo_1, 5 / (5 + 8))
        d2_eo_2 = self.outputs_2.delayed_edit_overhead(2)
        self.assertEqual(d2_eo_2, 1 / (1 + 3))

    def test_relative_correctness(self):
        self.assertEqual(self.outputs_1.relative_correctness, 5 / 10)
        self.assertEqual(self.outputs_2.relative_correctness, 2 / 5)

    def test_corretion_time_score(self):
        pass

    def test_corretion_time_per_token(self):
        pass

    def test_n_total_edits(self):
        self.assertEqual(self.outputs_1.n_total_edits, 11)
        self.assertEqual(self.outputs_2.n_total_edits, 4)

    def test_perc_total_edits(self):
        self.assertEqual(self.outputs_1.perc_total_edits, 100*11/45)
        self.assertEqual(self.outputs_2.perc_total_edits, 100*4/10)

    def test_n_edits_per_token(self):
        steps = [0, 2, 3, 1, 0, 2, 1, 2, 0, 0]
        self.assertEqual(self.outputs_1.n_edits_per_token.tolist(), steps)
        steps = [3, 0, 1, 0, 0]
        self.assertEqual(self.outputs_2.n_edits_per_token.tolist(), steps)

    def test_perc_edits_per_token(self):
        # ignore the last element because it is a nan
        steps = [0, 2, 3, 1, 0, 2, 1, 2, 0, 0]
        ps = [100 * x / (10-i-1) if i != 9 else 0 for i, x in enumerate(steps)]
        output = self.outputs_1.perc_edits_per_token.tolist()[:-1]
        self.assertEqual(output, ps[:-1])
        steps = [3, 0, 1, 0, 0]
        ps = [100 * x / (5-i-1) if i != 4 else 0 for i, x in enumerate(steps)]
        outputs = self.outputs_2.perc_edits_per_token.tolist()[:-1]
        self.assertEqual(outputs, ps[:-1])

    def test_n_edits_per_timestep(self):
        steps = [0, 0, 1, 0, 3, 0, 2, 1, 1, 3]
        self.assertEqual(self.outputs_1.n_edits_per_timestep.tolist(), steps)
        steps = [0, 1, 1, 2, 0]
        self.assertEqual(self.outputs_2.n_edits_per_timestep.tolist(), steps)

    def test_perc_edits_per_timestep(self):
        # ignore the first element because it is a nan
        steps = [0, 0, 1, 0, 3, 0, 2, 1, 1, 3]
        ps = [100 * x / i if i != 0 else 0 for i, x in enumerate(steps)]
        output = self.outputs_1.perc_edits_per_timestep.tolist()[1:]
        np.testing.assert_almost_equal(output, ps[1:])
        steps = [0, 1, 1, 2, 0]
        ps = [100 * x / i if i != 0 else 0 for i, x in enumerate(steps)]
        outputs = self.outputs_2.perc_edits_per_timestep.tolist()[1:]
        np.testing.assert_almost_equal(outputs, ps[1:])

    def test_n_edits_per_revision(self):
        steps = [1, 3, 2, 1, 1, 3]
        self.assertEqual(self.outputs_1.n_edits_per_revision.tolist(), steps)
        steps = [1, 1, 2]
        self.assertEqual(self.outputs_2.n_edits_per_revision.tolist(), steps)

    def test_n_edit_groups_per_timestep(self):
        pass

    def test_n_edit_groups_per_revision(self):
        pass

    def test_edit_distances(self):
        pass

    def test_label_diversity_per_token(self):
        pass

    def test_n_edits_with_quality_by_turn(self):
        pass

    def test_possible_edits_per_token(self):
        steps = np.array(list(range(10))[::-1])
        np.testing.assert_array_equal(self.outputs_1.possible_edits_per_token,
                                      steps)
        steps = np.array(list(range(5))[::-1])
        np.testing.assert_array_equal(self.outputs_2.possible_edits_per_token,
                                      steps)

    def test_possible_edits_per_timestep(self):
        steps = np.array(list(range(10)))
        np.testing.assert_array_equal(self.outputs_1.possible_edits_per_timestep,
                                      steps)
        steps = np.array(list(range(5)))
        np.testing.assert_array_equal(self.outputs_2.possible_edits_per_timestep,
                                      steps)

    def test_outputs_per_token(self):
        steps = np.array(list(range(1, 11))[::-1])
        np.testing.assert_array_equal(self.outputs_1.outputs_per_token, steps)
        steps = np.array(list(range(1, 6))[::-1])
        np.testing.assert_array_equal(self.outputs_2.outputs_per_token, steps)

    def test_total_possible_edits(self):
        self.assertEqual(self.outputs_1.total_possible_edits, 45)
        self.assertEqual(self.outputs_2.total_possible_edits, 10)

    def test_total_possible_outputs(self):
        self.assertEqual(self.outputs_1.total_possible_outputs, 55)
        self.assertEqual(self.outputs_2.total_possible_outputs, 15)

    def test_n_edits_with_quality_per_revision(self):
        pass

    def test_total_edits_with_quality(self):
        pass

    def test_perc_edits_with_quality(self):
        pass


if __name__ == '__main__':
    unittest.main()
