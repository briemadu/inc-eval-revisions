#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for revision.py.
"""

import unittest

import numpy as np

from inceval.revision import RevisionQualities, RevisionSeq


class TestRevisionQualities(unittest.TestCase):
    """Tests for all revision qualities."""

    def setUp(self):
        self.revisions = [RevisionQualities(range_param=i) for i in range(15)]

    def test_set_effectiveness(self):
        current = np.array([0, 1, 1, 2, 1])
        previous = np.array([0, 1, 1, 2, 0])
        gold = np.array([0, 1, 1, 2, 1])
        self.revisions[0].set_effectiveness(current, previous, gold)
        self.assertFalse(self.revisions[0].ineffective)
        self.assertFalse(self.revisions[0].defective)
        self.assertTrue(self.revisions[0].effective)

        previous = np.array([0, 2, 1, 2, 0])
        gold = np.array([0, 2, 1, 2, 0])
        self.revisions[1].set_effectiveness(current, previous, gold)
        self.assertFalse(self.revisions[1].ineffective)
        self.assertTrue(self.revisions[1].defective)
        self.assertFalse(self.revisions[1].effective)

        previous = np.array([0, 2, 1, 2, 0])
        gold = np.array([0, 3, 1, 2, 3])
        self.revisions[2].set_effectiveness(current, previous, gold)
        self.assertTrue(self.revisions[2].ineffective)
        self.assertFalse(self.revisions[2].defective)
        self.assertFalse(self.revisions[2].effective)

    def test_set_convenience(self):
        previous = np.array([0, 2, 4, 2, 0])
        gold = np.array([0, 2, 1, 2, 0])
        self.revisions[0].set_convenience(previous, gold)
        self.assertTrue(self.revisions[0].convenient)
        self.assertFalse(self.revisions[0].inconvenient)

        previous = np.array([0, 2, 4, 2, 0])
        gold = np.array([0, 2, 4, 2, 0])
        self.revisions[1].set_convenience(previous, gold)
        self.assertFalse(self.revisions[1].convenient)
        self.assertTrue(self.revisions[1].inconvenient)

    def test_set_recurrence(self):
        time_step = 5
        revsteps = [0, 4, 5, 8, 9]
        self.revisions[0].set_recurrence(time_step, revsteps)
        self.assertTrue(self.revisions[0].recurrent)
        self.assertFalse(self.revisions[0].steady)

        time_step = 5
        revsteps = [0, 3, 5, 6, 9]
        self.revisions[1].set_recurrence(time_step, revsteps)
        self.assertTrue(self.revisions[1].recurrent)
        self.assertFalse(self.revisions[1].steady)

        time_step = 5
        revsteps = [0, 3, 5, 8, 9]
        self.revisions[2].set_recurrence(time_step, revsteps)
        self.assertFalse(self.revisions[2].recurrent)
        self.assertTrue(self.revisions[2].steady)

        time_step = 0
        revsteps = [0, 3, 5, 8, 9]
        self.revisions[3].set_recurrence(time_step, revsteps)
        self.assertFalse(self.revisions[3].recurrent)
        self.assertTrue(self.revisions[3].steady)

        time_step = 0
        revsteps = [0, 1, 5, 8, 9]
        self.revisions[4].set_recurrence(time_step, revsteps)
        self.assertTrue(self.revisions[4].recurrent)
        self.assertFalse(self.revisions[4].steady)

    def test_set_oscillation(self):
        revsteps = [0, 1, 5, 8, 9]
        self.revisions[0].set_oscillation(revsteps)
        self.assertTrue(self.revisions[0].oscillating)
        self.assertFalse(self.revisions[0].stable)

        revsteps = [7]
        self.revisions[1].set_oscillation(revsteps)
        self.assertFalse(self.revisions[1].oscillating)
        self.assertTrue(self.revisions[1].stable)

    def test_set_edit_company(self):
        edits = np.array([0, 0, 1, 0, 1, 0])
        self.revisions[0].set_edit_company(edits)
        self.assertTrue(self.revisions[0].accompanied_edits)
        self.assertFalse(self.revisions[0].isolated_edit)

        edits = np.array([0, 0, 0, 0, 1, 0])
        self.revisions[1].set_edit_company(edits)
        self.assertFalse(self.revisions[1].accompanied_edits)
        self.assertTrue(self.revisions[1].isolated_edit)

    def test_set_edit_connectedness(self):
        edits = np.array([1, 1, 0, 0, 1])
        self.revisions[0].set_edit_connectedness(edits)
        self.assertFalse(self.revisions[0].connected_edits)
        self.assertFalse(self.revisions[0].disconnected_edits)
        self.assertTrue(self.revisions[0].dis_and_connected_edits)

        edits = np.array([0, 1, 1, 0, 0])
        self.revisions[1].set_edit_connectedness(edits)
        self.assertTrue(self.revisions[1].connected_edits)
        self.assertFalse(self.revisions[1].disconnected_edits)
        self.assertFalse(self.revisions[1].dis_and_connected_edits)

        edits = np.array([0, 0, 1, 0, 1])
        self.revisions[2].set_edit_connectedness(edits)
        self.assertFalse(self.revisions[2].connected_edits)
        self.assertTrue(self.revisions[2].disconnected_edits)
        self.assertFalse(self.revisions[2].dis_and_connected_edits)

    def test_set_distance(self):
        edits = np.array([1, 0, 0, 0, 0, 1, 0])
        self.revisions[2].set_distance(edits)
        self.assertFalse(self.revisions[2].long_range)
        self.assertFalse(self.revisions[2].short_range)
        self.assertTrue(self.revisions[2].short_and_long_range)

        edits = np.array([0, 0, 0, 0, 1, 0, 0])
        self.revisions[3].set_distance(edits)
        self.assertFalse(self.revisions[3].long_range)
        self.assertTrue(self.revisions[3].short_range)
        self.assertFalse(self.revisions[3].short_and_long_range)

        edits = np.array([1, 0, 1, 0, 0, 0, 0])
        self.revisions[4].set_distance(edits)
        self.assertTrue(self.revisions[4].long_range)
        self.assertFalse(self.revisions[4].short_range)
        self.assertFalse(self.revisions[4].short_and_long_range)

    def test_set_definiteness(self):
        time_step = 3
        revsteps = [0, 2, 3, 7, 8]
        self.revisions[0].set_definiteness(time_step, revsteps)
        self.assertTrue(self.revisions[0].temporary)
        self.assertFalse(self.revisions[0].definite)

        time_step = 8
        revsteps = [0, 2, 3, 7, 8]
        self.revisions[1].set_definiteness(time_step, revsteps)
        self.assertFalse(self.revisions[1].temporary)
        self.assertTrue(self.revisions[1].definite)

    def test_set_time(self):
        time_step = 5
        n_tokens = 8
        self.revisions[0].set_time(time_step, n_tokens)
        self.assertTrue(self.revisions[0].intermediate)
        self.assertFalse(self.revisions[0].final)

        time_step = 7
        n_tokens = 8
        self.revisions[1].set_time(time_step, n_tokens)
        self.assertFalse(self.revisions[1].intermediate)
        self.assertTrue(self.revisions[1].final)


class TestRevisionSeq:
    """Tests for the sequence of revisions."""
    ...


if __name__ == '__main__':
    unittest.main()
