#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
"""
Tests for edit.py.
"""

import unittest

import numpy as np

from inceval.edit import EditQualities, EditQualityChart
from inceval.aux import EMPTY


class TestEditQualities(unittest.TestCase):
    """Tests for all edit's qualities."""

    def setUp(self):
        self.edits = [EditQualities(range_param=i) for i in range(15)]

    def test_set_effectiveness(self):
        self.edits[0].set_effectiveness(0, 1, 1)
        self.assertFalse(self.edits[0].ineffective)
        self.assertTrue(self.edits[0].effective)
        self.assertFalse(self.edits[0].defective)

        self.edits[1].set_effectiveness(0, 1, 2)
        self.assertTrue(self.edits[1].ineffective)
        self.assertFalse(self.edits[1].effective)
        self.assertFalse(self.edits[1].defective)

        self.edits[2].set_effectiveness(0, 1, 0)
        self.assertFalse(self.edits[2].ineffective)
        self.assertFalse(self.edits[2].effective)
        self.assertTrue(self.edits[2].defective)

        with self.assertRaises(AssertionError):
            self.edits[3].set_effectiveness(1, 1, 1)

        with self.assertRaises(AssertionError):
            self.edits[4].set_effectiveness(1, 1, 0)

    def test_set_convenience(self):
        self.edits[0].set_convenience(0, 1, 1)
        self.assertTrue(self.edits[0].convenient)
        self.assertFalse(self.edits[0].inconvenient)

        self.edits[1].set_convenience(0, 1, 0)
        self.assertFalse(self.edits[1].convenient)
        self.assertTrue(self.edits[1].inconvenient)

        self.edits[2].set_convenience(0, 2, 1)
        self.assertTrue(self.edits[2].convenient)
        self.assertFalse(self.edits[2].inconvenient)

        with self.assertRaises(AssertionError):
            self.edits[3].set_convenience(1, 1, 1)

        with self.assertRaises(AssertionError):
            self.edits[4].set_convenience(1, 1, 0)

    def test_set_novelty(self):
        previous_labels = np.array([1, 2, 1, 1, 3])
        self.edits[0].set_novelty(previous_labels, 2)
        self.assertFalse(self.edits[0].innovative)
        self.assertTrue(self.edits[0].repetitive)

        previous_labels = np.array([1, 2, 1, 1, 3])
        self.edits[1].set_novelty(previous_labels, 4)
        self.assertTrue(self.edits[1].innovative)
        self.assertFalse(self.edits[1].repetitive)

    def test_set_recurrence(self):
        vertical_seq = np.array([EMPTY, EMPTY, EMPTY, 1, 1, 1, 0, 0])
        self.edits[0].set_recurrence(3, 4, vertical_seq, 8)
        self.assertFalse(self.edits[0].steady)
        self.assertTrue(self.edits[0].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, EMPTY, 1, 1, 0, 0, 0])
        self.edits[1].set_recurrence(3, 4, vertical_seq, 8)
        self.assertTrue(self.edits[1].steady)
        self.assertFalse(self.edits[1].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 1, 1, 0, 0, 0])
        self.edits[2].set_recurrence(2, 4, vertical_seq, 8)
        self.assertFalse(self.edits[2].steady)
        self.assertTrue(self.edits[2].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 0, 1, 0, 0, 0])
        self.edits[3].set_recurrence(2, 4, vertical_seq, 8)
        self.assertTrue(self.edits[3].steady)
        self.assertFalse(self.edits[3].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 0, 1, 0, 0, 1])
        self.edits[4].set_recurrence(2, 7, vertical_seq, 8)
        self.assertTrue(self.edits[4].steady)
        self.assertFalse(self.edits[4].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 0, 1, 1, 0, 1])
        self.edits[5].set_recurrence(2, 4, vertical_seq, 8)
        self.assertFalse(self.edits[5].steady)
        self.assertTrue(self.edits[5].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 0, 1, 1, 0, 1])
        self.edits[6].set_recurrence(2, 5, vertical_seq, 8)
        self.assertFalse(self.edits[6].steady)
        self.assertTrue(self.edits[6].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 0, 1, 0, 1, 1])
        self.edits[7].set_recurrence(2, 7, vertical_seq, 8)
        self.assertFalse(self.edits[7].steady)
        self.assertTrue(self.edits[7].recurrent)

        vertical_seq = np.array([EMPTY, EMPTY, 1, 0, 1, 0, 1, 1])
        self.edits[8].set_recurrence(2, 6, vertical_seq, 8)
        self.assertFalse(self.edits[8].steady)
        self.assertTrue(self.edits[8].recurrent)

        vertical_seq = np.array([EMPTY, 1, 1, 0, 0, 1, 0, 1])
        self.edits[9].set_recurrence(1, 5, vertical_seq, 8)
        self.assertTrue(self.edits[9].steady)
        self.assertFalse(self.edits[9].recurrent)

        with self.assertRaises(AssertionError):
            vertical_seq = np.array([EMPTY, 1, 1, 0, 0, 1, 0, 1])
            self.edits[10].set_recurrence(1, 1, vertical_seq, 8)

    def test_set_oscillation(self):
        vertical_seq = np.array([EMPTY, EMPTY, EMPTY, 1, 1, 1, 0, 0])
        self.edits[0].set_oscillation(vertical_seq)
        self.assertFalse(self.edits[0].stable)
        self.assertTrue(self.edits[0].oscillating)

        vertical_seq = np.array([EMPTY, EMPTY, EMPTY, 1, 0, 1, 0, 0])
        self.edits[1].set_oscillation(vertical_seq)
        self.assertTrue(self.edits[1].stable)
        self.assertFalse(self.edits[1].oscillating)

    def test_set_connectedness(self):
        current_prefix = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        self.edits[0].set_connectedness(0, 11, current_prefix)
        self.assertTrue(self.edits[0].connected)
        self.assertFalse(self.edits[0].disconnected)

        current_prefix = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        self.edits[1].set_connectedness(1, 11, current_prefix)
        self.assertTrue(self.edits[1].connected)
        self.assertFalse(self.edits[1].disconnected)

        current_prefix = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        self.edits[2].set_connectedness(4, 11, current_prefix)
        self.assertFalse(self.edits[2].connected)
        self.assertTrue(self.edits[2].disconnected)

        current_prefix = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        self.edits[3].set_connectedness(6, 11, current_prefix)
        self.assertTrue(self.edits[3].connected)
        self.assertFalse(self.edits[3].disconnected)

        current_prefix = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        self.edits[4].set_connectedness(8, 11, current_prefix)
        self.assertTrue(self.edits[4].connected)
        self.assertFalse(self.edits[4].disconnected)

        current_prefix = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        self.edits[5].set_connectedness(10, 11, current_prefix)
        self.assertFalse(self.edits[5].connected)
        self.assertTrue(self.edits[5].disconnected)

        current_prefix = np.array([1, 0, 0, 0, 1, 1])
        self.edits[6].set_connectedness(0, 6, current_prefix)
        self.assertFalse(self.edits[6].connected)
        self.assertTrue(self.edits[6].disconnected)

        current_prefix = np.array([1, 0, 0, 0, 1, 1])
        self.edits[7].set_connectedness(4, 6, current_prefix)
        self.assertTrue(self.edits[7].connected)
        self.assertFalse(self.edits[7].disconnected)

        current_prefix = np.array([1, 0, 0, 0, 1, 1])
        self.edits[8].set_connectedness(5, 6, current_prefix)
        self.assertTrue(self.edits[8].connected)
        self.assertFalse(self.edits[8].disconnected)

        with self.assertRaises(AssertionError):
            current_prefix = np.array([1, 0, 0, 0, 1, 0])
            self.edits[9].set_connectedness(5, 6, current_prefix)

    def test_set_company(self):
        self.edits[0].set_company(np.array([0, 1, 1, 0, 1]))
        self.assertTrue(self.edits[0].accompanied)
        self.assertFalse(self.edits[0].isolated)

        self.edits[1].set_company(np.array([0, 0, 0, 0, 1]))
        self.assertFalse(self.edits[1].accompanied)
        self.assertTrue(self.edits[1].isolated)

    def test_set_distance(self):
        self.edits[0].set_distance(5, 3)
        self.assertTrue(self.edits[0].long)
        self.assertFalse(self.edits[0].short)

        self.edits[1].set_distance(5, 3)
        self.assertTrue(self.edits[1].long)
        self.assertFalse(self.edits[1].short)

        self.edits[2].set_distance(5, 3)
        self.assertFalse(self.edits[2].long)
        self.assertTrue(self.edits[2].short)

        self.edits[4].set_distance(5, 3)
        self.assertFalse(self.edits[4].long)
        self.assertTrue(self.edits[4].short)

    def test_set_definiteness(self):
        edits = np.array([0, 0, 1, 1, 1])
        self.edits[0].set_definiteness(2, edits, 5)
        self.assertFalse(self.edits[0].definite)
        self.assertTrue(self.edits[0].temporary)

        edits = np.array([0, 0, 1, 1, 1])
        self.edits[1].set_definiteness(3, edits, 5)
        self.assertFalse(self.edits[1].definite)
        self.assertTrue(self.edits[1].temporary)

        edits = np.array([0, 0, 1, 1, 1])
        self.edits[2].set_definiteness(4, edits, 5)
        self.assertTrue(self.edits[2].definite)
        self.assertFalse(self.edits[2].temporary)

        edits = np.array([1, 0, 1, 0, 0])
        self.edits[3].set_definiteness(2, edits, 5)
        self.assertTrue(self.edits[3].definite)
        self.assertFalse(self.edits[3].temporary)

        edits = np.array([1, 0, 1, 0, 0])
        self.edits[4].set_definiteness(0, edits, 5)
        self.assertFalse(self.edits[4].definite)
        self.assertTrue(self.edits[4].temporary)

    def test_set_time(self):
        self.edits[0].set_time(0, 5)
        self.assertFalse(self.edits[0].final)
        self.assertTrue(self.edits[0].intermediate)

        self.edits[1].set_time(3, 5)
        self.assertFalse(self.edits[1].final)
        self.assertTrue(self.edits[1].intermediate)

        self.edits[2].set_time(4, 5)
        self.assertTrue(self.edits[2].final)
        self.assertFalse(self.edits[2].intermediate)

    def test_set_qualities(self):
        pass


class TestEditQualityChart:
    """Tests for the EditQualityChart."""
    ...


if __name__ == '__main__':
    unittest.main()
