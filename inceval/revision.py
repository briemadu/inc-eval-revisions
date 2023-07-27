#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two classes, one to represent the qualities of a single revision and one to
represent the complete revision sequence.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import List, Optional

import numpy as np
from sklearn.metrics import accuracy_score as accuracy

array = np.array


@dataclass
class RevisionQualities:
    """Characteristics of a revised prefix."""
    range_param: int
    # effectiveness
    effective: bool = False
    defective: bool = False
    ineffective: bool = False
    # convenience
    convenient: bool = False
    inconvenient: bool = False
    # recurrence
    recurrent: bool = False
    steady: bool = False
    # oscillation
    oscillating: bool = False
    stable: bool = False
    # edit company
    isolated_edit: bool = False
    accompanied_edits: bool = False
    # clusters of edits
    disconnected_edits: bool = False
    connected_edits: bool = False
    dis_and_connected_edits: bool = False
    # distance
    short_range: bool = False
    long_range: bool = False
    short_and_long_range: bool = False
    # definiteness
    definite: bool = False
    temporary: bool = False
    # time
    intermediate: bool = False
    final: bool = False

    def set_effectiveness(self, current: array, previous: array,
                          gold: array) -> None:
        """Set the effectiveness attribute of the revision.

        Args:
            current (array): _description_
            previous (array): _description_
            gold (array): _description_
        """
        current = current.astype(float)
        previous = previous.astype(float)
        gold = gold.astype(float)
        if accuracy(gold, current) > accuracy(gold, previous):
            self.effective = True
        elif accuracy(gold, current) < accuracy(gold, previous):
            self.defective = True
        else:
            self.ineffective = True

    def set_convenience(self, previous: array, gold: array) -> None:
        """Set the convenience attribute of the revision."""
        if np.array_equal(gold, previous):
            self.inconvenient = True
        else:
            self.convenient = True

    def set_recurrence(self, time_step: int, revsteps: List[int]) -> None:
        """Set the recurrence attribute of the revision."""
        if time_step - 1 in revsteps or time_step + 1 in revsteps:
            self.recurrent = True
        else:
            self.steady = True

    def set_oscillation(self, revsteps: List[int]) -> None:
        """Set the oscillation attribute of the revision."""
        if len(revsteps) > 1:
            self.oscillating = True
        elif len(revsteps) == 1:
            self.stable = True

    def set_edit_company(self, edits: array) -> None:
        """Set the edit company attribute of the revision."""
        if edits.sum() == 1.:
            self.isolated_edit = True
        else:
            self.accompanied_edits = True

    def set_edit_connectedness(self, edits: array) -> None:
        """Set the edit connectedness attribute of the revision."""
        edits = [len(list(group)) for key, group in groupby(edits) if key == 1]
        connected, disconnected = False, False
        if len(edits) != sum(edits):
            connected = True
        if 1 in edits:
            disconnected = True
        if connected and disconnected:
            # it can be that both are true, meaning that some edits are
            # disconnected and some are connected
            self.dis_and_connected_edits = True
        elif connected:
            self.connected_edits = True
        elif disconnected:
            self.disconnected_edits = True

    def set_distance(self, current_edits: array) -> None:
        """Set the distance attribute of the revision."""
        long_seq = current_edits[:-self.range_param]
        short_seq = current_edits[-self.range_param:]
        long, short = False, False
        if long_seq.sum() > 0.:
            long = True
        if short_seq.sum() > 0.:
            short = True
        if short and long:
            self.short_and_long_range = True
        elif short:
            self.short_range = True
        elif long:
            self.long_range = True

    def set_definiteness(self, time_step: int, revsteps: List[int]) -> None:
        """Set the definiteness attribute of the revision."""
        if time_step == max(revsteps):
            self.definite = True
        else:
            self.temporary = True

    def set_time(self, time_step: int, n_tokens: int) -> None:
        """Set the time attribute of the revision."""
        if time_step == n_tokens - 1:
            self.final = True
        else:
            self.intermediate = True

    def set_qualities(self, current_prefix: array, previous_prefix: array,
                      gold_prefix: array, current_edits: array,
                      time_step: int, revsteps: List[int], n_tokens: int):
        """Extract all revision's qualities."""
        self.set_effectiveness(current_prefix, previous_prefix, gold_prefix)
        self.set_convenience(previous_prefix, gold_prefix)
        self.set_recurrence(time_step, revsteps)
        self.set_oscillation(revsteps)
        self.set_edit_company(current_edits)
        self.set_edit_connectedness(current_edits)
        self.set_distance(current_edits)
        self.set_definiteness(time_step, revsteps)
        self.set_time(time_step, n_tokens)


class RevisionSeq:
    """Represent the sequence of characterised revisions."""
    def __init__(self, outputs: array, edits: array, gold: array,
                 revision_timesteps: List[int], range_param: int):
        """Initialise the sequence of revisions

        Args:
            outputs (array): the sequence of output prefixes (matrix of labels)
            edits (array): the sequence of edit prefixes (matrix of 0s and 1s)
            gold (array): the sequence of gold labels
            revision_timesteps (List[int]): time steps where revisions occurred
            range_param (int): the number of time steps used to differentiate
                between long and short range revisions
        """
        self._check_input(outputs, edits, gold)
        self.n_tokens = outputs.shape[0]
        self.range_param = range_param
        self.n_revisions = len(revision_timesteps)
        self.seq = self._fill_seq(outputs, edits, gold, revision_timesteps)

    @staticmethod
    def _check_input(outputs: array, edits: array, gold: array) -> None:
        """Check that sizes match.

        Args:
            outputs (array): the sequence of output prefixes (matrix)
            edits (array): the sequence of edit prefixes (matrix)
            gold (array): the gold standard sequence
        """
        assert outputs.shape == edits.shape
        assert gold.shape[0] == outputs.shape[0]

    def _fill_seq(self, outputs: array, edits: array, gold: array,
                  revision_timesteps: list) -> array:
        """Extract all revisions' qualities.

        Args:
            outputs (array): the sequence of output prefixes (matrix)
            edits (array): the sequence of edit prefixes (matrix)
            gold (array): the gold standard sequence
            revision_timesteps (list): the time steps where revisions occurred

        Returns:
            array: the sequence of characterised revisions
        """
        seq = np.full([self.n_tokens], None)
        for time_step in revision_timesteps:
            revision = self._build_revision(
                outputs, edits, gold, revision_timesteps, time_step)
            seq[time_step] = revision
        return seq

    def _build_revision(self, outputs: array, edits: array, gold: array,
                        revision_timesteps: List[int],
                        time_step: int) -> RevisionQualities:
        """Extract all subcomponents and build a revision object.

        Args:
            outputs (array): the sequence of output prefixes (matrix)
            edits (array): the sequence of edit prefixes (matrix)
            gold (array): the gold standard sequence
            revision_timesteps (List[int]): the time steps where revisions
                occurred
            time_step (int): the time step

        Returns:
            RevisionQualities: a characterised revision
        """
        current_prefix = outputs[time_step, : time_step]
        previous_prefix = outputs[time_step - 1, : time_step]
        gold_prefix = gold[:time_step]
        current_edits = edits[time_step, : time_step]
        assert not np.array_equal(current_prefix, previous_prefix)
        assert current_prefix.shape == previous_prefix.shape
        revision = RevisionQualities(self.range_param)
        revision.set_qualities(
            current_prefix, previous_prefix, gold_prefix, current_edits,
            time_step, revision_timesteps, self.n_tokens)
        return revision

    @staticmethod
    def _has_quality(var: Optional[RevisionQualities], quality: str) -> bool:
        """Check if a step contains a revision that has a certain quality.

        Args:
            var (Optional[RevisionQualities]): a cell in the revisions sequence
            quality (str): the name of a quality

        Returns:
            bool: True if cell is a revision and has the quality, else False
        """
        return var is not None and getattr(var, quality)

    def n_revisions_with_quality(self, quality: str) -> int:
        """Number of revisions with a certain quality in the sequence.

        Args:
            quality (str): the name of a quality

        Returns:
            int: how many revisions have a given quality
        """
        revs = [1 for step in self.seq if self._has_quality(step, quality)]
        return len(revs)

    def perc_revisions_with_quality(self, quality: str) -> float:
        """Percentage of revisions with a certain quality in the sequence.

        Args:
            quality (str): the name of a quality

        Returns:
            float: the % of revisions that have a quality
        """
        return self.n_revisions_with_quality(quality) / self.n_revisions
