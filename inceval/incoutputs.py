#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to represent the incremental outputs chart for one sequence and to
compute the evaluation metrics.
"""

from itertools import groupby
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from inceval.aux import (build_empty_chart, accuracy,
                         EMPTY, Criterion, SILVER, GOLD)
from inceval.edit import EditQualityChart, EditQualities
from inceval.revision import RevisionSeq

Label = Union[str, int, float]
Prefix = List[Label]


class IncOutputs:
    """A chart for incremental sequence labelling outputs and their metrics.

    The main chart is a lower triangular matrix in which each row represent
    one timestep. Cell (i, j) is the output label for token j at time i. The
    upper part is filled with the EMPTY constant as fillers.

    The edit chart contains 1 when a substitution/addition occured for a given
    label, else 0. It is a lower triangular matrix in which each row represent
    one timestep. The main diagonal is always filled with 1s, which are the
    additions. All other entries that are 1 are substitions. The upper part
    is filled with the EMPTY constant as fillers.
    """
    def __init__(self, n: int, gold: Optional[Prefix] = None,
                 recomputations: Optional[np.array] = None,
                 eval_mode: Criterion = SILVER, range_param: int = 2):
        """Initialise an empty incremental chart.

        After initialisation, the chart must be filled either with the method
        add_all_prefixes or step by step with the method add_prefix.

        Args:
            n (int): sequence length
            gold (Optional[Prefix], optional):
                Gold standard labels, if available. Defaults to None.
            recomputations (Optional[np.array], optional):
                An array containing the sequence of recomputations (True if
                there was a recomputation at that timestep). Defaults to None.
            eval_mode (Criterion, optional):
                Whether to evaluate using the gold or silver (i.e. the final
                output) standard. Defaults to SILVER.
            range_param (int):
                The number of time steps to distinguish short and long range
                edits and revisions.
        """
        self.eval_mode = eval_mode
        self.recomputations = recomputations
        self.chart = build_empty_chart(n)
        self.edits: Optional[np.array] = None
        self.edit_qualities: Optional[np.array] = None
        self.revision_qualities: Optional[np.array] = None
        self.silver: Optional[np.array] = None
        self.filled: bool = False
        self.standard: Optional[np.array] = None
        self.range_param: int = range_param
        if gold is not None:
            assert len(gold) == n
            self.gold = self._build_inc_gold(gold)
        if self.eval_mode is GOLD:
            assert gold is not None

    def add_all_prefixes(self, prefix_matrix: np.array) -> None:
        """Construct the chart from a complete output matrix.

        Args:
            prefix_matrix (np.array):
                Fill in the complete chart of incrementaloutputs at once
        """
        assert prefix_matrix.shape[0] == self.n_tokens
        for t, row in enumerate(prefix_matrix):
            prefix = list(row[: t+1].astype(int))
            self.add_prefix(t, prefix)

    def add_prefix(self, time_step: int, prefix: Prefix) -> None:
        """Construct the output chart row by row.

        Args:
            time_step (int): the time step, must by 1 + the current step
            prefix (Prefix): the output prefix of this time step
        """
        self._check_prefix_validity(time_step, prefix)
        self.chart[time_step][:time_step + 1] = prefix
        # upon addition of the last prefix, we create the edits and gold chart
        if time_step == self.n_tokens - 1:
            self._build_last_step()

    def _build_last_step(self) -> None:
        """Create edit charts, silver chart and set the evaluation standard."""
        self.edits = self._build_edits()
        self.silver = self._build_inc_silver()
        self._set_standard()
        self.filled = True
        self.edit_qualities = EditQualityChart(
            self.chart, self.edits, self.standard, self.range_param)
        self.revision_qualities = RevisionSeq(
            self.chart, self.edits, self.standard,
            self.revision_timesteps, self.range_param)

    def _check_prefix_validity(self, time_step: int, prefix: Prefix) -> None:
        """Ensure that prefixes are added in the right order.

        Args:
            time_step (int): the time step
            prefix (Prefix): the output prefix of this time step

        Raises:
            ValueError: if the chart is filled, no further prefix can be added
            ValueError: if the time step is not the expected next one
            ValueError: if the prefix length doesnt correspond to the time step
        """
        if self.filled:
            raise ValueError('The chart has already been filled!')
        # time_step starts from 0 and is the position to be filled
        if self._n_filled != time_step:
            raise ValueError(f'Add time step {self._n_filled} first!')
        if len(prefix) != time_step + 1:
            raise ValueError('Prefix length does not match currect time step!')

    def _build_inc_gold(self, gold: Prefix) -> np.array:
        """Create incremental gold chart.

        Args:
            gold (Prefix): the full gold sequence

        Returns:
            np.array: a chart with the incrementalised gold standard
        """
        inc_gold = build_empty_chart(self.n_tokens)
        for t in range(self.n_tokens):
            inc_gold[t][: t+1] = gold[: t+1]
        return inc_gold

    def _build_inc_silver(self) -> np.array:
        """Create incremental silver chart, with final output as gold.

        Returns:
            np.array: a chart with the incrementalised silver standard
        """
        inc_silver = build_empty_chart(self.n_tokens)
        for t in range(self.n_tokens):
            inc_silver[t][: t+1] = self.final_output[: t+1]
        return inc_silver

    def _build_edits(self) -> np.array:
        """Extract edits from incremental outputs by comparison rowwise.

        Returns:
            np.array: a chart with the edits (0: no edit, 1: edit)
        """
        edit_chart = np.full(self.chart.shape, 0.)
        for t, prefix in enumerate(self.chart):
            if t == 0:
                # the first label is an addition (by definition, an edit)
                edit_chart[t][0] = 1.
            else:
                previous_prefix = self.chart[t-1][:t]
                edited_labels = (prefix[:t] != previous_prefix).astype(float)
                edit_chart[t][:t] = edited_labels
                # the last label is an addition (by definition, an edit)
                edit_chart[t][t] = 1.
        return edit_chart

    @property
    def n_tokens(self) -> int:
        """Length of the input and final output sequence."""
        return self.chart.shape[0]

    @property
    def _n_filled(self) -> int:
        """How many timesteps have already been filled with output prefixes."""
        return (self.chart[:, 0] != EMPTY).sum()

    @property
    def final_output(self) -> np.array:
        """The non-incremental output (i.e. final label sequence)."""
        return self.chart[-1]

    @property
    def final_accuracy(self) -> float:
        """Correctness of final output wrt gold."""
        return accuracy(self.standard, self.final_output)

    def get_accuracy_at_t(self, t: int) -> float:
        """Correctness of a prefix at time step t."""
        return accuracy(self.standard[: t+1], self.chart[t][: t+1])

    @property
    def accuracy_by_turn(self) -> np.array:
        """Correctness of the prefix at each time step."""
        accs = [self.get_accuracy_at_t(t) for t in range(self.n_tokens)]
        return np.array(accs)

    def _set_standard(self) -> None:
        """Sets the internal gold standard, using gold or silver labels."""
        if self.eval_mode == GOLD:
            self.standard = self.gold[-1]
        if self.eval_mode == SILVER:
            self.standard = self.final_output

    @property
    def revision_timesteps(self) -> List[int]:
        """Return a list of timesteps where revisions occurred."""
        return [t for t, row in enumerate(self.edits) if (row == 1).sum() > 1]

    @property
    def effective_revision_timesteps(self) -> List[int]:
        """Return a list of timesteps where effective revisions occurred."""
        return [t for t in self.revision_timesteps
                if self.revision_qualities.seq[t].effective]

    @property
    def write_timesteps(self) -> List[int]:
        """Return a list of timesteps where revisions occurred."""
        return [t for t, row in enumerate(self.edits) if (row == 1).sum() == 1]

    @property
    def correct_prefixes(self) -> List[int]:
        """Which prefixes are correct with respect to the criterion."""
        return [t for t, row in enumerate(self.chart)
                if np.array_equal(row[:t+1], self.standard[:t+1])]

    @property
    def incorrect_prefixes(self) -> List[int]:
        """Which prefixes are correct with respect to the criterion."""
        return [t for t, row in enumerate(self.chart)
                if not np.array_equal(row[:t+1], self.standard[:t+1])]

    @property
    def n_correct_prefixes(self) -> int:
        """How many of the prefixes are correct."""
        return len(self.correct_prefixes)

    @property
    def n_correct_acted_prefixes(self) -> int:
        """How many prefixes upon which actions were made are correct."""
        # NOTE: using self.n_correct_prefixes directly does not work for the
        # appropriateness metrics denominator.
        # We need to ignore the last prefix, upon which no further
        # revision/addition was performed, and add the initial empty prefix
        if self.n_tokens - 1 in self.correct_prefixes:
            # if the last prefix is correct, we ignore it because no action
            # was made upon it
            return self.n_correct_prefixes
        # otherwise, we add one to account for the first (empty) prefix upon
        # which the first addition always occurs
        return self.n_correct_prefixes + 1

    @property
    def perc_correct_prefixes(self) -> float:
        """% of the prefixes that are correct."""
        return 100 * self.n_correct_prefixes / self.n_tokens

    @property
    def n_incorrect_prefixes(self) -> int:
        """How many of the prefixes are incorrect."""
        return len(self.incorrect_prefixes)

    @property
    def n_incorrect_acted_prefixes(self) -> int:
        """How many prefixes upon which actions were made are incorrect."""
        # NOTE: using self.n_incorrect_prefixes directly does not work here
        # because we need to ignore the last prefix, upon which no further
        # revision/addition was performed
        if self.n_tokens - 1 in self.incorrect_prefixes:
            # if the last prefix is incorrect, we ignore it because no action
            # was made upon it
            return self.n_incorrect_prefixes - 1
        return self.n_incorrect_prefixes

    @property
    def perc_incorrect_prefixes(self) -> float:
        """% of the prefixes that are incorrect."""
        return 100 * self.n_incorrect_prefixes / self.n_tokens

    @property
    def n_revisions(self) -> int:
        """How many revisions occurred."""
        return len(self.revision_timesteps)

    @property
    def perc_revisions(self) -> float:
        """% of timesteps with revision."""
        return 100 * self.n_revisions / self.n_tokens

    @property
    def n_effective_revisions(self) -> int:
        """How many effective revisions occurred."""
        return len(self.effective_revision_timesteps)

    @property
    def perc_effective_revisions(self) -> float:
        """% of timesteps with effective revision."""
        return 100 * self.n_effective_revisions / self.n_tokens

    @property
    def n_recomputations(self) -> int:
        """Total number of time steps where recomputations were performed."""
        if self.recomputations is None:
            return np.nan
        return self.recomputations.sum()

    @property
    def perc_recomputations(self) -> float:
        """% of timesteps with recomputations."""
        if self.recomputations is None:
            return np.nan
        return 100 * self.n_recomputations / self.n_tokens

    @property
    def recomputation_timesteps(self) -> List[int]:
        """Return a list of timesteps where recomputations occurred."""
        if self.recomputations is None:
            return np.nan
        return np.where(self.recomputations == True)[0].tolist()

    @property
    def n_active_recomputations(self) -> int:
        """Total number of recomputations that caused revisions."""
        if self.recomputations is None:
            return np.nan
        active = (set(self.revision_timesteps)
                  & set(self.recomputation_timesteps))
        return len(active)

    @property
    def perc_active_recomputations(self) -> float:
        """% of timesteps with recomputations that caused revisions."""
        if self.recomputations is None:
            return np.nan
        return 100 * self.n_active_recomputations / self.n_recomputations

    @property
    def n_inactive_recomputations(self) -> int:
        """Total number of recomputations that did not cause revisions."""
        if self.recomputations is None:
            return np.nan
        return self.n_recomputations - self.n_active_recomputations

    @property
    def perc_inactive_recomputations(self) -> float:
        """% of timesteps with recomputations that caused revisions."""
        if self.recomputations is None:
            return np.nan
        return 100 * self.n_inactive_recomputations / self.n_recomputations

    @property
    def n_writes(self) -> int:
        """How many writes (not revisions) occurred."""
        return len(self.write_timesteps)

    @property
    def perc_writes(self) -> float:
        """% of timesteps without revision."""
        return 100 * self.n_writes / self.n_tokens

    @property
    def n_revision_and_correct_prefix(self) -> int:
        """How many revisions on correct prefixes."""
        # NOTE: we need to compare whether a revision at time step t edited a
        # prefix that was correct at time step (t-1)
        shifted_steps = np.array(self.revision_timesteps) - 1
        intersect = set(shifted_steps) & set(self.correct_prefixes)
        return len(intersect)

    @property
    def n_revision_and_incorrect_prefix(self) -> int:
        """How many revisions on incorrect prefixes."""
        # NOTE: we need to compare whether a revision at time step t edited a
        # prefix that was incorrect at time step (t-1)
        shifted_steps = np.array(self.revision_timesteps) - 1
        intersect = set(shifted_steps) & set(self.incorrect_prefixes)
        return len(intersect)

    @property
    def n_effective_revision_and_correct_prefix(self) -> int:
        """How many effective revisions on correct prefixes."""
        # NOTE: we need to compare whether a revision at time step t edited a
        # prefix that was correct at time step (t-1)
        shifted_steps = np.array(self.effective_revision_timesteps) - 1
        intersect = set(shifted_steps) & set(self.correct_prefixes)
        return len(intersect)

    @property
    def n_effective_revision_and_incorrect_prefix(self) -> int:
        """How many effective revisions on incorrect prefixes."""
        # NOTE: we need to compare whether a revision at time step t edited a
        # prefix that was incorrect at time step (t-1)
        shifted_steps = np.array(self.effective_revision_timesteps) - 1
        intersect = set(shifted_steps) & set(self.incorrect_prefixes)
        return len(intersect)

    @property
    def n_write_and_correct_prefix(self) -> int:
        """How many writes on correct prefixes."""
        # NOTE: we need to compare whether a write at time step t extended a
        # prefix that was correct at time step (t-1)
        shifted_steps = np.array(self.write_timesteps) - 1
        intersect = set(shifted_steps) & set(self.correct_prefixes)
        # NOTE: we decided to add 1 here, i.e. to consider that the
        # emtpy prefix is correct and is always a write; this gives a bit
        # of advantage to the model evaluation
        # either we do this, or we have to change the total number of writes
        # otherwise the total won't sum to the number of time steps
        return len(intersect) + 1

    @property
    def n_write_and_incorrect_prefix(self) -> int:
        """How many writes on incorrect prefixes."""
        # NOTE: we need to compare whether a write at time step t extended a
        # prefix that was incorrect at time step (t-1)
        shifted_steps = np.array(self.write_timesteps) - 1
        intersect = set(shifted_steps) & set(self.incorrect_prefixes)
        return len(intersect)

    @property
    def r_pertinence(self) -> float:
        """Revisions on incorrect prefixes divided by all revisions."""
        if self.n_revisions == 0:
            return np.nan
        return self.n_revision_and_incorrect_prefix / self.n_revisions

    @property
    def r_pertinence_complement(self) -> float:
        """Revisions on correct prefixes divided by all revisions."""
        if self.n_revisions == 0:
            return np.nan
        return self.n_revision_and_correct_prefix / self.n_revisions

    @property
    def r_effective_pertinence(self) -> float:
        """Effective revisions on incorrect prefixes over all revisions."""
        if self.n_effective_revisions == 0:
            return np.nan
        return (self.n_effective_revision_and_incorrect_prefix
                / self.n_revisions)

    @property
    def a_pertinence(self) -> float:
        """Writes on correct prefixes divided by all writes."""
        if self.n_writes == 0:
            return np.nan
        return self.n_write_and_correct_prefix / self.n_writes

    @property
    def a_pertinence_complement(self) -> float:
        """Writes on incorrect prefixes divided by all writes."""
        if self.n_writes == 0:
            return np.nan
        return self.n_write_and_incorrect_prefix / self.n_writes

    @property
    def r_appropriateness(self) -> float:
        """Revisions on incorrect prefixes divided by all incorrect prefixes"""
        if self.n_incorrect_acted_prefixes == 0:
            return np.nan
        return (self.n_revision_and_incorrect_prefix
                / self.n_incorrect_acted_prefixes)

    @property
    def r_appropriateness_complement(self) -> float:
        """Writes on incorrect prefixes divided by all incorrect prefixes."""
        if self.n_incorrect_acted_prefixes == 0:
            return np.nan
        return (self.n_write_and_incorrect_prefix
                / self.n_incorrect_acted_prefixes)

    @property
    def r_effective_appropriateness(self) -> float:
        """Effective revisions on incorrect prefixes / incorrect prefixes"""
        if self.n_incorrect_acted_prefixes == 0:
            return np.nan
        return (self.n_effective_revision_and_incorrect_prefix
                / self.n_incorrect_acted_prefixes)

    @property
    def a_appropriateness(self) -> float:
        """Writes on correct prefixes divided by all correct prefixes."""
        if self.n_correct_acted_prefixes == 0:
            return np.nan
        return (self.n_write_and_correct_prefix
                / self.n_correct_acted_prefixes)

    @property
    def a_appropriateness_complement(self) -> float:
        """Revisions on correct prefixes divided by all correct prefixes."""
        if self.n_correct_acted_prefixes == 0:
            return np.nan
        return (self.n_revision_and_correct_prefix
                / self.n_correct_acted_prefixes)

    @property
    def edit_overhead(self) -> float:
        """Edit overhead metric on incremental chart."""
        necessary_edits = self.edits.diagonal().sum()
        unnecessary_edits = self.edits.sum() - necessary_edits
        return unnecessary_edits / (necessary_edits + unnecessary_edits)

    def delayed_edit_overhead(self, delay: int) -> float:
        """Edit overhead with delay."""
        assert delay > 0, "Delay must be a positive integer!"
        necessary_edits = self.edits.diagonal().sum() - delay
        diags = [self.edits.diagonal(-i).sum() for i in range(0, delay + 1)]
        unnecessary_edits = self.edits.sum() - np.sum(diags)
        return unnecessary_edits / (necessary_edits + unnecessary_edits)

    @property
    def relative_correctness(self) -> float:
        """Relative-correctness metric on incremental chart."""
        return self.n_correct_prefixes / self.n_tokens

    @property
    def correction_time_score(self) -> float:
        """Correction time score on incremental chart."""
        raise NotImplementedError

    @property
    def correction_time_per_token(self) -> np.array:
        """Correction time by label."""
        raise NotImplementedError

    @property
    def n_total_edits(self) -> int:
        """Total number of edits."""
        return np.tril(self.edits, k=-1).sum()

    @property
    def perc_total_edits(self) -> float:
        """% of edits."""
        return 100 * self.n_total_edits / self.total_possible_edits

    @property
    def n_edits_per_token(self) -> np.array:
        """Total number of edits per token."""
        return np.tril(self.edits, k=-1).sum(axis=0)

    @property
    def perc_edits_per_token(self) -> np.array:
        """% of edits per token."""
        return 100 * self.n_edits_per_token / self.possible_edits_per_token

    @property
    def n_edits_per_timestep(self) -> np.array:
        """Number of edits per timestep."""
        return np.tril(self.edits, k=-1).sum(axis=1)

    @property
    def perc_edits_per_timestep(self) -> np.array:
        """% of edits per timestep."""
        return 100 * (self.n_edits_per_timestep
                      / self.possible_edits_per_timestep)

    @property
    def n_edits_per_revision(self) -> np.array:
        """Number of edited labels for each time step where revision occurs."""
        return self.n_edits_per_timestep[self.revision_timesteps]

    @property
    def n_edit_groups_per_timestep(self) -> List[List[int]]:
        """Number of groups of edits for each time step."""
        n_edits = []
        for row in np.tril(self.edits, k=-1):
            n = [len(list(group)) for key, group in groupby(row) if key == 1]
            n_edits.append(n)
        return n_edits

    @property
    def n_edit_groups_per_revision(self) -> np.array:
        """Number of groups of edits for each time step with a revision."""
        return [x for i, x in enumerate(self.n_edit_groups_per_timestep)
                if i in self.revision_timesteps]

    @property
    def edit_distances(self) -> List[List]:
        """Distance of each edit to current step, for all steps."""
        distances = []
        for time_id, row in enumerate(np.tril(self.edits, k=-1)):
            row_dists = []
            for label_id in range(time_id):
                if row[label_id] == 1.:
                    row_dists.append(time_id - label_id)
            distances.append(row_dists)
        return distances

    @property
    def label_diversity_per_token(self) -> np.array:
        """Number of label types assigned to each token."""
        df = pd.DataFrame(self.chart).replace([EMPTY], np.nan)
        return df.nunique(dropna=True).to_list()

    def n_edits_with_quality_by_turn(self, quality: str):
        """Number of edits with a certain quality at each time step."""
        n_edits_quality = []
        for row in self.edit_qualities.chart:
            n = sum([1 for edit in row if self._has_quality(edit, quality)])
            n_edits_quality.append(n)
        return np.array(n_edits_quality)

    @staticmethod
    def _has_quality(var: Optional[EditQualities], quality: str) -> bool:
        """Check if a cell contains an edit that has a certain quality."""
        return var is not None and getattr(var, quality)

    @property
    def possible_edits_per_token(self) -> np.array:
        """Maximum number of edits that can happen for each token."""
        return np.arange(self.n_tokens - 1, -1, -1)

    @property
    def possible_edits_per_timestep(self) -> np.array:
        """Maximum number of edits that can happen at each time step."""
        return np.arange(0, self.n_tokens)

    @property
    def outputs_per_token(self) -> np.array:
        """Number of outputs for each token."""
        return np.arange(self.n_tokens, 0, -1)

    @property
    def total_possible_edits(self) -> np.array:
        """Maximum number of edits that can happen for a sequence."""
        return np.sum(np.arange(1, self.n_tokens))

    @property
    def total_possible_outputs(self) -> np.array:
        """Total output labels for a sequence."""
        return np.sum(np.arange(1, self.n_tokens + 1))

    def n_edits_with_quality_per_revision(self, quality: str) -> int:
        """Number of edits with a certain quality at revision steps."""
        edits = self.n_edits_with_quality_by_turn(quality)
        return edits[self.revision_timesteps]

    def total_edits_with_quality(self, quality: str) -> int:
        """Total number of edits with a certain quality in a chart."""
        return np.sum(self.n_edits_with_quality_by_turn(quality))

    def perc_edits_with_quality(self, quality: str) -> float:
        """% of all edits that have a certain quality."""
        return 100 * (self.total_edits_with_quality(quality)
                      / self.n_total_edits)

    def plot_inc_chart(self, tokens: List[str], figsize: Tuple[int]):
        """Plot the incremental chart with edits highlighted.

        Args:
            tokens (List[str]): input tokens
            figsize (Tuple[int]): a tuple with the matplotlib figure size
        """
        mock_edits = np.array([2 for x in range(self.n_tokens)])
        edits = np.vstack([self.edits, mock_edits])
        np.fill_diagonal(edits, 2)
        inc_labels = np.vstack([self.chart, self.standard])
        timesteps = [f"({token})    {i+1}" for i, token in enumerate(tokens)]
        timesteps += ['GOLD' if self.eval_mode == GOLD else 'SILVER']

        mask = np.triu(inc_labels)
        np.fill_diagonal(mask, False)

        _ = plt.figure(figsize=figsize)
        ax = sns.heatmap(edits, annot=inc_labels, mask=mask, cbar=False,
                         xticklabels=tokens, yticklabels=timesteps,
                         square=False, fmt='')
        ax.tick_params(left=False, bottom=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('(new token) timesteps')
        plt.xlabel('labeled input')
        plt.show()
