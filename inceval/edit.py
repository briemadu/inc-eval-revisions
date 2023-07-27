#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two classes, one to represent the qualities of a single edit and one to
represent all edits in an incremental chart.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

from inceval.aux import build_empty_chart, EMPTY

array = np.array
Label = Union[str, int, float]


@dataclass
class EditQualities:
    """Characteristics of an edit on a label."""
    range_param: int
    # effectiveness
    effective: bool = False
    defective: bool = False
    ineffective: bool = False
    # convenience
    convenient: bool = False
    inconvenient: bool = False
    # novelty
    innovative: bool = False
    repetitive: bool = False
    # recurrence (vertical)
    recurrent: bool = False
    steady: bool = False
    # oscillation (vertical)
    oscillating: bool = False
    stable: bool = False
    # connectedness (horizontal)
    connected: bool = False
    disconnected: bool = False
    # company
    accompanied: bool = False
    isolated: bool = False
    # distance
    short: bool = False
    long: bool = False
    # definiteness
    temporary: bool = False
    definite: bool = False
    # time
    intermediate: bool = False
    final: bool = False

    @staticmethod
    def edited(previous_label: Label, current_label: Label) -> bool:
        """Check if an edit occurred.

        Args:
            previous_label (Label): the label in the previous time step
            current_label (Label): the label in the current time step

        Returns:
            bool: Return True if labels differ.
        """
        return previous_label != current_label

    def set_effectiveness(self, previous_label: Label, current_label: Label,
                          gold_label: Label) -> None:
        """Set the effectiveness attribute of the edit.

        Args:
            previous_label (Label): the label in the previous time step
            current_label (Label): the label in the current time step
            gold_label (Label): the gold label
        """
        assert self.edited(previous_label, current_label), "Label not edited."
        if previous_label == gold_label and current_label != gold_label:
            self.defective = True
        elif previous_label != gold_label and current_label != gold_label:
            self.ineffective = True
        elif previous_label != gold_label and current_label == gold_label:
            self.effective = True

    def set_convenience(self, previous_label: Label, current_label: Label,
                        gold_label: Label) -> None:
        """Set the convenience attribute of the edit.

        Args:
            previous_label (Label): the label in the previous time step
            current_label (Label): the label in the current time step
            gold_label (Label): the gold label
        """
        assert self.edited(previous_label, current_label), "Label not edited."
        if previous_label == gold_label:
            self.inconvenient = True
        if previous_label != gold_label:
            self.convenient = True

    def set_novelty(self, previous_labels: array,
                    current_label: Label) -> None:
        """Set the novelty attribute of the edit.

        Args:
            previous_labels (array): all labels assigned to the token so far
            current_label (Label): the label in the current time step
        """
        if current_label in previous_labels:
            self.repetitive = True
        else:
            self.innovative = True

    def set_recurrence(self, label_id: int, time_id: int, vertical_seq: array,
                       n_tokens: int) -> None:
        """Set the recurrence attribute of the edit.

        Args:
            label_id (int): the label position
            time_id (int): the time step
            vertical_seq (array): all labels assigned to the token
            n_tokens (int): the total number of tokens in the sequence
        """
        assert label_id != time_id, "This is an addition, not an edit."
        assert vertical_seq[time_id] == 1., "Label not edited!"
        if label_id < (time_id-1) and vertical_seq[time_id - 1] == 1.:
            # check previous timestep, except for cases where the previous
            # time step was the addition
            self.recurrent = True
        elif time_id != (n_tokens - 1) and vertical_seq[time_id + 1] == 1.:
            # check the next time step, except for cases where the next
            # time step does not exist
            self.recurrent = True
        else:
            self.steady = True

    def set_oscillation(self, vertical_seq: array):
        """Set the oscillation attribute of the edit.

        Args:
            vertical_seq (array): all labels assigned to a token
        """
        edits = vertical_seq[vertical_seq != EMPTY]
        assert not edits.sum() < 2, "Label not edited!"
        if edits.sum() == 2:
            self.stable = True
        elif edits.sum() > 2:
            self.oscillating = True

    def set_connectedness(self, label_id: int, time_id: int,
                          current_edits: array) -> None:
        """Set the conectedness attribute of the edit.

        Args:
            label_id (int): the label position
            time_id (int): the time step
            current_edits (array):
                the edits made to the prefix, represented as a sequence of 0s
                and 1s (1 if an edit occurred)
        """
        assert current_edits[label_id] == 1., "Label not edited!"
        if label_id != 0 and current_edits[label_id - 1] == 1.:
            self.connected = True
        elif label_id < (time_id - 1) and current_edits[label_id + 1] == 1.:
            # main diagonal always an addition
            self.connected = True
        else:
            self.disconnected = True

    def set_company(self, edit_prefix: array) -> None:
        """Set the company attribute of the edit.

        Args:
            edit_prefix (array):
                the edits made to the prefix, represented as a sequence of 0s
                and 1s (1 if an edit occurred)
        """
        assert edit_prefix.sum() > 0, "No edits!"
        if edit_prefix.sum() > 1.:
            self.accompanied = True
        else:
            self.isolated = True

    def set_distance(self, time_id: int, label_id: int) -> None:
        """Set the distance attribute of the edit.

        Args:
            time_id (int): the time step
            label_id (int): the label position
        """
        if label_id < time_id - self.range_param:
            self.long = True
        else:
            self.short = True

    def set_definiteness(self, time_id: int, current_seq: array,
                         n_tokens: int) -> None:
        """Set the definiteness attribute of the edit.

        Args:
            time_id (int): the time step
            current_seq (array): the sequence of edits made to a label
            n_tokens (int): the number of tokens in the sequence
        """
        if time_id == n_tokens or current_seq[time_id + 1:].sum() == 0.:
            self.definite = True
        else:
            self.temporary = True

    def set_time(self, time_id: int, n_tokens: int) -> None:
        """Set the time attribute of the edit.

        Args:
            time_id (int): the time step
            n_tokens (int): the total number of tokens in the sequence
        """
        if time_id == n_tokens - 1:
            self.final = True
        else:
            self.intermediate = True

    def set_qualities(self, current_label: Label, previous_label: Label,
                      gold_label: Label, previous_labels: array,
                      current_edits: array, vertical_edits: array,
                      time_id: int, label_id: int, n_tokens: int) -> None:
        """Set all edit's qualities.

        Args:
            current_label (Label): the label currently assigned to the token
            previous_label (Label): the label assigned to the token in the
                last prefix
            gold_label (Label): the gold label of the token
            previous_labels (array): the sequence of labels assigned to the
                token
            current_edits (array): the edits in the current prefix,
                represented as a sequence of 0s (no edit) and 1s (edit)
            vertical_edits (array): the sequence of edits for the token
            time_id (int): the time step
            label_id (int): the label/token position
            n_tokens (int): the total number of tokens in the sequence
        """
        # initialise and fill in the qualities of the edit
        self.set_effectiveness(previous_label, current_label, gold_label)
        self.set_convenience(previous_label, current_label, gold_label)
        self.set_novelty(previous_labels, current_label)
        self.set_recurrence(label_id, time_id, vertical_edits, n_tokens)
        self.set_oscillation(vertical_edits)
        self.set_connectedness(label_id, time_id, current_edits)
        self.set_company(current_edits)
        self.set_distance(time_id, label_id)
        self.set_definiteness(time_id, vertical_edits, n_tokens)
        self.set_time(time_id, n_tokens)


class EditQualityChart:
    """Represent the incremental chart with the characteristics of edits."""
    def __init__(self, outputs: array, edits: array, gold: array,
                 range_param: int):
        """Initialise a chart containing all edits and their qualities.

        Args:
            outputs (array): the sequence of output prefixes (matrix)
            edits (array): the sequence of edit prefixes (matrix)
            gold (array): the gold standard sequence
            range_param (int): the number of time steps used to differentiate
                between long and short range edits
        """
        self._check_input(outputs, edits, gold)
        self.n_tokens = outputs.shape[0]
        self.range_param = range_param
        self.chart = self._fill_chart(outputs, edits, gold)

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

    def _fill_chart(self, outputs: array, edits: array, gold: array) -> array:
        """Extract and log qualities of each edit in the chart.

        Args:
            outputs (array): the sequence of output prefixes (matrix)
            edits (array): the sequence of edit prefixes (matrix)
            gold (array): the gold standard sequence

        Returns:
            array: the incremental edits chart filled with all edits
        """
        chart = build_empty_chart(self.n_tokens, filler=None)
        for time_id, label_id in zip(*np.tril_indices_from(edits, k=-1)):
            # looping over all elements in the lower portion of the matrix;
            # the main diagonal is ignored because it only contains additions
            if edits[time_id, label_id] == 0.:
                # label was not edited
                continue
            assert edits[time_id, label_id] == 1.
            edit = self._build_edit(time_id, label_id, outputs, edits, gold)
            chart[time_id, label_id] = edit
        return chart

    def _build_edit(self, time_id: int, label_id: int, outputs: array,
                    edits: array, gold: array) -> EditQualities:
        """Extract all subcomponents and build an edit object.

        Args:
            time_id (int): the time step
            label_id (int): the label/token position
            outputs (array): the sequence of output prefixes (matrix of labels)
            edits (array): the sequence of edit prefixes (matrix of 0s and 1s)
            gold (array): the sequence of gold labels

        Returns:
            EditQualities: a characterised edit
        """
        # get some specific components
        # the label assigned to the token at the current time step
        current_label = outputs[time_id, label_id]
        # the label assigned to the token in the previous time step
        previous_label = outputs[time_id - 1, label_id]
        # the label assigned to the token in the gold standard
        gold_label = gold[label_id]
        # all labels previously assigned to the token
        previous_labels = outputs[: time_id, label_id]
        # the edits that occurred in the current prefix
        current_edits = edits[time_id, : time_id]
        # the edits that occurred in all time steps for the current label
        vertical_edits = edits[:, label_id]
        assert previous_label != current_label
        # build the edit
        edit = EditQualities(range_param=self.range_param)
        edit.set_qualities(
            current_label, previous_label, gold_label, previous_labels,
            current_edits, vertical_edits, time_id, label_id, self.n_tokens)
        return edit
