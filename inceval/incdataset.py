#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to represent the incremental outputs charts for all sentences in a
dataset, and retrieve dataset-level metrics.
"""

from typing import List
import numpy as np


class IncData:
    """
    Represent a complete dataset, where each item in self.instances
    is the incremental chart of one sentence in the dataset.
    """
    def __init__(self, output_dic: dict):
        """Initialise a dataset

        Args:
            output_dic (dict): a dictionary mapping a sequence ID to an
                IncOutputs object build upon its incremental outputs
        """
        self.instances = output_dic
        self.seqs = self.instances.values()

    def edits_with_quality(self, quality: str) -> List[int]:
        """Number of edits with a given quality per sentence.

        Args:
            quality (str): the name of an edit quality

        Returns:
            List[int]: how many edits with a certain quality in each sequence
        """
        return [seq.total_edits_with_quality(quality) for seq in self.seqs]

    def perc_edits_with_quality(self, quality: str) -> float:
        """% edits with a given quality in the dataset.

        Args:
            quality (str): the name of an edit quality

        Returns:
            float: the percentage of edits with a quality (over all edits)
        """
        n_edits_quality = np.sum(self.edits_with_quality(quality))
        return 100 * n_edits_quality / self.get_total('n_total_edits')

    def revisions_with_quality(self, quality: str) -> List[int]:
        """Number of revisions with a given quality per sentence.

        Args:
            quality (str): the name of a revision quality

        Returns:
            List[int]: how many revision with a quality in each sequence
        """
        return [seq.revision_qualities.n_revisions_with_quality(quality)
                for seq in self.seqs]

    def perc_revisions_with_quality(self, quality: str) -> float:
        """% revisions with a given quality in the dataset.

        Args:
            quality (str): the name of a revision quality

        Returns:
            float: the percentage of revisions with a quality (over all edits)
        """
        n_revisions_quality = np.sum(self.revisions_with_quality(quality))
        return 100 * n_revisions_quality / self.get_total('n_revisions')

    def get_dist(self, attr: str) -> list:
        """Distribution of number of a given attribute in a sentence.

        Args:
            attr (str): the name of an attribute

        Returns:
            list: the attribute value for each element in the dataset
        """
        return [getattr(seq, attr) for seq in self.seqs]

    def get_total(self, attr: str) -> float:
        """Total number of a given attribute in the dataset.

        Args:
            attr (str): the name of an attribute

        Returns:
            float: the total number of the attribute in the dataset
        """
        return np.sum(self.get_dist(attr))

    def get_perc(self, attr: str, denominator: str) -> float:
        """% number of a given attribute in the dataset.

        Denominator required because we may want to divide by total number of
        tokens, or total number of edits, or revisions etc.

        Args:
            attr (str): the name of the attribute
            denominator (str): what attribute to use as denominator

        Returns:
            float: the fraction of observations of the attribute over the
                chose denominator
        """
        return 100 * self.get_total(attr) / self.get_total(denominator)

    def get_mean(self, attr: str) -> float:
        """Mean of a given attribute in the dataset.

        Args:
            attr (str): the name of the attribute

        Returns:
            float: the average of the attribute in the dataset, excluding
                cases where the attribute is undefined
        """
        return np.mean([x for x in self.get_dist(attr) if not np.isnan(x)])

    def get_std(self, attr: str) -> float:
        """Standard deviation of a given attribute in the dataset.

        Args:
            attr (str): the name of the attribute

        Returns:
            float: the standard deviation of the attribute in the dataset,
                excluding cases where the attribute is undefined
        """
        return np.std([x for x in self.get_dist(attr) if not np.isnan(x)])

    @property
    def perc_revisions(self) -> float:
        """% of timesteps with revisions in the dataset."""
        return 100 * self.get_total('n_revisions') / self.get_total('n_tokens')

    @property
    def perc_recomputations(self) -> float:
        """% of timesteps with recomputations in the dataset."""
        return 100 * (self.get_total('n_recomputations')
                      / self.get_total('n_tokens'))

    @property
    def perc_active_recomputations(self) -> float:
        """% of recomputations that caused revisions in the dataset."""
        return 100 * (self.get_total('n_active_recomputations')
                      / self.get_total('n_recomputations'))

    @property
    def r_pertinence(self) -> float:
        """Dataset-level revision-pertinence."""
        numerator = self.get_total('n_revision_and_incorrect_prefix')
        denominator = self.get_total('n_revisions')
        return numerator / denominator

    @property
    def a_pertinence(self) -> float:
        """Dataset-level addition-pertinence."""
        numerator = self.get_total('n_write_and_correct_prefix')
        denominator = self.get_total('n_writes')
        return numerator / denominator

    @property
    def r_appropriateness(self) -> float:
        """Dataset-level revision-appropriateness."""
        numerator = self.get_total('n_revision_and_incorrect_prefix')
        denominator = self.get_total('n_incorrect_acted_prefixes')
        return numerator / denominator

    @property
    def a_appropriateness(self) -> float:
        """Dataset-level addition-appropriateness."""
        numerator = self.get_total('n_write_and_correct_prefix')
        denominator = self.get_total('n_correct_acted_prefixes')
        return numerator / denominator

    @property
    def r_effective_pertinence(self) -> float:
        """Dataset-level effective revision-pertinence."""
        numerator = self.get_total('n_effective_revision_and_incorrect_prefix')
        denominator = self.get_total('n_revisions')
        return numerator / denominator

    @property
    def r_effective_appropriateness(self) -> float:
        """Dataset-level effective revision-appropriateness."""
        numerator = self.get_total('n_effective_revision_and_incorrect_prefix')
        denominator = self.get_total('n_incorrect_acted_prefixes')
        return numerator / denominator
