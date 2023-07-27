#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions and variables.
"""

import enum
from typing import Optional

import numpy as np

EMPTY = np.inf

# We can compare prefixes to the gold labels or to the final output.
Criterion = enum.Enum('Criterion', ['GOLD', 'SILVER'])
SILVER = Criterion.SILVER  # represents the final output
GOLD = Criterion.GOLD      # represents the real gold standard


def build_empty_chart(n: int, filler: Optional[float] = EMPTY) -> np.array:
    """Initialize a chart with a given shape, filled with EMPTY default.

    Args:
        n (int): the number of tokens (same as time steps)
        filler (Optional[float], optional):
            The symbol to fill in the chart with. Defaults to EMPTY.

    Returns:
        np.array: A matrix filled with the given symbol.
    """
    return np.full([n, n], filler, dtype='O')


def accuracy(x: np.array, y: np.array) -> float:
    """Fraction of labels that match in x and y.

    Args:
        x (np.array): a sequence
        y (np.array): another sequence

    Returns:
        float: the proportion of symbols that match in both sequences
    """
    return np.mean(x == y)
