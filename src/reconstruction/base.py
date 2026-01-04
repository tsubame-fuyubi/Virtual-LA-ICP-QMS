"""
Base classes for signal reconstruction algorithms.

This module defines the common interface that all reconstruction methods must implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconstructionResult:
    """
    Container for reconstruction results.
    
    Attributes:
        time: Unified time axis for reconstructed signals.
        signals: Dictionary mapping element names to reconstructed signal arrays.
    """
    time: np.ndarray
    signals: Dict[str, np.ndarray]


class Reconstructor:
    """
    Base interface for signal reconstruction algorithms.
    
    All reconstruction methods follow a two-step process:
    1. fit(): Learn from raw sampled data.
    2. reconstruct(): Generate synchronized signals on a unified time grid.
    
    Expected raw_df format (long table):
        - Time: float (seconds)
        - Element: str (element name)
        - Intensity: float (measured intensity)
    """

    def fit(self, raw_df: pd.DataFrame) -> "Reconstructor":
        """
        Fit the reconstructor to raw sampled data.
        
        Args:
            raw_df: DataFrame with columns Time, Element, Intensity.
        
        Returns:
            Self for method chaining.
        """
        raise NotImplementedError

    def reconstruct(self, unified_time: np.ndarray) -> ReconstructionResult:
        """
        Reconstruct synchronized signals on a unified time grid.
        
        Args:
            unified_time: Target time axis for reconstruction.
        
        Returns:
            ReconstructionResult containing time and signals.
        """
        raise NotImplementedError