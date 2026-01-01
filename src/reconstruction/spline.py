"""
Cubic spline interpolation for signal reconstruction.

This module implements a non-causal reconstruction method using cubic splines,
which can utilize future information and typically achieves high accuracy.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.reconstruction.base import ReconstructionResult, Reconstructor


class SplineReconstructor(Reconstructor):
    """
    Cubic spline reconstructor for signal interpolation.
    
    This method uses cubic spline interpolation to reconstruct signals.
    It is non-causal (uses future information), making it suitable for
    offline analysis. Typically achieves excellent peak alignment but may
    overshoot in sparse data regions.
    """
    
    def __init__(self):
        """Initialize the spline reconstructor."""
        self.raw_data: pd.DataFrame | None = None
        self.elements: list[str] = []

    def fit(self, raw_df: pd.DataFrame) -> "SplineReconstructor":
        """Fit the reconstructor to raw sampled data."""
        self.raw_data = raw_df
        self.elements = raw_df["Element"].unique().tolist()
        return self

    def reconstruct(self, unified_time: np.ndarray) -> ReconstructionResult:
        """
        Reconstruct signals using cubic spline interpolation.
        
        Args:
            unified_time: Target time axis for reconstruction.
        
        Returns:
            ReconstructionResult with synchronized signals.
        """
        signals = {}
        
        for el in self.elements:
            sub = self.raw_data[self.raw_data["Element"] == el]
            if len(sub) < 4:
                # Cubic spline requires at least 4 points
                signals[el] = np.zeros_like(unified_time)
                continue

            t_raw = sub["Time"].values
            y_raw = sub["Intensity"].values
            
            # Sort data (spline requires monotonically increasing x)
            idx = np.argsort(t_raw)
            t_raw = t_raw[idx]
            y_raw = y_raw[idx]

            # Build cubic spline with natural boundary conditions
            cs = CubicSpline(t_raw, y_raw, bc_type='natural')
            
            # Interpolate and enforce non-negativity
            y_interp = cs(unified_time)
            y_interp = np.maximum(0, y_interp)
            signals[el] = y_interp
            
        return ReconstructionResult(time=unified_time, signals=signals)