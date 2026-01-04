"""
Causal kernel-based signal reconstruction.

This module implements a causal (real-time) reconstruction method using
exponentially weighted averaging of historical data points.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.reconstruction.base import ReconstructionResult, Reconstructor


class CausalKernelReconstructor(Reconstructor):
    """
    Causal kernel reconstructor using exponentially weighted averaging.
    
    This method only uses past data points (causal constraint), making it
    suitable for real-time processing. Each reconstructed point is a weighted
    average of historical measurements, with weights decaying exponentially.
    """
    
    def __init__(self, kernel: str = 'exp', tau: float = 0.05):
        """
        Initialize the causal kernel reconstructor.
        
        Args:
            kernel: Kernel type (currently only 'exp' supported).
            tau: Decay time constant. Smaller values give more weight to recent data.
        """
        self.kernel = kernel
        self.tau = tau
        self.raw_data: pd.DataFrame | None = None
        self.elements: list[str] = []

    def fit(self, raw_df: pd.DataFrame) -> "CausalKernelReconstructor":
        """Fit the reconstructor to raw sampled data."""
        self.raw_data = raw_df
        self.elements = raw_df["Element"].unique().tolist()
        return self

    def reconstruct(self, unified_time: np.ndarray) -> ReconstructionResult:
        """
        Reconstruct signals using causal kernel weighting.
        
        Args:
            unified_time: Target time axis for reconstruction.
        
        Returns:
            ReconstructionResult with synchronized signals.
        """
        signals = {}
        
        for el in self.elements:
            sub = self.raw_data[self.raw_data["Element"] == el]
            if len(sub) == 0:
                signals[el] = np.zeros_like(unified_time)
                continue
            
            t_raw = sub["Time"].values
            y_raw = sub["Intensity"].values
            
            # Compute time differences: dt[i, j] = t_grid[i] - t_raw[j]
            # Shape: (M, 1) - (1, N) -> (M, N)
            dt_matrix = unified_time[:, None] - t_raw[None, :]
            
            # Causal constraint: only use past data (dt >= 0)
            mask = dt_matrix >= 0
            
            # Exponential decay weights: w = exp(-dt/tau)
            weights = np.exp(-dt_matrix / self.tau) * mask
            
            # Weighted average: y_recon = sum(w * y_raw) / sum(w)
            numerator = np.sum(weights * y_raw[None, :], axis=1)
            denominator = np.sum(weights, axis=1)
            
            # Handle cases with no past data available
            valid_mask = denominator > 1e-12
            recon_y = np.zeros_like(unified_time)
            recon_y[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            
            signals[el] = recon_y

        return ReconstructionResult(time=unified_time, signals=signals)