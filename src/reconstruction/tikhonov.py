"""
Tikhonov regularization for smooth signal reconstruction.

This module implements a global optimization approach that balances
data fidelity with signal smoothness through regularization.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.reconstruction.base import ReconstructionResult, Reconstructor


class TikhonovReconstructor(Reconstructor):
    """
    Tikhonov regularized reconstructor for smooth signal recovery.
    
    Solves the optimization problem:
        min ||A*x - y||² + λ||D*x||²
    
    where A is the observation matrix, D is the smoothness operator,
    and λ controls the trade-off between data fitting and smoothness.
    """
    
    def __init__(self, lam: float = 10.0):
        """
        Initialize the Tikhonov reconstructor.
        
        Args:
            lam: Regularization coefficient. Larger values produce smoother results.
        """
        self.lam = lam
        self.raw_data: pd.DataFrame | None = None
        self.elements: list[str] = []

    def fit(self, raw_df: pd.DataFrame) -> "TikhonovReconstructor":
        """Fit the reconstructor to raw sampled data."""
        self.raw_data = raw_df
        self.elements = raw_df["Element"].unique().tolist()
        return self

    def reconstruct(self, unified_time: np.ndarray) -> ReconstructionResult:
        """
        Reconstruct signals using Tikhonov regularization.
        
        Args:
            unified_time: Target time axis for reconstruction.
        
        Returns:
            ReconstructionResult with synchronized signals.
        """
        signals = {}
        
        for el in self.elements:
            sub = self.raw_data[self.raw_data["Element"] == el]
            if len(sub) < 2:
                signals[el] = np.zeros_like(unified_time)
                continue
                
            t_raw = sub["Time"].values
            y_raw = sub["Intensity"].values
            
            x_recon = self._solve_tikhonov(t_raw, y_raw, unified_time)
            signals[el] = np.maximum(0, x_recon)
            
        return ReconstructionResult(time=unified_time, signals=signals)

    def _solve_tikhonov(
        self, 
        t_meas: np.ndarray, 
        y_meas: np.ndarray, 
        t_grid: np.ndarray
    ) -> np.ndarray:
        """
        Solve Tikhonov regularized least squares problem.
        
        Args:
            t_meas: Measurement time points.
            y_meas: Measurement values.
            t_grid: Target reconstruction grid.
        
        Returns:
            Reconstructed signal on t_grid.
        """
        n_grid = len(t_grid)
        dt = t_grid[1] - t_grid[0]
        
        # Build observation matrix A (linear interpolation)
        # A[i, j] maps grid point j to measurement i
        # For measurement at time t between t_grid[k] and t_grid[k+1]:
        #   A[i, k] = 1 - w, A[i, k+1] = w, where w = (t - t_grid[k]) / dt
        
        idx_left = np.searchsorted(t_grid, t_meas) - 1
        idx_left = np.clip(idx_left, 0, n_grid - 2)
        
        t_left = t_grid[idx_left]
        w = (t_meas - t_left) / dt
        
        # Construct sparse matrix A
        rows = np.repeat(np.arange(len(t_meas)), 2)
        cols = np.dstack((idx_left, idx_left + 1)).flatten()
        data = np.dstack((1 - w, w)).flatten()
        
        A = sparse.coo_matrix((data, (rows, cols)), shape=(len(t_meas), n_grid))
        
        # Build smoothness operator D (first-order difference)
        data_D = np.ones((2, n_grid))
        data_D[0, :] = -1
        data_D[1, :] = 1
        D = sparse.spdiags(data_D, [0, 1], n_grid - 1, n_grid)
        
        # Solve: (A'*A + λ*D'*D) * x = A'*y
        ATA = A.T @ A
        DTD = D.T @ D
        LHS = ATA + self.lam * DTD
        RHS = A.T @ y_meas
        
        x = spsolve(LHS, RHS)
        return x