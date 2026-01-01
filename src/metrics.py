"""
Evaluation metrics for signal reconstruction quality.

This module provides metrics to assess the accuracy of reconstructed signals
compared to ground truth, including RMSE, peak timing, and area-under-curve errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate root mean square error.
    
    Args:
        y_true: Ground truth signal values.
        y_pred: Predicted signal values.
    
    Returns:
        RMSE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def peak_time_error(
    time: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Calculate peak time shift error.
    
    Args:
        time: Time axis.
        y_true: Ground truth signal.
        y_pred: Predicted signal.
    
    Returns:
        Peak time difference (predicted - true) in seconds.
    """
    time = np.asarray(time, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    t_true = float(time[int(np.argmax(y_true))])
    t_pred = float(time[int(np.argmax(y_pred))])
    return t_pred - t_true


def auc_error(
    time: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Calculate area-under-curve error.
    
    Args:
        time: Time axis.
        y_true: Ground truth signal.
        y_pred: Predicted signal.
    
    Returns:
        AUC difference (predicted - true).
    """
    time = np.asarray(time, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    auc_true = float(np.trapz(y_true, time))
    auc_pred = float(np.trapz(y_pred, time))
    return auc_pred - auc_true


def ratio_rmse(
    numerator_true: np.ndarray,
    denominator_true: np.ndarray,
    numerator_pred: np.ndarray,
    denominator_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Calculate RMSE of element ratio (e.g., Mg/Zn).
    
    Args:
        numerator_true: Ground truth numerator signal.
        denominator_true: Ground truth denominator signal.
        numerator_pred: Predicted numerator signal.
        denominator_pred: Predicted denominator signal.
        eps: Small value to avoid division by zero.
    
    Returns:
        RMSE of the ratio.
    """
    numerator_true = np.asarray(numerator_true, dtype=float)
    denominator_true = np.asarray(denominator_true, dtype=float)
    numerator_pred = np.asarray(numerator_pred, dtype=float)
    denominator_pred = np.asarray(denominator_pred, dtype=float)

    r_true = numerator_true / (denominator_true + eps)
    r_pred = numerator_pred / (denominator_pred + eps)
    return rmse(r_true, r_pred)


@dataclass(frozen=True)
class SignalMetrics:
    """Container for signal reconstruction metrics."""
    rmse: float
    peak_time_error: float
    auc_error: float


def compute_signal_metrics(
    time: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> SignalMetrics:
    """
    Compute all signal reconstruction metrics.
    
    Args:
        time: Time axis.
        y_true: Ground truth signal.
        y_pred: Predicted signal.
    
    Returns:
        SignalMetrics object containing RMSE, peak time error, and AUC error.
    """
    return SignalMetrics(
        rmse=rmse(y_true, y_pred),
        peak_time_error=peak_time_error(time, y_true, y_pred),
        auc_error=auc_error(time, y_true, y_pred),
    )