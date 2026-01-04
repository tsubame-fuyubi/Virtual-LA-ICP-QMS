"""
Command-line experiment pipeline for comparing reconstruction methods.

This script runs a complete comparison of all three reconstruction algorithms
on simulated LA-ICP-MS data and outputs performance metrics and visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.physics_engine import AerosolGenerator
from src.virtual_instrument import QuadrupoleMS
from src.metrics import compute_signal_metrics
from src.reconstruction.kernel import CausalKernelReconstructor
from src.reconstruction.spline import SplineReconstructor
from src.reconstruction.tikhonov import TikhonovReconstructor


def run_experiment():
    """
    Run complete comparison experiment for all reconstruction methods.
    
    The experiment:
    1. Generates ground truth aerosol pulse signals
    2. Simulates QMS sequential scanning
    3. Reconstructs signals using three different methods
    4. Evaluates performance and visualizes results
    """
    # Experiment configuration
    elements = ["Mg", "Zn"]
    dwell_time = 0.03  # QMS dwell time (seconds)
    dt_recon = 0.01    # Reconstruction time step (seconds)
    
    # Algorithm parameters
    tau = 0.06  # Kernel decay constant
    lam = 5.0   # Tikhonov regularization coefficient

    print("=== Experiment Configuration ===")
    print(f"Dwell time: {dwell_time}s, Reconstruction dt: {dt_recon}s")
    print(f"Parameters: Kernel tau={tau}, Tikhonov lambda={lam}")
    print("-" * 80)

    # Step 1: Generate ground truth signals
    print("Generating ground truth signals...")
    gen = AerosolGenerator()
    true_signals = {
        "Mg": gen.generate_pulse(onset_time=1.0, amplitude=10000, width=0.05, tailing=1.0),
        "Zn": gen.generate_pulse(onset_time=1.0, amplitude=8000, width=0.05, tailing=3.0),
    }
    t_truth = np.asarray(gen.time_axis, dtype=float)

    # Step 2: Simulate QMS sequential scanning
    print("Simulating QMS sequential scanning...")
    qms = QuadrupoleMS(elements, dwell_time=dwell_time)
    raw_df = qms.scan_event(t_truth, true_signals)

    # Create unified time grid for reconstruction
    t_min = float(raw_df["Time"].min())
    t_max = float(raw_df["Time"].max())
    unified_time = np.arange(t_min, t_max, dt_recon, dtype=float)

    # Interpolate ground truth onto unified grid for comparison
    truth_interp = {}
    for el in elements:
        truth_interp[el] = np.interp(unified_time, t_truth, true_signals[el])
    
    # Calculate gated ratio (only where Zn > 5% of peak)
    zn_thr = 0.05 * np.max(truth_interp["Zn"])
    mask = truth_interp["Zn"] > zn_thr
    ratio_true_gated = np.where(
        mask, 
        truth_interp["Mg"] / (truth_interp["Zn"] + 1e-12), 
        np.nan
    )

    # Step 3: Run all reconstruction methods
    print("Running reconstruction algorithms...")
    methods = {
        "Kernel(Exp)": CausalKernelReconstructor(kernel="exp", tau=tau),
        "Spline(Cubic)": SplineReconstructor(),
        "Tikhonov": TikhonovReconstructor(lam=lam)
    }

    results = {}
    
    # Print header
    print(f"\n{'Method':<15} | {'Element':<3} | {'RMSE':<8} | {'PeakShift':<9} | "
          f"{'AUC Err':<8} | {'Ratio RMSE(Gated)':<15}")
    print("-" * 80)

    for name, reconstructor in methods.items():
        reconstructor.fit(raw_df)
        res = reconstructor.reconstruct(unified_time)
        
        # Apply phase compensation for causal kernel method
        # (Spline is non-causal, Tikhonov minimizes lag globally)
        if "Kernel" in name:
            shift = dwell_time
            shifted_sigs = {}
            for el, y in res.signals.items():
                shifted_sigs[el] = np.interp(
                    unified_time, unified_time + shift, y, left=0, right=0
                )
            res.signals = shifted_sigs

        results[name] = res

        # Calculate and print metrics for each element
        for el in elements:
            m = compute_signal_metrics(unified_time, truth_interp[el], res.signals[el])
            
            # Calculate ratio metric (gated)
            ratio_pred = res.signals["Mg"] / (res.signals["Zn"] + 1e-12)
            ratio_pred_gated = np.where(mask, ratio_pred, np.nan)
            ratio_rmse = np.sqrt(np.nanmean((ratio_true_gated - ratio_pred_gated)**2))
            
            if el == "Mg":  # Print ratio metric only once per method
                print(f"{name:<15} | {el:<3} | {m.rmse:8.1f} | {m.peak_time_error:8.4f}s | "
                      f"{m.auc_error:8.1f} | {ratio_rmse:15.4f}")
            else:
                print(f"{'':<15} | {el:<3} | {m.rmse:8.1f} | {m.peak_time_error:8.4f}s | "
                      f"{m.auc_error:8.1f} | {'':<15}")
        print("-" * 80)

    # Step 4: Visualize results
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        ax.set_title(name)
        
        # Plot ground truth ratio
        ax.plot(unified_time, ratio_true_gated, 'k:', alpha=0.4, label="Truth Ratio")
        
        # Plot reconstructed ratio
        ratio_recon = res.signals["Mg"] / (res.signals["Zn"] + 1e-12)
        ratio_viz = np.where(mask, ratio_recon, np.nan)
        ax.plot(unified_time, ratio_viz, color="tab:red", label="Recon Ratio")
        
        ax.set_ylim(0, 3.0)
        ax.axhline(1.25, color='gray', alpha=0.2, linestyle='--', label="Ideal Ratio")
        
        if i == 0:
            ax.set_ylabel("Mg / Zn Ratio")
            ax.legend()
        if i == 1:
            ax.set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()
    
    print("Experiment completed.")


if __name__ == "__main__":
    run_experiment()