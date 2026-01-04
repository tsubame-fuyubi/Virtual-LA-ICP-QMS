import sys
import os

# Add project root to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.physics_engine import AerosolGenerator
from src.virtual_instrument import QuadrupoleMS
from src.metrics import compute_signal_metrics
from src.reconstruction.kernel import CausalKernelReconstructor
from src.reconstruction.spline import SplineReconstructor
from src.reconstruction.tikhonov import TikhonovReconstructor

# Page configuration
st.set_page_config(
    page_title="LA-ICP-MS Signal Reconstruction Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: parameter controls
st.sidebar.title("Experiment Console")

st.sidebar.subheader("1. Sampling Settings")
dwell_time = st.sidebar.slider("Dwell Time (s)", 0.01, 0.10, 0.03, 0.01)
dt_recon = st.sidebar.number_input("Reconstruction dt (s)", 0.001, 0.05, 0.01, 0.001)

st.sidebar.subheader("2. Reconstruction Method")
method_name = st.sidebar.radio(
    "Algorithm",
    ["Causal Kernel (Baseline)", "Cubic Spline (Ideal)", "Tikhonov (Optimization)"]
)

# Dynamic algorithm parameter display
algo_params = {}
if "Kernel" in method_name:
    st.sidebar.info("Causal kernel: weighted average based on historical data.")
    algo_params["tau"] = st.sidebar.slider("Tau (Decay)", 0.01, 0.20, 0.06, 0.01)
    algo_params["shift"] = st.sidebar.checkbox("Phase Compensation (Shift)", value=True)
elif "Spline" in method_name:
    st.sidebar.success("Cubic spline: non-causal control group utilizing future information.")
elif "Tikhonov" in method_name:
    st.sidebar.warning("Tikhonov: global optimization with smooth reconstruction.")
    algo_params["lam"] = st.sidebar.number_input("Lambda (Regularization)", 0.1, 100.0, 5.0, 1.0)

# Main logic
def run_simulation():
    # Generate ground truth signals
    gen = AerosolGenerator()
    true_signals = {
        "Mg": gen.generate_pulse(1.0, 10000, 0.05, 1.0),
        "Zn": gen.generate_pulse(1.0, 8000, 0.05, 3.0),
    }
    t_truth = np.asarray(gen.time_axis, dtype=float)

    # Virtual instrument sampling
    qms = QuadrupoleMS(elements=["Mg", "Zn"], dwell_time=dwell_time)
    raw_df = qms.scan_event(t_truth, true_signals)

    return t_truth, true_signals, raw_df

def run_reconstruction(raw_df, t_min, t_max):
    unified_time = np.arange(t_min, t_max, dt_recon)
    
    # Instantiate algorithm
    if "Kernel" in method_name:
        recon = CausalKernelReconstructor(tau=algo_params["tau"])
    elif "Spline" in method_name:
        recon = SplineReconstructor()
    elif "Tikhonov" in method_name:
        recon = TikhonovReconstructor(lam=algo_params["lam"])
    
    # Fit and reconstruct
    recon.fit(raw_df)
    res = recon.reconstruct(unified_time)
    
    # Phase compensation for kernel method
    if "Kernel" in method_name and algo_params["shift"]:
        shift_val = dwell_time
        shifted_sigs = {}
        for el, y in res.signals.items():
            shifted_sigs[el] = np.interp(unified_time, unified_time + shift_val, y, left=0, right=0)
        # Create new ReconstructionResult since the original is frozen
        from src.reconstruction.base import ReconstructionResult
        res = ReconstructionResult(time=unified_time, signals=shifted_sigs)

    return res, unified_time

# Execution and visualization
st.title("Virtual LA-ICP-MS Signal Reconstruction Lab")
st.markdown("""
This system addresses temporal misalignment in multi-element signals through physics-based modeling and virtual instrument simulation.
Adjust sidebar parameters to observe the effects of different reconstruction algorithms in real-time.
""")

try:
    # Parameter validation
    if dt_recon <= 0 or dt_recon > 1.0:
        st.error("Reconstruction time step must be between 0.001 and 1.0 seconds")
        st.stop()
    
    if dwell_time <= 0 or dwell_time > 1.0:
        st.error("Dwell time must be between 0.01 and 1.0 seconds")
        st.stop()
    
    # Run simulation
    with st.spinner("Generating physical signals and sampling data..."):
        t_truth, true_signals, raw_df = run_simulation()
    
    if raw_df.empty or len(raw_df) < 4:
        st.error("Insufficient sampling data. Please adjust parameters and retry.")
        st.stop()
    
    # Run reconstruction
    with st.spinner("Executing signal reconstruction..."):
        res, unified_time = run_reconstruction(raw_df, raw_df["Time"].min(), raw_df["Time"].max())
    
    # Project ground truth onto unified grid for metrics calculation
    truth_interp = {}
    for el in ["Mg", "Zn"]:
        truth_interp[el] = np.interp(unified_time, t_truth, true_signals[el])

    # Calculate metrics
    metrics_list = []
    for el in ["Mg", "Zn"]:
        m = compute_signal_metrics(unified_time, truth_interp[el], res.signals[el])
        metrics_list.append({
            "Element": el,
            "RMSE": f"{m.rmse:.1f}",
            "Peak Shift (s)": f"{m.peak_time_error:.4f}",
            "AUC Err (%)": f"{m.auc_error:.1f}"
        })

    # Ratio metric with gating threshold
    zn_thr = 0.05 * np.max(truth_interp["Zn"])
    mask = truth_interp["Zn"] > zn_thr

    ratio_true = np.where(mask, truth_interp["Mg"] / (truth_interp["Zn"] + 1e-12), np.nan)
    ratio_pred = np.where(mask, res.signals["Mg"] / (res.signals["Zn"] + 1e-12), np.nan)
    ratio_rmse = np.sqrt(np.nanmean((ratio_true - ratio_pred)**2))

    # Layout and visualization
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 1. Signal Reconstruction")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        
        # Plot ground truth
        ax1.plot(t_truth, true_signals["Mg"], 'b--', alpha=0.3, label="Mg True")
        ax1.plot(t_truth, true_signals["Zn"], 'r--', alpha=0.3, label="Zn True")
        
        # Plot raw samples
        raw_mg = raw_df[raw_df["Element"] == "Mg"]
        raw_zn = raw_df[raw_df["Element"] == "Zn"]
        ax1.scatter(raw_mg["Time"], raw_mg["Intensity"], color='b', s=10, alpha=0.4, label="Mg Raw")
        ax1.scatter(raw_zn["Time"], raw_zn["Intensity"], color='r', marker='s', s=10, alpha=0.4, label="Zn Raw")
        
        # Plot reconstruction
        ax1.plot(unified_time, res.signals["Mg"], color='b', linewidth=2, label="Mg Recon")
        ax1.plot(unified_time, res.signals["Zn"], color='r', linewidth=2, label="Zn Recon")
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Intensity (cps)")
        ax1.legend(loc='upper right')
        st.pyplot(fig1)
        
        st.markdown("### 2. Isotope Ratio (Mg / Zn)")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        
        ax2.plot(unified_time, ratio_true, 'k:', linewidth=2, alpha=0.5, label="True Ratio")
        ax2.plot(unified_time, ratio_pred, color='purple', linewidth=2, label="Recon Ratio")
        
        ax2.set_ylim(0, 3.0)
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        st.pyplot(fig2)

    with col2:
        st.markdown("### Metrics")
        
        st.table(pd.DataFrame(metrics_list).set_index("Element"))
        
        st.metric("Mg/Zn Ratio RMSE", f"{ratio_rmse:.4f}", delta_color="inverse")
        
        st.markdown("---")
        st.markdown("**Observation:**")
        if ratio_rmse < 0.1:
            st.success("Ratio is stable!")
        elif ratio_rmse < 0.5:
            st.warning("Ratio is acceptable.")
        else:
            st.error("Ratio deviates significantly.")
        
        st.markdown("---")
        st.markdown("### Statistics")
        st.info(f"""
        - Sample points: {len(raw_df)}
        - Reconstruction points: {len(unified_time)}
        - Time range: {raw_df['Time'].min():.2f} - {raw_df['Time'].max():.2f} s
        """)

except Exception as e:
    st.error(f"Runtime error: {str(e)}")
    st.exception(e)
    st.info("Please check parameter settings or refresh the page to retry.")