"""
Virtual quadrupole mass spectrometer simulator.

This module simulates the sequential scanning behavior of QMS instruments,
which causes temporal misalignment in multi-element signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class QuadrupoleMS:
    """
    Simulates sequential scanning characteristics of quadrupole mass spectrometer.
    
    Unlike continuous detectors, QMS instruments scan elements sequentially,
    causing temporal offsets between different element signals.
    """
    
    def __init__(
        self, 
        elements: list[str], 
        dwell_time: float = 0.01, 
        settling_time: float = 0.005
    ):
        """
        Initialize the virtual instrument.
        
        Args:
            elements: List of element names to scan, e.g., ['Mg', 'Zn'].
            dwell_time: Measurement duration per element in seconds.
            settling_time: Dead time for circuit stabilization when switching elements.
        """
        self.elements = elements
        self.dwell_time = dwell_time
        self.settling_time = settling_time
        
    def scan_event(
        self, 
        time_axis: np.ndarray, 
        true_signals_dict: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Simulate a sequential scanning measurement event.
        
        Args:
            time_axis: High-resolution time axis from physics engine.
            true_signals_dict: Ground truth signals, e.g., {'Mg': array, 'Zn': array}.
        
        Returns:
            DataFrame with columns: Time, Element, Intensity.
        """
        sampled_data = []
        current_time = 0.0
        max_time = time_axis[-1]
        
        # Sequential scanning: cycle through elements
        while current_time < max_time:
            for el in self.elements:
                # Settling time (dead time when switching elements)
                current_time += self.settling_time
                if current_time >= max_time:
                    break
                
                # Find corresponding index in ground truth
                idx = np.searchsorted(time_axis, current_time)
                
                if idx < len(time_axis):
                    true_intensity = true_signals_dict[el][idx]
                    
                    # Add measurement noise
                    # Shot noise: Poisson process inherent to counting
                    noisy_intensity = np.random.poisson(max(0, true_intensity))
                    
                    # Readout noise: Gaussian white noise from electronics
                    readout_noise = np.random.normal(0, 10)
                    final_intensity = noisy_intensity + readout_noise
                    
                    sampled_data.append({
                        'Time': current_time,
                        'Element': el,
                        'Intensity': max(0, final_intensity)
                    })
                
                # Dwell time (measurement duration)
                current_time += self.dwell_time
                
        return pd.DataFrame(sampled_data)

if __name__ == "__main__":
    """Example: Simulate QMS sampling and visualize temporal misalignment."""
    import matplotlib.pyplot as plt
    from src.physics_engine import AerosolGenerator
    
    print("Generating ground truth signals...")
    gen = AerosolGenerator(time_total=3.0)
    mg_true = gen.generate_pulse(1.0, 10000, 0.05, 1.0)
    zn_true = gen.generate_pulse(1.0, 8000, 0.05, 3.0)
    true_signals = {'Mg': mg_true, 'Zn': zn_true}
    
    print("Starting virtual mass spectrometer sampling...")
    qms = QuadrupoleMS(elements=['Mg', 'Zn'], dwell_time=0.02, settling_time=0.005)
    df = qms.scan_event(gen.time_axis, true_signals)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    data_mg = df[df['Element'] == 'Mg']
    plt.scatter(data_mg['Time'], data_mg['Intensity'], label='Mg samples', 
                color='blue', s=30, alpha=0.7)
    
    data_zn = df[df['Element'] == 'Zn']
    plt.scatter(data_zn['Time'], data_zn['Intensity'], label='Zn samples', 
                color='orange', marker='s', s=30, alpha=0.7)
    
    plt.plot(gen.time_axis, mg_true, '--', color='blue', alpha=0.3, label='Mg True')
    plt.plot(gen.time_axis, zn_true, '--', color='orange', alpha=0.3, label='Zn True')
    
    plt.title("Virtual QMS Sampling (Temporal Misalignment)")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (cps)")
    plt.legend()
    plt.show()