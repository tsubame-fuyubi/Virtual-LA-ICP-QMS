"""
Physics-based signal generator for laser ablation aerosol pulses.

This module simulates the physical process of aerosol generation and transport
in LA-ICP-MS systems using Ex-Gaussian distributions.
"""

import numpy as np
from scipy.stats import exponnorm


class AerosolGenerator:
    """
    Generates ground truth aerosol pulse signals using Ex-Gaussian model.
    
    The Ex-Gaussian distribution combines a Gaussian peak with an exponential tail,
    modeling both the initial particle transport and subsequent gas-phase diffusion.
    """
    
    def __init__(self, time_total: float = 5.0, sampling_rate: int = 1000):
        """
        Initialize the signal generator.
        
        Args:
            time_total: Total simulation duration in seconds.
            sampling_rate: Sampling rate in Hz (default: 1000 Hz = 1 ms resolution).
        """
        self.dt = 1.0 / sampling_rate
        self.time_axis = np.linspace(0, time_total, int(time_total * sampling_rate))
        
    def generate_pulse(
        self, 
        onset_time: float, 
        amplitude: float, 
        width: float, 
        tailing: float
    ) -> np.ndarray:
        """
        Generate a single aerosol pulse signal.
        
        Args:
            onset_time: Pulse onset time in seconds.
            amplitude: Peak intensity in counts per second (cps).
            width: Gaussian width parameter (sigma) controlling peak sharpness.
            tailing: Exponential tailing factor (K). Larger values produce longer tails.
        
        Returns:
            Signal intensity array matching the generator's time axis.
        """
        # Ex-Gaussian: convolution of Gaussian (width) and exponential (tailing)
        signal = exponnorm.pdf(
            self.time_axis, 
            K=tailing, 
            loc=onset_time, 
            scale=width
        )
        
        # Normalize to unit peak, then scale to desired amplitude
        signal = signal / np.max(signal) * amplitude
        
        return signal

if __name__ == "__main__":
    """Example: Generate and visualize pulse signals."""
    import matplotlib.pyplot as plt
    
    gen = AerosolGenerator()
    mg_signal = gen.generate_pulse(onset_time=1.0, amplitude=10000, width=0.05, tailing=1.0)
    zn_signal = gen.generate_pulse(onset_time=1.0, amplitude=8000, width=0.05, tailing=3.0)
    
    plt.plot(gen.time_axis, mg_signal, label='Mg (Particulate dominant)')
    plt.plot(gen.time_axis, zn_signal, label='Zn (Two-phase transport)')
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (cps)")
    plt.legend()
    plt.title("Physics Simulation: Single Pulse Response")
    plt.show()