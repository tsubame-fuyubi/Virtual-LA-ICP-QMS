# Virtual-LA-ICP-QMS

A simulation framework for signal reconstruction in Laser Ablation Inductively Coupled Plasma Mass Spectrometry (LA-ICP-MS).

## Overview

Quadrupole mass spectrometers scan elements sequentially, causing temporal misalignment in multi-element signals. This project provides a virtual simulation environment and three reconstruction algorithms to recover synchronized signals from sparse sampling data.

## Features

- **Physics Engine**: Ex-Gaussian model for aerosol pulse generation
- **Virtual Instrument**: QMS sequential scanning simulator
- **Reconstruction Algorithms**: Three methods for signal recovery
  - Causal Kernel: Real-time weighted averaging
  - Cubic Spline: High-precision interpolation
  - Tikhonov Regularization: Smooth global optimization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Virtual-LA-ICP-QMS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the comparison experiment:
```bash
python src/pipeline.py
```

This will:
1. Generate ground truth aerosol signals
2. Simulate QMS sequential scanning
3. Reconstruct signals using all three methods
4. Display performance metrics and visualizations

## Project Structure

```
src/
├── physics_engine.py      # Signal generator
├── virtual_instrument.py  # QMS simulator
├── metrics.py             # Evaluation metrics
├── pipeline.py            # Experiment script
└── reconstruction/        # Reconstruction algorithms
    ├── base.py
    ├── kernel.py
    ├── spline.py
    └── tikhonov.py
```

## Example

```python
from src.physics_engine import AerosolGenerator
from src.virtual_instrument import QuadrupoleMS
from src.reconstruction.tikhonov import TikhonovReconstructor
import numpy as np

# Generate signals
gen = AerosolGenerator()
mg_signal = gen.generate_pulse(onset_time=1.0, amplitude=10000, width=0.05, tailing=1.0)
zn_signal = gen.generate_pulse(onset_time=1.0, amplitude=8000, width=0.05, tailing=3.0)

# Simulate QMS sampling
qms = QuadrupoleMS(elements=["Mg", "Zn"], dwell_time=0.03)
raw_df = qms.scan_event(gen.time_axis, {"Mg": mg_signal, "Zn": zn_signal})

# Reconstruct signals
recon = TikhonovReconstructor(lam=5.0)
recon.fit(raw_df)
unified_time = np.arange(raw_df["Time"].min(), raw_df["Time"].max(), 0.01)
result = recon.reconstruct(unified_time)
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib

## License

MIT License
