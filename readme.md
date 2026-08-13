# PINN-PBE Model for Describing Gibbsite Bulk Crystallization Dynamics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Physics-Informed Neural Network for solving Population Balance Equations (PBE) describing gibbsite bulk crystallization dynamics.

## Overview

The model addresses vulnerabilities in purely data-driven approaches by incorporating first-principle physics into neural network training. The system of population balance equations was transformed into a linearized form, generating a PINN-PBE model as a set of interconnected neural networks approximating the solution under batch conditions.
This study presents a qualitatively new approach to linearizing the population balance equations and demonstrates the feasibility of their solution using PINNs. Model parameters are task‑dependent and require case‑specific tuning. The developed system of equations is amenable to solution via PINN under a wide range of coefficients and boundary conditions, though this implementation focuses on conditions representative of alumina refineries processing boehmite-diaspore bauxites.

### Industrial Context
The chosen case reflects typical operating conditions where the Sauter mean diameter of precipitated hydrate particles falls within **30-60 µm** and the growth rate ranges from **0.8 to 1.3 µm/day**. A validation experiment under these conditions was conducted earlier, with results presented in
[Golubev, V.O.; Vasiliev, V.V.; Brichkin, V.N.; Chistyakov, D.G. Method of Modeling the Precipitation Process of Aluminate Liquors
in Laboratory Experiment under Batch Conditions. In Proceedings of the 24th Conference “Aluminum of Siberia”, Krasnoyarsk,
Russia, 10–14 September 2018; pp. 202–205.].

## Architecture

The model predicts four variables:
- `θ(L,t)` – logarithm of particle size distribution
- `m₂(L,t)` – second moment of distribution
- `m₃(L,t)` – third moment of distribution  
- `dA(t)/dt` – rate of change of alumina concentration

The loss function incorporates:
- Population balance equation (linearized form)
- Mass balance closure (solid phase volume fraction)
- Supersaturation dynamics
- Kinetic coefficients: growth, secondary nucleation

### Neural Network Configuration
- **Hidden layers**: 4 layers
- **Activation function**: Hyperbolic tangent (tanh)
- **Regularization**: L2 regularization to prevent overfitting
- **Loss weighting**: Total loss computed as sum of all components with unit weights; parameters normalized to their respective approximation intervals (Table 1) to balance loss term contributions, with additional compensation for varying sampled dataset sizes

![PINN-PBE архитектура](images/architecture.png)


## Model Parameters

### Initial Distribution
θ(L,0) = Th0·exp(–L/393.3), where L is particle size [m]

### Kinetic and Process Coefficients
| Parameter | Value | Description |
|-----------|-------|-------------|
| k₀ | 4.7×10¹² | Pre-exponential factor, m/s |
| E_act | 73 | Activation energy, kJ/mol |
| z₁ | 2 | Oversaturation Exponent |
| ∂θ/∂t\|_agg | 0 | Agglomeration contribution |

```python
# Domain
Th0         = 28.2                # Initial θ = ln(u)
Lmin, Lmax  = 0.001, 350.0        # Crystal size [μm]
tmax        = 48.0                # Batch time [hours]

# Physical properties
rho         = 2420.0              # Hydrate density [kg/m³]
Thnuc_f     = 12.0                # Nucleation rate

# Liquor composition (Head Tank)
Al2O3_in, Na2Oo, Na2Ok, TC = 103.0, 161.0, 143.0, 62.0

# Molar weights
MW_Al2O3, MW_AlOH3 = 0.102, 0.078    # [kg/mol]
```

## Clone repository
git clone https://github.com/vladimirgolubev2-ship-it/pinn-pbe.git
cd PINN-PBE

# Install dependencies
pip install -r requirements.txt

# Usage
bash
jupyter notebook pinn-pbe_4nets.ipynb

# Requirements

## GPU Version (default)
- NVIDIA GPU with CUDA support
- CUDA 11.2 + cuDNN 8.1
- TensorFlow 2.10.0 (GPU version)

## CPU Version (alternative)
For systems without NVIDIA GPU, modify the notebook:
```python
# Replace GPU imports/settings with:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Or install CPU-only TensorFlow:
# pip install tensorflow-cpu==2.10.0
```

# Performance Comparison
The DPB solver requires a fine time step (0.5 s) to maintain numerical stability, resulting in 345,600 temporal steps and a computation time of 1.86 s for a 48‑hour simulation. In contrast, the LSTM and PINN models are data‑driven and physics‑informed surrogates, respectively, which do not require temporal discretization for stability; they provide direct predictions at any specified time horizon and particle size.
At the inference stage, for a forecast of 90 days with 10 interested particle size classes, the LSTM‑based model achieves a 15.6 times speed‑up over DPB (0.119 s). The proposed PINN model further reduces the inference time to 0.017 s on the same grid — a 110‑fold speed‑up compared to DPB and 7 times faster than LSTM (see Table 1).

### Table 1. Computational performance comparison

| Method | Grid Size (L × t) | Computation Time, s | Speed‑up |
|--------|-------------------|---------------------|----------|
| DPB    | 53 × 345,600      | 1.86                | 1×       |
| LSTM   | 10 × 90           | 0.119               | 15.6×    |
| **PINN** | **10 × 90**    | **0.017**           | **110×** |

This significant acceleration at the inference stage, combined with the physical consistency guaranteed by the PINN framework, makes the model particularly attractive for real‑time process control and optimization tasks in industrial alumina production.

*Note: The present work does not pursue exhaustive optimization of the PINN architecture; the configuration is limited to a basic selection of the number of neurons and hidden layers.*

See full text article [here](https://www.researchgate.net/publication/412197001)

# Citation
If you use this code in your research, please cite:
@article{If you use this code in your research, please cite: Litvinova, T.E.; Golubev, V.O.; Tuleshov, N.V. PINN-PBE Model for Describing Gibbsite Crystallization Dynamics. Metals 2026, 16(8): 903. https://doi.org/10.3390/met16080903}}
