# Studio 07 – Robust Regression with Heavy-Tailed Errors

This project investigates how **heavy-tailed error distributions** affect the performance of different linear regression estimators in high-dimensional settings.


---

## Objectives

- Simulate data from the linear model **y = Xβ + ε**, where errors follow a Student-t distribution
- Compare three estimators:
  - **Ordinary Least Squares (OLS)** — sensitive to heavy tails
  - **Least Absolute Deviations (LAD)** — robust to outliers
  - **Huber Regression** — compromise between OLS and LAD
- Evaluate estimator performance using the **Mean Squared Error (MSE)**
- Analyze how performance varies with:
  - Tail heaviness (degrees of freedom)
  - Signal-to-noise ratio (SNR)
  - Aspect ratio γ = p/n
  - Predictor correlation ρ

---

## Project Structure
```
studio-07/
├── src/
│   ├── dgps.py           # Data generation functions (simulate_dataset)
│   ├── methods.py        # Estimator implementations (fit_model)
│   ├── metrics.py        # Evaluation and results schema
│   ├── simulation.py     # Main simulation driver (runs Monte Carlo grid)
│   └── visualize.py      # Visualization of MSE vs df
├── results/
│   ├── raw/              # Saved CSV files of simulation results
│   └── figures/          # Publication-quality plots
├── venv/                 # Local virtual environment (ignored by git)
├── Makefile              # Reproducible workflow (simulate + visualize)
├── requirements.txt      # Package dependencies
├── .gitignore            # Ignored files (venv, caches, checkpoints)
└── README.md             # Project documentation
```

---

## Installation and Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\Activate.ps1     # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify setup
```bash
make help
```

---

## Usage

### Run the full simulation pipeline
```bash
make
```

This will:
1. Run the simulation study (`make simulate`)
2. Generate MSE vs df figures (`make visualize`)

### Individual commands
```bash
make simulate    # Run Monte Carlo experiments
make visualize   # Create plots from saved results
make clean       # Remove generated files
```

---

## Output

- **Results CSV**: `results/raw/simulation_results.csv`
- **Figures**: `results/figures/mse_vs_df_gamma*_snr*.png`

Each figure plots MSE vs degrees of freedom (df) for all estimators, with small multiples for different SNR and aspect ratios.

---

## Reproducibility Notes

- Randomness controlled by NumPy's generator (`np.random.default_rng(seed)`)
- `requirements.txt` locks exact package versions
- `.gitignore` excludes virtual environments, cache files, and generated results

---

## Authors

**Dili Maduabum**   
[University of Michigan / Department of Economics]

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
