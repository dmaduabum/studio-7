# ==========================================================
# Studio 07 — Robust Regression with Heavy-Tailed Errors
# Author: [Your Name or Group Name]
#
# This Makefile automates the full workflow:
#   1. Run simulation experiments
#   2. Generate tidy results CSV
#   3. Create publication-quality MSE vs df figures
#
# Directory structure:
#   src/                → source code
#   results/raw/        → simulation outputs (CSV files)
#   results/figures/    → generated plots
# ==========================================================

# Python interpreter
PYTHON := python3

# Output files and directories
RAW_RESULTS := results/raw/simulation_results.csv
FIGURE_DIR := results/figures

# ==========================================================
# Default target: run simulation and visualization
# ==========================================================
.PHONY: all
all: simulate visualize
	@echo "All tasks complete. Results and figures are ready."

# ==========================================================
# Run simulation experiments
# ==========================================================
.PHONY: simulate
simulate: $(RAW_RESULTS)

$(RAW_RESULTS): src/simulation.py src/dgps.py src/methods.py src/metrics.py
	@echo "Running Monte Carlo simulation..."
	$(PYTHON) -m src.simulation
	@echo "Simulation results saved to $(RAW_RESULTS)"

# ==========================================================
# Generate visualizations
# ==========================================================
.PHONY: visualize
visualize: $(RAW_RESULTS)
	@echo "Generating MSE vs df figures..."
	$(PYTHON) -m src.visualize
	@echo "Figures saved in $(FIGURE_DIR)"

# ==========================================================
# Remove generated results and figures
# ==========================================================
.PHONY: clean
clean:
	@echo "Removing generated results and figures..."
	rm -rf results/raw/*.csv results/figures/*.png
	@echo "Cleanup complete."

# ==========================================================
# Display available commands
# ==========================================================
.PHONY: help
help:
	@echo "Makefile for Studio 07 — Robust Regression Project"
	@echo ""
	@echo "Available commands:"
	@echo "  make              - Run the full pipeline (simulate + visualize)"
	@echo "  make simulate     - Run simulations and save results CSV"
	@echo "  make visualize    - Generate MSE vs df plots from results CSV"
	@echo "  make clean        - Remove generated results and figures"
