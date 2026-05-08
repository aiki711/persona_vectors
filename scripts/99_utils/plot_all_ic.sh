#!/bin/bash

echo "Running Pairwise Heatmap..."
python3 scripts/03_ic_adaptive/54_plot_ic_pairwise_analysis.py

echo "Running PPL Comparison Heatmap..."
python3 scripts/03_ic_adaptive/55_plot_ic_ppl_comparison.py

echo "Running Trade-off Scatter Plot..."
python3 scripts/03_ic_adaptive/56_plot_ic_delta_analysis.py

echo "All plots generated successfully!"
