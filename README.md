# Lightweight Offline Experiments (B×C only)

## Run
python scripts/00_prepare_vectors.py
python scripts/01_grid_sweep.py
python scripts/02_offline_safe_ucb.py

## Outputs
- exp/logs/baseline.jsonl
- exp/logs/grid_sweep.jsonl
- exp/logs/safe_ucb_offline.jsonl
- exp/figures/pareto.png
- exp/figures/learning_curve.png

## Notes
- rank -> top_k mapping
- alpha kept small (0.14–0.18) to match residual hook conventions
- Laplacian reg: a = (I + λL)^(-1) e
