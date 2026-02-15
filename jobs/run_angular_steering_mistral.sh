#!/bin/bash
# run_angular_steering_mistral.sh

# 仮想環境の有効化
source persona_steering/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/admin/work/s2550009/persona_vectors/src

# ログディレクトリの作成
mkdir -p log

# 実験の実行
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log/angular_steering_mistral_${TIMESTAMP}.log"

echo "Starting Angular Steering experiment with Mistral..." | tee -a $LOG_FILE
python scripts/26_run_angular_steering_expr.py 2>&1 | tee -a $LOG_FILE

echo "Experiment finished. Results are in exp_angular_steering/" | tee -a $LOG_FILE
