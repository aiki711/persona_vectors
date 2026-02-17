#!/bin/bash
# run_raw_pca_mistral_v2_improved.sh
# 改良版: 生成と LLM によるスコア計算を自動化し、引数でカスタマイズ可能に。

set -euo pipefail

# --- デフォルト設定 ---
WORKDIR="/home/admin/work/s2550009/persona_vectors"
cd "$WORKDIR"

# Venv と PYTHONPATH
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
source persona_steering/bin/activate

# 実験パラメータのデフォルト
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT="mistral_7b"
PROMPT_FILE="probe_inputs/personality_expression_30.json"
OUTPUT_BASE="exp_raw_pca"
LAYERS="15,16,17,18,19,20,21,22,23,24,25"
ALPHAS="-5.0,-2.5,0.0,2.5,5.0" # 段階を少し整理
SAMPLES=30 # データセット全件
AXES_BANK="exp_raw_pca/mistral_7b/vectors/mistral_7b_raw_pca.npz"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# 実行する特性 (デフォルトは Big Five 全て)
TRAITS=("openness" "extraversion" "agreeableness" "conscientiousness" "neuroticism")

# --- ログ設定 ---
mkdir -p log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log/raw_pca_mistral_v2_improved_${TIMESTAMP}.log"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "=== RAW PCA MISTRAL IMPROVED EXPERIMENT START ===" | tee -a "$LOG_FILE"
echo "TIME: $(date)" | tee -a "$LOG_FILE"
echo "MODEL: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "PROMPT: $PROMPT_FILE" | tee -a "$LOG_FILE"
echo "LAYERS: $LAYERS" | tee -a "$LOG_FILE"
echo "ALPHAS: $ALPHAS" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

for trait in "${TRAITS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo ">>> TARGET TRAIT: $trait <<<" | tee -a "$LOG_FILE"
    
    # ディレクトリ作成
    RESULTS_DIR="$OUTPUT_BASE/${MODEL_SHORT}/results_v2_improved"
    SCORES_DIR="$OUTPUT_BASE/${MODEL_SHORT}/scores_v2_improved"
    mkdir -p "$RESULTS_DIR" "$SCORES_DIR"
    
    OUT_JSONL="$RESULTS_DIR/${MODEL_SHORT}_raw_pca_${trait}_results.jsonl"
    OUT_CSV="$SCORES_DIR/scores_${trait}.csv"
    
    # 1. テキスト生成 (Steering)
    echo "[Step 1/2] Generating steered text..." | tee -a "$LOG_FILE"
    python scripts/01_run_probe.py \
        --model="$MODEL_NAME" \
        --trait="$trait" \
        --layers="$LAYERS" \
        --alpha_list="$ALPHAS" \
        --prompt_file="$PROMPT_FILE" \
        --out="$OUT_JSONL" \
        --axes_bank="$AXES_BANK" \
        --max_new_tokens=128 \
        --samples="$SAMPLES" \
        --seed=42 2>&1 | tee -a "$LOG_FILE"
    
    # 2. LLM-as-a-Judge によるスコア計算
    if [ -f "$OUT_JSONL" ]; then
        echo "[Step 2/2] Calculating personality scores with LLM-as-a-Judge..." | tee -a "$LOG_FILE"
        python scripts/14_calc_personality_score_llm.py \
            "$OUT_JSONL" \
            --model "$JUDGE_MODEL" \
            --output "$OUT_CSV" 2>&1 | tee -a "$LOG_FILE"
            
        # 相関の簡易表示
        echo "Correlation (alpha vs judge_score):" | tee -a "$LOG_FILE"
        python -c "import pandas as pd; df=pd.read_csv('$OUT_CSV'); print(f'  r = {df[\"alpha_total\"].corr(df[\"raw_score_${trait}\"]):.4f}')" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Generation failed, skipping scoring for $trait." | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=== ALL EXPERIMENTS FINISHED: $(date) ===" | tee -a "$LOG_FILE"
echo "Results are saved in $OUTPUT_BASE/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
