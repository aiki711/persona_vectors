#!/bin/bash
#PBS -N vectors_exp
#PBS -q GPU-1
#PBS -o log/alpha_sweep.o%j
#PBS -e log/alpha_sweep.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=20:00:00
#PBS -j oe

set -euo pipefail

WORKDIR="${PBS_O_WORKDIR:-$PWD}"
RUN_ID="${PBS_JOBID:-bash_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
mkdir -p log
LOG_FILE="log/alpha_sweep.${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== STARTING EFFICIENT FULL EXPERIMENT PIPELINE ==="
echo "START TIME: $(date)"
echo "Host: $(hostname)"
echo "CWD : $PWD"
echo "PBS_JOBID: ${PBS_JOBID:-}"
echo "RUN_ID: $RUN_ID"
echo "WORKDIR: $WORKDIR"
echo "LOG_FILE: $LOG_FILE"

# ==================== Project ====================
export PROJECT_DIR="$WORKDIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scripts:${PYTHONPATH:-}"

# キャッシュをプロジェクト内に固定（安定＆権限事故を減らす）
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export OMP_NUM_THREADS=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "[INFO] nvidia-smi:"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "(no nvidia-smi)"

# ==================== Tokens ====================
# トークンはログに出さない
set +x
if [ -f "$PROJECT_DIR/.hf_token" ]; then
  export HUGGINGFACE_HUB_TOKEN="$(head -n1 "$PROJECT_DIR/.hf_token" | tr -d '\r\n' | sed 's/^Bearer[[:space:]]\+//')"
fi

# ==================== Config backup & restore (safe) ====================
CONFIG_FILE="exp/configs/big5_vectors.yaml"
CONFIG_BAK="exp/configs/big5_vectors.yaml.bak"

restore_config() {
  set +e
  if [ -f "$CONFIG_BAK" ]; then
    echo "[CLEANUP] Restoring $CONFIG_FILE from $CONFIG_BAK..."
    cp "$CONFIG_BAK" "$CONFIG_FILE"
    rm -f "$CONFIG_BAK"
  fi
}
trap restore_config EXIT

if [ ! -f "$CONFIG_BAK" ]; then
  echo "[Step 0] Backing up $CONFIG_FILE to $CONFIG_BAK..."
  cp "$CONFIG_FILE" "$CONFIG_BAK"
else
  echo "[Step 0] Backup exists; restoring from it."
  cp "$CONFIG_BAK" "$CONFIG_FILE"
fi

# ==================== uv venv ====================
VENV="$PROJECT_DIR/persona_steering"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[ERROR] venv not found: $VENV"
  echo "Run on login node: uv venv persona_steering --python 3.10 && uv sync --active"
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.uv_cache}"

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

if [ "${RUN_UV_SYNC:-0}" = "1" ]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.uv_cache}"
  if [ -f "$PROJECT_DIR/uv.lock" ]; then
    uv sync --frozen --active
  else
    uv sync --active
  fi
fi

export PYTHON_BIN="$VENV/bin/python"

# --- Make pip-installed NVIDIA libs visible ---
export LD_LIBRARY_PATH="$($PYTHON_BIN - <<'PY'
import site, glob, os
paths=[]
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
seen=set(); out=[]
for p in paths:
    if p not in seen:
        out.append(p); seen.add(p)
print(":".join(out))
PY
):${LD_LIBRARY_PATH:-}"
echo "[INFO] injected nvidia libs into LD_LIBRARY_PATH"

"$PYTHON_BIN" -V
"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("compiled cuda", torch.version.cuda)
PY

# ==================== Experiment params ====================
TRAITS=("openness" "conscientiousness" "extraversion" "agreeableness" "neuroticism")

# 1行 = "TAG|BASE_ID|INSTR_ID|ALPHAS_BASE|ALPHAS_INSTR"
# ※ base/instructで同じなら同じ文字列を入れる
MODEL_SPECS=(
  "mistral_7b|mistralai/Mistral-7B-v0.3|mistralai/Mistral-7B-Instruct-v0.3|-5.5,-5,-4.5,-4,-3.5,-3,-2,-1,0,1,2,3,3.5,4,4.5,5,5.5|-5.5,-5,-4.5,-4,-3.5,-3,-2,-1,0,1,2,3,3.5,4,4.5,5,5.5"
  "llama3_8b|meta-llama/Meta-Llama-3-8B|meta-llama/Meta-Llama-3-8B-Instruct|-20,-16,-12,-8,-4,0,4,8,12,16,20|-20,-16,-12,-8,-4,0,4,8,12,16,20"
  "olmo3_7b|allenai/Olmo-3-1025-7B|allenai/Olmo-3-7B-Instruct|-50,-40,-30,-20,-10,0,10,20,30,40,50|-50,-40,-30,-20,-10,0,10,20,30,40,50"
  "qwen25_7b|Qwen/Qwen2.5-7B|Qwen/Qwen2.5-7B-Instruct|-160,-140,-120,-100,-80,0,80,100,120,140,160|-160,-140,-120,-100,-80,0,80,100,120,140,160"
  "gemma2_9b|google/gemma-2-9b|google/gemma-2-9b-it|-500,-400,-300,-200,-100,0,100,200,300,400,500|-500,-400,-300,-200,-100,0,100,200,300,400,500"
  "falcon3_7b|tiiuae/Falcon3-7B-Base|tiiuae/Falcon3-7B-Instruct|-500,-400,-300,-200,-100,0,100,200,300,400,500|-500,-400,-300,-200,-100,0,100,200,300,400,500"
)

# ---------- helpers ----------
is_nonempty_file() { local f="$1"; [[ -f "$f" && -s "$f" ]]; }

prepare_axes_if_needed() {
  local model_id="$1"
  local ax_bank="$2"
  local config_file="$3"

  local norm_bank="${ax_bank%.npz}_rawnorms.npz"

  sed -i "s|^model_name: .*|model_name: ${model_id}|g" "$config_file"

  # axes と rawnorms の両方があるときだけSKIP
  if is_nonempty_file "$ax_bank" && is_nonempty_file "$norm_bank"; then
    echo "[SKIP] axes+rawnorms exists: $ax_bank , $norm_bank"
  else
    echo "[RUN ] 00_prepare_vectors.py -> $ax_bank (+ $norm_bank)"
    "$PYTHON_BIN" scripts/00_prepare_vectors.py --config "$config_file" --bank_path "$ax_bank"
    if ! is_nonempty_file "$norm_bank"; then
      echo "[WARN] rawnorms not created (expected): $norm_bank"
    fi
  fi
}

run_probe_if_needed() {
  local split="$1"      # base | instruct
  local trait="$2"
  local model_id="$3"
  local axes_bank="$4"
  local out_jsonl="$5"
  local alpha_list="$6"

  if is_nonempty_file "$out_jsonl"; then
    echo "[SKIP] probe exists: $out_jsonl"
    return 0
  fi

  echo "[RUN ] 01_run_probe_with_rms.py -> $out_jsonl (split=$split trait=$trait)"
  "$PYTHON_BIN" scripts/01_run_probe_with_rms.py \
    --model       "$model_id" \
    --axes_bank   "$axes_bank" \
    --trait       "$trait" \
    --alpha_list="$alpha_list" \
    --out         "$out_jsonl"
}

concat_alltraits() {
  local results_dir="$1"
  local split="$2"       # base | instruct
  local out_all="$3"

  rm -f "$out_all"
  for trait in "${TRAITS[@]}"; do
    local f="${results_dir}/${tag}_${split}_${trait}_with_rms.jsonl"
    if is_nonempty_file "$f"; then
      cat "$f" >> "$out_all"
    else
      echo "[WARN] missing/empty: $f (not included in alltraits)"
    fi
  done
  echo "[INFO] wrote: $out_all ($(wc -l < "$out_all" 2>/dev/null || echo 0) lines)"
}

run_text_analysis() {
  local tag="$1"
  local results_dir="$2"
  local split="$3"      # base | instruct
  
  local input_jsonl="${results_dir}/${tag}_${split}_alltraits.jsonl"
  
  if ! is_nonempty_file "$input_jsonl"; then
    echo "[WARN] Text analysis skipped: $input_jsonl not found or empty"
    return 0
  fi

  # --- 13: Edit Distance (CPU) ---
  # 出力: ${tag}_${split}_edit_distance.csv
  local dist_csv="${results_dir}/${tag}_${split}_edit_distance.csv"
  if [ -s "$dist_csv" ]; then
    echo "[SKIP] 13_text_change_vs_alpha.py exists: $dist_csv"
  else
    echo "[RUN ] 13_text_change_vs_alpha.py -> $dist_csv"
    "$PYTHON_BIN" scripts/13_text_change_vs_alpha.py "$input_jsonl" \
      --output "$dist_csv" \
      --baseline 0
  fi

  # --- 14: Personality Score (GPU) ---
  # 出力: ${tag}_${split}_personality_scores.csv
  local score_csv="${results_dir}/${tag}_${split}_personality_scores.csv"
  if [ -s "$score_csv" ]; then
    echo "[SKIP] 14_calc_personality_score.py exists: $score_csv"
  else
    echo "[RUN ] 14_calc_personality_score.py -> $score_csv"
    "$PYTHON_BIN" scripts/14_calc_personality_score.py "$input_jsonl" \
      --output "$score_csv" \
      --batch_size 32 \
      --model "Minej/bert-base-personality"
  fi
}

run_alpha_select_and_viz() {
  local tag="$1"
  local results_dir="$2"
  local sel_rng_root="$3"

  mkdir -p "$sel_rng_root"

  for SPLIT in base instruct; do
    for trait in "${TRAITS[@]}"; do
      local in_jsonl="${results_dir}/${tag}_${SPLIT}_${trait}_with_rms.jsonl"
      [ -s "$in_jsonl" ] || { echo "[WARN] missing $in_jsonl (skip)"; continue; }

      "$PYTHON_BIN" scripts/06_alpha_eval_v13.py \
        --in "$in_jsonl" \
        --out_root "$sel_rng_root" \
        --per_prompt \
        --pass_rate_min 0.8 \
        --sem_min 0.0 \
        --len_ratio_min 0.5 \
        --len_ratio_max 2.0 \
        --distinct2_min 0.3 \
        --punct_ratio_max 0.85 \
        --min_phrase_tokens 3 \
        --max_run_token_max 8 \
        --max_run_phrase_max 2
    done
  done

  mkdir -p "$sel_rng_root/_summary"
  "$PYTHON_BIN" scripts/07_alpha_visualize.py \
    --globs "$sel_rng_root/range/*_per_prompt.jsonl" \
    --out_csv "$sel_rng_root/_summary/per_prompt_summary.csv" \
    --out_dir "$sel_rng_root/_summary/per_prompt_figs"

  mkdir -p "$sel_rng_root/_corr"
  "$PYTHON_BIN" scripts/08_corr_range_vs_rms_v8.py \
    --range_csv "$sel_rng_root/_summary/per_prompt_summary.csv" \
    --probe_jsonl_glob "${results_dir}/${tag}_*_*_with_rms.jsonl" \
    --rawnorm_npz_glob "exp/${tag}/axes_*_rawnorms.npz" \
    --out_dir "$sel_rng_root/_corr" \
    --make_plots \
    --corr_group split
}

run_slopes_and_viz() {
  local tag="$1"
  local results_dir="$2"

  local base_all="${results_dir}/${tag}_base_alltraits.jsonl"
  local instr_all="${results_dir}/${tag}_instruct_alltraits.jsonl"

  if ! is_nonempty_file "$base_all" || ! is_nonempty_file "$instr_all"; then
    echo "[WARN] missing alltraits jsonl for slopes: base=$base_all instr=$instr_all (skip 02/03)"
    return 0
  fi

  local slopes_dir="${results_dir}/slopes"
  local figs_dir="${slopes_dir}/figs"
  mkdir -p "$slopes_dir" "$figs_dir"

  local out_csv="${slopes_dir}/slopes_${tag}_asst_pairwise.csv"

  # --- 02: compute slopes CSV ---
  if is_nonempty_file "$out_csv"; then
    echo "[SKIP] slopes csv exists: $out_csv"
  else
    echo "[RUN ] 02_probe_slopes_from_logs.py -> $out_csv"
    "$PYTHON_BIN" scripts/02_probe_slopes_from_logs.py \
      --base_json  "$base_all" \
      --instr_json "$instr_all" \
      --out_csv    "$out_csv" \
      --pooling asst \
      --axis_mode pairwise
  fi

  # --- 03: visualize slopes ---
  echo "[RUN ] 03_slopes_visualize.py (alpha01 slope) -> $figs_dir"
  "$PYTHON_BIN" scripts/03_slopes_visualize.py \
    --input_dir "$out_csv" \
    --out_dir   "$figs_dir" \
    --value_col slope_delta_score_vs_alpha01

  echo "[RUN ] 03_slopes_visualize.py (raw slope) -> $figs_dir"
  "$PYTHON_BIN" scripts/03_slopes_visualize.py \
    --input_dir "$out_csv" \
    --out_dir   "$figs_dir" \
    --value_col slope_delta_score_vs_alpha
}

run_one_model_pair() {
  local tag="$1"
  local base_id="$2"
  local instr_id="$3"
  local alphas_base="$4"
  local alphas_instr="$5"

  echo "==== ${tag} / pooling=asst / mode=pairwise ===="

  local results_dir="exp/${tag}/asst_pairwise_results"
  mkdir -p "$results_dir"

  # --- Base axes ---
  echo "[${tag}] Base axes"
  local ax_base="exp/${tag}/axes_base_asst_pairwise.npz"
  prepare_axes_if_needed "$base_id" "$ax_base" "$CONFIG_FILE"

  for trait in "${TRAITS[@]}"; do
    run_probe_if_needed "base" "$trait" "$base_id" "$ax_base" \
      "${results_dir}/${tag}_base_${trait}_with_rms.jsonl" "$alphas_base"
  done
  concat_alltraits "$results_dir" "base" "${results_dir}/${tag}_base_alltraits.jsonl"

  run_text_analysis "$tag" "$results_dir" "base"

  # --- Instruct axes ---
  echo "[${tag}] Instruct axes"
  local ax_instr="exp/${tag}/axes_instruct_asst_pairwise.npz"
  prepare_axes_if_needed "$instr_id" "$ax_instr" "$CONFIG_FILE"

  for trait in "${TRAITS[@]}"; do
    run_probe_if_needed "instruct" "$trait" "$instr_id" "$ax_instr" \
      "${results_dir}/${tag}_instruct_${trait}_with_rms.jsonl" "$alphas_instr"
  done
  concat_alltraits "$results_dir" "instruct" "${results_dir}/${tag}_instruct_alltraits.jsonl"

  run_text_analysis "$tag" "$results_dir" "instruct"

  # --- 02/03 ---
  run_slopes_and_viz "$tag" "$results_dir"

  # --- 06/07 ---
  local sel_rng_root="${results_dir}/selected_range"
  run_alpha_select_and_viz "$tag" "$results_dir" "$sel_rng_root"

  echo "==== ${tag} END TIME: $(date)===="
  echo "-----------------------------------------"
}

# ==================== Run all ====================
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r TAG BASE_ID INSTR_ID ALPHAS_BASE ALPHAS_INSTR <<< "$spec"
  run_one_model_pair "$TAG" "$BASE_ID" "$INSTR_ID" "$ALPHAS_BASE" "$ALPHAS_INSTR"
done

echo "==== GLOBAL CORR (all models) ===="

GLOBAL_DIR="exp/_all/asst_pairwise_results/selected_range"
mkdir -p "$GLOBAL_DIR/_summary" "$GLOBAL_DIR/_corr"

# 1) range_summary を全モデルから結合
"$PYTHON_BIN" - <<'PY'
import glob, os
import pandas as pd

paths = sorted(glob.glob("exp/*/asst_pairwise_results/selected_range/_summary/range_summary.csv"))
assert paths, "range_summary.csv not found"

dfs=[]
for p in paths:
    df=pd.read_csv(p)
    if "kind" in df.columns:
        df=df[df["kind"].astype(str).str.lower()=="range"].copy()
    dfs.append(df)

out=pd.concat(dfs, ignore_index=True)
os.makedirs("exp/_all/asst_pairwise_results/selected_range/_summary", exist_ok=True)
out_path="exp/_all/asst_pairwise_results/selected_range/_summary/range_summary.csv"
out.to_csv(out_path, index=False)
print("saved:", out_path, "rows:", len(out))
print("unique tags:", out["tag"].nunique(), "splits:", out["split"].nunique(), "traits:", out["trait"].nunique())
PY

# 2) 全モデル横断で 08 を実行（ここがメイン）
"$PYTHON_BIN" scripts/08_corr_range_vs_rms_v9.py \
  --per_prompt_jsonl_glob "exp/mistral_7b/asst_pairwise_results/selected_range/range/*_per_prompt.jsonl" \
  --probe_jsonl_glob "exp/mistral_7b/asst_pairwise_results/*with_rms.jsonl" \
  --out_dir "exp/mistral_7b/asst_pairwise_results/selected_range/_corr_v9" \
  --make_plots \
  --min_n 8

"$PYTHON_BIN" scripts/09_corr_and_scaling_from_merged.py \
  --in_csv "exp/_all/asst_pairwise_results/selected_range/_corr_v9/merged_metrics.csv" \
  --out_dir "exp/_all/asst_pairwise_results/selected_range/_corr_v9"

"$PYTHON_BIN" scripts/09_visualize_summaries.py \
  --corr "exp/_all/asst_pairwise_results/selected_range/_corr_v9/corr_pred_summary.csv" \
  --disp "exp/_all/asst_pairwise_results/selected_range/_corr_v9/scaling_dispersion.csv" \
  --outdir "exp/_all/asst_pairwise_results/selected_range/_corr_v9/plot" \
  --methods spearman

"$PYTHON_BIN" scripts/10_scatter_from_merged.py \
  --merged "exp/_all/asst_pairwise_results/selected_range/_corr_v9/merged_metrics.csv" \
  --outdir "exp/_all/asst_pairwise_results/selected_range/_corr_v9/plot"

"$PYTHON_BIN" scripts/10_within_model_and_lomo.py \
  --merged "exp/_all/asst_pairwise_results/selected_range/_corr_v9/merged_metrics.csv" \
  --outdir "exp/_all/within_model_and_lomo" \
  --xcols rms0_mean \
  --min_train_points 6 \
  --min_test_points 3

"$PYTHON_BIN" scripts/11_build_prompt_alpha_table.py \
  --jsonl_glob 'exp/*/asst_pairwise_results/*_with_rms.jsonl' \
  --out_csv exp/_all/prompt_alpha_table.csv

"$PYTHON_BIN" scripts/12_join_boundaries_and_label.py \
  --table exp/_all/prompt_alpha_table.csv \
  --merged "exp/_all/asst_pairwise_results/selected_range/_corr_v9/merged_metrics.csv" \
  --out_csv exp/_all/prompt_alpha_labeled.csv

echo "=== PIPELINE COMPLETED ==="