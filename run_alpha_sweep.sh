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
ALPHAS="-5.5,-5,-4.5,-4,-3.5,-3,-2,-1,0,1,2,3,3.5,4,4.5,5,5.5"
TRAITS=("openness" "conscientiousness" "extraversion" "agreeableness" "neuroticism")

MODEL_PAIRS=(
  "mistral_7b mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3"
)

# ---------- helpers ----------
is_nonempty_file() {
  local f="$1"
  [[ -f "$f" && -s "$f" ]]
}

run_probe_if_needed() {
  local split="$1"      # base | instruct
  local trait="$2"
  local model_id="$3"   # HF model id
  local axes_bank="$4"
  local out_jsonl="$5"

  if is_nonempty_file "$out_jsonl"; then
    echo "[SKIP] probe exists: $out_jsonl"
    return 0
  fi

  echo "[RUN ] 01_run_probe_with_rms.py -> $out_jsonl"
  "$PYTHON_BIN" scripts/01_run_probe_with_rms.py \
    --model      "$model_id" \
    --axes_bank  "$axes_bank" \
    --trait      "$trait" \
    --alpha_list="$ALPHAS" \
    --out        "$out_jsonl"
}

# ==================== Run: select alpha ====================
for line in "${MODEL_PAIRS[@]}"; do
  set -- $line
  TAG=$1
  BASE_ID=$2
  INSTR_ID=$3

  echo "==== ${TAG} / pooling=asst / mode=pairwise ===="
  mkdir -p exp/configs "exp/${TAG}"
  RESULTS_DIR_CLUSTER="exp/${TAG}/asst_pairwise_results"
  mkdir -p "${RESULTS_DIR_CLUSTER}"

BASE_CFG="exp/configs/_tmp_${TAG}_base.yaml"
cat > "$BASE_CFG" <<EOF
model_name: "${BASE_ID}"
quant: "8bit"
pooling: "asst"
per_axis: 1000
batch_size: 8
max_length: 160
axes_bank_path: "exp/${TAG}/axes_base_asst_pairwise.npz"
seed: 2025
EOF

  # Base: prepare
  AX_BANK="exp/${TAG}/axes_base_asst_pairwise.npz"
  if [ -s "$AX_BANK" ]; then
    echo "[SKIP] axes exists: $AX_BANK"
  else
    "$PYTHON_BIN" scripts/00_prepare_vectors.py --config "$BASE_CFG"
  fi
      
  for trait in "${TRAITS[@]}"; do
    run_probe_if_needed "base" "$trait" "${BASE_ID}" "${AX_BANK}" "${RESULTS_DIR_CLUSTER}/${TAG}_base_${trait}_with_rms.jsonl"
  done
  BASE_ALL="${RESULTS_DIR_CLUSTER}/${TAG}_base_alltraits.jsonl"
  rm -f "$BASE_ALL"
  for trait in "${TRAITS[@]}"; do
    f="${RESULTS_DIR_CLUSTER}/${TAG}_base_${trait}_with_rms.jsonl"
    if is_nonempty_file "$f"; then
      cat "$f" >> "$BASE_ALL"
    else
      echo "[WARN] missing/empty: $f (not included in alltraits)"
    fi
  done
  echo "[INFO] wrote: $BASE_ALL ($(wc -l < "$BASE_ALL" 2>/dev/null || echo 0) lines)"
  
INSTR_CFG="exp/configs/_tmp_${TAG}_instruct.yaml"
cat > "$INSTR_CFG" <<EOF
model_name: "${INSTR_ID}"
quant: "8bit"
pooling: "asst"
per_axis: 1000
batch_size: 8
max_length: 160
axes_bank_path: "exp/${TAG}/axes_instruct_asst_pairwise.npz"
seed: 2025
EOF

  # Instruct: prepare  
  AX_BANK="exp/${TAG}/axes_instruct_asst_pairwise.npz"
  if [ -s "$AX_BANK" ]; then
    echo "[SKIP] axes exists: $AX_BANK"
  else
    "$PYTHON_BIN" scripts/00_prepare_vectors.py --config "$INSTR_CFG"
  fi
  
  # Instruct: probe
  for trait in "${TRAITS[@]}"; do
    run_probe_if_needed "instruct" "$trait" "${INSTR_ID}" "${AX_BANK}" "${RESULTS_DIR_CLUSTER}/${TAG}_instruct_${trait}_with_rms.jsonl"
  done
  INSTR_ALL="${RESULTS_DIR_CLUSTER}/${TAG}_instruct_alltraits.jsonl"
  rm -f "$INSTR_ALL"
  for trait in "${TRAITS[@]}"; do
    f="${RESULTS_DIR_CLUSTER}/${TAG}_instruct_${trait}_with_rms.jsonl"
    if is_nonempty_file "$f"; then
      cat "$f" >> "$INSTR_ALL"
    else
      echo "[WARN] missing/empty: $f (not included in alltraits)"
    fi
  done
  echo "[INFO] wrote: $INSTR_ALL ($(wc -l < "$INSTR_ALL" 2>/dev/null || echo 0) lines)"
    
  SEL_RNG_ROOT="${RESULTS_DIR_CLUSTER}/selected_range"
  mkdir -p "$SEL_RNG_ROOT"

  # ★ base/instruct × 5 traits 全部に対して 06/07 を回す
  for SPLIT in base instruct; do
    for trait in "${TRAITS[@]}"; do
      IN_JSONL="${RESULTS_DIR_CLUSTER}/${TAG}_${SPLIT}_${trait}_with_rms.jsonl"
      [ -s "$IN_JSONL" ] || { echo "[WARN] missing $IN_JSONL (skip)"; continue; }

      "$PYTHON_BIN" scripts/06_alpha_eval_v13.py \
        --in "$IN_JSONL" \
        --out_root "$SEL_RNG_ROOT" \
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
  # ==================== Summarize (CSV + plots) ====================
  # range の結果だけ集計（selected は後で同様に可能）
  mkdir -p "$SEL_RNG_ROOT/_summary"
  "$PYTHON_BIN" scripts/07_alpha_visualize.py \
    --globs "$SEL_RNG_ROOT/range/alpha_range_*.jsonl" \
    --out_csv "$SEL_RNG_ROOT/_summary/range_summary.csv" \
    --out_dir "$SEL_RNG_ROOT/_summary/range_figs" \
    --compare_splits

  mkdir -p "$SEL_RNG_ROOT/_corr"
  "$PYTHON_BIN" scripts/08_corr_range_vs_rms_v8.py \
    --range_csv "$SEL_RNG_ROOT/_summary/range_summary.csv" \
    --probe_jsonl_glob "${RESULTS_DIR_CLUSTER}/${TAG}_*_*_with_rms.jsonl" \
    --out_dir "$SEL_RNG_ROOT/_corr" \
    --make_plots \
    --corr_group split

  set +x
  echo "==== ${TAG} END ===="
  echo "-----------------------------------------"
done

echo "=== EFFICIENT PIPELINE COMPLETED ==="
echo "END TIME: $(date)"
