#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_compare_vector_norms.py

目的:
- Base / Instruct モデルについて、Big5Chat 上の各性格軸ごとに
  v_diff(L) を計算し、ノルムと Base vs Instruct のコサイン類似度を
  「全レイヤー」について比較する。
- axis_mode によって v_diff の定義を切り替える:
  - cluster  : high/low クラスタ平均の差 v_pos - v_neg
  - pairwise : 同一シナリオ high/low 応答の差 d_i = h_pos - h_neg の平均
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

try:
    import yaml
except Exception:
    yaml = None

from persona_vectors.live_axes import (
    AXES,                      # ["openness", ...]
    load_model_and_tokenizer,
    _ensure_pad_token,
    _ensure_dialog_tokens,
    _infer_main_device,
)


# =========================
# Big5Chat 読み込み
# =========================

def _canon_axes() -> List[str]:
    return ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def _parse_level(raw) -> int:
    """
    Big5Chat の level フィールドを +1 / -1 に正規化するヘルパ。
    想定:
      - "high" / "low" などの文字列
      - 1 / 0 / -1 などの数値

    戻り値:
      >0: high (POS)
      <=0: low (NEG)
    """
    if raw is None:
        return 0

    # すでに数値の場合
    if isinstance(raw, (int, float)):
        try:
            return int(raw)
        except Exception:
            return 0

    # 文字列の場合
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("high", "pos", "positive", "hi", "1"):
            return 1
        if s in ("low", "neg", "negative", "lo", "0", "-1"):
            return -1
        # それ以外はとりあえず 0 扱いにしておく
        try:
            return int(s)
        except Exception:
            return 0

    # 想定外型
    return 0

def extract_big5_texts_cluster(per_axis: int = 1000) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    axis_mode=cluster 用:
    Big5Chat から traitごとの high/low テキスト集合を作る。
    """
    if not _HAS_DATASETS:
        raise RuntimeError("datasets が無いため Big5Chat を読み込めません。pip install datasets してください。")

    ds_all = load_dataset("wenkai-li/big5_chat")
    ds = ds_all[next(iter(ds_all.keys()))] if isinstance(ds_all, dict) else ds_all

    AXES_CANON = _canon_axes()
    POS = {ax: [] for ax in AXES_CANON}
    NEG = {ax: [] for ax in AXES_CANON}

    for ex in ds:
        tr = (ex.get("trait") or "").strip().lower()
        level = _parse_level(ex.get("level"))
        # trait 名を Canonical に
        ax_map = {
            "openness": "openness",
            "conscientiousness": "conscientiousness",
            "extraversion": "extraversion",
            "agreeableness": "agreeableness",
            "neuroticism": "neuroticism",
        }
        ax = ax_map.get(tr)
        if ax is None:
            continue

        out = (ex.get("train_output") or "").strip()
        if not out:
            continue

        # asst プーリングで使えるよう、<asst> を付けておく
        text = "<asst> " + out
        if level > 0:
            POS[ax].append(text)
        else:
            NEG[ax].append(text)

    # per_axis 上限
    for ax in AXES_CANON:
        if per_axis > 0:
            POS[ax] = POS[ax][:per_axis]
            NEG[ax] = NEG[ax][:per_axis]
        print(f"[cluster] {ax}: +{len(POS[ax])} / -{len(NEG[ax])}")

    return POS, NEG


def extract_big5_pairs_pairwise(per_axis: int = 1000) -> Dict[str, List[Tuple[str, str]]]:
    """
    axis_mode=pairwise 用:
    Big5Chat から (同じ original_index, trait) の high/low ペアを作る。
    戻り値: PAIRS[axis] = [(pos_text, neg_text), ...]
    """
    if not _HAS_DATASETS:
        raise RuntimeError("datasets が無いため Big5Chat を読み込めません。pip install datasets してください。")

    ds_all = load_dataset("wenkai-li/big5_chat")
    ds = ds_all[next(iter(ds_all.keys()))] if isinstance(ds_all, dict) else ds_all

    AXES_CANON = _canon_axes()
    buckets: Dict[Tuple[str, int], Dict[str, str]] = {}

    for ex in ds:
        tr = (ex.get("trait") or "").strip().lower()
        level = _parse_level(ex.get("level"))
        original_idx = ex.get("original_index")
        if original_idx is None:
            continue

        ax_map = {
            "openness": "openness",
            "conscientiousness": "conscientiousness",
            "extraversion": "extraversion",
            "agreeableness": "agreeableness",
            "neuroticism": "neuroticism",
        }
        ax = ax_map.get(tr)
        if ax is None:
            continue

        out = (ex.get("train_output") or "").strip()
        if not out:
            continue

        key = (ax, int(original_idx))
        d = buckets.setdefault(key, {})
        if level > 0:
            d["pos"] = "<asst> " + out
        else:
            d["neg"] = "<asst> " + out

    PAIRS: Dict[str, List[Tuple[str, str]]] = {ax: [] for ax in AXES_CANON}
    for (ax, idx), d in buckets.items():
        if "pos" in d and "neg" in d:
            PAIRS[ax].append((d["pos"], d["neg"]))

    for ax in AXES_CANON:
        if per_axis > 0:
            PAIRS[ax] = PAIRS[ax][:per_axis]
        print(f"[pairwise] {ax}: {len(PAIRS[ax])} pairs")

    return PAIRS


# =========================
# プーリングヘルパ
# =========================

def _pool_hidden(
    hs: torch.Tensor,
    attn_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    mode: str,
) -> torch.Tensor:
    """
    hs: (B, T, H)
    attn_mask: (B, T), 1=有効
    input_ids: (B, T)
    """
    B, T, H = hs.shape
    device = hs.device

    if mode == "last":
        # 各サンプルの最後の非PADトークン
        lengths = attn_mask.sum(dim=1)  # (B,)
        idx = (lengths - 1).clamp(min=0)  # (B,)
        out = hs[torch.arange(B, device=device), idx]  # (B, H)
        return out

    if mode == "mean":
        # 有効トークンの平均
        mask = attn_mask.unsqueeze(-1)  # (B, T, 1)
        summed = (hs * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / counts

    if mode == "asst":
        # <asst> 以降のトークンの平均（なければ全体平均）
        try:
            asst_id = tokenizer.convert_tokens_to_ids("<asst>")
        except Exception:
            asst_id = None

        if asst_id is None or asst_id == tokenizer.unk_token_id:
            # fallback: mean
            return _pool_hidden(hs, attn_mask, input_ids, tokenizer, "mean")

        out_vecs = []
        for b in range(B):
            ids_b = input_ids[b]          # (T,)
            hs_b = hs[b]                  # (T, H)
            mask_b = attn_mask[b]         # (T,)
            pos = (ids_b == asst_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                start = int(pos[0].item()) + 1
                if start < T:
                    valid_mask = (mask_b[start:] == 1)
                    if valid_mask.any():
                        hs_sel = hs_b[start:][valid_mask]
                        out_vecs.append(hs_sel.mean(dim=0))
                        continue
            # fallback: 有効トークン平均
            valid_mask = (mask_b == 1)
            hs_sel = hs_b[valid_mask]
            out_vecs.append(hs_sel.mean(dim=0))
        return torch.stack(out_vecs, dim=0)

    # デフォルトで mean
    return _pool_hidden(hs, attn_mask, input_ids, tokenizer, "mean")


# =========================
# v_diff 計算（cluster / pairwise）
# =========================

def _collect_vdiff_cluster_for_model(
    model_name: str,
    texts_pos: Dict[str, List[str]],
    texts_neg: Dict[str, List[str]],
    pooling: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    cluster モード用: v_diff(L) = mean_pos(L) - mean_neg(L)
    戻り値: vdiff[axis][layer_idx] = np.ndarray(H,)
    """
    print(f"[cluster] Loading model: {model_name}")
    model, tok = load_model_and_tokenizer(model_name)
    tok = _ensure_pad_token(tok, model)
    if pooling == "asst":
        tok = _ensure_dialog_tokens(tok, model)

    device = _infer_main_device(model)
    model.eval()

    vdiff_by_axis: Dict[str, Dict[int, np.ndarray]] = {}
    AXES_CANON = _canon_axes()

    for ax in AXES_CANON:
        pos_list = texts_pos.get(ax, [])
        neg_list = texts_neg.get(ax, [])
        if not pos_list or not neg_list:
            print(f"[cluster] {ax}: no data, skip")
            continue

        print(f"[cluster] axis={ax}: pos={len(pos_list)}, neg={len(neg_list)}")

        def _collect_mean_for_texts(texts: List[str]) -> Dict[int, torch.Tensor]:
            layer_sums = None
            count = 0

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True, use_cache=False)
                hs_all = out.hidden_states  # tuple(len_layers) of (B, T, H)
                attn = enc["attention_mask"]
                input_ids = enc["input_ids"]

                if layer_sums is None:
                    num_layers = len(hs_all)
                    layer_sums = [
                        torch.zeros(hs_all[0].shape[-1], device=device, dtype=hs_all[0].dtype)
                        for _ in range(num_layers)
                    ]

                B = attn.shape[0]
                for L, hs_L in enumerate(hs_all):
                    v_batch = _pool_hidden(hs_L, attn, input_ids, tok, pooling)  # (B, H)
                    layer_sums[L] += v_batch.sum(dim=0)
                count += B

            means = {L: (layer_sums[L] / max(count, 1)).detach().cpu() for L in range(len(layer_sums))}
            return means

        pos_means = _collect_mean_for_texts(pos_list)
        neg_means = _collect_mean_for_texts(neg_list)

        vdiff_by_axis[ax] = {}
        for L in pos_means.keys():
            v_pos = pos_means[L].to(dtype=torch.float32)
            v_neg = neg_means[L].to(dtype=torch.float32)
            v_diff = (v_pos - v_neg).cpu().numpy()
            vdiff_by_axis[ax][L] = v_diff

    return vdiff_by_axis

def _collect_vdiff_pairwise_for_model(
    model_name: str,
    pairs: Dict[str, List[Tuple[str, str]]],
    pooling: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    pairwise モード用:
    v_diff(L) = mean_i [ h_pos_i(L) - h_neg_i(L) ]
    """
    print(f"[pairwise] Loading model: {model_name}")
    model, tok = load_model_and_tokenizer(model_name)
    tok = _ensure_pad_token(tok, model)
    if pooling == "asst":
        tok = _ensure_dialog_tokens(tok, model)

    # accelerate 側の device を推定（model.to(device) はしない）
    device = _infer_main_device(model)
    model.eval()

    # emb の語彙サイズを取得しておく
    vocab_size = model.get_input_embeddings().weight.shape[0]
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else 0

    vdiff_by_axis: Dict[str, Dict[int, np.ndarray]] = {}
    AXES_CANON = _canon_axes()

    for ax in AXES_CANON:
        pair_list = pairs.get(ax, [])
        if not pair_list:
            print(f"[pairwise] {ax}: no pairs, skip")
            continue

        print(f"[pairwise] axis={ax}: pairs={len(pair_list)}")

        layer_sums = None
        pair_count = 0

        for i in range(0, len(pair_list), batch_size):
            batch_pairs = pair_list[i : i + batch_size]
            pos_texts = [p[0] for p in batch_pairs]
            neg_texts = [p[1] for p in batch_pairs]

            texts = pos_texts + neg_texts  # 先に pos, 後ろに neg
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # --- ★ 語彙外 ID を unk に潰すガード ---
            ids = enc["input_ids"]  # CPU tensor
            bad_mask = (ids < 0) | (ids >= vocab_size)
            if bad_mask.any():
                n_bad = int(bad_mask.sum().item())
                max_id = int(ids.max().item())
                min_id = int(ids.min().item())
                print(
                    f"[Warning][{model_name}][pairwise] "
                    f"found {n_bad} token ids out of range "
                    f"(min={min_id}, max={max_id}, vocab_size={vocab_size}). "
                    f"Replacing them with unk_id={unk_id}."
                )
                ids[bad_mask] = unk_id
                enc["input_ids"] = ids
            # --------------------------------------

            # ここでまとめて device へ
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            hs_all = out.hidden_states
            attn = enc["attention_mask"]
            input_ids = enc["input_ids"]

            if layer_sums is None:
                num_layers = len(hs_all)
                layer_sums = [
                    torch.zeros(
                        hs_all[0].shape[-1],
                        device=device,
                        dtype=hs_all[0].dtype,
                    )
                    for _ in range(num_layers)
                ]

            B2 = attn.shape[0]
            B = B2 // 2  # pos/neg なので偶数前提
            for L, hs_L in enumerate(hs_all):
                v_batch = _pool_hidden(hs_L, attn, input_ids, tok, pooling)  # (B2, H)
                v_pos = v_batch[:B]
                v_neg = v_batch[B:B + B]
                diff = v_pos - v_neg  # (B, H)
                layer_sums[L] += diff.sum(dim=0)

            pair_count += B

        vdiff_by_axis[ax] = {}
        for L in range(len(layer_sums)):
            v_mean = (layer_sums[L] / max(pair_count, 1)).to(dtype=torch.float32)
            vdiff_by_axis[ax][L] = v_mean.detach().cpu().numpy()

    return vdiff_by_axis


# =========================
# メイン
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True, help="Base model ID (e.g., mistralai/Mistral-7B-v0.1)")
    ap.add_argument("--model_instruct", required=True, help="Instruct model ID (e.g., mistralai/Mistral-7B-Instruct-v0.3)")
    ap.add_argument("--config", default="exp/configs/exp.yaml", help="Path to config YAML (per_axis, batch_size, etc.)")
    ap.add_argument("--pooling", choices=["last", "asst", "mean"], default="last", help="Pooling strategy")
    ap.add_argument("--axis_mode", choices=["cluster", "pairwise"], default="cluster", help="How to define v_diff")
    ap.add_argument("--out", default="exp/norm_comparison_all_layers.jsonl", help="Output JSONL file path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- config 読み込み ---
    per_axis = 1000
    batch_size = 8
    max_length = 256

    cfg_path = Path(args.config)
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # できるだけ柔軟に拾う
        per_axis = cfg.get("big5chat", {}).get("per_axis", cfg.get("per_axis", per_axis))
        batch_size = cfg.get("batch_size", batch_size)
        max_length = cfg.get("max_length", max_length)

    print(f"[INFO] pooling={args.pooling}, axis_mode={args.axis_mode}, per_axis={per_axis}, batch_size={batch_size}, max_length={max_length}")

    # --- テキスト / ペアの準備 ---
    if args.axis_mode == "cluster":
        POS_TEXTS, NEG_TEXTS = extract_big5_texts_cluster(per_axis=per_axis)
        PAIRS = None
    else:
        PAIRS = extract_big5_pairs_pairwise(per_axis=per_axis)
        POS_TEXTS = NEG_TEXTS = None

    # --- 各モデルで v_diff を計算 ---
    if args.axis_mode == "cluster":
        vdiff_base = _collect_vdiff_cluster_for_model(
            args.model_base, POS_TEXTS, NEG_TEXTS, args.pooling, batch_size, max_length
        )
        vdiff_instr = _collect_vdiff_cluster_for_model(
            args.model_instruct, POS_TEXTS, NEG_TEXTS, args.pooling, batch_size, max_length
        )
    else:
        vdiff_base = _collect_vdiff_pairwise_for_model(
            args.model_base, PAIRS, args.pooling, batch_size, max_length
        )
        vdiff_instr = _collect_vdiff_pairwise_for_model(
            args.model_instruct, PAIRS, args.pooling, batch_size, max_length
        )

    # --- ノルム / cos を計算して JSONL 書き出し ---
    records = []
    AXES_CANON = _canon_axes()
    for ax in AXES_CANON:
        vb_layers = vdiff_base.get(ax, {})
        vi_layers = vdiff_instr.get(ax, {})
        for L in sorted(vb_layers.keys()):
            if L not in vi_layers:
                continue
            vb = vb_layers[L].astype(np.float64)
            vi = vi_layers[L].astype(np.float64)

            norm_base = float(np.linalg.norm(vb))
            norm_instr = float(np.linalg.norm(vi))
            denom = max(norm_base * norm_instr, 1e-12)
            cos = float(np.dot(vb, vi) / denom)
            ratio = float(norm_instr / max(norm_base, 1e-12))

            rec = {
                "layer": int(L),
                "axis": ax,
                "norm_base": norm_base,
                "norm_instruct": norm_instr,
                "norm_ratio_instruct_to_base": ratio,
                "cosine_similarity": cos,
                "axis_mode": args.axis_mode,
                "pooling": args.pooling,
                "model_base": args.model_base,
                "model_instruct": args.model_instruct,
            }
            records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Done] Saved comparison results to {out_path}")


if __name__ == "__main__":
    main()
