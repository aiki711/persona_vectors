#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 00_prepare_vectors.py (Big5Chat Pairwise / All-Layers Edition)
#
# 目的:
#  - Big5Chat データセットを用いて、各 Big Five 軸について
#      「同じシナリオ + 同じ trait」での (High 応答 − Low 応答)
#    のペア差分ベクトルを全レイヤーで計算する。
#  - 各レイヤー L / trait ax の軸ベクトル a_{L,ax} を unit ベクトルとして
#    .npz (axes_bank) に保存する。
#

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import os
import numpy as np
import torch
import yaml
from datasets import load_dataset

# ★ live_axes_and_hook から共通ヘルパを import
from live_axes_and_hook import (
    AXES as AXES_CANON,
    load_model_and_tokenizer,
    _ensure_pad_token,
    _ensure_dialog_tokens,
    _infer_main_device,
    _is_bnb_quantized,
    get_layer_stack,
)


# ==============================================
#  Big5Chat から「ペア差分」用データを取り出す
# ==============================================

def extract_big5_pairs_from_hf(per_axis: int = 1000) -> Dict[str, List[Tuple[str, str]]]:
    """
    HuggingFace の Big5Chat から、(同じ original_index, 同じ trait) の
    high/low 応答ペアを抽出して返す。

    戻り値:
        PAIRS[axis_name] = [(text_high, text_low), ...]
        axis_name は "openness" など AXES_CANON に合わせた正規化名。
    """
    ds_all = load_dataset("wenkai-li/big5_chat")
    # {'train': Dataset} 形式が多いので、その最初の split を使う
    if isinstance(ds_all, dict):
        split_name = next(iter(ds_all.keys()))
        ds = ds_all[split_name]
    else:
        ds = ds_all

    # (trait, original_index) ごとに high/low の候補をためる
    buckets: Dict[tuple, Dict[str, List[str]]] = defaultdict(
        lambda: {"high": [], "low": []}
    )

    for ex in ds:
        tr_raw = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr_raw not in AXES_CANON or lv not in {"high", "low"}:
            continue

        orig_idx = ex.get("original_index")
        if orig_idx is None:
            continue

        to = (ex.get("train_output") or "").strip()
        if not to:
            continue

        # <asst> プレフィックスをつけておく（後で asst プーリングしやすいように）
        text = f"<asst> {to}"
        buckets[(tr_raw, orig_idx)][lv].append(text)

    # buckets から実際の (high, low) ペアを作る
    PAIRS: Dict[str, List[Tuple[str, str]]] = {ax: [] for ax in AXES_CANON}
    for (tr, orig_idx), d in buckets.items():
        highs = d["high"]
        lows = d["low"]
        if not highs or not lows:
            continue
        # シンプルに 1 組だけ使う（必要なら複数組にしてもOK）
        text_high = highs[0]
        text_low = lows[0]
        PAIRS[tr].append((text_high, text_low))

    # 軸ごとにシャッフル & per_axis でカット
    for ax in AXES_CANON:
        random.shuffle(PAIRS[ax])
        if per_axis > 0:
            PAIRS[ax] = PAIRS[ax][:per_axis]
        print(f"[big5chat-pairs] {ax}: {len(PAIRS[ax])} pairs")

    return PAIRS


# ==============================================
#  メインロジック: ペア差分で全レイヤー軸を作る
# ==============================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML config path")
    ap.add_argument("--bank_path", "-b", required=True, help="Axes bank path")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 1. コンフィグ ----
    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("config に model_name または model.name が必要です。")

    quant = cfg.get("quant", "auto")
    pooling = os.environ.get("POOLING", cfg.get("pooling", "asst")) # "asst" 推奨
    per_axis = int(cfg.get("per_axis", 1000))
    batch_size = int(cfg.get("batch_size", 8))
    max_length = int(cfg.get("max_length", 160))
    bank_path_str = args.bank_path

    print("=== 00_prepare_vectors (Big5Chat Pairwise Edition) ===")
    print(f"  model_name : {model_name}")
    print(f"  quant      : {quant}")
    print(f"  pooling    : {pooling}")
    print(f"  per_axis   : {per_axis}")
    print(f"  batch_size : {batch_size}")
    print(f"  max_length : {max_length}")
    print(f"  bank_path  : {bank_path_str}")

    # 再現性のためシードを固定
    seed = int(cfg.get("seed", 2025))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- 2. モデル / トークナイザ ----
    print("\n[Step 1] Loading model/tokenizer...")
    t0 = time.time()
    model, tok = load_model_and_tokenizer(model_name, quant=quant)
    load_sec = time.time() - t0
    print(f"  Loaded in {load_sec:.1f} sec")

    # レイヤー情報
    layers_stack, N_layers, kind = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    print(f"\n[Model] {kind}")
    print(f"  Total layers: {N_layers}")
    print(f"  Targeting ALL {len(layer_indices)} hidden states (L=0..{N_layers-1})")

    H_dim = model.config.hidden_size
    print(f"  Hidden size (H): {H_dim}")

    # device 推定
    device = _infer_main_device(model)
    if _is_bnb_quantized(model):
        model.eval()
    else:
        model.to(device).eval()

    # トークナイザ調整
    tok = _ensure_pad_token(tok, model)
    if pooling == "asst":
        tok = _ensure_dialog_tokens(tok, model)

    # ---- 3. Big5Chat からペア抽出 ----
    print("\n[Step 2] Loading Big5Chat and building pairs...")
    PAIRS = extract_big5_pairs_from_hf(per_axis=per_axis)

    # ---- 4. プーリングヘルパ ----
    @torch.no_grad()
    def _pool_hidden_per_sample(
        hs: torch.Tensor,         # (B, T, H)
        attn: torch.Tensor,       # (B, T)
        input_ids: torch.Tensor,  # (B, T)
        pooling_mode: str = "last",
    ) -> torch.Tensor:
        """
        各サンプルごとのプーリング結果 (B, H) を返す。
        pooling_mode: "last" / "mean" / "asst"
        """
        if pooling_mode == "mean":
            denom = attn.sum(dim=1, keepdim=True).clamp_min(1)
            x = (hs * attn.unsqueeze(-1)).sum(dim=1) / denom  # (B, H)
            return x

        if pooling_mode == "asst":
            # <asst> 以降の平均。見つからなければ全体平均。
            asst_id = tok.convert_tokens_to_ids("<asst>")
            B, T, H = hs.size()
            out = torch.empty((B, H), device=hs.device, dtype=hs.dtype)
            for b in range(B):
                ids_b = input_ids[b]
                idx = (ids_b == asst_id).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    j = int(idx[0].item())
                    mask_b = attn[b, j:].bool()
                    vecs_b = hs[b, j:][mask_b]
                else:
                    mask_b = attn[b].bool()
                    vecs_b = hs[b][mask_b]
                if vecs_b.size(0) == 0:
                    out[b] = hs[b, -1]  # フォールバックで最後のトークン
                else:
                    out[b] = vecs_b.mean(dim=0)
            return out

        # デフォルト: "last"
        idx = attn.sum(dim=1) - 1       # (B,)
        idx = idx.clamp(min=0)
        gathered = hs[torch.arange(hs.size(0), device=hs.device), idx]  # (B, H)
        return gathered

    # ---- 5. ペア差分で各レイヤーの平均軸を計算 ----
    @torch.no_grad()
    def calculate_pairwise_axis_for_trait(
        pairs: List[Tuple[str, str]]
    ) -> Dict[int, torch.Tensor]:
        """
        与えられた (text_pos, text_neg) のペアリストについて、
        各レイヤー L で
            d_mean(L) = mean_i (h_pos_i(L) - h_neg_i(L))
        を計算して {L: (H,)} を返す。
        """
        layer_diff_sum: Dict[int, torch.Tensor] = {
            L: torch.zeros(H_dim, device=device, dtype=torch.float32)
            for L in layer_indices
        }
        total_pairs = 0

        # 1バッチ = N ペア ⇒ テキストは 2N 本
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i: i + batch_size]
            if not batch_pairs:
                continue

            texts: List[str] = []
            signs: List[int] = []      # +1: pos, -1: neg
            pair_idx: List[int] = []   # どのペアに属するか

            for j, (t_pos, t_neg) in enumerate(batch_pairs):
                texts.append(t_pos); signs.append(+1); pair_idx.append(j)
                texts.append(t_neg); signs.append(-1); pair_idx.append(j)

            tok_d = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            out = model(
                **tok_d,
                output_hidden_states=True,
                use_cache=False,
            )
            attn = tok_d["attention_mask"]
            input_ids = tok_d["input_ids"]

            B = input_ids.size(0)
            assert B == len(texts)

            # レイヤーごとに、各サンプルのベクトル→ペア差分に集約
            for L, hs in enumerate(out.hidden_states):  # L = 0..N
                if L not in layer_indices:
                    continue

                # (B, H)
                v_batch = _pool_hidden_per_sample(
                    hs, attn, input_ids, pooling_mode=pooling
                )

                num_batch_pairs = len(batch_pairs)
                pos_vecs = [
                    torch.zeros(H_dim, device=device, dtype=torch.float32)
                    for _ in range(num_batch_pairs)
                ]
                neg_vecs = [
                    torch.zeros(H_dim, device=device, dtype=torch.float32)
                    for _ in range(num_batch_pairs)
                ]

                for b, (p_idx, sgn) in enumerate(zip(pair_idx, signs)):
                    if sgn == +1:
                        pos_vecs[p_idx] = v_batch[b]
                    else:
                        neg_vecs[p_idx] = v_batch[b]

                # このバッチ内の各ペアについて差分を足し込む
                for j in range(num_batch_pairs):
                    d = pos_vecs[j] - neg_vecs[j]
                    layer_diff_sum[L] += d.to(torch.float32)

            total_pairs += len(batch_pairs)

        if total_pairs == 0:
            return {
                L: torch.zeros(H_dim, dtype=torch.float32) for L in layer_indices
            }

        final_axes: Dict[int, torch.Tensor] = {
            L: (v_sum / total_pairs) for L, v_sum in layer_diff_sum.items()
        }
        return final_axes

    # ---- 6. 各軸について pairwise ベクトルを計算 & 正規化して保存 ----
    axes_by_layer: Dict[Tuple[int, str], np.ndarray] = {}
    rawnorm_by_layer: Dict[Tuple[int, str], float] = {}

    for ax in AXES_CANON:
        pairs_ax = PAIRS[ax]
        print(f"\n=== Axis (pairwise): {ax} ===")
        print(f"  #pairs = {len(pairs_ax)}")

        v_diff_all_layers = calculate_pairwise_axis_for_trait(pairs_ax)

        for L in layer_indices:
            v = v_diff_all_layers[L]  # (H,)
            raw_norm = float(v.norm().item())
            rawnorm_by_layer[(L, ax)] = raw_norm
            norm = v.norm() + 1e-12
            v_unit = (v / norm).cpu().to(torch.float32).numpy()
            axes_by_layer[(L, ax)] = v_unit

    # ---- 7. .npz に保存 ----
    bank_path = Path(bank_path_str)
    bank_path.parent.mkdir(parents=True, exist_ok=True)

    npz_dict = {
        f"{L}|{ax}": vec
        for (L, ax), vec in axes_by_layer.items()
    }

    np.savez_compressed(bank_path, **npz_dict)
    print(f"\n[Done] Saved ALL-LAYER pairwise axes bank to: {bank_path}")
    print(f"  Total vectors saved: {len(npz_dict)}")

    norm_path = Path(str(bank_path).replace(".npz", "_rawnorms.npz"))
    npz_norm = {
        f"{L}|{ax}": np.array(val, dtype=np.float32)
        for (L, ax), val in rawnorm_by_layer.items()
    }
    np.savez_compressed(norm_path, **npz_norm)
    print(f"[Done] Saved RAW (pre-normalization) norms bank to: {norm_path}")


if __name__ == "__main__":
    main()
