# scripts/01_run_probe.py
# (★ 全レイヤー同時介入 (All-Layer Intervention) 版 - 純粋ステアリング用)

import argparse, os, json, numpy as np, random
import torch
import re
import math
from typing import Dict, List, Tuple, Iterable, Any, Optional, Sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

from contextlib import ExitStack
from dataclasses import dataclass, field

from persona_vectors.live_axes import (
    AXES,
    get_layer_stack,
    _infer_main_device,
    load_model_and_tokenizer,
)
from datasets import load_dataset
from collections import defaultdict

@dataclass
class ResidualSteerer:
    """
    指定の層の residual stream に一定ベクトルを加算する context manager。
    Batched alpha に対応: alpha が list/tensor の場合、(B, 1, 1) に拡張して適用する。
    """
    model: torch.nn.Module
    layer: int
    v_mix: np.ndarray
    alpha: Any # float or List[float] or torch.Tensor
    answer_only: bool = False
    
    def __post_init__(self):
        self.handle = None

    def __enter__(self):
        stack, _, _ = get_layer_stack(self.model)
        target_mod = stack[self.layer]
        v = torch.tensor(self.v_mix, dtype=torch.float32) # (H,)

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # hs: (B, T, H)
            
            if self.answer_only:
                # cacheあり: decodeは T=1 → そのまま通す
                if hs.size(1) != 1:
                    # cacheなし: 最初に見た長さを prefill とみなしてスキップ
                    if not hasattr(self, "_prefill_T"):
                        self._prefill_T = int(hs.size(1))
                        return out
                    # それ以降(=生成が進んで長さが増えた)は適用してOK
                    if int(hs.size(1)) == self._prefill_T:
                        return out
            
            # キャッシュ作成 (device/dtype合わせ)
            if not hasattr(self, "_add_cache") or self._add_cache.device != hs.device or self._add_cache.dtype != hs.dtype:
                # v: (H) -> (1, 1, H)
                self._add_cache = v.to(device=hs.device, dtype=hs.dtype).view(1, 1, -1)
                
                # Alphaの処理
                if isinstance(self.alpha, (float, int)):
                     #Scalar
                     self._alpha_cache = torch.tensor(self.alpha, device=hs.device, dtype=hs.dtype)
                else:
                     # Batch: (B,) -> (B, 1, 1) or Tensor
                     if isinstance(self.alpha, torch.Tensor):
                         a_t = self.alpha.detach().clone().to(device=hs.device, dtype=hs.dtype)
                     else:
                         a_t = torch.tensor(self.alpha, device=hs.device, dtype=hs.dtype)
                     
                     if a_t.ndim == 1:
                         a_t = a_t.view(-1, 1, 1)
                     self._alpha_cache = a_t

            # 加算: hs (B,T,H) + alpha (B,1,1) * v (1,1,H) -> Broadcast
            # alphaのバッチサイズがBと一致している必要あり
            hs2 = hs + self._alpha_cache * self._add_cache

            if isinstance(out, tuple):
                return (hs2, *out[1:])
            return hs2

        self.handle = target_mod.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

# ==============================
# プロンプトテンプレート
# ==============================

PROMPT_TEMPLATE = """You are a helpful, polite assistant.
Please answer in 3–5 sentences of natural English.
Do NOT output bullet points, numbered lists, source code, or URLs.

User: {question}
Assistant:"""

def build_prompt(question: str) -> str:
    """
    01_run_probe 内で完結するプロンプト整形関数。
    """
    return PROMPT_TEMPLATE.format(question=question)

def get_unseen_prompts(trait: str, cnt: int, vector_seed: int = 2025, vector_count: int = 1000) -> List[str]:
    """
    big5-chat データセットから、ベクトル作成(00_prepare_vectors.py)に使われていない
    プロンプト(train_input)をランダムに cnt 件抽出して返す。
    """
    print(f"[get_unseen_prompts] Loading big5-chat dataset to exclude first {vector_count} pairs (seed={vector_seed})...")
    
    # 1. データセット読み込み
    ds_all = load_dataset("wenkai-li/big5_chat")
    if isinstance(ds_all, dict):
        split_name = next(iter(ds_all.keys()))
        ds = ds_all[split_name]
    else:
        ds = ds_all

    # 2. 00_prepare_vectors.py と同じロジックでペア抽出 & シャッフル
    #    (全軸に対して行わないと乱数平仄が合わない可能性があるため、全軸回す)
    
    # buckets: (trait, orig_idx) -> {high:[], low:[]}
    buckets = defaultdict(lambda: {"high": [], "low": [], "input": []})
    
    # dataset は streamモードでないと仮定(メモリに乗るサイズ)
    for ex in ds:
        tr_raw = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr_raw not in AXES: 
            continue
            
        orig_idx = ex.get("original_index")
        if orig_idx is None:
            continue
            
        to = (ex.get("train_output") or "").strip()
        ti = (ex.get("train_input") or "").strip() # ここが必要
        
        if not to: continue
        
        # 00_prepare では <asst> 接頭辞をつけるが、ID特定には影響しない
        # buckets には input も保存しておく
        if lv in ["high", "low"]:
            buckets[(tr_raw, orig_idx)][lv].append(to)
            # inputはhigh/low共通のはずだが、エントリ毎に入っている
            buckets[(tr_raw, orig_idx)]["input"].append(ti)

    # PAIRS構築: {trait: [(orig_idx, high_to, low_to), ...]}
    # ※ 00_prepare では (text_high, text_low) のリストだが、
    #    ここでは除外判定のために orig_idx を保持したい。
    #    random.shuffle の挙動を合わせるため、リストの長さを合わせる必要がある。
    #    00_prepare: PAIRS[tr].append((text_high, text_low))
    #    ここでも要素数が同じになるように構成する。
    
    temp_pairs = {ax: [] for ax in AXES}
    
    # 辞書のイテレーション順序は挿入順(Python 3.7+)。
    # ds のイテレーション順が一定なら buckets のキー順も一定。
    for (tr, orig_idx), d in buckets.items():
        highs = d["high"]
        lows = d["low"]
        if not highs or not lows:
            continue
        
        # 00_prepare: text_high = highs[0], text_low = lows[0]
        # 要素として (orig_idx, ...) を入れているが、shuffle はリストの「要素の位置」を入れ替えるだけなので
        # 要素の中身がオブジェクトかタプルかで乱数消費は変わらない(はず)。
        # ただし、sort順などが絡むと変わるが、random.shuffleはin-placeランダムスワップ。
        
        # input テキストも保持しておく (候補として使うため)
        inp_text = d["input"][0] if d["input"] else ""
        
        temp_pairs[tr].append({
            "orig_idx": orig_idx,
            "input": inp_text
        })
        
    # 3. 再現のためのシャッフル
    #    00_prepare_vectors.py:
    #      random.seed(seed)
    #      for ax in AXES_CANON: ... random.shuffle(PAIRS[ax]) ...
    
    rng_state_backup = random.getstate()
    try:
        random.seed(vector_seed)
        
        used_indices = set()
        
        for ax in AXES:
            lst = temp_pairs[ax]
            random.shuffle(lst)
            
            # ターゲット特性の場合、先頭 vector_count 件が「使用済み」
            if ax == trait:
                seen_list = lst[:vector_count]
                for item in seen_list:
                    used_indices.add(item["orig_idx"])
                    
        # 4. 未使用プロンプトの抽出
        #    ターゲット特性のリストのうち、vector_count 以降のもの、
        #    または buckets に入らなかったもの... はペアが成立しなかったものなので
        #    「ベクトル作成に使われていない」定義ならOKだが、
        #    品質担保のため「ペア成立したが採用されなかったもの」から選ぶのが無難。
        #    ここでは temp_pairs[trait] の残りの部分から選ぶ。
        
        candidates = []
        target_list = temp_pairs[trait]
        
        # vector_count以降を候補とする
        # リスト自体は既にシャッフルされているので、そのまま上から取ればランダム
        remaining = target_list[vector_count:]
        
        print(f"  [trait={trait}] Total pairs: {len(target_list)}, Used: {vector_count}, Remaining: {len(remaining)}")
        
        for item in remaining:
             if item["input"] and item["input"].strip():
                 candidates.append(item["input"])
        
        if len(candidates) < cnt:
            print(f"Warning: Only {len(candidates)} unseen prompts found (requested {cnt}). Returning all.")
            return candidates
            
        # 既にシャッフルされているので先頭から取るだけでよいが、
        # 念のため今回の実行用の乱数で再選出してもよい。
        # ここではシンプルに先頭から取る(ランダム性はshuffleで担保されている)
        return candidates[:cnt]

    finally:
        random.setstate(rng_state_backup)

# ---- 引数ユーティリティ ----
def parse_floats_csv(s: str):
    """カンマ区切りの浮動小数点数をパースする"""
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip()]

def load_axes_bank(path_npz: str):
    """00_prepare_vectors.py で作成した .npz ファイルを読み込む"""
    try:
        bank = np.load(path_npz)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Axes bank file not found: {path_npz}")
        print("Please run 00_prepare_vectors.py first.")
        raise
        
    axes_by_layer = {}
    for k in bank.files:              # 形式: f"{L}|{ax}"
        try:
            Ls, ax = k.split("|")
            axes_by_layer[(int(Ls), ax)] = bank[k]
        except Exception as e:
            print(f"Warning: Skipping invalid key in axes_bank: {k} ({e})")
            continue
    print(f"Loaded {len(axes_by_layer)} vectors from {path_npz}")
    return axes_by_layer

# ---- 最終トークンの隠れ状態を測定する関数 ----
@torch.no_grad()
def style_from_y(model, tok, text: str, *, v_axes: dict, layer: int, trait: str, k: float = 3.0) -> float:
    """
    text のトークン列の *最終トークン* の隠れ状態を (layer, trait) の軸に投影 → [-1,1]
    """
    dev = _infer_main_device(model)
    max_len = getattr(model.config, "max_position_embeddings", 4096) - 10
    
    if not text or not text.strip():
        return 0.0
        
    tokd = tok(text, return_tensors="pt", max_length=max_len, truncation=True).to(dev)

    with torch.no_grad():
        out = model(**tokd, output_hidden_states=True, use_cache=False)
        
        if layer >= len(out.hidden_states):
            print(f"Warning: Layer {layer} out of range for hidden_states (max: {len(out.hidden_states)-1}). Returning 0.0.")
            return 0.0
            
        hs  = out.hidden_states[layer]          # (B,T,H)
        
        att = tokd["attention_mask"][0].bool()  # (T)
        h_all = hs[0][att]                      # (T_valid, H)
        
        if h_all.shape[0] == 0:
            return 0.0
        
        h = h_all[-1].detach().cpu().float().numpy() # (H)

    h_norm = np.linalg.norm(h)
    if h_norm < 1e-12:
        return 0.0
    h = h / (h_norm + 1e-12)
    
    v_key = (layer, trait)
    if v_key not in v_axes:
        raise KeyError(f"Vector for {v_key} not found in axes bank. Available keys: {list(v_axes.keys())[:10]}...")
        
    v = v_axes[v_key]
    v_norm_val = np.linalg.norm(v)
    if v_norm_val > 1e-12:
        v = v / v_norm_val
    
    cos_sim = float(np.dot(h, v))             # [-1, 1]
    
    return cos_sim

def _prepare_axes_tensor(
    v_axes: dict,
    layers: Sequence[int],
    trait: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, int]]:
    """
    axes_bank (dict[(L, trait)] -> np.ndarray) から、
    指定 layers の軸ベクトルを (n_layers, H) の torch.Tensor にまとめて返す。
    ついでに layer -> row_index の対応も返す。
    """
    layer2i = {L: i for i, L in enumerate(layers)}
    vecs = []
    for L in layers:
        v_key = (L, trait)
        if v_key not in v_axes:
            raise KeyError(f"Vector for {v_key} not found in axes bank.")
        v = v_axes[v_key]
        vt = torch.tensor(v, dtype=torch.float32, device=device)  # (H,)
        vt = vt / (vt.norm(p=2) + 1e-12)
        vecs.append(vt)
    V = torch.stack(vecs, dim=0)  # (n_layers, H)
    return V, layer2i


@torch.no_grad()
def style_from_text_all_layers(
    model,
    tok,
    texts: List[str],
    *,
    layers: Sequence[int],         # state layer index (L>0)
    V_axes: torch.Tensor,           # (n_layers, H) 正規化済み
    layer2i: dict[int, int],
    max_len_margin: int = 10,
) -> List[dict[int, float]]:
    """
    Batched version: texts をまとめて forward して、
    各バッチ要素の「最終有効トークン hidden」を軸に射影した cos を返す。
    戻り値: List[{L: cos_sim}] (len = len(texts))
    """
    if not texts:
        return []

    dev = _infer_main_device(model)
    max_len = getattr(model.config, "max_position_embeddings", 4096) - max_len_margin
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Paddingありでバッチ化
    tokd = tok(texts, return_tensors="pt", max_length=max_len, truncation=True, padding=True).to(dev)
    out = model(**tokd, output_hidden_states=True, use_cache=False)

    # 最終有効トークン index
    # paddingは右側に入ると仮定 (tokenizerのデフォルト)
    att = tokd["attention_mask"] # (B, T)
    last_indices = att.sum(dim=1) - 1 # (B,)
    
    batch_size = len(texts)
    
    # 結果のコンテナ: {L: [score_b0, score_b1, ...]}
    batched_scores = {L: [] for L in layers}

    for L in layers:
        if L >= len(out.hidden_states):
            batched_scores[L] = [0.0] * batch_size
            continue
            
        hs_all = out.hidden_states[L] # (B, T, H)
        
        # torch.gather等使ってもいいが、Indexingで取得
        # (B, H)
        hs_last = hs_all[torch.arange(batch_size, device=dev), last_indices].to(dtype=torch.float32)
        
        # Normalize
        hs_last = hs_last / (hs_last.norm(p=2, dim=1, keepdim=True) + 1e-12)

        # 該当層の軸ベクトル
        v_idx = layer2i[L]
        v_vec = V_axes[v_idx] # (H,)
        
        # Cosine similarity: (B, H) @ (H,) -> (B,)
        # v_vecは既に正規化済み
        cos_b = torch.matmul(hs_last, v_vec).detach().cpu().numpy()
        batched_scores[L] = cos_b.tolist()

    # List[Dict] に変換
    ret = []
    for i in range(batch_size):
        d = {}
        for L in layers:
            d[L] = batched_scores[L][i]
        ret.append(d)
        
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--axes_bank", required=True, help="npz from 00_prepare_vectors.py")
    ap.add_argument("--layers", default=None, help="同時介入する層 (e.g. '16,20')。省略すると全レイヤー。")
    ap.add_argument("--trait", choices=AXES, default="openness", help="操舵・測定対象の特性")
    ap.add_argument("--alpha_list", type=str, required=True,
                    help="comma-separated floats (e.g. '-2,0,2')。全レイヤーに「分配」される合計の強さ。")
    ap.add_argument("--alpha_mode", choices=["distribute", "additive"], default="distribute",
                    help="[distribute]: alphaを全層に分配 (安全) / [additive]: alphaを各層に適用 (危険)")
    ap.add_argument("--samples", type=int, default=12, help="プローブ用プロンプトの数")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.90)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--include_prefill", action="store_true", help="prefill(プロンプト処理)段階にも介入する。デフォルトはdecode(T=1)のみ")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="exp/<trait>_probe_results.jsonl")
    ap.add_argument("--layer_start", type=int, default=0, 
                        help="Intervention start layer index (default: 0)")
    ap.add_argument("--layer_end", type=int, default=None, 
                        help="Intervention end layer index (default: None -> All layers)")
    ap.add_argument("--use_dataset", action="store_true", help="hardcoded prompt ではなくデータセットから未使用プロンプトをサンプリングする")
    ap.add_argument("--vector_seed", type=int, default=2025, help="00_prepare_vectors.py で使ったシード (未使用データ特定のため)")
    ap.add_argument("--prompt_file", type=str, default=None, help="JSON file containing list of prompts to use (overrides --use_dataset)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    layers_arg = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None
    alphas = parse_floats_csv(args.alpha_list)
    if 0.0 not in alphas:
        alphas = sorted(alphas + [0.0])
    
    # alphasソート (一貫性のため)
    alphas = sorted(list(set(alphas)))

    print(f"Target trait: {args.trait}")
    print(f"Target alphas (total): {alphas}")
    print(f"Alpha mode: {args.alpha_mode}")

    mdl, tok = load_model_and_tokenizer(
        args.model, quant="8bit", device_map="auto"
    )
    device = _infer_main_device(mdl)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    axes_bank = load_axes_bank(args.axes_bank)
    
    all_available_layers = sorted(list(set([
        L for (L, ax) in axes_bank.keys() if ax == args.trait and L > 0
    ])))
    
    # 1. まず候補リストを作成 (--layers引数があればそれを優先、なければ全層)
    if layers_arg:
        candidates = [L for L in all_available_layers if L in layers_arg]
    else:
        candidates = all_available_layers

    # 2. 層の範囲フィルタリング (Start/End)
    target_end = args.layer_end if args.layer_end is not None else 99999
    
    layers_to_process = []
    for L in candidates:
        L_idx = L - 1  # 0-based index (実際のモデル層インデックス)
        if args.layer_start <= L_idx < target_end:
            layers_to_process.append(L)

    print(f"Layer filter applied: Index [{args.layer_start}, {target_end}). Processing {len(layers_to_process)} layers: {layers_to_process}")
        
    if not layers_to_process:
        raise ValueError(f"No valid layers found for trait '{args.trait}' in axes_bank. Check --layers or 00_prepare_vectors.py.")
        
    num_steered_layers = len(layers_to_process)
    V_axes, layer2i = _prepare_axes_tensor(
        axes_bank, layers_to_process, args.trait, device=device
    )
    out_path = args.out.replace("<trait>", args.trait)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        
        print(f"Starting [probe] mode (Batched). Simultaneous intervention on {num_steered_layers} layers.")
        
        if args.prompt_file:
             print(f"Loading prompts from file: {args.prompt_file}")
             with open(args.prompt_file, "r", encoding="utf-8") as f:
                 prompts = json.load(f)
             if not isinstance(prompts, list):
                 raise ValueError("prompt_file must contain a JSON list of strings")
             if args.samples < len(prompts):
                  print(f"Warning: --samples {args.samples} is smaller than file count {len(prompts)}. Truncating.")
                  prompts = prompts[:args.samples]
        elif args.use_dataset:
            print(f"Sampling {args.samples} unseen prompts from dataset for trait '{args.trait}'...")
            prompts = get_unseen_prompts(args.trait, args.samples, vector_seed=args.vector_seed)
        else:
            prompts = [
                "What is a good weekend plan?",
                "How would you approach learning a new skill quickly?",
                "Suggest a gift for a friend with unique tastes.",
                "How do you handle unexpected changes at work?",
                "Convince me to try something outside my comfort zone.",
                "Discuss pros and cons of unconventional ideas in research.",
                "What are some creative ways to improve team collaboration?",
                "How do you evaluate new ideas objectively?",
                "Propose an unusual but practical solution to reduce stress.",
                "How would you explain a complex topic in a novel way?",
                "Share a time you adapted to something unexpected.",
                "What inspires you to try new experiences?",
            ][:args.samples]
        
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=(args.temperature > 0),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )

        # Batch Steering Setup
        alphas_tensor = torch.tensor(alphas, dtype=torch.float32, device=device)
        if args.alpha_mode == "distribute":
            alphas_per_layer = alphas_tensor / num_steered_layers
        else:
            alphas_per_layer = alphas_tensor
            
        try:
             idx_zero = alphas.index(0.0)
        except ValueError:
             idx_zero = 0

        for i, x in enumerate(prompts):
            if (i + 1) % 25 == 0 or (i + 1) == len(prompts):
                print(f"  Probe sample {i+1}/{len(prompts)}...")

            # 1. Prepare Batch Input (Prompt repeated for each alpha)
            # 全てのalphaについて同じプロンプトを使用
            batch_prompts = [build_prompt(x) for _ in alphas]
            
            # Tokenize batch
            inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(device)
            inputs.pop("token_type_ids", None)
            
            # 2. Steered Generation
            try:
                with ExitStack() as stack:
                    for L_state in layers_to_process:
                        L_stack = L_state - 1
                        
                        v_key = (L_state, args.trait)
                        if v_key not in axes_bank:
                            continue
                        vec = axes_bank[v_key]
                        
                        stack.enter_context(
                            ResidualSteerer(
                                mdl, L_stack, vec, alphas_per_layer, 
                                answer_only=(not args.include_prefill)
                            )
                        )
                    
                    # Generate batch
                    out_ids = mdl.generate(**inputs, **gen_kwargs)

            except Exception as e:
                print(f"!! ERROR during batched generation at index {i}: {e}")
                continue

            # 3. Decode
            prompt_len = inputs["input_ids"].shape[1]
            generated_texts = []
            
            # バッチサイズ分デコード
            for j in range(len(alphas)):
                gen_id = out_ids[j, prompt_len:]
                y = tok.decode(gen_id, skip_special_tokens=True).strip()
                generated_texts.append(y)

            # 4. Scoring (Batched)
            scores_list = style_from_text_all_layers(
                mdl, tok, generated_texts,
                layers=layers_to_process,
                V_axes=V_axes,
                layer2i=layer2i
            )
            
            # 5. Extract Baseline info
            baseline_res = scores_list[idx_zero]
            y0 = generated_texts[idx_zero]
            s0_avg = np.mean(list(baseline_res.values())) if baseline_res else 0.0

            # 6. Write Records
            for j, a_total in enumerate(alphas):
                y = generated_texts[j]
                s_dict = scores_list[j]
                s_avg = np.mean(list(s_dict.values())) if s_dict else 0.0
                
                a_per = alphas_per_layer[j].item()

                rec = {
                    "i": i, "trait": args.trait, "layers": layers_to_process,
                    "alpha_total": float(a_total),
                    "alpha_mode": args.alpha_mode,
                    "alpha_per_layer": float(a_per),
                    "x": x, "y": y,
                    "s_avg": s_avg,
                    "s0_avg": s0_avg,
                    "ds_avg": s_avg - s0_avg,
                    "s_by_layer": s_dict,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            fout.flush()

    print(f"[done/probe_batched] wrote {out_path} (layers={layers_to_process}, trait={args.trait})")


if __name__ == "__main__":
    main()
