# scripts/01_run_probe.py
# (★ 全レイヤー同時介入 (All-Layer Intervention) 版)

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



# ==============================
# RMS ログ用ユーティリティ
# ==============================

class _OnlineStats:
    """Welford で平均/分散をオンライン更新（floatのみ）"""
    __slots__ = ("n", "mean", "m2")
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
    def add(self, x: float):
        if not math.isfinite(x):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
    def as_dict(self):
        if self.n == 0:
            return {"n": 0, "mean": float("nan"), "var": float("nan")}
        var = (self.m2 / (self.n - 1)) if self.n > 1 else 0.0
        return {"n": int(self.n), "mean": float(self.mean), "var": float(var)}

class RMSCollector:
    """
    層ごとに「介入前/後」の last-token RMS を集計する。
    - answer_only=True の場合、decodeステップ(T=1)だけ呼ばれる想定。
    """
    def __init__(self):
        # layer -> stats
        self.before = {}
        self.after = {}
    def reset(self):
        self.before = {}
        self.after = {}
    def add(self, layer: int, rms_before: float, rms_after: Optional[float]):
        if layer not in self.before:
            self.before[layer] = _OnlineStats()
            self.after[layer] = _OnlineStats()
        self.before[layer].add(rms_before)
        if rms_after is not None:
            self.after[layer].add(rms_after)
    def snapshot(self, alpha_per_layer: float, eps: float = 1e-12):
        out = {}
        for L in sorted(self.before.keys()):
            b = self.before[L].as_dict()
            a = self.after[L].as_dict()

            m = b["mean"]
            if isinstance(m, (int, float)) and math.isfinite(m) and m > eps:
                denom = m
                ratio = float(alpha_per_layer / denom)
            else:
                denom = float("nan")
                ratio = float("nan")

            out[str(L)] = {
                "rms_before": b,
                "rms_after": a,
                "alpha_over_rms_before_mean": ratio,
            }
        return out

@dataclass
class ResidualSteererLogged:
    """
    ResidualSteerer（live_axes_and_hook）と同じ位置に加算しつつ、
    介入前/後の last-token RMS を collector に記録する。
    """
    model: torch.nn.Module
    layer: int
    v_mix: np.ndarray
    alpha: float
    answer_only: bool = False
    collector: RMSCollector | None = None
    layer_key: int | None = None

    def __post_init__(self):
        self.handle = None

    def __enter__(self):
        stack, _, _ = get_layer_stack(self.model)
        target_mod = stack[self.layer]
        v = torch.tensor(self.v_mix, dtype=torch.float32)

        def _rms_last_token(hs: torch.Tensor) -> float:
            # hs: (B, T, D). last token の RMS（B=1前提でOK）
            x = hs[:, -1, :].float()
            return float(torch.sqrt(torch.mean(x * x)).detach().cpu().item())

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self.answer_only:
                # cacheあり: decodeは T=1 → そのまま通す
                if hs.size(1) != 1:
                    # cacheなし: 最初に見た長さを prefill とみなしてスキップ
                    if not hasattr(self, "_prefill_T"):
                        self._prefill_T = int(hs.size(1))
                        return out
                    # それ以降(=生成が進んで長さが増えた)はログしてOK
                    if int(hs.size(1)) == self._prefill_T:
                        return out
            # ここで初回だけキャッシュ
            if not hasattr(self, "_add_cache") or self._add_cache.device != hs.device or self._add_cache.dtype != hs.dtype:
                self._add_cache = v.to(device=hs.device, dtype=hs.dtype).view(1, 1, -1)
                self._alpha_cache = torch.tensor(self.alpha, device=hs.device, dtype=hs.dtype)

            rms_b = _rms_last_token(hs)
            hs2 = hs + self._alpha_cache * self._add_cache
            rms_a = _rms_last_token(hs2)

            if self.collector is not None:
                key = self.layer_key if self.layer_key is not None else self.layer
                self.collector.add(key, rms_b, rms_a)

            if isinstance(out, tuple):
                return (hs2, *out[1:])
            return hs2

        stack, _, _ = get_layer_stack(self.model)
        target_mod = stack[self.layer]
        self.handle = target_mod.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

# ==============================
# プロンプトテンプレート（★ここを追加）
# ==============================

PROMPT_TEMPLATE = """You are a helpful, polite assistant.
Please answer in 3–5 sentences of natural English.
Do NOT output bullet points, numbered lists, source code, or URLs.

User: {question}
Assistant:"""

def build_prompt(question: str) -> str:
    """
    01_run_probe 内で完結するプロンプト整形関数。
    question: 純粋な質問文 (prompts の各要素) を受け取り、
              上記テンプレートにはめ込んだ最終プロンプト文字列を返す。
    """
    return PROMPT_TEMPLATE.format(question=question)

# ---- 引数ユーティリティ ----
def parse_floats_csv(s: str):
    """カンマ区切りの浮動小数点数をパースする"""
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip()]

def build_e(trait: str, sign: float):
    """操舵方向の辞書 e を作成する (generate_with_steer 互換)"""
    e = {ax: 0.0 for ax in AXES}
    e[trait] = 1.0 if sign >= 0 else -1.0
    return e

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
    
    v_axes: load_axes_bank で読み込んだ辞書
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
    text: str,
    *,
    layers: Sequence[int],         # state layer index (L>0)
    V_axes: torch.Tensor,           # (n_layers, H) 正規化済み
    layer2i: dict[int, int],
    max_len_margin: int = 10,
) -> dict[int, float]:
    """
    text を 1回 forward して、指定 layers すべての
    「最終有効トークン hidden」を軸に射影した cos を返す。
    戻り値: {L: cos_sim}
    """
    if not text or not text.strip():
        return {int(L): 0.0 for L in layers}

    dev = _infer_main_device(model)
    max_len = getattr(model.config, "max_position_embeddings", 4096) - max_len_margin

    tokd = tok(text, return_tensors="pt", max_length=max_len, truncation=True).to(dev)
    out = model(**tokd, output_hidden_states=True, use_cache=False)

    # 最終有効トークン index（pad を除外）
    att = tokd["attention_mask"][0].bool()  # (T,)
    idxs = torch.nonzero(att, as_tuple=False).view(-1)
    if idxs.numel() == 0:
        return {int(L): 0.0 for L in layers}
    last_idx = int(idxs[-1].item())

    # 各層の last token hidden を積む
    hs_list = []
    for L in layers:
        if L >= len(out.hidden_states):
            # 念のため
            hs_list.append(torch.zeros_like(V_axes[0]))
            continue
        hsL = out.hidden_states[L][0, last_idx, :].to(dtype=torch.float32)  # (H,)
        hs_list.append(hsL)

    H = torch.stack(hs_list, dim=0)  # (n_layers, H)
    H = H / (H.norm(p=2, dim=1, keepdim=True) + 1e-12)  # 行ごと正規化

    # cos: (n_layers,)
    cos = torch.sum(H * V_axes, dim=1).detach().cpu().numpy().tolist()

    return {int(L): float(cos[layer2i[L]]) for L in layers}


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
    ap.add_argument("--rank", type=int, default=1, help="mix_top_k (単軸検証は1推奨)")
    ap.add_argument("--samples", type=int, default=12, help="プローブ用プロンプトの数")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.90)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--include_prefill", action="store_true", help="prefill(プロンプト処理)段階にも介入/ログする。デフォルトはdecode(T=1)のみ")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="exp/<trait>_probe_all_layers.jsonl")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    layers_arg = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None
    alphas = parse_floats_csv(args.alpha_list)
    if 0.0 not in alphas:
        alphas = sorted(alphas + [0.0])

    print(f"Target trait: {args.trait}")
    print(f"Target alphas (total): {alphas}")
    print(f"Alpha mode: {args.alpha_mode}")

    mdl, tok = load_model_and_tokenizer(
        args.model, quant="8bit", device_map="auto"
    )
    device = _infer_main_device(mdl)

    axes_bank = load_axes_bank(args.axes_bank)
    
    all_available_layers = sorted(list(set([
        L for (L, ax) in axes_bank.keys() if ax == args.trait and L > 0
    ])))
    
    if layers_arg:
        layers_to_process = [L for L in all_available_layers if L in layers_arg]
        print(f"Filtering to --layers argument. Processing {len(layers_to_process)} layers: {layers_to_process}")
    else:
        layers_to_process = all_available_layers
        print(f"No --layers specified. Processing ALL {len(layers_to_process)} available layers (L>0): {layers_to_process}")
        
    if not layers_to_process:
        raise ValueError(f"No valid layers found for trait '{args.trait}' in axes_bank. Check --layers or 00_prepare_vectors.py.")
        
    num_steered_layers = len(layers_to_process)
    V_axes, layer2i = _prepare_axes_tensor(
        axes_bank, layers_to_process, args.trait, device=device
    )
    out_path = args.out.replace("<trait>", args.trait)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    collector = RMSCollector()

    with open(out_path, "w", encoding="utf-8") as fout:
        
        print(f"Starting [probe] mode. Simultaneous intervention on {num_steered_layers} layers.")
        
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

        # --- ベースライン (alpha=0) ---
        baseline_ys = {}
        baseline_rms = {}
        print("Generating baseline (alpha=0) responses...")
        for i, x in enumerate(prompts):
            full_prompt = build_prompt(x)
            inputs = tok(full_prompt, return_tensors="pt").to(device)
            inputs.pop("token_type_ids", None)

            collector.reset()
            with ExitStack() as stack:
                for L_state in layers_to_process:
                    L_stack = L_state - 1
                    v_key = (L_state, args.trait)
                    if v_key not in axes_bank:
                        continue
                    vec = axes_bank[v_key]
                    stack.enter_context(
                        ResidualSteererLogged(
                            mdl, L_stack, vec, alpha=0.0,  # ★alpha=0でログだけ取る
                            answer_only=(not args.include_prefill),
                            collector=collector,
                            layer_key=L_state,
                        )
                    )
                out_ids = mdl.generate(**inputs, **gen_kwargs)
            
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[0, prompt_len:]
            y0 = tok.decode(gen_ids, skip_special_tokens=True).strip()
            baseline_ys[i] = y0
            baseline_rms[i] = collector.snapshot(alpha_per_layer=0.0)

        print("Generating steered responses (ALL-LAYER intervention)...")
        for i, x in enumerate(prompts):
            print(f"  Probe sample {i+1}/{len(prompts)}...")
            y0 = baseline_ys[i]
            full_prompt = build_prompt(x)
            inputs = tok(full_prompt, return_tensors="pt").to(device)
            inputs.pop("token_type_ids", None)
            baseline_scores = style_from_text_all_layers(
                mdl, tok, y0,
                layers=layers_to_process,
                V_axes=V_axes,
                layer2i=layer2i
            )
            s0_avg = np.mean(list(baseline_scores.values())) if baseline_scores else 0.0

            for a_total in alphas:
                if a_total == 0.0:
                    rec = {
                        "i": i, "trait": args.trait, "layers": layers_to_process,
                        "alpha_total": 0.0, "alpha_mode": args.alpha_mode,
                        "alpha_per_layer": 0.0,
                        "x": x, "y": y0,
                        "s_avg": s0_avg, "s0_avg": s0_avg, "ds_avg": 0.0,
                        "s_by_layer": baseline_scores,
                        "rms_by_layer": baseline_rms[i],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                if args.alpha_mode == "distribute":
                    alpha_per_layer = a_total / num_steered_layers
                else:
                    alpha_per_layer = a_total

                L_state = None
                y = ""
                try:
                    collector.reset()
                    with ExitStack() as stack:
                        for L_state in layers_to_process:
                            L_stack = L_state - 1
                            
                            v_key = (L_state, args.trait)
                            if v_key not in axes_bank:
                                print(f"Warning: Skipping state {L_state}, vector not in bank.")
                                continue
                                
                            vec = axes_bank[v_key]
                            
                            stack.enter_context(
                                ResidualSteererLogged(
                                    mdl, L_stack, vec, alpha_per_layer,
                                    answer_only=(not args.include_prefill),
                                    collector=collector,
                                    layer_key=L_state,
                                )
                            )

                        out_ids = mdl.generate(**inputs, **gen_kwargs)
                    
                    prompt_len = inputs["input_ids"].shape[1]
                    gen_ids = out_ids[0, prompt_len:]
                    y = tok.decode(gen_ids, skip_special_tokens=True).strip()
                
                except Exception as e:
                    print(f"!! ERROR during generation (alpha={a_total}, last_L_state={L_state}): {e}")
                    y = f"GENERATION_ERROR: {e}"

                steered_scores = {}
                if y.strip() and "GENERATION_ERROR" not in y:
                    steered_scores = style_from_text_all_layers(
                        mdl, tok, y,
                        layers=layers_to_process,
                        V_axes=V_axes,
                        layer2i=layer2i
                    )

                s_avg = np.mean(list(steered_scores.values())) if steered_scores else 0.0

                rec = {
                    "i": i, "trait": args.trait, "layers": layers_to_process,
                    "alpha_total": float(a_total),
                    "alpha_mode": args.alpha_mode,
                    "alpha_per_layer": float(alpha_per_layer),
                    "x": x, "y": y,
                    "s_avg": s_avg,
                    "s0_avg": s0_avg,
                    "ds_avg": s_avg - s0_avg,
                    "s_by_layer": steered_scores,
                    "rms_by_layer": collector.snapshot(alpha_per_layer),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            fout.flush()

    print(f"[done/probe_all_layers] wrote {out_path} (layers={layers_to_process}, trait={args.trait})")

if __name__ == "__main__":
    main()
