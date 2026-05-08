# live_axes_and_hook.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
import os, re
from typing import Dict, List, Tuple, Iterable, Optional
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList, PreTrainedModel, PreTrainedTokenizerBase
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None

AXES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# === generation 引数をクリーンに組み立てる（greedy時は sampling系を外す） ===
def build_gen_kwargs(tokenizer, raw: dict | None):
    gen = dict(raw) if raw else {}
    # 安全な既定
    gen.setdefault("pad_token_id", tokenizer.eos_token_id)
    gen.setdefault("eos_token_id", tokenizer.eos_token_id)

    # do_sample の判定（temperature>0 または明示 do_sample=True）
    temp = float(gen.get("temperature", 0) or 0)
    do_sample = bool(gen.get("do_sample", False) or (temp > 0.0))

    if do_sample:
        gen["do_sample"] = True
        # top_p / temperature は sampling時のみ意味がある
        if "top_p" not in gen:
            gen["top_p"] = 0.9
        # top_k は指定があれば使う（なければ入れない）
    else:
        gen["do_sample"] = False
        gen.pop("temperature", None)
        gen.pop("top_p", None)
        gen.pop("top_k", None)
    return gen

def _is_bnb_quantized(model) -> bool:
    # transformers が付けるフラグ
    return bool(getattr(model, "is_loaded_in_8bit", False) or
                getattr(model, "is_loaded_in_4bit", False))

def _infer_main_device(model) -> torch.device:
    """
    device_map=auto でも安全に “入力を置く先” を推定する。
    まず埋め込みのデバイス、なければ最初のマップ値、最終手段で model.device/cuda/cpu。
    """
    def _norm_dev(x):
        # int/str の 0/“0” → "cuda:0"、"cuda" → "cuda:0"
        if isinstance(x, int):
            return torch.device(f"cuda:{x}") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(x, str):
            s = x.strip().lower()
            if s.isdigit():
                return torch.device(f"cuda:{s}") if torch.cuda.is_available() else torch.device("cpu")
            if s == "cuda":
                return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            try:
                return torch.device(s)
            except Exception:
                return torch.device("cpu")
        if isinstance(x, torch.device):
            return x
        return torch.device("cpu")

    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for key in ("model.embed_tokens", "model.decoder.embed_tokens",
                    "transformer.wte", "model.model.embed_tokens"):
            if key in model.hf_device_map:
                return _norm_dev(model.hf_device_map[key])
        first = next(iter(model.hf_device_map.values()))
        return _norm_dev(first)

    if hasattr(model, "device"):
        return _norm_dev(model.device)

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_pad_token(tokenizer, model=None):
    """
    - できるだけ『語彙サイズを変えずに』pad_token を確保する。
    - それでも新規トークンが必要な場合は、必ず model.resize_token_embeddings を呼んで
      model 側の埋め込み行列と vocab_size を同期させる。
    """
    # すでに pad_token があれば何もしない（vocab_size も変えない）
    if getattr(tokenizer, "pad_token", None) is not None:
        return tokenizer

    # eos_token があれば、それを pad_token として流用（語彙サイズは増えないので安全）
    if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # ここに来るモデルはかなりレア。
    # 新しい [PAD] トークンを追加するが、このときは必ず model 側もリサイズしたい。
    if model is None:
        raise ValueError(
            "[_ensure_pad_token] pad_token を新規追加する必要がありますが、"
            "model が None のため埋め込みを拡張できません。\n"
            "必ず `model` を渡して呼び出してください。"
        )

    num_before = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    num_after = len(tokenizer)

    if num_after != num_before:
        # 語彙サイズが増えたので、埋め込み行列も拡張する
        model.resize_token_embeddings(num_after)
        # config.vocab_size も更新しておくと後続のチェックがずれない
        if hasattr(model, "config"):
            model.config.vocab_size = num_after

    return tokenizer


def _ensure_dialog_tokens(tokenizer, model=None):
    """
    対話用の特別トークン <usr>, <asst> を語彙に追加（未登録なら）。
    追加した場合は、model に対して resize_token_embeddings を一度だけ呼ぶ。

    - すでにその文字列トークンが vocab に存在する場合は再追加しない。
    - 追加が発生したのに model が None の場合は例外で止める（不公平な状態を避ける）。
    """
    # 既にあるかどうかを vocab / convert_tokens_to_ids で確認
    to_add = []
    for sp in ["<usr>", "<asst>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(sp)
        except Exception:
            tid = None

        # unk と同じ / None の場合は、実質「未登録」とみなす
        if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
            to_add.append(sp)

    # 追加不要ならそのまま返す（vocab_size も変わらない）
    if not to_add:
        return tokenizer

    if model is None:
        raise ValueError(
            "[_ensure_dialog_tokens] <usr>, <asst> を追加する必要がありますが、"
            "model が None のため埋め込みを拡張できません。\n"
            "必ず `model` を渡して呼び出してください。"
        )

    num_before = len(tokenizer)
    # まとめて一回だけ追加（重複トークンが生まれないように）
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    num_after = len(tokenizer)

    if num_after != num_before:
        model.resize_token_embeddings(num_after)
        if hasattr(model, "config"):
            model.config.vocab_size = num_after

    return tokenizer


def _resolve_hf_token() -> str | None:
    """
    取得優先順位：
      1) 環境変数 HUGGINGFACE_HUB_TOKEN
      2) このファイル(scripts/)の一つ外側 (= プロジェクト直下) の .hf_token
      3) 見つからなければ None
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()

    # scripts/ の一つ外側をプロジェクト直下とみなす
    proj_root = Path(__file__).resolve().parent.parent
    tokfile = proj_root / ".hf_token"
    if tokfile.exists():
        try:
            return tokfile.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None

def _format_prompt(tokenizer, prompt: str) -> str:
    # 1) chat template が使えれば最優先（Instruct系モデル）
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # 2) フォールバック：シンプルな Instruct 風プロンプト
    return f"### Instruction:\n{prompt}\n\n### Response:\n"

# --- Any-size loader (fp16/bf16, 8bit/4bit) ---
def load_model_and_tokenizer(
    name: str,
    quant: Optional[str] = "auto",      # "auto" | "8bit" | "4bit" | None
    torch_dtype=None,                    # 例: torch.bfloat16
    device_map: str = "auto",            # 大きいモデルは基本 "auto"
):
    kwargs = {}
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif quant == "auto":
        # GPUがあれば bf16/float16 + device_map=auto
        if torch.cuda.is_available():
            kwargs.update(device_map=device_map, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        else:
            kwargs.update(device_map=None)  # CPUの場合
    else:
        if torch_dtype is not None:
            kwargs.update(torch_dtype=torch_dtype)
        kwargs.update(device_map=device_map)

    token = _resolve_hf_token()
    if token:
        if "token" not in kwargs:
            kwargs["token"] = token
        # 念のため旧キーが混在していたら消す
        kwargs.pop("use_auth_token", None)
        
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    tok = AutoTokenizer.from_pretrained(name, token=token) if token else AutoTokenizer.from_pretrained(name)
    tok = _ensure_pad_token(tok, model)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tok))  # ← eos が無い超レアケースのみ必要
    return model, tok

# --- get_layer_stack ---
def get_layer_stack(model):
    """
    各種モデルから「デコーダブロックのスタック」を取り出すヘルパ。
    戻り値: (layers_stack, num_layers, kind_str)
    """

    # 1) GPT-2 / Falcon 系: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        return layers, len(layers), "gpt2/falcon系"

    # 2) Gemma3 マルチモーダル版:
    #    Gemma3ForConditionalGeneration の場合は language_model.model.layers の中に
    #    テキスト用のブロックが入っていることが多い。
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        layers = model.language_model.model.layers
        return layers, len(layers), "gemma3-language_model系"

    # 3) LLaMA / Mistral / Gemma2 / Qwen2 / Gemma3(CausalLM) など:
    #    model.model.layers の中にブロックが並んでいるパターン
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        return layers, len(layers), "llama/mistral/gemma/qwen系"

    # 4) GPT-NeoX 系
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
        return layers, len(layers), "neox系"

    # 5) 最後のフォールバック: とりあえず model.layers だけ持っているモデル
    if hasattr(model, "layers"):
        layers = model.layers
        try:
            n = len(layers)
        except TypeError:
            n = None
        return layers, n, "generic_layers系"

    # ここまで来たら本当に未対応
    raise RuntimeError(f"未知の層スタックです（type={type(model)}）。対応を追加してください。")


# ---------- ユーティリティ ----------

# ★ 修正: _to_device 関数は _infer_main_device で代替されるため削除 (余分なコード)

def _mean_pool_last_token(hidden: torch.Tensor, attn_mask: torch.Tensor, mode: str = "last") -> torch.Tensor:
    """
    hidden: (B, T, H), attn_mask: (B, T)
    mode="last": 各サンプルの「最後の非PADトークン」のベクトルを集めて平均
    mode="mean": 各サンプルでマスク平均 → さらにバッチ平均
    """
    if mode == "mean":
        # マスク平均 → (B, H) → バッチ平均 (H)
        denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
        x = (hidden * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        return x.mean(dim=0)
    else:
        # 最後の非PADインデックスを取る
        idx = attn_mask.sum(dim=1) - 1  # (B,)
        gathered = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]  # (B, H)
        return gathered.mean(dim=0)  # (H,)


@torch.no_grad()
def _collect_layer_mean(
    model, tokenizer, texts: List[str], layer: int, device: torch.device,
    batch_size: int = 8, pooling: str = "asst", max_length: int = 512
) -> torch.Tensor:
    """
    指定 layer の hidden_states を取り出し、バッチ内平均＋全体平均で (H,) を返す。
    """
    vecs = []
    tokenizer = _ensure_pad_token(tokenizer, model)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        out = model(**tok, output_hidden_states=True, use_cache=False)
        # huggingface は hidden_states[0] = embeddings、[1..n] = 各層
        hs = out.hidden_states[layer + 1]  # (B, T, H)
        attn = tok["attention_mask"]
        if pooling == "asst":
            # <asst> 以降のみの平均（行ごとに位置を探す）
            asst_id = tokenizer.convert_tokens_to_ids("<asst>")
            # 見つからない行は全体平均のフォールバックにする
            v_list = []
            for b in range(hs.size(0)):
                ids_b = tok["input_ids"][b]
                idx = (ids_b == asst_id).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    j = int(idx[0].item())
                    mask_b = attn[b, j:].bool()
                    vecs_b = hs[b, j:][mask_b]
                    if vecs_b.numel() == 0:               # ★追加
                        mask_b2 = attn[b].bool()
                        vecs_b = hs[b][mask_b2]
                else:
                    mask_b = attn[b].bool()
                    vecs_b = hs[b][mask_b]
                v_b = vecs_b.mean(dim=0)  # (H,)
                v_list.append(v_b)
            v = torch.stack(v_list, dim=0).mean(dim=0)  # (H,)
        else:
            v = _mean_pool_last_token(hs, attn, mode=pooling)  # (H,)
        vecs.append(v)
    V = torch.stack(vecs, dim=0).mean(dim=0)  # (H,)
    V = V / (V.norm() + 1e-12)
    return V

@torch.no_grad()
def build_axes_for_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_list: Iterable[int] = (),
    pos_texts: Dict[str, List[str]] = None,
    neg_texts: Dict[str, List[str]] = None,
    pooling: str = "asst",
    batch_size: int = 8,
    max_length: int = 512,
):
    """
    対象LLMの各 layer で Big5 の +/− テキスト平均差を取り、正規化した軸ベクトルを返す。
    返り値: axes_by_layer: Dict[(layer, axis)] -> np.ndarray[H]
    """
    device = _infer_main_device(model)
    model.eval()
    tokenizer = _ensure_pad_token(tokenizer, model)

    if pooling == "asst":
        tokenizer = _ensure_dialog_tokens(tokenizer, model)

    # 事前チェック
    assert set(pos_texts.keys()) == set(AXES) and set(neg_texts.keys()) == set(AXES), "pos/neg_texts のキーが AXES と一致しません。"

    axes_by_layer: Dict[Tuple[int, str], np.ndarray] = {}
    for L in layer_list:
        for ax in AXES:
            v_pos = _collect_layer_mean(model, tokenizer, pos_texts[ax], L, device, batch_size, pooling, max_length)
            v_neg = _collect_layer_mean(model, tokenizer, neg_texts[ax], L, device, batch_size, pooling, max_length)
            v = (v_pos - v_neg)
            v = v / (v.norm() + 1e-12)
            v_np = v.detach().cpu().to(torch.float32).numpy()
            if not np.isfinite(v_np).all():
                raise ValueError(f"non-finite axis: layer={L}, axis={ax}")
            axes_by_layer[(L, ax)] = v_np
    return axes_by_layer


def make_vmix_live(
    axes_by_layer: Dict[Tuple[int, str], np.ndarray],
    layer: int,
    e: Dict[str, float],
    top_k: int = 1,
    temp: float = 1.0,
) -> np.ndarray:
    """
    e = u_now - s_now （あなたの既存パイプラインで計算済みの“望ましい方向”）
    → その符号と大きさで、対象LLM空間の軸を混合して v_mix を作る。
    """
    if e is None:
        e = {} # 空辞書
    
    # e.keys() (例: 'pc0', 'pc1'...) をイテレートする
    keys = [ax for ax in AXES if ax in e]
    items = sorted(
        [(ax, abs(float(e.get(ax, 0.0))), 1 if float(e.get(ax, 0.0)) >= 0 else -1) for ax in keys], 
        key=lambda t: t[1], 
        reverse=True
    )[:top_k]
    
    mags = np.array([m for _, m, _ in items], dtype=np.float32)
    if temp > 0:
        z = mags / max(temp, 1e-6)
        z = z - z.max()
        w = np.exp(z)
        w = w / (w.sum() + 1e-9)
    else:
        # 均等
        w = np.ones_like(mags) / len(mags)

    H = len(next(iter(axes_by_layer.values())))
    if len(items) == 0:
        return np.zeros((H,), dtype=np.float32)
    v = np.zeros((H,), dtype=np.float32)
    for (ax, mag, sign), wi in zip(items, w):
        v_ax = axes_by_layer[(layer, ax)]
        v += wi * v_ax

    v = v / (np.linalg.norm(v) + 1e-12)
    return v

@dataclass
class ResidualSteerer:
    """
    指定層の出力（hidden_states）に α·v_mix を加算するフック。
    ★修正: 演算を float32 で行い、NaN発生時の安全策を追加。
    """
    model: torch.nn.Module
    layer: int
    v_mix: np.ndarray
    alpha: float
    answer_only: bool = False

    def __post_init__(self):
        self.handle = None

    def __enter__(self):
        # v_mix をあらかじめ float32 で持っておく
        v = torch.tensor(self.v_mix, dtype=torch.float32)

        def hook(mod, inp, out):
            # hidden のテンソルを取り出す
            hs = out[0] if isinstance(out, tuple) else out
            
            # 入力がすでに死んでいる場合は何もしない
            if not torch.isfinite(hs).all():
                # ここでログを出しても良いですが、すでに壊れているのでスルーします
                return out

            # ★ 修正ポイント1: 計算はすべて float32 で行う
            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            
            # shape 合わせ
            add = v.to(device=hs.device).view(1, 1, -1)
            
            # ★ 回答生成ステップのみ適用するロジック
            if self.answer_only and hs.size(1) != 1:
                return out
            
            # 加算 (float32)
            steered_hs_f32 = hs_f32 + self.alpha * add
            
            # ★ 修正ポイント2: 書き戻す前に有限値チェック
            # もし加算で NaN/Inf になったら、ステアリングを諦めて元の値を返す（クラッシュ回避）
            if not torch.isfinite(steered_hs_f32).all():
                # 必要なら print(f"Warning: Steering caused NaN at layer {self.layer}, skipping.")
                return out

            # 元の型に戻す
            steered_hs = steered_hs_f32.to(orig_dtype)

            if isinstance(out, tuple):
                return (steered_hs, *out[1:])
            return steered_hs

        # モデルの層オブジェクトを推定
        stack, _, _ = get_layer_stack(self.model)
        target_mod = stack[self.layer]
        self.handle = target_mod.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            
class StopOnSeq(StoppingCriteria):
    def __init__(self, tokenizer, stop_texts):
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_texts]
    def __call__(self, input_ids, scores, **kwargs):
        for sid in self.stop_ids:
            if len(input_ids[0]) >= len(sid) and input_ids[0].tolist()[-len(sid):] == sid:
                return True
        return False


# ★★★ ここから修正 ★★★
@torch.no_grad()
def generate_with_steer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    axes_by_layer: Dict[Tuple[int, str], np.ndarray],
    layer: int,
    alpha: float,
    e: Dict[str, float],
    *,
    mix_top_k: int = 1,
    mix_temp: float = 1.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    **kwargs,
) -> str:
    # ---- 安全化：e は全軸キーを持たせ、alpha は強さ(絶対値)に統一 ----
    if e is None:
        e = {}
    alpha_val = float(alpha)
    # モデル/トークナイザ準備
    device = _infer_main_device(model)
    model.eval()
    tokenizer = _ensure_pad_token(tokenizer, model)

    # v_mix を作る（対象LLM空間）
    v_mix = make_vmix_live(axes_by_layer, layer, e, top_k=mix_top_k, temp=mix_temp)

    # 生成
    prompt_text = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ---- 生成引数：greedy/samplingを自動で切り分け ----
    do_sample = (temperature is not None) and (float(temperature) > 0)
    gen = build_gen_kwargs(tokenizer, {
        "max_new_tokens": max_new_tokens,
        "temperature":    temperature,           # 0 or None なら greedy になる
        "top_p":          top_p,
        "repetition_penalty":  repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "do_sample": do_sample,
        #"top_k":          gen_top_k,             # 指定があれば使う
    })
    
    # ★ 修正: 呼び出し元からの追加引数 (logits_processorなど) をマージ
    gen.update(kwargs)

    # ★ 回答以降のみ加算（CAA流）
    # 方向は e（v_mix 側）に持たせているので alpha は強さのみ
    with ResidualSteerer(model, layer, v_mix, alpha_val, answer_only=True):
        out_ids = model.generate(**inputs, **gen)


    # 2) 新規生成ぶんだけを取り出して表示
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text