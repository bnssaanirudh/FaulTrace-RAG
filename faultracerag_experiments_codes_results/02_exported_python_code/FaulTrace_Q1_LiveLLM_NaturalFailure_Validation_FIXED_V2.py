"""Exported code cells from FaulTrace_Q1_LiveLLM_NaturalFailure_Validation_FIXED_V2.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
# ============================================================
# 0. CONFIGURATION
# ============================================================

FAST_MODE = False
PAPER_MODE = True

# 8+8 examples per model for a smoke test, 50+50 for the paper run.
N_PER_DATASET = 8 if FAST_MODE else 50

DATASETS_TO_RUN = [
    "hotpotqa",
    "2wikimultihopqa",
]

MODEL_SPECS = [
    {
        "name": "Qwen2.5-3B-Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": False,
        "enabled": True,
    },
    {
        "name": "Phi-4-mini-instruct",
        "model_id": "microsoft/Phi-4-mini-instruct",
        # Use Transformers' native Phi-3 implementation.
        # The repo-side custom modeling_phi3.py can be incompatible with
        # some Colab Transformers builds (e.g. missing LossKwargs).
        "trust_remote_code": False,
        "enabled": True,
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "enabled": True,
    },
]

TOP_K_SENTENCES = 8
MAX_INPUT_TOKENS = 4096
MAX_EXTRACT_TOKENS = 128
MAX_ANSWER_TOKENS = 40

USE_4BIT = True
FAILURE_F1_THRESHOLD = 0.80
MCR_SUCCESS_THRESHOLD = 0.80

FORCE_RERUN = False
HUMAN_AUDIT_N = 50

SEED = 20260914

print("Examples/model =", N_PER_DATASET * len(DATASETS_TO_RUN))
print("Models enabled =", [m["name"] for m in MODEL_SPECS if m["enabled"]])

# ============================================================
# NOTEBOOK CELL 3 / CODE CELL 2
# ============================================================
%%capture
!pip -q install -U \
    "datasets>=3.0" \
    "transformers>=4.48" \
    "accelerate>=1.2" \
    "bitsandbytes>=0.45" \
    "rank-bm25>=0.2.2" \
    "huggingface_hub>=0.27" \
    "pandas>=2.0" \
    "pyarrow>=15" \
    "scipy>=1.11" \
    "tqdm>=4.66" \
    "matplotlib>=3.8"

# ============================================================
# NOTEBOOK CELL 5 / CODE CELL 3
# ============================================================
from __future__ import annotations

import os
import sys
import re
import gc
import json
import time
import math
import string
import random
import hashlib
import platform
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy import stats

import torch

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/FaulTrace_RAG_Experiments")
except Exception:
    DRIVE = Path("/content/FaulTrace_RAG_Experiments")

ROOT = DRIVE / "LIVE_LLM_NATURAL_FAILURE_VALIDATION"
CACHE = ROOT / "cache"
RAW = ROOT / "raw_examples"
TABLES = ROOT / "paper_tables"
PLOTS = ROOT / "plots"
AUDIT = ROOT / "human_audit"
EXPORT = ROOT / "github_export"

for p in [ROOT, CACHE, RAW, TABLES, PLOTS, AUDIT, EXPORT]:
    p.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is strongly recommended for this notebook. "
        "In Colab choose Runtime > Change runtime type > T4/L4/A100 GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))

# Snapshot the FaulTrace source version used for the experiment.
REPO = Path("/content/FaulTrace-RAG")

if REPO.exists():
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin"], check=False)
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=False)
else:
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/bnssaanirudh/FaulTrace-RAG.git",
            str(REPO),
        ],
        check=True,
    )

COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"],
    text=True,
).strip()

print("FaulTrace commit:", COMMIT)

(ROOT / "environment.txt").write_text(
    f"utc={datetime.now(timezone.utc).isoformat()}\n"
    f"python={sys.version}\n"
    f"platform={platform.platform()}\n"
    f"torch={torch.__version__}\n"
    f"cuda={torch.version.cuda}\n"
    f"gpu={torch.cuda.get_device_name(0)}\n"
    f"faultrace_commit={COMMIT}\n"
)

# ============================================================
# NOTEBOOK CELL 7 / CODE CELL 4
# ============================================================
# ============================================================
# CORE FUNCTIONS
# ============================================================

def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = "".join(
        ch if ch not in string.punctuation else " "
        for ch in s
    )
    tokens = [
        t for t in s.split()
        if t not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> float:
    return float(
        normalize_answer(pred)
        ==
        normalize_answer(gold)
    )


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()

    if not p and not g:
        return 1.0

    if not p or not g:
        return 0.0

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def best_answer_scores(pred: str, gold_answers):
    gold_answers = [
        str(x)
        for x in gold_answers
        if str(x).strip()
    ] or [""]

    em = max(
        exact_match(pred, g)
        for g in gold_answers
    )

    f1 = max(
        token_f1(pred, g)
        for g in gold_answers
    )

    return float(em), float(f1)


def two_player_shapley(values):
    empty = frozenset()
    R = frozenset({"R"})
    E = frozenset({"E"})
    RE = frozenset({"R", "E"})

    phi_r = 0.5 * (
        (values[R] - values[empty])
        +
        (values[RE] - values[E])
    )

    phi_e = 0.5 * (
        (values[E] - values[empty])
        +
        (values[RE] - values[R])
    )

    return {
        "R": float(phi_r),
        "E": float(phi_e),
    }


def pair_interaction(values):
    return float(
        values[frozenset({"R", "E"})]
        - values[frozenset({"R"})]
        - values[frozenset({"E"})]
        + values[frozenset()]
    )


def minimal_repair(scores, threshold=MCR_SUCCESS_THRESHOLD):
    valid = [
        S for S, score in scores.items()
        if float(score) >= threshold
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda S:(
            len(S),
            tuple(sorted(S))
        )
    )


# ============================================================
# REGRESSION TESTS
# ============================================================

assert normalize_answer(
    "The, Eiffel Tower!"
) == "eiffel tower"

assert token_f1(
    "The Eiffel Tower",
    "eiffel tower"
) == 1.0

assert abs(
    token_f1(
        "Paris France",
        "Paris"
    )
    -
    (2 / 3)
) < 1e-9

_test_values = {
    frozenset(): 0.0,
    frozenset({"R"}): 0.2,
    frozenset({"E"}): 0.4,
    frozenset({"R", "E"}): 0.9,
}

_test_phi = two_player_shapley(
    _test_values
)

assert abs(
    _test_phi["R"]
    +
    _test_phi["E"]
    -
    0.9
) < 1e-12

_test_scores = {
    frozenset(): 0.2,
    frozenset({"R"}): 0.85,
    frozenset({"E"}): 0.70,
    frozenset({"R", "E"}): 0.95,
}

assert minimal_repair(
    _test_scores,
    0.80
) == frozenset({"R"})

print("Core regression tests: PASS")

# ============================================================
# NOTEBOOK CELL 9 / CODE CELL 5
# ============================================================
from datasets import load_dataset


def deterministic_take(ds, n, seed):
    n = min(n, len(ds))

    rg = np.random.default_rng(seed)

    idx = sorted(
        rg.choice(
            len(ds),
            size=n,
            replace=False,
        ).tolist()
    )

    return ds.select(idx)


def normalize_hotpot(ex):
    ctx = ex["context"]

    docs = {
        str(title): [
            str(sentence)
            for sentence in sentences
        ]
        for title, sentences in zip(
            ctx["title"],
            ctx["sentences"],
        )
    }

    sf = ex["supporting_facts"]

    supports = {
        (str(title), int(sent_id))
        for title, sent_id in zip(
            sf["title"],
            sf["sent_id"],
        )
    }

    return {
        "id": str(ex["id"]),
        "question": str(ex["question"]),
        "answers": [
            str(ex.get("answer", ""))
        ],
        "docs": docs,
        "supports": supports,
    }


def normalize_2wiki(ex):
    m = ex.get("metadata", ex)

    ctx = m["context"]

    sentence_lists = ctx.get("content")

    if sentence_lists is None:
        sentence_lists = ctx.get(
            "sentences"
        )

    if sentence_lists is None:
        raise KeyError(
            "Unsupported 2Wiki context schema. "
            f"Available keys: {list(ctx.keys())}"
        )

    titles = ctx.get("title")

    if titles is None:
        raise KeyError(
            "2Wiki context is missing title. "
            f"Available keys: {list(ctx.keys())}"
        )

    docs = {
        str(title): [
            str(sentence)
            for sentence in sentences
        ]
        for title, sentences in zip(
            titles,
            sentence_lists,
        )
    }

    sf = m.get(
        "supporting_facts",
        ex.get(
            "supporting_facts",
            {}
        ),
    )

    supports = {
        (str(title), int(sent_id))
        for title, sent_id in zip(
            sf.get("title", []),
            sf.get("sent_id", []),
        )
    }

    answers = ex.get(
        "golden_answers"
    )

    if answers is None:
        a = ex.get("answer", "")
        answers = (
            a
            if isinstance(a, list)
            else [a]
        )

    return {
        "id": str(ex["id"]),
        "question": str(ex["question"]),
        "answers": [
            str(x)
            for x in answers
        ] or [""],
        "docs": docs,
        "supports": supports,
    }


def load_benchmarks():
    out = {}

    if "hotpotqa" in DATASETS_TO_RUN:
        hp = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
        )

        hp = deterministic_take(
            hp,
            N_PER_DATASET,
            SEED,
        )

        out["hotpotqa"] = [
            normalize_hotpot(x)
            for x in hp
        ]

    if "2wikimultihopqa" in DATASETS_TO_RUN:
        tw = load_dataset(
            "cmriat/2wikimultihopqa",
            split="validation",
        )

        tw = deterministic_take(
            tw,
            N_PER_DATASET,
            SEED + 1,
        )

        normalized = []

        for x in tw:
            ex = normalize_2wiki(x)

            if ex["supports"]:
                normalized.append(ex)

        out["2wikimultihopqa"] = normalized

    return out


BENCHMARKS = load_benchmarks()

for name, examples in BENCHMARKS.items():
    print(
        name,
        "examples =",
        len(examples),
        "| first supports =",
        len(examples[0]["supports"]),
    )

# ============================================================
# NOTEBOOK CELL 11 / CODE CELL 6
# ============================================================
from rank_bm25 import BM25Okapi


def words(text):
    return re.findall(
        r"[A-Za-z0-9]+",
        str(text).lower(),
    )


def flatten_context(ex):
    rows = []

    for title, sentences in ex["docs"].items():
        for i, sentence in enumerate(sentences):
            rows.append({
                "title": title,
                "sent_id": i,
                "key": (title, i),
                "text": sentence,
            })

    return rows


def bm25_retrieve(ex, top_k=TOP_K_SENTENCES):
    rows = flatten_context(ex)

    if not rows:
        return []

    bm25 = BM25Okapi(
        [
            words(r["text"])
            for r in rows
        ]
    )

    scores = np.asarray(
        bm25.get_scores(
            words(ex["question"])
        ),
        dtype=float,
    )

    order = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        rows[int(i)]
        for i in order
    ]


def oracle_retrieve(ex):
    flat = {
        r["key"]: r
        for r in flatten_context(ex)
    }

    return [
        flat[key]
        for key in sorted(ex["supports"])
        if key in flat
    ]


def oracle_extract_from_retrieved(
    retrieved,
    ex,
):
    support = ex["supports"]

    rows = [
        r
        for r in retrieved
        if r["key"] in support
    ]

    if not rows:
        return (
            "<NO_GOLD_SUPPORT_PRESENT_"
            "IN_RETRIEVED_CONTEXT>"
        )

    return "\n".join(
        f"- [{r['title']} #{r['sent_id']}] "
        f"{r['text']}"
        for r in rows
    )


def render_passages(rows):
    if not rows:
        return "<NO_PASSAGES>"

    return "\n\n".join(
        (
            f"[{i+1}] "
            f"{r['title']} "
            f"(sentence {r['sent_id']}): "
            f"{r['text']}"
        )
        for i, r in enumerate(rows)
    )


# Sanity check on support-preserving oracle extraction.
for dataset, examples in BENCHMARKS.items():
    ex = examples[0]

    oracle_r = oracle_retrieve(ex)

    assert len(oracle_r) > 0
    assert all(
        r["key"] in ex["supports"]
        for r in oracle_r
    )

print("Retrieval/oracle sanity tests: PASS")

# ============================================================
# NOTEBOOK CELL 13 / CODE CELL 7
# ============================================================
# ============================================================
# PHI / TRANSFORMERS COMPATIBILITY PREFLIGHT
# ============================================================

import transformers

print("transformers version:", transformers.__version__)

try:
    from transformers import Phi3ForCausalLM
    print("Native Phi3ForCausalLM support: PASS")
except Exception as e:
    raise RuntimeError(
        "This runtime does not expose native Phi3ForCausalLM. "
        "Restart the runtime after the dependency-install cell, then rerun "
        "from the imports cell. Original import error: " + repr(e)
    )

phi_spec = next(
    x for x in MODEL_SPECS
    if x["model_id"] == "microsoft/Phi-4-mini-instruct"
)

assert phi_spec["trust_remote_code"] is False

print("Phi native-loading configuration: PASS")
print("Existing Qwen JSON checkpoints will be reused automatically.")

# ============================================================
# NOTEBOOK CELL 14 / CODE CELL 8
# ============================================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from huggingface_hub import HfApi


def release_model(
    model=None,
    tokenizer=None,
):
    try:
        del model
    except Exception:
        pass

    try:
        del tokenizer
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_manifest_entry(spec):
    row = {
        "name": spec["name"],
        "model_id": spec["model_id"],
    }

    try:
        info = HfApi().model_info(
            spec["model_id"]
        )
        row["revision"] = info.sha
    except Exception as e:
        row["revision"] = "unresolved"
        row["revision_error"] = repr(e)

    return row


MODEL_MANIFEST = [
    model_manifest_entry(spec)
    for spec in MODEL_SPECS
    if spec["enabled"]
]

(ROOT / "model_manifest.json").write_text(
    json.dumps(
        MODEL_MANIFEST,
        indent=2,
    )
)

display(
    pd.DataFrame(
        MODEL_MANIFEST
    )
)


def load_model(spec):
    print(
        "\nLoading:",
        spec["model_id"],
    )

    print(
        "Loader mode:",
        "remote custom code"
        if spec["trust_remote_code"]
        else "native Transformers architecture",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        trust_remote_code=spec[
            "trust_remote_code"
        ],
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": spec[
            "trust_remote_code"
        ],
        "low_cpu_mem_usage": True,
    }

    if USE_4BIT:
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )
        )
    else:
        kwargs["torch_dtype"] = (
            torch.float16
        )

    try:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                spec["model_id"],
                **kwargs,
            )
        )

    except Exception as e:
        if not USE_4BIT:
            raise

        print(
            "4-bit load failed; "
            "retrying float16:",
            repr(e),
        )

        kwargs.pop(
            "quantization_config",
            None,
        )

        kwargs["torch_dtype"] = (
            torch.float16
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                spec["model_id"],
                **kwargs,
            )
        )

    model.eval()

    return tokenizer, model


def model_device(model):
    try:
        return next(
            model.parameters()
        ).device
    except Exception:
        return torch.device("cuda:0")


@torch.inference_mode()
def generate_chat(
    tokenizer,
    model,
    messages,
    max_new_tokens,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    dev = model_device(model)

    inputs = {
        k: v.to(dev)
        for k, v in inputs.items()
    }

    in_tokens = int(
        inputs["input_ids"].shape[-1]
    )

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=(
            tokenizer.pad_token_id
        ),
        eos_token_id=(
            tokenizer.eos_token_id
        ),
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    new_ids = outputs[
        0,
        in_tokens:
    ]

    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()

    out_tokens = int(
        new_ids.shape[-1]
    )

    return {
        "text": text,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "latency_s": elapsed,
    }

# ============================================================
# NOTEBOOK CELL 16 / CODE CELL 9
# ============================================================
EXTRACT_SYSTEM = """You are an evidence-extraction component in a RAG pipeline.
Extract only facts from the supplied passages that are directly useful for answering the question.
Do not use outside knowledge.
Do not answer the question directly.
Return short evidence statements only."""

ANSWER_SYSTEM = """You are the final answer-synthesis component in a RAG pipeline.
Use only the supplied extracted evidence.
Answer the question with the shortest correct phrase possible.
Do not explain your reasoning.
If the evidence is genuinely insufficient, answer exactly INSUFFICIENT_EVIDENCE."""


def extract_with_model(
    tokenizer,
    model,
    question,
    passages,
):
    user = (
        f"Question:\n{question}\n\n"
        f"Passages:\n{render_passages(passages)}\n\n"
        "Extract the minimal evidence."
    )

    return generate_chat(
        tokenizer,
        model,
        [
            {
                "role": "system",
                "content": EXTRACT_SYSTEM,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        MAX_EXTRACT_TOKENS,
    )


def answer_with_model(
    tokenizer,
    model,
    question,
    evidence,
):
    user = (
        f"Question:\n{question}\n\n"
        f"Extracted evidence:\n{evidence}\n\n"
        "Answer:"
    )

    return generate_chat(
        tokenizer,
        model,
        [
            {
                "role": "system",
                "content": ANSWER_SYSTEM,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        MAX_ANSWER_TOKENS,
    )


def evaluate_example(
    tokenizer,
    model,
    ex,
):
    # --------------------------------------------------------
    # Retrieval states
    # --------------------------------------------------------

    r0 = bm25_retrieve(
        ex,
        TOP_K_SENTENCES,
    )

    rR = oracle_retrieve(
        ex
    )

    # --------------------------------------------------------
    # Model extraction under baseline and repaired retrieval
    # --------------------------------------------------------

    e0_call = extract_with_model(
        tokenizer,
        model,
        ex["question"],
        r0,
    )

    eR_call = extract_with_model(
        tokenizer,
        model,
        ex["question"],
        rR,
    )

    # --------------------------------------------------------
    # Oracle extraction restricted to evidence actually
    # present in each retrieval state.
    # --------------------------------------------------------

    eE = oracle_extract_from_retrieved(
        r0,
        ex,
    )

    eRE = oracle_extract_from_retrieved(
        rR,
        ex,
    )

    # --------------------------------------------------------
    # Same answer model in all four worlds.
    # No gold-answer oracle is used.
    # --------------------------------------------------------

    a0 = answer_with_model(
        tokenizer,
        model,
        ex["question"],
        e0_call["text"],
    )

    aR = answer_with_model(
        tokenizer,
        model,
        ex["question"],
        eR_call["text"],
    )

    aE = answer_with_model(
        tokenizer,
        model,
        ex["question"],
        eE,
    )

    aRE = answer_with_model(
        tokenizer,
        model,
        ex["question"],
        eRE,
    )

    worlds = {
        frozenset(): a0["text"],
        frozenset({"R"}): aR["text"],
        frozenset({"E"}): aE["text"],
        frozenset({"R", "E"}): aRE["text"],
    }

    scores = {}

    for S, answer in worlds.items():
        em, f1 = best_answer_scores(
            answer,
            ex["answers"],
        )

        scores[S] = {
            "em": em,
            "f1": f1,
            "loss": 1.0 - f1,
        }

    baseline_loss = scores[
        frozenset()
    ]["loss"]

    values = {
        S: (
            baseline_loss
            -
            result["loss"]
        )
        for S, result in scores.items()
    }

    phi = two_player_shapley(
        values
    )

    interaction = pair_interaction(
        values
    )

    repair_scores = {
        S: result["f1"]
        for S, result in scores.items()
    }

    mcr = minimal_repair(
        repair_scores,
        MCR_SUCCESS_THRESHOLD,
    )

    natural_failure = (
        scores[frozenset()]["f1"]
        <
        FAILURE_F1_THRESHOLD
    )

    upstream_repairable = (
        natural_failure
        and
        scores[
            frozenset({"R", "E"})
        ]["f1"]
        >=
        MCR_SUCCESS_THRESHOLD
    )

    residual_generation_failure = (
        natural_failure
        and
        scores[
            frozenset({"R", "E"})
        ]["f1"]
        <
        MCR_SUCCESS_THRESHOLD
    )

    if not natural_failure:
        diagnosis = "none"

    elif residual_generation_failure:
        diagnosis = "G_residual"

    elif abs(phi["R"]) > abs(phi["E"]):
        diagnosis = "R"

    elif abs(phi["E"]) > abs(phi["R"]):
        diagnosis = "E"

    else:
        diagnosis = "R+E"

    calls = [
        e0_call,
        eR_call,
        a0,
        aR,
        aE,
        aRE,
    ]

    return {
        "retrieval_baseline": [
            {
                "title": r["title"],
                "sent_id": r["sent_id"],
                "text": r["text"],
                "is_support": (
                    r["key"]
                    in ex["supports"]
                ),
            }
            for r in r0
        ],
        "retrieval_oracle": [
            {
                "title": r["title"],
                "sent_id": r["sent_id"],
                "text": r["text"],
            }
            for r in rR
        ],
        "extraction_baseline": e0_call["text"],
        "extraction_retrieval_repaired": eR_call["text"],
        "extraction_oracle_on_baseline_retrieval": eE,
        "extraction_oracle_full": eRE,
        "answer_baseline": a0["text"],
        "answer_R": aR["text"],
        "answer_E": aE["text"],
        "answer_RE": aRE["text"],
        "em_baseline": scores[frozenset()]["em"],
        "f1_baseline": scores[frozenset()]["f1"],
        "em_R": scores[frozenset({"R"})]["em"],
        "f1_R": scores[frozenset({"R"})]["f1"],
        "em_E": scores[frozenset({"E"})]["em"],
        "f1_E": scores[frozenset({"E"})]["f1"],
        "em_RE": scores[frozenset({"R", "E"})]["em"],
        "f1_RE": scores[frozenset({"R", "E"})]["f1"],
        "phi_R": phi["R"],
        "phi_E": phi["E"],
        "interaction_RE": interaction,
        "mcr": (
            None
            if mcr is None
            else "+".join(
                sorted(mcr)
            )
            or "none"
        ),
        "natural_failure": bool(
            natural_failure
        ),
        "upstream_repairable": bool(
            upstream_repairable
        ),
        "residual_generation_failure": bool(
            residual_generation_failure
        ),
        "diagnosis": diagnosis,
        "total_input_tokens": int(
            sum(
                c["input_tokens"]
                for c in calls
            )
        ),
        "total_output_tokens": int(
            sum(
                c["output_tokens"]
                for c in calls
            )
        ),
        "total_latency_s": float(
            sum(
                c["latency_s"]
                for c in calls
            )
        ),
    }

# ============================================================
# NOTEBOOK CELL 18 / CODE CELL 10
# ============================================================
def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )


all_rows = []

for spec in MODEL_SPECS:

    if not spec["enabled"]:
        continue

    model_name = spec["name"]

    model_dir = (
        RAW
        /
        safe_name(model_name)
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.cuda.reset_peak_memory_stats()

    tokenizer = None
    model = None

    try:
        tokenizer, model = load_model(
            spec
        )

        for dataset_name, examples in BENCHMARKS.items():

            for ex in tqdm(
                examples,
                desc=(
                    f"{model_name} / "
                    f"{dataset_name}"
                ),
            ):
                uid = (
                    f"{dataset_name}"
                    f"__{ex['id']}"
                )

                out_path = (
                    model_dir
                    /
                    f"{safe_name(uid)}.json"
                )

                if (
                    out_path.exists()
                    and
                    not FORCE_RERUN
                ):
                    try:
                        row = json.loads(
                            out_path.read_text()
                        )

                        all_rows.append(
                            row
                        )

                        continue

                    except Exception:
                        pass

                result = evaluate_example(
                    tokenizer,
                    model,
                    ex,
                )

                row = {
                    "model": model_name,
                    "model_id": spec["model_id"],
                    "dataset": dataset_name,
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold_answers": ex["answers"],
                    **result,
                }

                tmp = (
                    out_path
                    .with_suffix(".tmp")
                )

                tmp.write_text(
                    json.dumps(
                        row,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                os.replace(
                    tmp,
                    out_path,
                )

                all_rows.append(
                    row
                )

        peak_mb = (
            torch.cuda.max_memory_allocated()
            /
            1024
            /
            1024
        )

        (
            model_dir
            /
            "gpu_peak_memory_mb.txt"
        ).write_text(
            str(peak_mb)
        )

        print(
            model_name,
            "peak allocated GPU MB:",
            round(peak_mb, 1),
        )

    finally:
        release_model(
            model,
            tokenizer,
        )


results = pd.DataFrame(
    all_rows
)

results.to_parquet(
    ROOT / "live_llm_all.parquet",
    index=False,
)

results.to_csv(
    ROOT / "live_llm_all.csv",
    index=False,
)

print(
    "Completed rows:",
    len(results),
)

display(
    results[
        [
            "model",
            "dataset",
            "f1_baseline",
            "f1_RE",
            "phi_R",
            "phi_E",
            "diagnosis",
        ]
    ].head()
)

# ============================================================
# NOTEBOOK CELL 20 / CODE CELL 11
# ============================================================
if len(results) == 0:
    raise RuntimeError(
        "No completed live-model results found."
    )

summary = (
    results
    .groupby(
        [
            "model",
            "dataset",
        ],
        as_index=False,
    )
    .agg(
        n=("id", "size"),
        baseline_em=("em_baseline", "mean"),
        baseline_f1=("f1_baseline", "mean"),
        R_f1=("f1_R", "mean"),
        E_f1=("f1_E", "mean"),
        RE_f1=("f1_RE", "mean"),
        natural_failure_rate=(
            "natural_failure",
            "mean",
        ),
        upstream_repairable_rate=(
            "upstream_repairable",
            "mean",
        ),
        residual_generation_failure_rate=(
            "residual_generation_failure",
            "mean",
        ),
        mean_phi_R=("phi_R", "mean"),
        mean_phi_E=("phi_E", "mean"),
        mean_interaction_RE=(
            "interaction_RE",
            "mean",
        ),
        mean_latency_s=(
            "total_latency_s",
            "mean",
        ),
        mean_input_tokens=(
            "total_input_tokens",
            "mean",
        ),
        mean_output_tokens=(
            "total_output_tokens",
            "mean",
        ),
    )
)

summary.to_csv(
    TABLES /
    "LIVE_LLM_model_dataset_summary.csv",
    index=False,
)

display(summary)


overall = (
    results
    .groupby(
        "model",
        as_index=False,
    )
    .agg(
        n=("id", "size"),
        baseline_em=("em_baseline", "mean"),
        baseline_f1=("f1_baseline", "mean"),
        R_f1=("f1_R", "mean"),
        E_f1=("f1_E", "mean"),
        RE_f1=("f1_RE", "mean"),
        natural_failure_rate=(
            "natural_failure",
            "mean",
        ),
        upstream_repairable_rate=(
            "upstream_repairable",
            "mean",
        ),
        residual_generation_failure_rate=(
            "residual_generation_failure",
            "mean",
        ),
        mean_phi_R=("phi_R", "mean"),
        mean_phi_E=("phi_E", "mean"),
        mean_interaction_RE=(
            "interaction_RE",
            "mean",
        ),
        mean_latency_s=(
            "total_latency_s",
            "mean",
        ),
        mean_input_tokens=(
            "total_input_tokens",
            "mean",
        ),
        mean_output_tokens=(
            "total_output_tokens",
            "mean",
        ),
    )
)

overall.to_csv(
    TABLES /
    "LIVE_LLM_overall_by_model.csv",
    index=False,
)

display(overall)

# ============================================================
# NOTEBOOK CELL 22 / CODE CELL 12
# ============================================================
failures = results[
    results["natural_failure"]
].copy()

print(
    "Natural failures:",
    len(failures),
    "/",
    len(results),
)

if len(failures) > 0:

    failure_summary = (
        failures
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            natural_failures=(
                "id",
                "size",
            ),
            baseline_f1=(
                "f1_baseline",
                "mean",
            ),
            R_f1=(
                "f1_R",
                "mean",
            ),
            E_f1=(
                "f1_E",
                "mean",
            ),
            RE_f1=(
                "f1_RE",
                "mean",
            ),
            repairable_fraction=(
                "upstream_repairable",
                "mean",
            ),
            residual_G_fraction=(
                "residual_generation_failure",
                "mean",
            ),
            mean_phi_R=(
                "phi_R",
                "mean",
            ),
            mean_phi_E=(
                "phi_E",
                "mean",
            ),
            interaction_RE=(
                "interaction_RE",
                "mean",
            ),
        )
    )

    failure_summary.to_csv(
        TABLES /
        "LIVE_LLM_natural_failures.csv",
        index=False,
    )

    display(
        failure_summary
    )


    mcr_table = (
        failures
        .assign(
            mcr_label=(
                failures["mcr"]
                .fillna("unrecoverable")
            )
        )
        .groupby(
            [
                "model",
                "mcr_label",
            ],
            as_index=False,
        )
        .size()
    )

    mcr_table["fraction"] = (
        mcr_table["size"]
        /
        mcr_table
        .groupby("model")["size"]
        .transform("sum")
    )

    mcr_table.to_csv(
        TABLES /
        "LIVE_LLM_MCR_distribution.csv",
        index=False,
    )

    display(
        mcr_table
    )

# ============================================================
# NOTEBOOK CELL 24 / CODE CELL 13
# ============================================================
def bootstrap_mean_ci(
    values,
    B=2000,
    seed=SEED,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    rg = np.random.default_rng(
        seed
    )

    means = np.empty(
        B,
        dtype=float,
    )

    for b in range(B):
        idx = rg.integers(
            0,
            len(values),
            len(values),
        )

        means[b] = (
            values[idx].mean()
        )

    return (
        float(values.mean()),
        float(
            np.quantile(
                means,
                .025,
            )
        ),
        float(
            np.quantile(
                means,
                .975,
            )
        ),
    )


ci_rows = []

for model_name, group in results.groupby("model"):

    for metric in [
        "f1_baseline",
        "f1_R",
        "f1_E",
        "f1_RE",
    ]:
        mean, lo, hi = bootstrap_mean_ci(
            group[metric].values
        )

        ci_rows.append({
            "model": model_name,
            "metric": metric,
            "mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })


ci_table = pd.DataFrame(
    ci_rows
)

ci_table.to_csv(
    TABLES /
    "LIVE_LLM_bootstrap_CI.csv",
    index=False,
)

display(
    ci_table
)


test_rows = []

for model_name, group in results.groupby("model"):

    diff = (
        group["f1_RE"]
        -
        group["f1_baseline"]
    ).values

    if np.allclose(
        diff,
        0,
    ):
        W = 0.0
        p = 1.0
    else:
        W, p = stats.wilcoxon(
            group["f1_RE"],
            group["f1_baseline"],
            zero_method="zsplit",
            alternative="two-sided",
        )

    # Exact paired EM improvement / degradation counts.
    improved = int(
        (
            (group["em_baseline"] == 0)
            &
            (group["em_RE"] == 1)
        ).sum()
    )

    degraded = int(
        (
            (group["em_baseline"] == 1)
            &
            (group["em_RE"] == 0)
        ).sum()
    )

    if improved + degraded:
        mcnemar_p = stats.binomtest(
            min(improved, degraded),
            improved + degraded,
            p=.5,
            alternative="two-sided",
        ).pvalue
    else:
        mcnemar_p = 1.0

    test_rows.append({
        "model": model_name,
        "n": len(group),
        "baseline_f1": (
            group["f1_baseline"].mean()
        ),
        "RE_f1": (
            group["f1_RE"].mean()
        ),
        "mean_f1_improvement": (
            diff.mean()
        ),
        "wilcoxon_W": W,
        "wilcoxon_p": p,
        "EM_improved_cases": improved,
        "EM_degraded_cases": degraded,
        "paired_exact_p": mcnemar_p,
    })


tests = pd.DataFrame(
    test_rows
)

tests.to_csv(
    TABLES /
    "LIVE_LLM_significance.csv",
    index=False,
)

display(
    tests
)

# ============================================================
# NOTEBOOK CELL 27 / CODE CELL 14
# ============================================================
import matplotlib.pyplot as plt

def safe_save_figure(stem, dpi=220):
    stem = Path(stem)

    # Always save the raster artifact first.
    png_path = stem.with_suffix(".png")
    plt.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    print("saved:", png_path)

    # PDF is optional because some Colab sessions can have mixed
    # Matplotlib modules after an in-place package upgrade.
    pdf_path = stem.with_suffix(".pdf")
    try:
        plt.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        print("saved:", pdf_path)
    except (ImportError, AttributeError) as e:
        print(
            "PDF backend unavailable in this runtime; "
            "PNG was saved successfully."
        )
        print("PDF backend error:", repr(e))


# ------------------------------------------------------------
# Baseline vs fully repaired upstream evidence
# ------------------------------------------------------------

plot_df = (
    results
    .groupby("model")[
        [
            "f1_baseline",
            "f1_RE",
        ]
    ]
    .mean()
)

ax = plot_df.plot(
    kind="bar",
    figsize=(8,4),
)

ax.set_ylim(0,1)
ax.set_ylabel("Answer token F1")
ax.set_xlabel("Live model")
ax.set_title(
    "Natural-error performance before and after R+E repair"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.tight_layout()

safe_save_figure(
    PLOTS /
    "LIVE_LLM_baseline_vs_RE"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# R/E attribution on natural failures
# ------------------------------------------------------------

if len(failures) > 0:

    attr = (
        failures
        .groupby("model")[
            [
                "phi_R",
                "phi_E",
            ]
        ]
        .mean()
    )

    ax = attr.plot(
        kind="bar",
        figsize=(8,4),
    )

    ax.set_ylabel(
        "Mean recoverable-loss attribution"
    )

    ax.set_xlabel(
        "Live model"
    )

    ax.set_title(
        "FaulTrace attribution on natural LLM failures"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    safe_save_figure(
        PLOTS /
        "LIVE_LLM_R_E_attribution"
    )

    plt.show()
    plt.close()

print("Paper figure section complete.")

# ============================================================
# NOTEBOOK CELL 29 / CODE CELL 15
# ============================================================
# A human audit is deliberately exported rather than fabricated.
# Fill human_stage_label with one of:
# R, E, G, compound, unclear
#
# This can later be compared against FaulTrace's diagnosis.

if len(failures) > 0:

    audit_parts = []

    models = sorted(
        failures["model"]
        .unique()
        .tolist()
    )

    per_model = max(
        1,
        math.ceil(
            HUMAN_AUDIT_N
            /
            len(models)
        ),
    )

    for model_name in models:

        g = failures[
            failures["model"]
            ==
            model_name
        ].copy()

        g = g.sort_values(
            [
                "f1_baseline",
                "f1_RE",
            ],
            ascending=[
                True,
                False,
            ],
        )

        audit_parts.append(
            g.head(
                per_model
            )
        )

    audit_df = (
        pd.concat(
            audit_parts,
            ignore_index=True,
        )
        .head(
            HUMAN_AUDIT_N
        )
        .copy()
    )

    audit_df["human_stage_label"] = ""
    audit_df["human_actionable"] = ""
    audit_df["human_notes"] = ""

    audit_columns = [
        "model",
        "dataset",
        "id",
        "question",
        "gold_answers",
        "answer_baseline",
        "answer_R",
        "answer_E",
        "answer_RE",
        "f1_baseline",
        "f1_R",
        "f1_E",
        "f1_RE",
        "phi_R",
        "phi_E",
        "interaction_RE",
        "mcr",
        "diagnosis",
        "human_stage_label",
        "human_actionable",
        "human_notes",
    ]

    audit_df[
        audit_columns
    ].to_csv(
        AUDIT /
        "LIVE_LLM_50_case_human_audit.csv",
        index=False,
    )

    print(
        "Human audit sheet:",
        AUDIT /
        "LIVE_LLM_50_case_human_audit.csv",
    )

# ============================================================
# NOTEBOOK CELL 31 / CODE CELL 16
# ============================================================
evidence_lines = [
    "# FaulTrace-RAG Live-LLM External Validation",
    "",
    f"FaulTrace source commit: `{COMMIT}`",
    f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
    "",
    "## Models",
]

for row in MODEL_MANIFEST:
    evidence_lines.append(
        f"- {row['name']}: `{row['model_id']}` @ `{row.get('revision','unresolved')}`"
    )

evidence_lines += [
    "",
    "## Experimental rule",
    "- Retrieval and extraction can be repaired counterfactually.",
    "- The final answer model is never replaced with the gold answer.",
    "- Therefore residual loss after R+E is reported as generation/reasoning residual.",
    "",
]

for _, row in overall.iterrows():
    evidence_lines += [
        f"## {row['model']}",
        f"- n: {int(row['n'])}",
        f"- baseline EM: {row['baseline_em']:.4f}",
        f"- baseline F1: {row['baseline_f1']:.4f}",
        f"- R-repaired F1: {row['R_f1']:.4f}",
        f"- E-repaired F1: {row['E_f1']:.4f}",
        f"- R+E-repaired F1: {row['RE_f1']:.4f}",
        f"- natural failure rate: {row['natural_failure_rate']:.4f}",
        f"- upstream-repairable rate (all cases): {row['upstream_repairable_rate']:.4f}",
        f"- residual generation-failure rate (all cases): {row['residual_generation_failure_rate']:.4f}",
        f"- mean phi_R: {row['mean_phi_R']:.4f}",
        f"- mean phi_E: {row['mean_phi_E']:.4f}",
        "",
    ]

evidence_lines += [
    "## Interpretation guardrail",
    "Natural failures do not provide independent component labels automatically.",
    "Use the exported human-audit sheet if the manuscript claims stage-label accuracy on natural failures.",
    "Without human labels, report counterfactual repairability, attribution, residual loss, and efficiency only.",
]

evidence_md = "\n".join(
    evidence_lines
)

(ROOT / "LIVE_LLM_FINAL_EVIDENCE.md").write_text(
    evidence_md
)

print(evidence_md)

# ============================================================
# NOTEBOOK CELL 33 / CODE CELL 17
# ============================================================
if EXPORT.exists():
    shutil.rmtree(
        EXPORT
    )

EXPORT.mkdir(
    parents=True,
    exist_ok=True,
)

for source in [
    ROOT / "live_llm_all.csv",
    ROOT / "live_llm_all.parquet",
    ROOT / "LIVE_LLM_FINAL_EVIDENCE.md",
    ROOT / "model_manifest.json",
    ROOT / "environment.txt",
]:
    if source.exists():
        shutil.copy2(
            source,
            EXPORT / source.name,
        )

for folder in [
    TABLES,
    PLOTS,
    AUDIT,
]:
    if folder.exists():
        shutil.copytree(
            folder,
            EXPORT / folder.name,
        )


(EXPORT / "README.md").write_text(
    f"""# FaulTrace-RAG Live-LLM Natural-Failure Validation

FaulTrace source commit: `{COMMIT}`

This experiment evaluates naturally occurring errors from real instruction-tuned
language models on HotpotQA and 2WikiMultihopQA.

Counterfactual repairs are restricted to retrieval (R) and extraction (E).
The answer model is never replaced by a gold-answer oracle, so residual loss
after R+E represents model-side answer synthesis / reasoning failure.

Use LIVE_LLM_FINAL_EVIDENCE.md for the headline experiment summary.
Use the human-audit CSV only after a human has actually filled the label columns.
"""
)

archive = (
    ROOT /
    "FaulTrace_LIVE_LLM_External_Validation_GitHub.zip"
)

if archive.exists():
    archive.unlink()

shutil.make_archive(
    str(
        archive.with_suffix("")
    ),
    "zip",
    EXPORT,
)

print(
    "Final GitHub package:",
    archive,
)
