"""Small shared statistics, RNG replay, and bounded selection rules."""
from collections import Counter
import hashlib
import math
import random
import statistics

import numpy as np
import torch


SAMPLERS = {"primary": (32, 1.5), "secondary": (64, 1.0)}
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
METRICS = ["gen_ppl", "entropy", "repeated_4gram_fraction", "adjacent_repetition_fraction",
           "duplicate_text_rate", "empty_fraction", "gpt2_length_before", "gpt2_length_after"]


def reset_rng(seed, rank=0):
    value = seed + rank * 1_000_003
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return torch.Generator(device="cpu").manual_seed(value)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def rng_hash(generator):
    return {"cpu": tensor_hash(torch.get_rng_state()), "generator": tensor_hash(generator.get_state()),
            "cuda": tensor_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}


def token_diagnostics(tokens):
    n = len(tokens)
    counts = np.array(list(Counter(tokens).values()), dtype=np.int64)
    probs = counts.astype(np.float32) / n if n else np.array([], dtype=np.float32)
    return {
        "entropy": float(-np.sum(probs * np.log(probs + 1e-10))) if n else None,
        "repeated_4gram_fraction": 1 - len(set(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))) / (n - 3) if n >= 4 else None,
        "adjacent_repetition_fraction": sum(a == b for a, b in zip(tokens, tokens[1:])) / (n - 1) if n >= 2 else None,
    }


def ppl_weights(tokens, padded_width, eos):
    # Match the repository metric, including its first-EOS padding weight.
    sequence = list(tokens) + [eos] * (padded_width - len(tokens))
    seen, weight = 0, 0
    for i, token in enumerate(sequence):
        seen += token == eos
        if i:
            weight += int(seen == 1) + int(token != eos)
    return weight


def summarize(values):
    values = list(values)
    if not values or any(v is None or not math.isfinite(v) for v in values):
        return {"mean": None, "sample_sd": None, "se": None, "n": len(values), "values": values}
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {"mean": statistics.mean(values), "sample_sd": sd,
            "se": sd / math.sqrt(len(values)) if sd is not None else None, "n": len(values), "values": values}


def mean_defined(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def score_texts(rows, scorer, t5_tokenizer):
    assert [row["id"] for row in rows] == list(range(len(rows)))
    texts = [row["generated"] for row in rows]
    assert all(isinstance(t, str) for t in texts)
    keep = [i for i, text in enumerate(texts) if text.strip()]
    full = scorer.tokenizer(texts, add_special_tokens=False)["input_ids"]
    t5_lengths = [len(ids) for ids in t5_tokenizer(texts, add_special_tokens=False)["input_ids"]]
    scorer.reset()
    scored = scorer.record_generative_perplexity([texts[i] for i in keep], max_length=1024, retokenize=True) if keep else None
    per_ppl = dict(zip(keep, scored["per_sample_ppl"])) if scored else {}
    padded_width = max((min(len(full[i]), 1024) for i in keep), default=0)
    per_sample = []
    for i, ids in enumerate(full):
        tokens = ids[:1024]
        eligible = i in per_ppl
        diag = token_diagnostics(tokens) if eligible else {k: None for k in token_diagnostics([])}
        value = per_ppl.get(i)
        valid_ppl = value is not None and math.isfinite(value)
        weight = ppl_weights(tokens, padded_width, scorer.tokenizer.eos_token_id) if eligible else 0
        per_sample.append({"id": i, "nonempty": eligible, "ppl": value if valid_ppl else None,
                           "ppl_token_weight": weight, "scoring_failure": None if valid_ppl or not eligible else "nonfinite_or_no_scored_tokens",
                           "gpt2_length_before": len(ids), "gpt2_length_after": len(tokens), "t5_retokenized_length": t5_lengths[i],
                           "text_sha256": hashlib.sha256(texts[i].encode()).hexdigest(), **diag})
    result = {"requested_count": len(rows), "generated_count": len(rows), "nonempty_count": len(keep),
              "ppl_scored_count": sum(r["ppl"] is not None for r in per_sample),
              "entropy_scored_count": sum(r["entropy"] is not None for r in per_sample),
              "scoring_failures": [r["id"] for r in per_sample if r["scoring_failure"]],
              "empty_sample_ids": [i for i in range(len(rows)) if i not in per_ppl],
              "empty_fraction": 1 - len(keep) / len(rows),
              "gen_ppl": scored["ppl"] if scored and math.isfinite(scored["ppl"]) else None,
              "entropy": scored["mean_entropy"] if scored else None,
              "duplicate_text_rate": 1 - len({texts[i] for i in keep}) / len(keep) if keep else None}
    for field in ("repeated_4gram_fraction", "adjacent_repetition_fraction", "gpt2_length_before", "gpt2_length_after", "t5_retokenized_length"):
        result[field] = mean_defined(r[field] for r in per_sample)
        result[field + "_defined_count"] = sum(r[field] is not None for r in per_sample)
    if keep:
        assert abs(result["entropy"] - mean_defined(r["entropy"] for r in per_sample)) < 2e-6
    if result["gen_ppl"] is not None and not result["scoring_failures"]:
        weighted = [r for r in per_sample if r["ppl"] is not None]
        rebuilt = math.exp(math.fsum(math.log(r["ppl"]) * r["ppl_token_weight"] for r in weighted) /
                           sum(r["ppl_token_weight"] for r in weighted))
        assert math.isclose(rebuilt, result["gen_ppl"], rel_tol=1e-9)
    return result, per_sample


def frontier(rows):
    valid = [r for r in rows if r["gen_ppl"] is not None and r["entropy"] is not None]
    return [r for r in valid if not any(
        q["gen_ppl"] <= r["gen_ppl"] and q["entropy"] >= r["entropy"] and
        (q["gen_ppl"] < r["gen_ppl"] or q["entropy"] > r["entropy"]) for q in valid)]


def refinements(rows, target=5.15):
    ordered = sorted((r for r in rows if 0.5 <= r["sc_cfg"] <= 5 and r["entropy"] is not None), key=lambda r: r["sc_cfg"])
    pairs = [(a, b) for a, b in zip(ordered, ordered[1:]) if (a["entropy"] - target) * (b["entropy"] - target) < 0]
    if not pairs:
        return {"values": [], "reason": "No adjacent in-range screening scales bracket entropy 5.15; no speculative in-range refinement added."}
    a, b = min(pairs, key=lambda pair: abs(pair[0]["entropy"] - pair[1]["entropy"]))
    interpolated = a["sc_cfg"] + (b["sc_cfg"] - a["sc_cfg"]) * (target - a["entropy"]) / (b["entropy"] - a["entropy"])
    existing = {r["sc_cfg"] for r in rows}
    values = sorted({round(interpolated, 4), round((a["sc_cfg"] + b["sc_cfg"]) / 2, 4)} - existing)
    values = [v for v in values if a["sc_cfg"] < v < b["sc_cfg"]][:2]
    return {"values": values, "reason": f"Refine the {a['sc_cfg']}–{b['sc_cfg']} in-range bracket around entropy 5.15: linear estimate and midpoint; interpolation only chooses a new measured setting."}


def select_confirmation(rows):
    nd = frontier(rows)
    assert nd, "No finite screening points available for selection"
    selected = []
    def add(row, reason):
        if any(v["sc_cfg"] == row["sc_cfg"] for v in selected):
            return
        selected.append({"sc_cfg": row["sc_cfg"], "reason": reason,
                         "screening_diagnostics": {k: row.get(k) for k in METRICS},
                         "outside_trained_range": not 0.5 <= row["sc_cfg"] <= 5})
    add(next(r for r in rows if r["sc_cfg"] == 3), "Mandatory SC-CFG=3 reference, even if dominated.")
    add(max(nd, key=lambda r: (r["entropy"], -r["gen_ppl"])), "Highest-entropy empirical nondominated point; repetition and lengths remain visible and are not evidence of coherence.")
    band = [r for r in nd if 5.10 <= r["entropy"] <= 5.20]
    target = min(band, key=lambda r: r["gen_ppl"]) if band else min(nd, key=lambda r: (abs(r["entropy"] - 5.15), r["gen_ppl"]))
    add(target, "Lowest screening GenPPL among nondominated points in entropy [5.10,5.20]." if band else "A entropy neighborhood unreachable in screening; closest nondominated entropy point.")
    if len(selected) < 3:
        remaining = [r for r in nd if r["sc_cfg"] not in {v["sc_cfg"] for v in selected}]
        if remaining:
            add(max(remaining, key=lambda r: min(abs(r["entropy"] - s["screening_diagnostics"]["entropy"]) for s in selected)),
                "Deduplicated selection left a slot; retain a distinct frontier point with the largest entropy separation.")
    return {"selected": selected, "frontier_scales": sorted(r["sc_cfg"] for r in nd),
            "entropy_neighborhood": [5.10, 5.20], "confirmation_seeds": [20, 21, 22],
            "selection_limit": 3, "frozen_before_confirmation": True}
