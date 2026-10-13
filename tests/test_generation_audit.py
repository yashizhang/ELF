"""Check the final auditor against independent examples and production masks."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.metrics_utils import Metrics


spec = importlib.util.spec_from_file_location(
    "generation_audit", Path(__file__).resolve().parents[1] / "scripts/audit_final_generation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_repository_ppl_mask_and_independent_entropy_reconstruction():
    ids = np.array([[3, 3, 0, 0, 0], [3, 0, 4, 0, 0], [3, 4, 3, 4, 3]])
    mask = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    weights, entropy = audit.token_statistics(ids, mask, eos_token_id=0)
    assert weights == [2, 3, 4]
    expected = [0, math.log(3), -(3 / 5) * math.log(3 / 5) - (2 / 5) * math.log(2 / 5)]
    np.testing.assert_allclose(entropy, expected, atol=1e-12)

    class FixedLogits:
        def __call__(self, input_ids, attention_mask):
            logits = torch.tensor([0., .3, .5, 1., -1.]).expand(*input_ids.shape, 5)
            return SimpleNamespace(logits=logits)

    scorer = object.__new__(Metrics)
    scorer.tokenizer = SimpleNamespace(eos_token_id=0)
    scorer._eval_model = FixedLogits()
    nll, actual_weights = scorer._compute_batch_nlls(torch.from_numpy(ids), torch.from_numpy(mask))
    assert actual_weights.sum(dim=1).tolist() == weights
    nll_sums = (nll.double() * actual_weights).sum(dim=1).tolist()
    per_sample = [math.exp(n / w) for n, w in zip(nll_sums, weights)]
    assert audit.reconstruct_ppl(per_sample, weights) == pytest.approx(math.exp(sum(nll_sums) / 9))


def test_ppl_reconstruction_uses_token_weighted_logs_and_preserves_unscored_samples():
    assert audit.reconstruct_ppl([math.exp(1), math.exp(3)], [1, 3]) == pytest.approx(math.exp(2.5))
    assert audit.reconstruct_ppl([None, math.exp(2)], [0, 2]) == pytest.approx(math.exp(2))
    assert audit.reconstruct_ppl([None], [0]) is None
    with pytest.raises(ValueError, match="scored tokens"):
        audit.reconstruct_ppl([None], [1])
    with pytest.raises(ValueError, match="null PPL"):
        audit.reconstruct_ppl([1.0], [0])


class TinyTokenizer:
    eos_token_id = 0
    tokens = {"same": [4, 4], "mixed": [4, 5, 4], "": [], " ": [6], "tiny": [4]}

    def __call__(self, texts, **kwargs):
        sequences = [self.tokens[text] for text in texts]
        if kwargs.get("return_tensors") != "np":
            return {"input_ids": sequences}
        width = max(map(len, sequences))
        return {"input_ids": np.array([ids + [0] * (width - len(ids)) for ids in sequences]),
                "attention_mask": np.array([[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences])}


def seed_fixture(texts, *, ppl, scored_count, entropy):
    rows = [{"id": i, "generated": text} for i, text in enumerate(texts)]
    lengths = [len(TinyTokenizer.tokens[text]) for text in texts]
    nonempty = sum(bool(text.strip()) for text in texts)
    item = {"requested_count": 1000, "generated_count": 1000, "nonempty_count": nonempty,
            "entropy_scored_count": nonempty, "ppl_scored_count": scored_count,
            "empty_output_fraction": 1 - nonempty / 1000,
            "gpt2_lengths_before_truncation": lengths, "gpt2_lengths_after_truncation": lengths,
            "t5_retokenized_lengths": lengths, "qualitative_samples": rows[:3],
            "per_sample_ppl_nonempty_order": [ppl] * nonempty,
            "gen_ppl": ppl, "mean_sample_unigram_entropy": entropy}
    return item, rows


def test_seed_audit_exposes_empty_exclusions_and_rejects_missing_or_duplicate_samples():
    entropy = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)) / 2
    item, rows = seed_fixture(["same", "mixed", "", " "] * 250,
                              ppl=math.exp(2), scored_count=500, entropy=entropy)
    tokenizer = TinyTokenizer()
    result = audit.check_seed_metrics(item, rows, tokenizer, tokenizer)
    assert result["empty_sample_ids"] == [i for i in range(1000) if i % 4 in (2, 3)]
    assert result["ppl_token_weight_total"] == 1000
    with pytest.raises(ValueError, match="1,000"):
        audit.check_rows(rows[:-1])
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["id"] = 998
    with pytest.raises(ValueError, match="duplicate"):
        audit.check_rows(duplicate)
    with pytest.raises(ValueError, match="PPL-scored"):
        audit.check_seed_metrics(item | {"ppl_scored_count": 1000}, rows, tokenizer, tokenizer)
    with pytest.raises(ValueError, match="aggregate PPL"):
        audit.check_seed_metrics(item | {"gen_ppl": 2.0}, rows, tokenizer, tokenizer)


@pytest.mark.parametrize("text,scored_count,entropy", [(" ", 0, None), ("tiny", 0, 0.0)])
def test_empty_or_unscorable_outputs_remain_valid_negative_results(text, scored_count, entropy):
    item, rows = seed_fixture([text] * 1000, ppl=None, scored_count=scored_count, entropy=entropy)
    result = audit.check_seed_metrics(item, rows, TinyTokenizer(), TinyTokenizer())
    assert result["reconstructed_gen_ppl"] is None
    assert result["nonempty_count"] == (0 if text == " " else 1000)
    assert len(result["nonempty_samples_without_scored_tokens"]) == (0 if text == " " else 1000)


def aggregate_fixture():
    per_seed, aggregates = [], []
    for config in ("primary", "secondary"):
        for seed, value in enumerate((10., 20., 30.)):
            per_seed.append({"seed": seed, "configuration": config,
                             "gen_ppl": value, "mean_sample_unigram_entropy": value / 10,
                             "empty_output_fraction": 0.0})
        aggregate = {"arm": "C", "configuration": config, "origin": "measured_this_project",
                     "training_seed_count": 1, "inference_seeds": [0, 1, 2]}
        for field, scale in [("gen_ppl", 1), ("mean_sample_unigram_entropy", .1), ("empty_output_fraction", 0)]:
            aggregate[field] = {"values": [10 * scale, 20 * scale, 30 * scale], "mean": 20 * scale,
                                "sample_std": 10 * scale, "sem": 10 * scale / math.sqrt(3)}
        aggregates.append(aggregate)
    return {"per_seed": per_seed, "aggregates": aggregates}


def test_aggregate_audit_rejects_population_std_and_duplicate_inference_seeds():
    results = aggregate_fixture()
    audit.check_aggregates(results, "C")
    wrong = copy.deepcopy(results)
    wrong["aggregates"][0]["gen_ppl"]["sample_std"] = math.sqrt(200 / 3)
    with pytest.raises(ValueError, match="sample_std"):
        audit.check_aggregates(wrong, "C")
    wrong = copy.deepcopy(results)
    wrong["per_seed"][1]["seed"] = 0
    with pytest.raises(ValueError, match="exactly seeds"):
        audit.check_aggregates(wrong, "C")
    # One undefined inference-seed score leaves the aggregate explicitly null.
    results["per_seed"][0]["gen_ppl"] = None
    results["aggregates"][0]["gen_ppl"] = {"values": [None, 20., 30.],
                                            "mean": None, "sample_std": None, "sem": None}
    audit.check_aggregates(results, "C")
