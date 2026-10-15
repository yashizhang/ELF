"""Focused tests for guidance semantics, paired RNGs, scores and selection."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from guidance_sweep.common import (frontier, ppl_weights, refinements, reset_rng, rng_hash,
                                  score_texts, select_confirmation, summarize, token_diagnostics)
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch
from utils.sampling_utils import _forward_sample, get_sampling_steps


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, z, t, deterministic=True, self_cond_cfg_scale=None, decoder_step_active=None):
        self.calls.append((self_cond_cfg_scale.clone(), decoder_step_active))
        width = z.shape[-1] // 2
        out = .2 * z[..., :width] + .1 * z[..., width:] + .03 * z[..., :width].mean(1, keepdim=True)
        out = out + self_cond_cfg_scale[:, None, None] * torch.arange(1, width + 1)[None, None] * .01
        return out, torch.cat((out, -out), dim=-1) if decoder_step_active else None


def config():
    return SimpleNamespace(t_eps=.05, self_cond_prob=.5, num_self_cond_cfg_tokens=4, use_bf16=False, denoiser_noise_scale=2.)


def test_inactive_ordinary_cfg_active_scale_and_nonzero_prefix_control():
    reset_rng(17)
    model, cfg = TinyModel(), config()
    z, previous = torch.randn(2, 8, 4), torch.randn(2, 8, 4)
    t, cond, mask = torch.tensor([.3, .7]), torch.zeros_like(z), torch.zeros(2, 8)
    outputs = [_forward_sample(model, z, t, previous, cfg, c, 3., cond, mask) for c in [0, .5, 1, 2, 3, 5]]
    for value in outputs:
        for a, b in zip(value, outputs[2]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    different = _forward_sample(model, z, t, previous, cfg, 1, .5, cond, mask)
    assert not torch.equal(different[1], outputs[2][1])
    mask[:, :2] = 1
    a = _forward_sample(model, z, t, previous, cfg, 0, 3., cond, mask)
    b = _forward_sample(model, z, t, previous, cfg, 3, 3., cond, mask)
    assert not torch.equal(a[1][:, 2:], b[1][:, 2:])


def test_full_rng_reset_makes_configuration_order_irrelevant_and_decode_gets_scale():
    model, cfg = TinyModel(), config()
    def run(scale):
        generator = reset_rng(10, 2)
        times = get_sampling_steps(8, "logit_normal", -1.5, .8)
        z = torch.randn(2, 8, 4, generator=generator) * 2
        latent = _generate_samples_single_batch(model, generator, z, times, None, None, cfg,
            SimpleNamespace(sampling_method="sde", sde_gamma=1.5), 1, scale)
        ids = _dlm_decode_batch(latent, model, 1., cfg, scale)
        return latent, ids, rng_hash(generator)
    first = run(3.)
    run(.5)
    replay = run(3.)
    assert torch.equal(first[0], replay[0]) and torch.equal(first[1], replay[1]) and first[2] == replay[2]
    assert model.calls[-1][1] is True and model.calls[-1][0].tolist() == [3., 3.]
    _dlm_decode_batch(first[0], model, 1., cfg, .75)
    assert model.calls[-1][0].tolist() == [.75, .75]


def test_repetition_undefined_lengths_and_original_ppl_eos_weights():
    assert token_diagnostics([1, 1, 1])["repeated_4gram_fraction"] is None
    assert token_diagnostics([1])["adjacent_repetition_fraction"] is None
    assert token_diagnostics([1] * 6)["repeated_4gram_fraction"] == pytest.approx(2 / 3)
    assert token_diagnostics([1, 1, 2, 2])["adjacent_repetition_fraction"] == pytest.approx(2 / 3)
    assert ppl_weights([1, 2], 4, 99) == 2
    assert ppl_weights([1, 99, 2], 4, 99) == 3


class Tokenizer:
    eos_token_id = 99
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[int(v) for v in text.split()] for text in texts]}


class Scorer:
    tokenizer = Tokenizer()
    def reset(self):
        pass
    def record_generative_perplexity(self, texts, max_length, retokenize):
        assert max_length == 1024 and retokenize
        tokens = [ids[:1024] for ids in self.tokenizer(texts)["input_ids"]]
        return {"ppl": math.exp(1), "per_sample_ppl": [math.exp(1)] * len(tokens),
                "mean_entropy": np.mean([token_diagnostics(ids)["entropy"] for ids in tokens])}


def test_score_all_nonempty_counts_duplicates_cap_and_per_sample_outputs():
    rows = [{"id": i, "generated": text} for i, text in enumerate(["1 2 1 2", "", "1 2 1 2", " ", " ".join(["3"] * 1100)])]
    summary, samples = score_texts(rows, Scorer(), Tokenizer())
    assert summary["requested_count"] == summary["generated_count"] == 5
    assert summary["ppl_scored_count"] == summary["entropy_scored_count"] == summary["nonempty_count"] == 3
    assert summary["empty_sample_ids"] == [1, 3] and summary["empty_fraction"] == .4
    assert summary["duplicate_text_rate"] == pytest.approx(1 / 3)
    assert samples[-1]["gpt2_length_before"] == 1100 and samples[-1]["gpt2_length_after"] == 1024
    assert samples[1]["ppl"] is None and samples[1]["entropy"] is None
    assert not summary["scoring_failures"] and len(samples) == 5


def test_seed_statistics_use_sample_sd_and_preserve_invalid_values():
    result = summarize([1., 2., 3.])
    assert result["mean"] == 2 and result["sample_sd"] == 1
    assert result["se"] == pytest.approx(1 / math.sqrt(3))
    assert summarize([1., None, 3.])["mean"] is None


def test_bounded_refinement_frontier_and_frozen_selection_retains_reference():
    rows = [{"sc_cfg": s, "gen_ppl": p, "entropy": h} for s, p, h in [(.5, 50, 5.3), (1., 25, 5.1), (2., 20, 4.9), (3., 22, 4.7), (4., 30, 4.5)]]
    assert {r["sc_cfg"] for r in frontier(rows)} == {.5, 1., 2.}
    refine = refinements(rows)
    assert 0 < len(refine["values"]) <= 2
    assert all(.5 < s < 1 for s in refine["values"])
    selected = select_confirmation(rows)
    assert selected["frozen_before_confirmation"] and selected["confirmation_seeds"] == [20, 21, 22]
    assert {r["sc_cfg"] for r in selected["selected"]} == {3., .5, 1.}
    assert len(rows) == 5


def test_report_uses_seed_means_and_paired_seed_differences():
    from guidance_sweep.common import METRICS
    from guidance_sweep.report import aggregate, paired_differences
    rows = []
    for arm, scale, values in [("C", 3, [10, 20, 30]), ("C", 1, [8, 19, 27]), ("A_measured_local", 3, [15, 21, 34])]:
        for seed, value in zip((20, 21, 22), values):
            rows.append({"arm": arm, "sc_cfg": scale, "sampler": "primary", "stage": "confirmation", "seed": seed,
                         "requested_count": 1000, "generated_count": 1000, "nonempty_count": 999, "scoring_failures": [],
                         **{key: value for key in METRICS + ["t5_retokenized_length"]}})
    rows.reverse()
    result = next(r for r in aggregate(rows) if r["arm"] == "C" and r["sc_cfg"] == 3)
    assert result["gen_ppl"]["mean"] == 20 and result["gen_ppl"]["sample_sd"] == 10
    assert result["generated_count"] == 3000 and result["nonempty_count"] == 2997
    pair = next(r for r in paired_differences(rows) if r["comparison"] == "within_arm_vs_sc3")
    assert pair["gen_ppl"]["values"] == [-2, -1, -3]
    assert pair["gen_ppl"]["mean"] == -2 and pair["gen_ppl"]["sample_sd"] == 1
    assert pair["gen_ppl"]["se"] == pytest.approx(1 / math.sqrt(3))
