"""Focused checks for errors that would invalidate C/D, using tiny T5 fixtures."""
import ast
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration

from t5_adaptation.common import ROOT, load_config, rng_state, restore_rng
from t5_adaptation.corruption import (SpanCollator, corrupt_source, random_spans_mask,
                                      sample_positions)
from t5_adaptation.objectives import AdaptationObjective, make_sigreg


class ToyTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    all_special_ids = [0, 1, 2, *range(100, 200)]

    def __len__(self):
        return 200

    def convert_tokens_to_ids(self, token):
        return 199 - int(token.removeprefix("<extra_id_").removesuffix(">"))


def reference_collator():
    # Execute the exact pinned reference class's NumPy methods, without
    # installing JAX/Flax or copying an independently implemented oracle.
    path = ROOT / "third_party/transformers/examples/flax/language-modeling/run_t5_mlm_flax.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlaxDataCollatorForT5MLM")
    cls.decorator_list = []
    module = ast.Module(body=[cls], type_ignores=[])
    ns = {"np": np, "PreTrainedTokenizerBase": object, "List": list, "Dict": dict, "BatchEncoding": dict}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    ref = ns[cls.name]()
    ref.tokenizer, ref.noise_density, ref.mean_noise_span_length = ToyTokenizer(), 0.15, 3.0
    return ref


def test_canonical_matches_pinned_reference():
    ref = reference_collator()
    source = (np.arange(568) % 90 + 3).astype(np.int64)
    for seed in range(12):
        np.random.seed(seed)
        mask = ref.random_spans_noise_mask(568)
        np.testing.assert_array_equal(random_spans_mask(568, np.random.RandomState(seed)), mask)
        expected_input = ref.filter_input_ids(source[None], ref.create_sentinel_ids(mask[None].astype(np.int8)))[0]
        expected_labels = ref.filter_input_ids(source[None], ref.create_sentinel_ids((~mask)[None].astype(np.int8)))[0]
        got = corrupt_source(source, ref.tokenizer, np.random.RandomState(seed))
        np.testing.assert_array_equal(got["corrupt"], expected_input)
        np.testing.assert_array_equal(got["labels"], expected_labels)
        assert (len(got["clean"]), len(got["corrupt"]), len(got["labels"])) == (569, 512, 114)


def reconstruct(corrupt, labels):
    spans, current = {}, None
    for token in labels[:-1]:
        if token >= 100:
            current = int(token)
            spans[current] = []
        else:
            spans[current].append(token)
    return np.array([value for token in corrupt[:-1] for value in spans.get(int(token), [token])])


def test_boundaries_alignment_and_no_truncation():
    tokenizer = ToyTokenizer()
    # Repeated identities must remain separate occurrences; protected EOS,
    # pad, unknown and pre-existing sentinels remain unmasked.
    source = np.tile([9, 9, 9, 9, 1, 9, 10, 11, 1, 12, 12, 12, 12, 12], 40)
    for seed in range(10):
        row = corrupt_source(source, tokenizer, np.random.RandomState(seed))
        np.testing.assert_array_equal(reconstruct(row["corrupt"], row["labels"]), source)
        for clean_pos, corrupt_pos in row["pairs"]:
            assert source[clean_pos] == row["corrupt"][corrupt_pos]
            assert source[clean_pos] not in tokenizer.all_special_ids
        # Recover occurrence mapping directly by walking removed span lengths.
        src_pos, compressed_pos, span = 0, 0, 0
        expected = []
        while src_pos < len(source):
            if row["corrupt"][compressed_pos] >= 100:
                src_pos += row["span_lengths"][span]
                span += 1
            else:
                if source[src_pos] not in tokenizer.all_special_ids:
                    expected.append((src_pos, compressed_pos))
                src_pos += 1
            compressed_pos += 1
        assert expected == row["pairs"]
        assert sum(row["corrupt"] == 1) == sum(source == 1) + 1
        assert len(row["corrupt"]) > 512  # Never force to canonical shape.
    # Highly fragmented documents cannot overflow the 100 sentinel tokens.
    fragmented = np.tile([7, 8, 1], 189)
    row = corrupt_source(fragmented, tokenizer, np.random.RandomState(0))
    assert row["noise_tokens"] == round((fragmented != 1).sum() * 0.15)
    np.testing.assert_array_equal(reconstruct(row["corrupt"], row["labels"]), fragmented)


def test_rng_independence_and_samples():
    tokenizer = ToyTokenizer()
    collator = SpanCollator(tokenizer, samples_per_rank=8)
    examples = [{"id": 10, "input_ids": [4, 1]}, {"id": 11, "input_ids": list(range(3, 25))}]
    first = collator(examples)
    np.random.seed(328)
    np.random.randn(999)
    torch.rand(777)
    second = collator(examples)
    for k in first:
        torch.testing.assert_close(first[k], second[k])
    for view in ("clean", "corrupt"):
        selected = first[view + "_sample"].tolist()
        assert len(set(map(tuple, selected))) == 8
        assert all(first[view][b, p] not in tokenizer.all_special_ids for b, p in selected)


def tiny_model(arm, dropout=0):
    cfg = T5Config(vocab_size=200, d_model=16, d_kv=4, d_ff=32, num_layers=1,
                   num_decoder_layers=1, num_heads=4, dropout_rate=dropout,
                   pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    return T5ForConditionalGeneration(cfg) if arm == "C" else T5EncoderModel(cfg)


def fixture_examples():
    return [{"id": i, "input_ids": [3 + j % 20 if j % 9 else 1 for j in range(n)]}
            for i, n in enumerate([27, 35, 19, 51])]


@pytest.mark.parametrize("arm", ["C", "D"])
def test_gradient_routes_and_clean_determinism(arm):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    objective = AdaptationObjective(tiny_model(arm, dropout=0.3), arm, 0.03, 0.8)
    objective.train()
    batch = SpanCollator(ToyTokenizer(), samples_per_rank=16)(fixture_examples())
    h1, h2 = objective.clean_states(batch), objective.clean_states(batch)
    assert h1.requires_grad and h2.requires_grad
    torch.testing.assert_close(h1, h2, rtol=0, atol=0)
    output = objective(batch)
    assert output["sigreg"].dtype == torch.float32
    torch.testing.assert_close(output["loss"], output["task"] + 0.05 * output["sigreg"])
    output["sigreg"].backward(retain_graph=True)
    assert sum(p.grad.abs().sum() for p in objective.encoder.parameters() if p.grad is not None) > 0
    if arm == "C":
        assert all(p.grad is None for n, p in objective.model.decoder.named_parameters() if n != "embed_tokens.weight")
    objective.zero_grad(set_to_none=True)
    output["task"].backward()
    assert all(p.requires_grad and p.grad is not None for p in objective.model.parameters())
    assert objective.model.shared.weight.grad.abs().sum() > 0
    assert objective.encoder.encoder.final_layer_norm.weight.grad.abs().sum() > 0 if arm == "D" else objective.encoder.final_layer_norm.weight.grad.abs().sum() > 0


def _distributed_worker(rank, init_file, expected_path, arm):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        model = AdaptationObjective(tiny_model(arm), arm, 0.03, 0.8)
        wrapped = DDP(model, broadcast_buffers=False)
        examples = fixture_examples()[rank * 2:rank * 2 + 2]
        batch = SpanCollator(ToyTokenizer(), samples_per_rank=8)(examples)
        output = wrapped(batch)
        output["loss"].backward()
        expected = torch.load(expected_path, weights_only=False)
        torch.testing.assert_close(output["task_global"], expected["task"], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(output["sigreg"], expected["sigreg"], rtol=2e-5, atol=2e-6)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, expected["grads"][name], rtol=3e-4, atol=3e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("arm", ["C", "D"])
def test_distributed_loss_and_parameter_gradients(tmp_path, arm):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = AdaptationObjective(tiny_model(arm), arm, 0.03, 0.8)
    batch = SpanCollator(ToyTokenizer(), samples_per_rank=16)(fixture_examples())
    output = model(batch)
    output["loss"].backward()
    expected = {"task": output["task_global"].detach(), "sigreg": output["sigreg"].detach(),
                "grads": {n: p.grad for n, p in model.named_parameters()}}
    path = tmp_path / "expected.pt"
    torch.save(expected, path)
    mp.spawn(_distributed_worker, args=(str(tmp_path / "pg"), str(path), arm), nprocs=2, join=True)


def test_view_separation_and_projection_resume():
    torch.set_num_threads(1)
    clean, corrupt = torch.randn(256, 512), 2 * torch.randn(256, 512) + 0.3
    reg = make_sigreg()
    state = {k: v.clone() for k, v in reg.state_dict().items()}
    combined = reg(torch.stack([clean, corrupt]))
    reg.load_state_dict(state)
    a = reg(clean)
    reg.load_state_dict(state)
    b = reg(corrupt)
    torch.testing.assert_close(combined, (a + b) / 2)
    reg.load_state_dict(state)
    assert not torch.isclose(combined, reg(torch.cat([clean, corrupt])))
    saved = {k: v.clone() for k, v in reg.state_dict().items()}
    expected = reg(clean)
    restored = make_sigreg()
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored(clean), expected, rtol=0, atol=0)


def test_nominal_budget():
    cfg = load_config(ROOT / "src/configs/t5_adaptation/arm_c.yml")
    assert cfg["updates"] == 38147
    assert cfg["source_examples"] == 19531264
    assert cfg["source_examples"] * 512 == 10000007168
