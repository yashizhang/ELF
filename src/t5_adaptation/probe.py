"""Small matched frozen-encoder token-readout diagnostic (500 updates)."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

from .common import atomic_json, object_hash, read_json, state_hash
from .corruption import example_rng
from .data import PackedC4


def batch_features(model, dataset, indices, tokenizer, mu, std, device):
    ids = torch.tensor(np.stack([np.r_[dataset[i]["input_ids"], tokenizer.eos_token_id] for i in indices]), device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float()
    states, targets = [], []
    for b, i in enumerate(indices):
        valid = torch.nonzero(~torch.isin(ids[b], torch.tensor(tokenizer.all_special_ids, device=device))).flatten().cpu().numpy()
        sampled = example_rng(1618, i, "readout").permutation(valid)[:32]
        states.append((hidden[b, sampled] - mu) / std)
        targets.append(ids[b, sampled])
    return torch.cat(states), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shared, encoder, output = Path(args.shared), Path(args.encoder), Path(args.output)
    normalization = read_json(shared / "c4_normalization.json")
    encoder_manifest = read_json(encoder / "manifest.json")
    train = PackedC4(shared / "c4/train", count=3072)
    validation = PackedC4(shared / "c4/validation", source_length=568, count=512)
    identity = {"encoder_hash": encoder_manifest["content_hash"], "normalization": normalization["content_hash"],
                "train_data": train.manifest["content_hash"], "validation_data": validation.manifest["content_hash"],
                "train_ids": [0, 3072], "validation_ids": [0, 512], "id_intervals": "half-open",
                "updates": 500, "batch_size": 8, "ordinary_tokens_per_sequence": 32,
                "readout_seed": 12345, "data_order_seed": 123, "noise_seed": 56789,
                "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0, "train_condition": "clean normalized latents",
                "evaluation_conditions": ["clean normalized latents", "normalized latents plus N(0, 0.1^2 I)"]}
    if output.exists():
        if read_json(output)["identity"] != identity:
            raise ValueError("Readout probe output has different provenance")
        return
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained(encoder).to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    torch.manual_seed(12345)
    head = torch.nn.Linear(512, model.config.vocab_size).to(device)
    initial_hash = state_hash(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    order = np.random.RandomState(123).permutation(len(train))
    mu, std = normalization["mean"], normalization["std"]
    for step in range(500):
        indices = [int(order[(step * 8 + i) % len(order)]) for i in range(8)]
        x, y = batch_features(model, train, indices, tokenizer, mu, std, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite readout probe loss")
        loss.backward()
        optimizer.step()
    head.eval()
    sums = {"clean": [0.0, 0, 0], "noise_0.1": [0.0, 0, 0]}
    generator = torch.Generator(device=device).manual_seed(56789)
    with torch.no_grad():
        for start in range(0, len(validation), 8):
            x, y = batch_features(model, validation, list(range(start, min(start + 8, len(validation)))),
                                   tokenizer, mu, std, device)
            noise = torch.randn(x.shape, generator=generator, device=device) * 0.1
            for name, features in (("clean", x), ("noise_0.1", x + noise)):
                logits = head(features)
                sums[name][0] += F.cross_entropy(logits, y, reduction="sum").item()
                sums[name][1] += (logits.argmax(-1) == y).sum().item()
                sums[name][2] += len(y)
    metrics = {name: {"token_ce": ce / n, "token_accuracy": correct / n, "scored_tokens": n}
               for name, (ce, correct, n) in sums.items()}
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError("Probe updated the frozen encoder")
    atomic_json(output, {"status": "completed", "identity": identity, "initial_readout_hash": initial_hash,
                         "metrics": metrics, "wall_seconds": time.monotonic() - started,
                         "interpretation": "practical token decodability probe, not generation quality or an encoder-training arm"})


if __name__ == "__main__":
    main()
