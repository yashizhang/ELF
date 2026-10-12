"""Position-preserving T5 span corruption, with protected document boundaries.

The boundary-free algorithm matches Transformers v4.44.2's Flax collator.
For packed documents we apportion round(0.15 * eligible_tokens) noise tokens
to eligible regions by largest remainder, then draw spans inside each region.
Every nonempty region retains its first token; singleton regions stay clean.
There is one sentinel per removed span and one terminal decoder EOS, with no
length truncation. Source EOS and other specials survive only in the input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch


def example_rng(seed, example_id, stream):
    raw = f"{seed}:{example_id}:{stream}".encode()
    value = int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "little")
    return np.random.RandomState(value)


def random_spans_mask(length, rng, noise_density=0.15, mean_span=3.0, noise_tokens=None):
    if length < 2 or noise_tokens == 0:
        return np.zeros(length, dtype=bool)
    n = int(np.round(length * noise_density)) if noise_tokens is None else int(noise_tokens)
    n = min(max(n, 1), length - 1)
    spans = max(1, int(np.round(min(n, length - n) / mean_span)))

    def segment(items):
        starts = np.arange(items - 1) < spans - 1
        rng.shuffle(starts)
        _, lengths = np.unique(np.cumsum(np.pad(starts, (1, 0))), return_counts=True)
        return lengths

    noise = segment(n)
    clean = segment(length - n)
    lengths = np.stack([clean, noise], axis=1).reshape(-1)
    starts = np.zeros(length, dtype=np.int8)
    starts[np.cumsum(lengths)[:-1]] = 1
    return np.cumsum(starts) % 2 == 1


def protected_mask(ordinary, rng, noise_density=0.15, mean_span=3.0):
    ordinary = np.asarray(ordinary, dtype=bool)
    if ordinary.all():
        return random_spans_mask(len(ordinary), rng, noise_density, mean_span)
    starts = np.flatnonzero(ordinary & ~np.r_[False, ordinary[:-1]])
    ends = np.flatnonzero(ordinary & ~np.r_[ordinary[1:], False]) + 1
    lengths = ends - starts
    capacities = np.maximum(lengths - 1, 0)
    target = min(int(np.round(ordinary.sum() * noise_density)), int(capacities.sum()))
    quotas = np.minimum(lengths * noise_density, capacities)
    counts = np.floor(quotas).astype(int)
    # Largest remainder, deterministic random tie breaking. At most 85 noise
    # tokens for 568 positions, so the 100 original sentinels always suffice.
    priorities = np.lexsort((rng.uniform(size=len(lengths)), -(quotas - counts)))
    while counts.sum() < target:
        for i in priorities:
            if counts[i] < capacities[i] and counts.sum() < target:
                counts[i] += 1
    mask = np.zeros(len(ordinary), dtype=bool)
    for start, end, count in zip(starts, ends, counts):
        mask[start:end] = random_spans_mask(end - start, rng, noise_density, mean_span, count)
    return mask


def corrupt_source(source, tokenizer, rng, noise_density=0.15, mean_span=3.0):
    source = np.asarray(source, dtype=np.int64)
    ordinary = ~np.isin(source, tokenizer.all_special_ids)
    mask = protected_mask(ordinary, rng, noise_density, mean_span)
    sentinel_ids = [tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]
    clean = np.r_[source, tokenizer.eos_token_id]
    corrupt, labels, pairs, span_lengths = [], [], [], []
    i = 0
    while i < len(source):
        if mask[i]:
            end = i + 1
            while end < len(source) and mask[end]:
                end += 1
            if len(span_lengths) >= len(sentinel_ids):
                raise ValueError("More noise spans than original T5 sentinels")
            sentinel = sentinel_ids[len(span_lengths)]
            corrupt.append(sentinel)
            labels.extend([sentinel, *source[i:end].tolist()])
            span_lengths.append(end - i)
            i = end
        else:
            if ordinary[i]:
                pairs.append((i, len(corrupt)))
            corrupt.append(int(source[i]))
            i += 1
    corrupt.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    return {"clean": clean, "corrupt": np.asarray(corrupt), "labels": np.asarray(labels),
            "pairs": pairs, "noise_tokens": int(mask.sum()), "eligible_tokens": int(ordinary.sum()),
            "source_tokens": len(source), "span_count": len(span_lengths), "span_lengths": span_lengths}


def sample_positions(sequences, example_ids, tokenizer, seed, stream, total):
    """Equal per-example quotas, with deterministic spillover for rare short inputs."""
    eligible = [np.flatnonzero(~np.isin(x, tokenizer.all_special_ids)) for x in sequences]
    if sum(map(len, eligible)) < total:
        raise ValueError("Insufficient ordinary tokens for the fixed SIGReg sample count")
    pools = [example_rng(seed, eid, stream).permutation(pos) for eid, pos in zip(example_ids, eligible)]
    selected = [[] for _ in pools]
    quota, remainder = divmod(total, len(pools))
    for i, pool in enumerate(pools):
        selected[i] = pool[:min(len(pool), quota + (i < remainder))].tolist()
    missing = total - sum(map(len, selected))
    while missing:
        for i, pool in enumerate(pools):
            if len(selected[i]) < len(pool) and missing:
                selected[i].append(int(pool[len(selected[i])]))
                missing -= 1
    return [(i, p) for i, positions in enumerate(selected) for p in positions]


@dataclass
class SpanCollator:
    tokenizer: object
    corruption_seed: int = 1729
    sampling_seed: int = 2718
    samples_per_rank: int = 256
    noise_density: float = 0.15
    mean_span: float = 3.0

    def __call__(self, examples):
        ids = [int(e["id"]) for e in examples]
        rows = [corrupt_source(e["input_ids"], self.tokenizer,
                               example_rng(self.corruption_seed, e["id"], "span"),
                               self.noise_density, self.mean_span) for e in examples]
        batch = {"example_ids": torch.tensor(ids)}
        for view in ("clean", "corrupt", "labels"):
            width = max(len(r[view]) for r in rows)
            values = torch.full((len(rows), width), -100 if view == "labels" else self.tokenizer.pad_token_id)
            attention = torch.zeros_like(values)
            for i, row in enumerate(rows):
                n = len(row[view])
                values[i, :n] = torch.from_numpy(row[view].copy())
                attention[i, :n] = 1
            batch[view] = values
            batch[view + "_mask"] = attention
        pairs = [(b, c, k) for b, r in enumerate(rows) for c, k in r["pairs"]]
        batch["pairs"] = torch.tensor(pairs, dtype=torch.long).reshape(-1, 3)
        for view in ("clean", "corrupt"):
            batch[view + "_sample"] = torch.tensor(sample_positions(
                [r[view] for r in rows], ids, self.tokenizer, self.sampling_seed,
                view, self.samples_per_rank), dtype=torch.long)
        # Counts are globally summed once per update by the training loop.
        batch["counts"] = torch.tensor([len(rows), sum(r["source_tokens"] for r in rows),
                                        sum(len(r["labels"]) for r in rows),
                                        sum(r["noise_tokens"] for r in rows),
                                        sum(r["eligible_tokens"] for r in rows),
                                        sum(r["span_count"] for r in rows),
                                        sum(len(r["clean"]) + len(r["corrupt"]) for r in rows)])
        return batch
