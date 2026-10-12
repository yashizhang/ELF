"""Immutable, contiguous packed C4 tokens; disjoint deterministic rank slices."""
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Sampler

from .common import read_json


class PackedC4(Dataset):
    def __init__(self, path, source_length=None, count=None):
        self.root = Path(path)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["status"] != "complete":
            raise ValueError("C4 preparation is incomplete")
        self.length = source_length or self.manifest["source_length"]
        maximum = self.manifest["tokens"] // self.length
        self.count = maximum if count is None else count
        if self.count > maximum:
            raise ValueError("Requested more examples than prepared")
        self.tokens = np.memmap(self.root / "tokens.u16", dtype="<u2", mode="r")
        if len(self.tokens) != self.manifest["tokens"]:
            raise ValueError("Token file length does not match manifest")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"id": index, "input_ids": self.tokens[index * self.length:(index + 1) * self.length].astype(np.int64)}


class ContiguousRankBatches(Sampler):
    def __init__(self, updates, local_batch, rank, world, start_update=0):
        self.updates, self.local_batch = updates, local_batch
        self.rank, self.world, self.start = rank, world, start_update

    def __iter__(self):
        for step in range(self.start, self.updates):
            offset = step * self.local_batch * self.world + self.rank * self.local_batch
            yield list(range(offset, offset + self.local_batch))

    def __len__(self):
        return self.updates - self.start
