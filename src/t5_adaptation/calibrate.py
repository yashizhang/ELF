"""Streaming scalar population normalization for C4 adaptation and frozen OWT."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from .common import atomic_json, file_hash, object_hash, read_json, state_hash
from .data import PackedC4


class PopulationMoments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, values):
        values = values.detach().double()
        count = values.numel()
        if count == 0:
            return
        variance, mean = torch.var_mean(values, unbiased=False)
        mean, m2 = mean.item(), variance.item() * count
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError("No calibration coordinates")
        std = (self.m2 / self.count) ** 0.5
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Invalid calibration")
        return {"mean": self.mean, "std": std, "coordinate_count": self.count,
                "estimator": "FP64 streaming Chan merge, population variance M2/N"}


@torch.no_grad()
def calibrate(model, sequences, special_ids, device, *, ordinary_only, use_bf16, batch_size=16):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    moments = PopulationMoments()
    token_count = 0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        width = max(map(len, batch))
        ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for i, seq in enumerate(batch):
            ids[i, :len(seq)] = torch.as_tensor(seq, device=device)
            valid[i, :len(seq)] = True
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_bf16 and device.type == "cuda"):
            hidden = model(input_ids=ids, attention_mask=valid).last_hidden_state
        if ordinary_only:
            valid &= ~torch.isin(ids, torch.tensor(special_ids, device=device))
        if not torch.isfinite(hidden[valid]).all():
            raise ValueError("Non-finite calibration representations")
        moments.update(hidden[valid])
        token_count += valid.sum().item()
    return moments.result() | {"token_count": token_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument("--encoder", help="Exported adapted encoder for OWT normalization")
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["c4", "owt"], required=True)
    args = parser.parse_args()
    shared = Path(args.shared)
    encoder_path = Path(args.encoder) if args.encoder else shared / "t5"
    if args.kind == "owt" and not args.encoder:
        raise ValueError("OWT calibration requires an adapted encoder export")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained(encoder_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    if args.kind == "c4":
        data_root = shared / "c4/train"
        # A shard-committed prefix is already immutable, so calibration and
        # the GPU correctness pilot need not wait for the remaining 11B tokens.
        manifest_path = data_root / "manifest.json"
        data_manifest = read_json(manifest_path if manifest_path.exists() else data_root / "preparation.json")
        needed = 4096 * 568
        if data_manifest["tokens"] < needed:
            raise ValueError("The first 4096 C4 sequences have not been committed yet")
        with (data_root / "tokens.u16").open("rb") as f:
            prefix = f.read(needed * 2)
        if len(prefix) != needed * 2:
            raise ValueError("Incomplete C4 calibration prefix")
        packed = np.frombuffer(prefix, dtype="<u2").reshape(4096, 568)
        sequences = [np.r_[row.astype(np.int64), tokenizer.eos_token_id] for row in packed]
        identity = {"kind": "c4", "data_identity": data_manifest["identity"],
                    "data_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                    "encoder_hash": read_json(encoder_path / "manifest.json")["content_hash"],
                    "ids": list(range(4096)), "mask": "ordinary tokens only, all specials excluded",
                    "use_bf16": False, "length": 569}
    else:
        from datasets import load_from_disk
        data = load_from_disk(str(shared / "owt"))
        sequences = [np.asarray(data[i]["input_ids"][:1024], dtype=np.int64) for i in range(4096)]
        identity = {"kind": "owt", "data_hash": read_json(shared / "owt_manifest.json")["content_hash"],
                    "encoder_hash": read_json(encoder_path / "manifest.json")["content_hash"],
                    "ids": list(range(4096)), "mask": "all original non-padding positions, including EOS; same as ELF validity mask",
                    "use_bf16": True, "length": 1024}
    output = Path(args.output)
    if output.exists():
        if read_json(output)["identity"] != identity:
            raise ValueError("Existing normalization has different provenance")
        return
    stats = calibrate(model, sequences, tokenizer.all_special_ids, device,
                      ordinary_only=args.kind == "c4", use_bf16=identity["use_bf16"])
    result = stats | {"identity": identity, "model_state_hash": state_hash(model.state_dict())}
    result["content_hash"] = object_hash(result)
    atomic_json(output, result)
    print({k: v for k, v in result.items() if k != "identity"}, flush=True)


if __name__ == "__main__":
    main()
