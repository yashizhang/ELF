"""Export raw final encoders and verify them through ELF's actual wrapper."""
import argparse
import copy
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration

from modules.t5_encoder import get_encoder
from .common import atomic_json, directory_manifest, file_hash, read_json, state_hash
from .data import PackedC4
from .objectives import AdaptationObjective, dropout_disabled


def export_encoder(objective, tokenizer, destination):
    if objective.arm == "C":
        config = copy.deepcopy(objective.model.config)
        exported = T5EncoderModel(config)
        exported.shared.load_state_dict(objective.model.shared.state_dict())
        exported.encoder.load_state_dict(objective.model.encoder.state_dict())
    else:
        exported = objective.model
    exported.eval()
    exported.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    return exported


@torch.no_grad()
def verify_reload(objective, destination, ids, attention_mask, device):
    _, wrapper = get_encoder(str(destination), torch.float32)
    wrapper.to(device).eval()
    with dropout_disabled(objective):
        expected = objective.encoder(input_ids=ids, attention_mask=attention_mask).last_hidden_state
    actual = wrapper(input_ids=ids, attention_mask=attention_mask, deterministic=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if actual.shape != (*ids.shape, objective.model.config.d_model):
        raise ValueError("Encoder export shape mismatch")
    return {"max_absolute_error": (actual - expected).abs().max().item(),
            "rtol": 1e-5, "atol": 1e-6, "shape": list(actual.shape), "precision": "fp32"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out, shared = Path(args.output), Path(args.shared)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity, counters = payload["identity"], payload["counters"]
    cfg = identity["config"]
    if not payload["final"] or identity["smoke"] or counters["updates"] != cfg["updates"]:
        raise ValueError("Only the final fixed-budget adaptation checkpoint may feed ELF")
    if counters["nominal_positions"] != 10_000_007_168:
        raise ValueError("Final adaptation budget mismatch")
    provenance = {"source_checkpoint_sha256": file_hash(args.checkpoint), "identity": identity,
                  "counters": counters, "raw_outputs": True}
    if (out / "manifest.json").exists():
        existing = read_json(out / "manifest.json")
        if existing["provenance"] != provenance:
            raise ValueError("Encoder output already belongs to a different checkpoint")
        if directory_manifest(out)["content_hash"] != existing["content_hash"]:
            raise ValueError("Immutable encoder export changed")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = T5ForConditionalGeneration if identity["arm"] == "C" else T5EncoderModel
    model = model_cls.from_pretrained(shared / "t5")
    calibration = read_json(shared / "c4_normalization.json")
    objective = AdaptationObjective(model, identity["arm"], calibration["mean"], calibration["std"])
    objective.load_state_dict(payload["model"])
    objective.to(device).eval()
    del payload
    tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    temp = out.with_name(out.name + ".incomplete")
    temp.mkdir(parents=True, exist_ok=True)
    export_encoder(objective, tokenizer, temp)
    raw = PackedC4(shared / "c4/train", count=2)
    ids = torch.tensor([list(raw[i]["input_ids"]) + [tokenizer.eos_token_id] for i in range(2)], device=device)
    reload = verify_reload(objective, temp, ids, torch.ones_like(ids), device)
    original = T5EncoderModel.from_pretrained(shared / "t5")
    original_hash = state_hash(original.state_dict())
    adapted_hash = state_hash(T5EncoderModel.from_pretrained(temp).state_dict())
    if adapted_hash == original_hash:
        raise ValueError("Final adapted encoder is unchanged from original weights")
    manifest = directory_manifest(temp) | {"provenance": provenance, "reload_check": reload,
                                           "original_state_hash": original_hash, "state_hash": adapted_hash}
    atomic_json(temp / "manifest.json", manifest)
    os.rename(temp, out)


if __name__ == "__main__":
    main()
