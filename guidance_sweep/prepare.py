"""Resolve and verify immutable study inputs, without any compute submission."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import torch
import yaml
from configs.config import load_config_from_yaml
from modules.model import ELF_models
from transformers import AutoTokenizer
from t5_adaptation.common import atomic_json, directory_manifest, file_hash, read_json, state_hash

BASE = "1d96853d8e9e78400ff1b3ac57500e058db30155"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/guidance_sweep_20261015")
    args = parser.parse_args()
    repo, root = Path.cwd(), Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    original = {}
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], text=True).splitlines()
    for name in names:
        path = repo / name
        if path.is_file():
            expected = subprocess.check_output(["git", "show", f"{BASE}:{name}"])
            assert expected == path.read_bytes(), f"Original file changed: {name}"
            original[name] = file_hash(path)
    inputs = {"inspected_revision": BASE, "actual_checkout_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "original_tracked_files": original, "original_artifacts": {}, "arms": {},
              "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "artifact_root": str(root),
              "sources": read_json(repo / "src/configs/t5_adaptation/sources.json")}
    def keep(path, expected=None):
        path = Path(path).resolve()
        digest = file_hash(path)
        assert expected is None or digest == expected, path
        inputs["original_artifacts"][str(path)] = digest
        return digest
    for arm in ("C", "D"):
        audit = read_json(repo / f"evidence/generation-{arm}.json")
        stage = audit["stage_records"]["generation"]["command"]
        config_path = Path(stage[stage.index("--config") + 1])
        keep(config_path, audit["evaluation_config_sha256"])
        cfg = load_config_from_yaml(str(config_path))
        checkpoint = Path(audit["checkpoint"])
        keep(checkpoint, audit["checkpoint_sha256"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        assert payload["epoch"] == 5 and payload["step"] == 95090
        assert state_hash(payload["ema_params1"]) == audit["ema_state_hash"]
        encoder = Path(cfg.encoder_model_name)
        manifest = read_json(encoder / "manifest.json")
        assert directory_manifest(encoder)["content_hash"] == audit["encoder_hash"]
        for name, digest in manifest["files"].items():
            keep(encoder / name, digest)
        normalization = encoder.parent / "owt_normalization.json"
        norm = read_json(normalization)
        keep(normalization)
        assert cfg.latent_mean == norm["mean"] and cfg.latent_std == norm["std"]
        for name, digest in audit["source_hashes"].items():
            assert file_hash(repo / name) == digest
        inputs["arms"][arm] = {"config": str(config_path), "checkpoint": str(checkpoint),
            "checkpoint_sha256": audit["checkpoint_sha256"], "ema_state_hash": audit["ema_state_hash"],
            "encoder": str(encoder), "encoder_hash": audit["encoder_hash"], "normalization": str(normalization),
            "latent_mean": cfg.latent_mean, "latent_std": cfg.latent_std,
            "trained_sc_cfg_interval": [cfg.self_cond_cfg_min, cfg.self_cond_cfg_max],
            "range_evidence": "Actual training YAML plus unchanged pinned Config defaults and train_step.sample_cfg_scale; no per-update scale trace.",
            "vocab_size": len(AutoTokenizer.from_pretrained(encoder)), "step": 95090, "epoch": 5}
        assert [cfg.self_cond_cfg_min, cfg.self_cond_cfg_max] == [.5, 5.0]
    for name in ("t5", "evaluator"):
        path = repo / "artifacts/shared" / name
        manifest = read_json(path / "manifest.json")
        assert directory_manifest(path)["content_hash"] == manifest["content_hash"]
        for file, digest in manifest["files"].items():
            keep(path / file, digest)
        inputs[name] = {"path": str(path), "content_hash": manifest["content_hash"], "source": manifest["source"]}
    anchor = root / "anchor"
    download = read_json(anchor / "download.json")
    for name, digest in download["files"].items():
        assert file_hash(anchor / name) == digest
    payload = torch.load(anchor / "checkpoint_95085", map_location="cpu", weights_only=False, mmap=True)
    cfg = load_config_from_yaml(str(anchor / "config.yml"))
    vocab_size = len(AutoTokenizer.from_pretrained(inputs["t5"]["path"]))
    model = ELF_models[cfg.model](text_encoder_dim=512, max_length=cfg.max_length, attn_drop=cfg.attn_dropout,
        proj_drop=cfg.proj_dropout, num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens, bottleneck_dim=cfg.bottleneck_dim)
    model.load_state_dict(payload["ema_params1"], strict=True)
    inputs["arms"]["A_measured_local"] = {"config": str(anchor / "config.yml"), "checkpoint": str(anchor / "checkpoint_95085"),
        "checkpoint_sha256": download["files"]["checkpoint_95085"], "ema_state_hash": state_hash(model.state_dict()),
        "encoder": inputs["t5"]["path"], "encoder_hash": inputs["t5"]["content_hash"],
        "latent_mean": cfg.latent_mean, "latent_std": cfg.latent_std, "vocab_size": vocab_size,
        "source": download, "step": payload["step"], "epoch": payload["epoch"], "strict_ema_load": True}
    assert cfg.latent_mean == 0 and cfg.latent_std == .2
    atomic_json(repo / "guidance_sweep/inputs.json", inputs)
    plan = {"artifact_root": str(root), "ordinary_cfg": 1, "ordinary_cfg_audit_values": [0, .5, 1, 2, 3, 5],
            "screening_sc_cfg": [.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5], "screening_seed": 10, "screening_samples": 250,
            "samplers": {"primary": {"steps": 32, "gamma": 1.5}, "secondary": {"steps": 64, "gamma": 1.0}},
            "fixed": {"length": 1024, "time_schedule": "logit_normal", "P_mean": -1.5, "P_std": .8,
                      "denoiser_noise_scale": 2, "t_eps": .05, "forward_bf16": True, "parameter_and_latent_dtype": "float32",
                      "decode": "existing final argmax and mask_after_eos", "global_batch_size": 512, "world_size": 8},
            "entropy_neighborhood": [5.10, 5.20], "extrapolation": {"values": [0, .25], "trigger": "screening entropy at 0.5 < 5.10", "outside_trained_range": True},
            "refinement": {"max_in_range_per_arm_sampler": 2, "rule": "If adjacent in-range points bracket 5.15, measure rounded linear estimate and midpoint of narrowest entropy bracket; otherwise zero refinements."},
            "confirmation": {"max_selections_per_arm_sampler": 3, "seeds": [20, 21, 22], "samples_per_seed": 1000,
                "rule": "Always 3; maximum-entropy nondominated point; minimum-PPL nondominated point in [5.10,5.20] or closest entropy. Deduplicate; spare slot uses largest entropy separation on frontier. Include all repetition/length/empty diagnostics; no confirmation-driven search."},
            "rng": "Reset Python, NumPy, torch CPU, CUDA global RNGs and explicit CPU generator before every configuration, seed+rank*1000003. Fixed ceil(N/8) contiguous rank blocks and local batch 64; the last screening rank produces exactly the remaining 26 samples instead of generating/trimming six extras. Confirmation uses the original 125/rank, batches64+61. Warmup/audit streams are separate. Replay checks and batch noise/time-grid/RNG hashes enforce order independence.",
            "submission_phases": "Two independent 8-H100 screening jobs, then two independent 8-H100 confirmation jobs after all four selections are frozen. C confirmation also evaluates the released A anchor; no separate A allocation.",
            "diagnostics": "4-gram and adjacent repetition on the same post-cap GPT-2 token sequence as entropy; undefined below 4/2 tokens. Duplicate rate=1-unique exact nonempty strings/nonempty count. Length summaries include empties.",
            "a_anchor": "Run released A at SC-CFG=3 for both samplers/seeds20,21,22 in the C allocation with its original 0/.2 normalization.",
            "account": "snow.research.er", "resources_per_arm": {"gpus": 8, "gpu_model": "H100", "cpu": 64, "ram_gb": 512}}
    atomic_json(repo / "guidance_sweep/plan.json", plan)
    print({"inputs": "guidance_sweep/inputs.json", "anchor": "compatible, strict EMA load", "original_files": len(original),
           "immutable_artifacts": len(inputs["original_artifacts"])})


if __name__ == "__main__":
    main()
