"""Revalidate final artifacts and render the reviewed GOAL requirement mapping.

Read-only with respect to runs: no submissions, model forwards, or training.
Reuses deep tensor/generation audits only after checking their artifact hashes.
Human/source-review findings live in evidence/acceptance-review.json; this
script does not pretend that file-presence checks prove those judgments.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import yaml

from t5_adaptation.common import atomic_json, file_hash, load_config, object_hash


ROOT = Path(__file__).resolve().parents[1]
RELEASE = "1591a0a1719e3e7422439cf366f8229b63fcafd4"
BASE = "b29d8833609e9ab7f67cd9da39435ac5cea04837"
VERIFIED = {}
DATA_RETAINED = {}


def read(path):
    return json.loads((ROOT / path).read_text())


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd).decode().strip()


def verify(path, expected=None):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    key = str(path.relative_to(ROOT))
    if key not in VERIFIED:
        assert path.is_file(), path
        VERIFIED[key] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    if expected is not None:
        assert VERIFIED[key]["sha256"] == expected, f"Hash mismatch: {key}"
    return VERIFIED[key]["sha256"]


def snapshot(path):
    directory = ROOT / path
    manifest = read(directory / "manifest.json")
    assert object_hash(manifest["files"]) == manifest["content_hash"]
    for name, digest in manifest["files"].items():
        verify(directory / name, digest)
    verify(directory / "manifest.json")
    return manifest


def data_file(path, expected):
    """Record preparation hashes honestly, without another full data scan."""
    path = ROOT / path
    assert path.is_file() and path.stat().st_size > 0, path
    DATA_RETAINED[str(path.relative_to(ROOT))] = {
        "bytes": path.stat().st_size, "sha256": expected,
        "verification": "Preparation manifest hash retained; current file presence/size checked, full bytes not reread in this final audit."}


def main():
    reviewed = read("evidence/acceptance-review.json")
    assert len(reviewed["requirements"]) == 51
    assert len({r["id"] for r in reviewed["requirements"]}) == 51
    assert git("branch", "--show-current") == "t5-sigreg-cd"
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    git("merge-base", "--is-ancestor", "5098bf28b5e9b52c329970a7e4e1cc28251c76e6", "main")
    main_ref = git("rev-parse", "main")
    assert main_ref == "398ac406be5e7a0c61fbfdc3d854faf86d46ad96"
    source_pins = read("src/configs/t5_adaptation/sources.json")
    for name in ("transformers", "lejepa"):
        path = ROOT / "third_party" / name
        assert git("rev-parse", "HEAD", cwd=path) == source_pins[name]["commit"]
        assert not git("status", "--porcelain", cwd=path)
        verify(path / "LICENSE")
    unchanged_upstream = ["src/modules/model.py", "src/modules/layers.py", "src/modules/t5_encoder.py",
                          "src/utils/encoder_utils.py", "src/utils/metrics_utils.py", "src/utils/sampling_utils.py",
                          "src/generation.py", "src/configs/training_configs/train_owt_ELF-B.yml",
                          "src/configs/sampling_configs/uncond_sampling_configs.yml"]
    for path in unchanged_upstream:
        original = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
        verify(path, hashlib.sha256(original).hexdigest())
    for path in ("tests/test_adaptation.py", "tests/test_elf_reproducibility.py"):
        original = subprocess.check_output(["git", "show", f"{RELEASE}:{path}"], cwd=ROOT)
        verify(path, hashlib.sha256(original).hexdigest())
    test_record = read("evidence/generation-auditor-checks.json")
    verify(test_record["auditor"], test_record["auditor_sha256"])
    verify(test_record["tests"], test_record["tests_sha256"])
    assert test_record["unit_tests"]["exit_code"] == 0
    envs = read("evidence/final-environments.json")["environments"]
    for env in envs.values():
        verify(env["lock"], env["lock_sha256"])
        assert env["lock_matches"] and env["pip_check_exit_code"] == 0

    shared = ROOT / "artifacts/shared"
    original = snapshot(shared / "t5")
    evaluator = snapshot(shared / "evaluator")
    assert original["source"] == source_pins["t5"]
    assert evaluator["source"] == source_pins["evaluator"]
    c4 = {}
    for split in ("train", "validation"):
        manifest = read(shared / f"c4/{split}/manifest.json")
        assert manifest == read(f"evidence/c4-{split}-manifest.json")
        assert manifest["status"] == "complete" and manifest["identity"]["source"] == source_pins["c4"]
        assert object_hash({k: v for k, v in manifest.items() if k != "content_hash"}) == manifest["content_hash"]
        c4[split] = manifest
        verify(shared / f"c4/{split}/manifest.json")
        for name, key in (("tokens.u16", "token_sha256"), ("documents.jsonl", "documents_sha256")):
            data_file(shared / f"c4/{split}/{name}", manifest[key])
        assert (shared / f"c4/{split}/tokens.u16").stat().st_size == 2 * manifest["tokens"]
    calibration = read(shared / "c4_normalization.json")
    assert calibration == read("evidence/normalization_c4.json")
    assert object_hash({k: v for k, v in calibration.items() if k != "content_hash"}) == calibration["content_hash"]
    assert calibration["identity"]["ids"] == list(range(4096))
    assert calibration["identity"]["encoder_hash"] == original["content_hash"]
    assert calibration["identity"]["data_identity"] == c4["train"]["identity"]
    with (shared / "c4/train/tokens.u16").open("rb") as stream:
        assert hashlib.sha256(stream.read(4096 * 568 * 2)).hexdigest() == calibration["identity"]["data_prefix_sha256"]
    owt = read(shared / "owt_manifest.json")
    assert owt == read("evidence/owt-data-manifest.json")
    assert owt["source"] == source_pins["owt"] and owt["examples"] == 9737184
    assert object_hash(owt["files"]) == owt["content_hash"]
    assert owt["calibration_ids"] == list(range(4096))
    for name, digest in owt["files"].items():
        data_file(shared / "owt" / name, digest)
    verify(shared / "owt_manifest.json")
    assert read("evidence/owt-loader-check.json")["status"] == "passed"
    assert read("evidence/preflight-eight-h100.json")["status"] == "passed"
    assert read("evidence/compile-check.json")["status"] == "passed"

    scheduler = read("evidence/final-scheduler.json")
    submission = read("artifacts/submission.json")
    total_samples, initial_hashes, adaptation_counters, arm_proofs = 0, [], [], {}
    required_cfg = yaml.safe_load((ROOT / "src/configs/t5_adaptation/common.yml").read_text())
    template = yaml.safe_load((ROOT / "src/configs/training_configs/train_owt_ELF-B.yml").read_text())
    allowed_overrides = {"data_path", "encoder_model_name", "latent_mean", "latent_std", "sampling_configs_path",
                         "eval_freq", "output_dir", "use_wandb", "wandb_run_name"}
    for arm in ("C", "D"):
        root = ROOT / "artifacts" / arm
        sched = next(j for j in scheduler["arms"] if j["arm"] == arm)
        assert sched["state"] == "SUCCEEDED" and sched["account"] == "snow.research.er"
        assert len(sched["runs"]) == 1 and sched["runs"][0]["exitCode"] == 0
        assert sched["resources"]["gpu"] == 8 and sched["resources"]["gpuModel"] == "H100"
        assert len(submission["arms"][arm]["jobs"]) == 1
        assert submission["arms"][arm]["jobs"][0]["id"] == sched["id"]
        assert submission["arms"][arm]["scheduler_state"] == "SUCCEEDED"
        runtime = read(root / "runtime.json")
        assert runtime["git_commit"] == RELEASE and len(runtime["gpus"]) == 8
        assert all("H100" in name for name in runtime["gpus"])
        assert runtime["torch"] == "2.5.1+cu124" and runtime["cuda"] == "12.4"
        assert len(runtime["project_python_sha256"]) == 37
        for path, digest in runtime["project_python_sha256"].items():
            verify(path, digest)
            verify(ROOT / "artifacts/releases" / RELEASE / path, digest)
        packages = {k.lower().replace("_", "-"): v for k, v in runtime["packages"].items()}
        for line in (ROOT / "requirements-adaptation-gpu.lock").read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                name, version = line.split("==")
                assert packages[name.lower().replace("_", "-")] == version

        identity = read(root / "adaptation/identity.json")
        cfg = load_config(ROOT / f"src/configs/t5_adaptation/arm_{arm.lower()}.yml")
        assert identity["config"] == cfg and not identity["smoke"]
        assert all(cfg[k] == v for k, v in required_cfg.items())
        assert cfg["updates"] == 38147 and cfg["source_examples"] == 19531264
        assert identity["sources"] == source_pins
        assert identity["data_hash"] == c4["train"]["content_hash"]
        assert identity["normalization_hash"] == calibration["content_hash"]
        assert identity["original_model_hash"] == original["content_hash"]
        for name, digest in identity["implementation"].items():
            verify(ROOT / "src/t5_adaptation" / name, digest)
        final = read(f"evidence/final-adaptation-{arm}.json")
        verify(final["checkpoint"], final["checkpoint_sha256"])
        assert final["identity_sha256"] == object_hash(identity)
        assert final["all_floating_model_and_optimizer_tensors_finite"]
        assert final["model_parameter_dtypes"] == ["torch.float32"]
        assert final["optimizer_step_min"] == final["optimizer_step_max"] == 38147
        assert final["rank_rng_count"] == 8 and final["final_learning_rates"] == [1e-5, 1e-5]
        for step in (3815, 11445, 38147):
            meta = read(root / f"adaptation/milestone-{step:06d}.json")
            verify(root / f"adaptation/milestone-{step:06d}.pt", meta["sha256"])
            assert meta["counters"]["updates"] == step and meta["identity"] == identity
        final_meta = read(root / "adaptation/milestone-038147.json")
        assert final_meta["final"] and final_meta["counters"] == final["counters"]
        expected_counters = {"updates": 38147, "nominal_positions": 10000007168,
                             "source_examples": 19531264, "source_tokens": 11093757952,
                             "encoder_sequence_passes": 39062528,
                             "decoder_target_tokens": 2230695230 if arm == "C" else 0}
        assert all(final["counters"][k] == v for k, v in expected_counters.items())
        adaptation_counters.append({k: v for k, v in final["counters"].items() if k not in ("decoder_target_tokens", "gpu_hours", "wall_seconds")})
        for step in (0, 3815, 11445, 38147):
            diagnostic = read(root / f"adaptation/diagnostics-{step:06d}.json")
            assert set(diagnostic) == {"569", "1024"}
            for stats in diagnostic.values():
                assert stats["sample_count"] == 2048 and stats["finite_fraction"] == 1
                assert len(stats["covariance_eigenvalues"]) == 512
                assert all(math.isfinite(v) for v in stats["covariance_eigenvalues"])
        train_log = [json.loads(line) for line in (root / "adaptation/train.jsonl").read_text().splitlines()]
        assert any(r["updates"] > 500 and r["seconds_per_update"] > 0 and r["eta_seconds"] > 0 for r in train_log)
        assert any(r.get("task_encoder_grad_norm", 0) > 0 and r.get("sigreg_encoder_grad_norm", 0) > 0 for r in train_log)

        encoder = snapshot(root / "encoder")
        handoff = read(f"evidence/encoder-handoff-{arm}.json")
        assert encoder["content_hash"] == handoff["encoder_export_hash"]
        assert encoder["provenance"]["source_checkpoint_sha256"] == final["checkpoint_sha256"]
        assert encoder["provenance"]["raw_outputs"] is True
        assert encoder["state_hash"] != encoder["original_state_hash"] == calibration["model_state_hash"]
        assert handoff["reload_check"]["max_absolute_error"] == 0
        assert handoff["tokenizer_vocabulary_and_special_ids_unchanged"]
        norm = read(root / "owt_normalization.json")
        assert norm == read(f"evidence/normalization_owt_{arm}.json")
        assert object_hash({k: v for k, v in norm.items() if k != "content_hash"}) == norm["content_hash"]
        assert norm["identity"]["ids"] == list(range(4096))
        assert norm["identity"]["length"] == 1024 and norm["identity"]["use_bf16"] is True
        assert norm["identity"]["encoder_hash"] == encoder["content_hash"]
        assert norm["token_count"] == 3915472
        elf_cfg = yaml.safe_load((root / "elf.yml").read_text())
        assert all(elf_cfg[k] == v for k, v in template.items() if k not in allowed_overrides)
        assert elf_cfg["eval_freq"] == 0 and elf_cfg["hf_repo_id"] is None and elf_cfg["resume"] is None
        assert elf_cfg["encoder_model_name"] == elf_cfg["tokenizer_name"] == str(root / "encoder")
        assert elf_cfg["latent_mean"] == norm["mean"] and elf_cfg["latent_std"] == norm["std"]
        init, complete = read(root / "elf/initialization.json"), read(root / "elf/completed.json")
        assert init["fresh_initialization"] and init["seed"] == 42 and init["elf_parameters"] == 104579940
        initial_hashes.append(init["initial_elf_state_hash"])
        assert complete["epochs"] == 5 and complete["updates"] == 95090 and complete["encoder_unchanged"]
        deep = read(f"evidence/elf-{arm}-epoch-5.json")
        verify(deep["checkpoint"], deep["sha256"])
        verify("scripts/audit_elf_epoch.py", deep["auditor_sha256"])
        assert deep["step"] == 95090 and deep["epoch"] == 5 and deep["model_and_ema_finite"]
        assert deep["identity_sha256"] == object_hash(init["identity"])
        for shard in deep["optimizer_shards"]:
            verify(root / "elf" / shard["file"], shard["sha256"])
            assert shard["all_optimizer_tensors_finite"] and shard["optimizer_step_min"] == shard["optimizer_step_max"] == 95090
        assert len(deep["optimizer_shards"]) == deep["rank_rng_count"] == 8
        gen = read(f"evidence/generation-{arm}.json")
        verify(f"evidence/elf-{arm}-epoch-5.json", gen["checkpoint_audit_sha256"])
        verify(root / "results.json", gen["results_sha256"])
        verify("scripts/audit_final_generation.py", gen["auditor_sha256"])
        verify(root / "evaluation.yml", gen["evaluation_config_sha256"])
        assert gen["checkpoint_sha256"] == deep["sha256"] and gen["ema_state_hash"] == deep["ema_state_hash"]
        assert gen["source_hashes"] == runtime["project_python_sha256"]
        assert gen["encoder_hash"] == encoder["content_hash"] and gen["evaluator_hash"] == evaluator["content_hash"]
        assert gen["aggregate_statistics_verified"] and gen["generated_total"] == gen["requested_total"] == 6000
        for item in gen["per_seed"]:
            sample = Path(item["samples_file"])
            verify(sample, item["samples_sha256"])
            verify(sample.parent / "project_metrics.json", item["metrics_sha256"])
        total_samples += gen["generated_total"]
        previous_end = 0
        for stage in ("adaptation", "export", "owt_normalization", "readout", "elf", "generation", "metrics"):
            record = read(root / f"stage-{stage}.json")
            assert record["state"] == "completed" and record["exit_code"] == 0
            assert record["source_commit"] == RELEASE and record["started"] >= previous_end
            previous_end = record["ended"]
        assert read(root / "completed.json")["state"] == "completed"
        for pattern in ("*.json", "*.yml", "*.log", "compute.jsonl", "stage_attempts/*.json",
                        "adaptation/*.json", "adaptation/train.jsonl", "elf/*.json", "elf/*.yml",
                        "elf/train.jsonl", "elf/initial_smoke/*/*.jsonl"):
            for path in root.glob(pattern):
                verify(path)
        arm_proofs[arm] = {"adaptation_checkpoint_sha256": final["checkpoint_sha256"], "encoder_hash": encoder["content_hash"],
                           "elf_checkpoint_sha256": deep["sha256"], "ema_state_hash": deep["ema_state_hash"],
                           "elf_updates": 95090, "elf_epochs": 5, "generated_samples": 6000,
                           "execution_source_files_verified": 37, "retained_elf_checkpoint_count": len(list((root / "elf").glob("checkpoint_*")))}
        print(f"{arm}: source, budget, handoff, final state and generation artifact hashes verified", flush=True)

    assert initial_hashes[0] == initial_hashes[1] and adaptation_counters[0] == adaptation_counters[1]
    assert total_samples == 12000
    probes = [read(f"evidence/readout-{arm}.json") for arm in ("original", "C", "D")]
    assert len({p["initial_readout_hash"] for p in probes}) == 1
    assert all(p["status"] == "completed" and p["identity"]["updates"] == 500 for p in probes)
    compute = read("compute.json")
    assert len(compute["scheduler_runs"]) == 6
    events = []
    for job in compute["scheduler_runs"]:
        assert job["account"] == "snow.research.er" and not job["provisional"] and job["ended_at"]
        assert math.isclose(job["allocated_gpu_hours"], job["gpus"] * job["wall_seconds"] / 3600)
        if job["gpus"] == 8:
            assert job["cpu"] == 64 and job["ram_gb"] == 512
            events.extend([(job["started_at"], 1), (job["ended_at"], -1)])
        elif job["gpus"] == 1:
            assert job["cpu"] == 16 and job["ram_gb"] == 128
        else:
            assert job["gpus"] == 0 and job["cpu"] == 32 and job["ram_gb"] == 128
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert active == 0 and peak == 2
    assert math.isclose(compute["allocated_gpu_hours"], sum(j["allocated_gpu_hours"] for j in compute["scheduler_runs"]))
    assert math.isclose(compute["allocated_gpu_hours"], 405.74666666666667)
    results = read("results.json")
    assert results["complete"] and results["B"] == "intentionally omitted" and len(results["rows"]) == 5
    reference = results["rows"][0]
    assert reference["origin"] == "published_external" and reference["measured_here"] is False
    assert [reference[k] for k in ("gen_ppl_mean", "gen_ppl_se", "entropy_mean", "entropy_se")] == [24.08, .16, 5.15, .002]
    assert len(read("qualitative_samples.json")["groups"]) == len(read("lengths.json")["rows"]) == 4
    for row in results["rows"][1:]:
        measured = read(f"artifacts/{row['arm']}/results.json")
        aggregate = next(v for v in measured["aggregates"] if v["configuration"] == row["configuration"])
        assert row["measured_here"] and row["origin"] == "measured_this_project"
        for prefix, metric in (("gen_ppl", "gen_ppl"), ("entropy", "mean_sample_unigram_entropy")):
            for suffix, key in (("mean", "mean"), ("se", "sem"), ("std", "sample_std")):
                assert row[f"{prefix}_{suffix}"] == aggregate[metric][key]
        seeds = [v for v in measured["per_seed"] if v["configuration"] == row["configuration"]]
        for key in ("requested_count", "generated_count", "nonempty_count", "ppl_scored_count", "entropy_scored_count"):
            assert row[key] == sum(v[key] for v in seeds)
        length_row = next(v for v in read("lengths.json")["rows"] if v["arm"] == row["arm"] and v["configuration"] == row["configuration"])
        for field in ("gpt2_lengths_before_truncation", "gpt2_lengths_after_truncation", "t5_retokenized_lengths"):
            values = [n for seed in seeds for n in seed[field]]
            assert length_row[field] == {"count": len(values), "min": min(values), "max": max(values),
                                         "mean": statistics.mean(values), "median": statistics.median(values)}
        group = next(v for v in read("qualitative_samples.json")["groups"] if v["arm"] == row["arm"] and v["configuration"] == row["configuration"])
        seed_zero = next(v for v in seeds if v["seed"] == 0)
        assert group["samples"] == seed_zero["qualitative_samples"] and group["seed"] == 0
        assert group["samples_sha256"] == seed_zero["samples_sha256"]
    for path in ("artifacts/submission.json", "artifacts/shared/c4_normalization.json", "artifacts/shared/original_probe.json"):
        verify(path)

    timestamp = datetime.now(timezone.utc).isoformat()
    index = {"recorded_at_utc": timestamp, "root": str(ROOT), "storage_root": str(ROOT / "artifacts"),
             "retention": "Selected final products listed below were freshly hashed. All prior recovery checkpoints remain in place. Git does not contain the large artifact payloads; use the designated experiment storage.",
             "verified_files": {k: v for k, v in VERIFIED.items() if k.startswith("artifacts/")},
             "shared_data_files": DATA_RETAINED, "arms": arm_proofs}
    atomic_json(ROOT / "evidence/final-artifacts.json", index)
    for requirement in reviewed["requirements"]:
        for path in requirement["evidence"]:
            assert (ROOT / path).exists(), f"Missing requirement {requirement['id']} evidence: {path}"
            if (ROOT / path).is_file():
                verify(path)
    for path in [ROOT / "GOAL.md", ROOT / "EXPERIMENT.md", ROOT / "RESULTS.md", Path(__file__),
                 ROOT / "scripts/finalize_cd_report.py", *ROOT.glob("*.csv"), *ROOT.glob("*.json"), *ROOT.glob("evidence/*")]:
        if path.is_file() and path.name != "acceptance.json":
            verify(path)
    outcome = {"status": "accepted", "audited_at_utc": timestamp, "goal_sha256": verify("GOAL.md"),
               "execution_commit": RELEASE, "reviewed_head_before_closeout_commit": git("rev-parse", "HEAD"),
               "protected_main": main_ref, "upstream_base": BASE, "reviewed_requirements": reviewed["requirements"],
               "scope": reviewed["scope"], "current_verified_files": VERIFIED,
               "arms": arm_proofs, "allocated_gpu_hours": compute["allocated_gpu_hours"],
               "max_simultaneous_eight_gpu_allocations": peak,
               "limits": ["Historical numerical tests are reused after source-hash continuity checks; no new training jobs were launched.",
                          "Final deep checkpoint and generation audits are reused after rehashing their bound artifacts; GPT-2 model forward passes are not rerun.",
                          "ELF EMA use is supported by inspected source and the audited state, not an independent runtime tensor trace.",
                          "Full-epoch exposure is inferred from saved steps and the tested sampler, not a per-example runtime trace.",
                          "Large C4/OWT files retain preparation hashes; final auditing checks their presence and sizes plus the actual C4 calibration prefix, not a full data rescan.",
                          "Git commit/push verification is a subsequent delivery step and is not inferred from scheduler success."]}
    atomic_json(ROOT / "evidence/acceptance.json", outcome)
    lines = ["# Final GOAL.md acceptance audit", "", f"Accepted on {timestamp}. All **51 reviewed requirement groups** are satisfied for the two completed pipelines. This assessment combines source review, focused checks, execution records, and current artifact verification; it is not just the scheduler's success flag.", "",
             "The [machine-readable audit](evidence/acceptance.json) binds the goal, source, evidence and report hashes. The [artifact index](evidence/final-artifacts.json) identifies the retained local products. The scientific findings are in [RESULTS.md](RESULTS.md).", "",
             "Final facts: 38,147 T5 updates and 10,000,007,168 nominal positions per arm; five ELF epochs / 95,090 updates per arm; 12,000 final generated texts; 405.74666666666667 allocated GPU hours. Both original ER jobs succeeded with exit 0. No required diagnostic or evaluation remains outstanding.", "",
             "The seven completion criteria in GOAL.md are covered by requirement groups 08–25 (objectives/correctness), 05–11 (same initialization/data budget), 27–29 (exports), 30–33 (fresh five-epoch ELF), 42–46 (final evaluation), 26/40/47–48 (reporting/limits), and 04/25/33/49 (environment/reproduction/resume).", "",
             "| ID / GOAL section | Requirement and reviewed finding | Evidence |", "| --- | --- | --- |"]
    for row in reviewed["requirements"]:
        refs = ", ".join(f"[{Path(p).name}]({p})" for p in row["evidence"])
        lines.append(f"| {row['id']} / {row['section']} | **{row['requirement']}** — {row['finding']} | {refs} |")
    lines += ["", "Audit limits:", ""] + [f"- {limit}" for limit in outcome["limits"]]
    lines += ["", "One training seed per arm and no matched-entropy sweep limit the scientific conclusion. A is external, B is omitted, and the results do not isolate SIGReg's causal effect or demonstrate overall generation improvement. These are interpretation limits, not unfinished experimental stages.", ""]
    (ROOT / "ACCEPTANCE.md").write_text("\n".join(lines))
    print(json.dumps({"status": "accepted", "reviewed_requirement_groups": 51,
                      "freshly_hashed_files": len(VERIFIED), "retained_shared_data_files": len(DATA_RETAINED),
                      "generated_samples": total_samples, "gpu_hours": compute["allocated_gpu_hours"]}))


if __name__ == "__main__":
    main()
