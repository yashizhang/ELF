"""Audit completed C/D generation artifacts on CPU without rerunning GPT-2."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from transformers import AutoTokenizer
import yaml

from t5_adaptation.common import (
    ROOT, atomic_json, directory_manifest, file_hash, object_hash, read_json, sources,
)


PROTOCOL = {
    "primary": {"sampling_method": "sde", "num_sampling_steps": [32],
                "cfgs": [1], "sde_gamma": 1.5, "self_cond_cfg_scales": [3],
                "time_schedule": "logit_normal"},
    "secondary": {"sampling_method": "sde", "num_sampling_steps": [64],
                  "cfgs": [1], "sde_gamma": 1.0, "self_cond_cfg_scales": [3],
                  "time_schedule": "logit_normal"},
}
METRIC_FIELDS = ("gen_ppl", "mean_sample_unigram_entropy", "empty_output_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, *, abs_tol=1e-10, rel_tol=1e-9):
    if expected is None:
        require(actual is None, f"{name}: expected an explicit null")
    else:
        require(isinstance(actual, (int, float)) and math.isfinite(actual),
                f"{name}: missing or non-finite value")
        require(math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
                f"{name}: {actual!r} != {expected!r}")


def check_rows(rows):
    require(len(rows) == 1000, "Expected exactly 1,000 saved samples")
    require(all(set(row) == {"id", "generated"} for row in rows), "Unexpected sample schema")
    require(all(type(row["id"]) is int for row in rows), "Sample IDs must be integers")
    require([row["id"] for row in rows] == list(range(1000)), "Missing, reordered or duplicate sample IDs")
    require(all(isinstance(row["generated"], str) for row in rows), "Sample text must be a string")


def token_statistics(input_ids, attention_mask, eos_token_id):
    """Independently count the repository's PPL weights and per-sample entropy.

    Its PPL mask includes the first EOS padding position; it is not simply
    attention_mask[:, 1:]. After a literal EOS it can also give weight two to
    ordinary tokens until the next EOS. Preserve this existing metric exactly.
    Entropy uses the attention mask and natural logarithms, one sample at a time.
    """
    weights, entropies = [], []
    for ids, mask in zip(input_ids, attention_mask):
        eos_seen, weight = 0, 0
        for position, token in enumerate(ids):
            eos_seen += int(token == eos_token_id)
            if position:
                weight += int(eos_seen == 1) + int(token != eos_token_id)
        weights.append(weight)
        tokens = ids[:int(sum(mask))]
        require(len(tokens) > 0, "A nonempty text retokenized to no GPT-2 tokens")
        counts = Counter(map(int, tokens))
        entropies.append(-math.fsum((n / len(tokens)) * math.log(n / len(tokens))
                                   for n in counts.values()))
    return weights, entropies


def reconstruct_ppl(per_sample, weights):
    require(len(per_sample) == len(weights), "Per-sample PPL count mismatch")
    weighted_logs = []
    for value, weight in zip(per_sample, weights):
        if not weight:
            require(value is None, "A sample with no scored token must have null PPL")
        else:
            require(value is not None and math.isfinite(value) and value > 0,
                    "Non-finite per-sample PPL with scored tokens needs separate review")
            weighted_logs.append(weight * math.log(value))
    return math.exp(math.fsum(weighted_logs) / sum(weights)) if sum(weights) else None


def length_summary(values):
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def check_seed_metrics(item, rows, gpt2_tokenizer, t5_tokenizer):
    check_rows(rows)
    texts = [row["generated"] for row in rows]
    keep_ids = [i for i, text in enumerate(texts) if text.strip()]
    kept = [texts[i] for i in keep_ids]
    require(item["requested_count"] == item["generated_count"] == 1000, "Requested/generated count mismatch")
    require(item["nonempty_count"] == item["entropy_scored_count"] == len(kept), "Nonempty/entropy count mismatch")
    close(item["empty_output_fraction"], 1 - len(kept) / 1000, "empty fraction")
    full_lengths = [len(ids) for ids in gpt2_tokenizer(texts, add_special_tokens=False)["input_ids"]]
    t5_lengths = [len(ids) for ids in t5_tokenizer(texts, add_special_tokens=False)["input_ids"]]
    require(item["gpt2_lengths_before_truncation"] == full_lengths, "GPT-2 lengths mismatch")
    require(item["gpt2_lengths_after_truncation"] == [min(1024, n) for n in full_lengths],
            "Truncated GPT-2 lengths mismatch")
    require(item["t5_retokenized_lengths"] == t5_lengths, "T5 retokenized lengths mismatch")
    require(item["qualitative_samples"] == rows[:3], "Qualitative samples must be the first three saved rows")
    if kept:
        encoded = gpt2_tokenizer(kept, return_tensors="np", return_token_type_ids=False,
                                 return_attention_mask=True, truncation=True, padding=True, max_length=1024)
        weights, entropies = token_statistics(encoded["input_ids"], encoded["attention_mask"],
                                              gpt2_tokenizer.eos_token_id)
    else:
        weights, entropies = [], []
    expected_ppl = reconstruct_ppl(item["per_sample_ppl_nonempty_order"], weights)
    expected_entropy = statistics.mean(entropies) if entropies else None
    close(item["gen_ppl"], expected_ppl, "token-weighted aggregate PPL")
    # The production entropy estimator uses FP32 counts/logs; this independent
    # reconstruction uses Python FP64 arithmetic without its 1e-10 log offset.
    close(item["mean_sample_unigram_entropy"], expected_entropy, "mean sample entropy", abs_tol=2e-6)
    require(item["ppl_scored_count"] == sum(weight > 0 for weight in weights), "PPL-scored count mismatch")
    return {
        "requested_count": 1000, "generated_count": len(rows), "nonempty_count": len(kept),
        "ppl_scored_count": item["ppl_scored_count"], "entropy_scored_count": len(kept),
        "empty_output_fraction": item["empty_output_fraction"],
        "empty_sample_ids": [row["id"] for row in rows if not row["generated"].strip()],
        "nonempty_samples_without_scored_tokens": [i for i, w in zip(keep_ids, weights) if not w],
        "ppl_token_weight_total": sum(weights), "reconstructed_gen_ppl": expected_ppl,
        "reconstructed_mean_sample_unigram_entropy": expected_entropy,
        "gpt2_lengths_before_truncation": length_summary(full_lengths),
        "gpt2_lengths_after_truncation": length_summary([min(1024, n) for n in full_lengths]),
        "t5_retokenized_lengths": length_summary(t5_lengths),
    }


def check_aggregates(results, arm):
    require(len(results["per_seed"]) == 6 and len(results["aggregates"]) == 2, "Missing seed/configuration results")
    require({row["configuration"] for row in results["aggregates"]} == set(PROTOCOL), "Aggregate configurations mismatch")
    for config in PROTOCOL:
        group = sorted((row for row in results["per_seed"] if row["configuration"] == config),
                       key=lambda row: row["seed"])
        require([row["seed"] for row in group] == [0, 1, 2], "Each configuration requires exactly seeds 0, 1, 2")
        aggregate = next(row for row in results["aggregates"] if row["configuration"] == config)
        require(aggregate["arm"] == arm and aggregate["origin"] == "measured_this_project", "Aggregate origin mismatch")
        require(aggregate["training_seed_count"] == 1 and aggregate["inference_seeds"] == [0, 1, 2], "Seed uncertainty mismatch")
        for field in METRIC_FIELDS:
            values = [row[field] for row in group]
            actual = aggregate[field]
            require(actual["values"] == values, f"{field}: seed values mismatch")
            expected = ({"mean": None, "sample_std": None, "sem": None} if any(v is None for v in values)
                        else {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                              "sem": statistics.stdev(values) / math.sqrt(3)})
            for statistic, value in expected.items():
                close(actual[statistic], value, f"{config}/{field}/{statistic}")


def verify_snapshot(directory):
    manifest = read_json(directory / "manifest.json")
    actual = directory_manifest(directory)
    require(actual["files"] == manifest["files"] and actual["content_hash"] == manifest["content_hash"],
            f"Snapshot content mismatch: {directory}")
    return manifest


def completed_stage(arm_root, name, source_commit):
    path = arm_root / f"stage-{name}.json"
    record = read_json(path)
    require(record["stage"] == name and record["state"] == "completed" and record["exit_code"] == 0,
            f"Stage not successfully completed: {name}")
    require(record["source_commit"] == source_commit, f"Stage source release mismatch: {name}")
    require(record["allocated_gpus"] == 8 and record["ended"] >= record["started"], f"Invalid stage record: {name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["C", "D"], required=True)
    parser.add_argument("--elf-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    arm_root, shared = ROOT / "artifacts" / args.arm, ROOT / "artifacts/shared"
    output = args.output or ROOT / "artifacts" / f"generation-{args.arm}-audit.json"
    require(not output.exists(), f"Preserve prior audit: {output}")
    require((arm_root / "completed.json").is_file(), f"Arm {args.arm} has not completed its pipeline yet")
    completed = read_json(arm_root / "completed.json")
    require(completed["state"] == "completed" and completed["arm"] == args.arm, "Pipeline completion mismatch")
    elf_done = read_json(arm_root / "elf/completed.json")
    require(elf_done["epochs"] == 5 and elf_done["updates"] == 95090 and elf_done["encoder_unchanged"] is True,
            "ELF training or frozen-encoder evidence is incomplete")
    require(elf_done["final_checkpoint"] == "checkpoint_95090", "Generation must use the final fixed-budget checkpoint")
    checkpoint = arm_root / "elf" / elf_done["final_checkpoint"]
    checkpoint_hash = file_hash(checkpoint)
    require(checkpoint_hash == completed["final_checkpoint_sha256"], "Final checkpoint hash mismatch")
    audit_path = args.elf_audit or ROOT / "artifacts" / f"elf-{args.arm}-epoch-5-audit.json"
    audit = read_json(audit_path)
    require(audit["arm"] == args.arm and audit["step"] == 95090 and audit["epoch"] == 5.0,
            "A matching final ELF checkpoint audit is required")
    require(audit["sha256"] == checkpoint_hash and audit["model_and_ema_finite"] is True,
            "Final checkpoint does not match its model/EMA audit")
    require(object_hash(elf_done["identity"]) == audit["identity_sha256"], "ELF completion identity mismatch")
    require(audit["auditor_sha256"] == file_hash(ROOT / "scripts/audit_elf_epoch.py"), "Checkpoint auditor source mismatch")
    runtime = read_json(arm_root / "runtime.json")
    release = ROOT / "artifacts/releases" / runtime["git_commit"]
    required_sources = {"src/eval.py", "src/generation.py", "src/configs/config.py",
                        "src/modules/model.py", "src/modules/layers.py",
                        "src/utils/checkpoint_utils.py", "src/utils/generation_utils.py",
                        "src/utils/sampling_utils.py", "src/utils/metrics_utils.py",
                        "src/t5_adaptation/pipeline.py", "src/t5_adaptation/evaluate.py"}
    source_files = sorted(runtime["project_python_sha256"])
    require(required_sources <= set(source_files), "Incomplete execution source manifest")
    source_hashes = {name: file_hash(release / name) for name in source_files}
    for name, digest in source_hashes.items():
        require(digest == runtime["project_python_sha256"][name] == file_hash(ROOT / name),
                f"Execution source mismatch: {name}")
    stages = {name: completed_stage(arm_root, name, runtime["git_commit"])
              for name in ("elf", "generation", "metrics")}
    require(stages["generation"]["started"] >= stages["elf"]["ended"], "Generation preceded ELF completion")
    require(stages["metrics"]["started"] >= stages["generation"]["ended"], "Scoring preceded generation completion")
    config_path = arm_root / "evaluation.yml"
    config = yaml.safe_load(config_path.read_text())
    training = yaml.safe_load((arm_root / "elf.yml").read_text())
    expected_config = training | {"output_dir": str(arm_root / "samples"), "online_eval": False,
                                  "eval_ppl_model": str(shared / "evaluator"), "num_samples": 1000, "use_compile": True}
    require(config == expected_config, "Evaluation changed training defaults or declared overrides")
    require(config["model"] == "ELF-B" and config["max_length"] == 1024 and config["global_batch_size"] == 512,
            "Evaluation model, length or batch mismatch")
    require(config["encoder_model_name"] == config["tokenizer_name"] == str(arm_root / "encoder"), "Encoder/tokenizer mismatch")
    require(config["hf_repo_id"] is None and config["eval_freq"] == 0, "Unexpected upload or intermediate sampling")
    sampling_path = Path(config["sampling_configs_path"])
    require(sampling_path == release / "src/configs/sampling_configs/uncond_sampling_configs.yml", "Sampling config source mismatch")
    require(yaml.safe_load(sampling_path.read_text()) == list(PROTOCOL.values()), "Sampling protocol mismatch")
    python = str(ROOT / ".venv-gpu/bin/python")
    expected_command = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
                        str(release / "src/eval.py"), "--config", str(config_path),
                        "--checkpoint_path", str(checkpoint), "--seeds", "0,1,2"]
    require(stages["generation"]["command"] == expected_command, "Generation launch protocol mismatch")
    require(stages["metrics"]["command"] == [python, "-m", "t5_adaptation.evaluate", "--samples-root",
            str(arm_root / "samples"), "--shared", str(shared), "--arm", args.arm,
            "--output", str(arm_root / "results.json")], "Metric command mismatch")
    encoder = verify_snapshot(arm_root / "encoder")
    evaluator = verify_snapshot(shared / "evaluator")
    require(evaluator["status"] == "complete" and evaluator["source"] == sources()["evaluator"], "Evaluator revision mismatch")
    require(evaluator["source"] == {"id": "openai-community/gpt2-large", "revision": "32b71b12589c2f8d625668d2335a01cac3249519"},
            "The declared GPT-2 Large revision is required")
    norm = read_json(arm_root / "owt_normalization.json")
    require(norm["content_hash"] == object_hash({k: v for k, v in norm.items() if k != "content_hash"}), "Normalization hash mismatch")
    require(norm["identity"]["encoder_hash"] == encoder["content_hash"], "Normalization encoder mismatch")
    require(config["latent_mean"] == norm["mean"] and config["latent_std"] == norm["std"], "Latent normalization mismatch")
    gpt2 = AutoTokenizer.from_pretrained(shared / "evaluator", local_files_only=True)
    gpt2.pad_token = gpt2.eos_token
    t5 = AutoTokenizer.from_pretrained(shared / "t5", local_files_only=True)
    exported_tokenizer = AutoTokenizer.from_pretrained(arm_root / "encoder", local_files_only=True)
    require(exported_tokenizer.get_vocab() == t5.get_vocab() and exported_tokenizer.all_special_ids == t5.all_special_ids,
            "Exported tokenizer changed the original vocabulary or special IDs")
    results_path = arm_root / "results.json"
    require(Path(completed["results"]) == results_path, "Pipeline result path mismatch")
    results = read_json(results_path)
    require(results["arm"] == args.arm and results["status"] == "completed", "Metric stage result is incomplete")
    check_aggregates(results, args.arm)
    summaries, expected_files = [], []
    for seed in (0, 1, 2):
        for name, spec in PROTOCOL.items():
            steps, gamma = spec["num_sampling_steps"][0], spec["sde_gamma"]
            run_name = f"sde-steps{steps}-cfg1-sccfg3-ts_logit_normal-gamma{gamma}-uncond"
            path = arm_root / "samples" / f"seed_{seed}" / run_name / "all_generated_5_95090.jsonl"
            expected_files.append(path)
            item = next(row for row in results["per_seed"] if row["seed"] == seed and row["configuration"] == name)
            expected = {"arm": args.arm, "origin": "measured_this_project", "steps": steps, "gamma": gamma,
                        "self_cond_cfg": 3.0, "cfg": 1.0, "time_schedule": "logit_normal", "sequence_length": 1024,
                        "evaluator": evaluator["source"], "evaluator_hash": evaluator["content_hash"],
                        "samples_file": str(path), "samples_sha256": file_hash(path)}
            require(all(item.get(k) == v for k, v in expected.items()), f"Per-seed protocol/provenance mismatch: {seed}/{name}")
            metric_path = path.parent / "project_metrics.json"
            require(read_json(metric_path) == item, "Per-seed metric file differs from aggregate input")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            summary = check_seed_metrics(item, rows, gpt2, t5)
            summaries.append(summary | {"seed": seed, "configuration": name, "samples_file": str(path),
                                        "samples_sha256": expected["samples_sha256"], "metrics_sha256": file_hash(metric_path)})
    require(set((arm_root / "samples").rglob("all_generated_*.jsonl")) == set(expected_files), "Unexpected/missing final sample files")
    record = {
        "arm": args.arm, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Final generation command/configuration and checkpoint provenance, raw samples, tokenization/exclusions, reconstructed entropy/PPL aggregation, and inference-seed statistics.",
        "limits": ["GPT-2 model forward passes are not rerun; PPL reconstruction uses the saved per-sample PPL values.",
                   "EMA selection is evidenced by the inspected generation source and audited final checkpoint, not an independent runtime tensor trace.",
                   "This artifact does not replace the complete GOAL.md acceptance audit."],
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_audit": str(audit_path), "checkpoint_audit_sha256": file_hash(audit_path),
        "ema_state_hash": audit["ema_state_hash"], "source_commit": runtime["git_commit"], "source_hashes": source_hashes,
        "evaluation_config_sha256": file_hash(config_path), "sampling_config_sha256": file_hash(sampling_path),
        "encoder_hash": encoder["content_hash"], "evaluator": evaluator["source"], "evaluator_hash": evaluator["content_hash"],
        "results_sha256": file_hash(results_path), "stage_records": stages,
        "requested_total": 6000, "generated_total": sum(row["generated_count"] for row in summaries),
        "per_seed": summaries, "aggregate_statistics_verified": True,
        "training_seed_count": 1, "inference_seeds": [0, 1, 2], "auditor_sha256": file_hash(__file__),
    }
    atomic_json(output, record)
    print(json.dumps({"audit": str(output), "arm": args.arm, "generated_total": record["generated_total"],
                      "sha256": file_hash(output)}))


if __name__ == "__main__":
    main()
