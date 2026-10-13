"""Write factual result and compute tables from saved measurements and job state."""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import statistics
import time

from .common import ROOT, atomic_json, read_json

PUBLISHED_A = {
    "arm": "A", "origin": "published_external", "configuration": "primary",
    "gen_ppl_mean": 24.08, "gen_ppl_se": 0.16, "entropy_mean": 5.15, "entropy_se": 0.002,
    "uncertainty": "reported standard error over evaluation seeds 0–5, n=6",
    "source": "https://arxiv.org/html/2605.10938v2#A5.T9", "table": "Appendix E.4, Table 9",
    "steps": 32, "gamma": 1.5, "self_cond_cfg": 3.0,
    "status": "published", "measured_here": False,
}


def summarize_jobs(artifacts):
    paths = [artifacts / "shared/preparation-job.json", *sorted((artifacts / "preflight").rglob("job.json")),
             *sorted((artifacts / "compile-check").rglob("job.json"))]
    job_ids = {}
    for path in paths:
        if path.exists():
            job = read_json(path)
            job_ids[job["id"]] = "shared_preparation" if "shared" in path.parts else "compile_check" if "compile-check" in path.parts else "preflight"
    submission = artifacts / "submission.json"
    if submission.exists():
        for arm, record in read_json(submission)["arms"].items():
            for job in record["jobs"]:
                job_ids[job["id"]] = arm
    records = []
    for job_id, stage in job_ids.items():
        output = subprocess.check_output(["eai", "job", "get", job_id, "--account", "snow.research.er",
                                          "--format", "json"], text=True)
        job = json.loads(output)
        for run in job["runs"]:
            seconds = 0.0
            if run["startedOn"]:
                begin = dt.datetime.fromisoformat(run["startedOn"].replace("Z", "+00:00"))
                end = dt.datetime.fromisoformat(run["endedOn"].replace("Z", "+00:00")) if run["endedOn"] else dt.datetime.now(dt.timezone.utc)
                seconds = max(0, (end - begin).total_seconds())
            records.append({"job_id": job_id, "run_id": run["id"], "arm_or_stage": stage,
                            "account": job["account"], "state": run["state"], "state_info": run["stateInfo"],
                            "gpus": job["resources"]["gpu"], "gpu_model_filter": job["resources"]["gpuModel"],
                            "cpu": job["resources"]["cpu"], "ram_gb": job["resources"]["mem"],
                            "node": run["nodeName"], "wall_seconds": seconds,
                            "started_at": begin.timestamp() if run["startedOn"] else None,
                            "ended_at": end.timestamp() if run["endedOn"] and run["startedOn"] else None,
                            "allocated_gpu_hours": seconds * job["resources"]["gpu"] / 3600,
                            "provisional": run["endedOn"] is None, "exit_code": run["exitCode"]})
    return records


def optional_json(path):
    return read_json(path) if path.exists() else {}


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_training(artifacts):
    records = []
    for arm in ("C", "D"):
        root = artifacts / arm
        checkpoint = optional_json(root / "adaptation/latest.json")
        adaptation = checkpoint.get("counters", {})
        latest_status = optional_json(root / "adaptation/status.json")
        initial_t5 = optional_json(root / "adaptation/initial_encoder.json")
        encoder = optional_json(root / "encoder/manifest.json")
        normalization = optional_json(root / "owt_normalization.json")
        initialization = optional_json(root / "elf/initialization.json")
        elf = optional_json(root / "elf/completed.json")
        elf_progress = optional_json(root / "elf/status.json")
        record = {"arm": arm, "adaptation_final_checkpoint": checkpoint.get("final", False),
                  "initial_t5_encoder_state_hash": initial_t5.get("state_hash"),
                  "adaptation_data_hash": checkpoint.get("identity", {}).get("data_hash"),
                  "adaptation_normalization_hash": checkpoint.get("identity", {}).get("normalization_hash"),
                  "adaptation_checkpoint_sha256": checkpoint.get("sha256"),
                  "adaptation_saved_updates": adaptation.get("updates"),
                  "adaptation_observed_updates": latest_status.get("updates"),
                  "nominal_positions": adaptation.get("nominal_positions"),
                  "source_examples": adaptation.get("source_examples"),
                  "source_tokens": adaptation.get("source_tokens"),
                  "decoder_target_tokens": adaptation.get("decoder_target_tokens"),
                  "corruption_target_tokens": adaptation.get("corruption_target_tokens"),
                  "eligible_tokens": adaptation.get("eligible_tokens"),
                  "noise_tokens": adaptation.get("noise_tokens"),
                  "noise_spans": adaptation.get("noise_spans"),
                  "encoder_sequence_passes": adaptation.get("encoder_sequence_passes"),
                  "encoder_nonpadding_positions": adaptation.get("encoder_nonpadding_positions"),
                  "diagnostic_encoder_sequence_passes": adaptation.get("diagnostic_encoder_sequence_passes"),
                  "owt_latent_mean": normalization.get("mean"), "owt_latent_std": normalization.get("std"),
                  "owt_calibration_tokens": normalization.get("token_count"),
                  "owt_normalization_hash": normalization.get("content_hash"),
                  "adapted_encoder_export_hash": encoder.get("content_hash"),
                  "encoder_state_hash": initialization.get("encoder_state_hash"),
                  "fresh_elf_initialization": initialization.get("fresh_initialization", False),
                  "initial_elf_state_hash": initialization.get("initial_elf_state_hash"),
                  "elf_parameters": initialization.get("elf_parameters"),
                  "elf_epochs_completed": elf.get("epochs"), "elf_updates_completed": elf.get("updates"),
                  "elf_observed_updates": elf_progress.get("step"),
                  "elf_observed_epoch": elf_progress.get("epoch"),
                  "owt_dataset_examples": elf.get("dataset_examples"),
                  "elf_steps_per_epoch": elf.get("steps_per_epoch"),
                  "padding_repeats_per_epoch": elf.get("padding_repeats_per_epoch"),
                  "encoder_unchanged": elf.get("encoder_unchanged", False),
                  "final_elf_checkpoint": elf.get("final_checkpoint")}
        for stage, field in (("adaptation", "seconds_per_update"), ("elf", "steps_per_second")):
            path = root / stage / "train.jsonl"
            logs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            values = [entry[field] for entry in logs if field in entry]
            record[f"{stage}_logged_{field}_median"] = statistics.median(values) if values else None
            record[f"{stage}_rank0_peak_memory_bytes"] = max((entry["max_memory_bytes"] for entry in logs), default=None)
        if adaptation.get("eligible_tokens"):
            record["actual_noise_density"] = adaptation["noise_tokens"] / adaptation["eligible_tokens"]
            record["actual_mean_noise_span_length"] = adaptation["noise_tokens"] / adaptation["noise_spans"]
        records.append(record)
    return records


def summarize_stage_compute(artifacts, scheduler_runs):
    rows = []
    for arm in ("C", "D"):
        root = artifacts / arm
        paths = sorted((root / "stage_attempts").glob("*.json"))
        if paths:
            entries = [read_json(path) for path in paths]
        else:
            path = root / "compute.jsonl"
            entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            entries += [record for path in root.glob("stage-*.json") if (record := read_json(path))["state"] == "running"]
        for entry in sorted(entries, key=lambda item: item["started"]):
            ended = entry.get("ended")
            state = entry["state"]
            timing_source = "stage_process"
            if ended is None:
                runs = [run for run in scheduler_runs if run["arm_or_stage"] == arm
                        and run["started_at"] is not None and run["started_at"] <= entry["started"]
                        and (run["ended_at"] is None or entry["started"] <= run["ended_at"])]
                run = max(runs, key=lambda item: item["started_at"]) if runs else None
                if run and run["ended_at"] is not None:
                    ended, state, timing_source = run["ended_at"], "interrupted", "scheduler_run_end"
                elif not run or run["state"] != "RUNNING":
                    state = "unresolved"
            seconds = (ended - entry["started"] if ended is not None else
                       time.time() - entry["started"] if state == "running" else None)
            rows.append({"arm": arm, "stage": entry["stage"], "state": state,
                         "started": entry["started"], "ended": ended, "allocated_gpus": entry["allocated_gpus"],
                         "wall_seconds": seconds, "allocated_gpu_hours": seconds * entry["allocated_gpus"] / 3600 if seconds is not None else None,
                         "exit_code": entry.get("exit_code"), "source_commit": entry["source_commit"],
                         "provisional": ended is None, "timing_source": timing_source})
    return rows


def summarize_diagnostics(artifacts):
    probes = {"original": artifacts / "shared/original_probe.json", "C": artifacts / "C/readout.json", "D": artifacts / "D/readout.json"}
    results = {"readout": {key: read_json(path) if path.exists() else {"status": "not_run_yet"}
                            for key, path in probes.items()}, "representations": []}
    for arm in ("C", "D"):
        for path in sorted((artifacts / arm / "adaptation").glob("diagnostics-*.json")):
            update = int(path.stem.split("-")[-1])
            for length, values in read_json(path).items():
                results["representations"].append({"arm": arm, "updates": update,
                                                   "nominal_positions": update * 512 * 512,
                                                   "clean_length": int(length), "source_file": str(path), **values})
    return results


def recorded_completion(artifacts, training, evaluations):
    """Conservative artifact-level status; final scientific audit is separate."""
    checks = {}
    for record in training:
        arm = record["arm"]
        root = artifacts / arm
        expected_samples = {(seed, config) for seed in (0, 1, 2) for config in ("primary", "secondary")}
        result = evaluations.get(arm, {})
        samples = result.get("per_seed", [])
        checks[arm] = {
            "pipeline_finished": optional_json(root / "completed.json").get("state") == "completed",
            "fixed_adaptation_budget": record["adaptation_final_checkpoint"] and record["adaptation_saved_updates"] == 38147
                and record["nominal_positions"] == 10000007168 and record["source_examples"] == 19531264
                and record["source_tokens"] == 11093757952,
            "fresh_frozen_elf": record["fresh_elf_initialization"] and record["encoder_unchanged"],
            "five_full_elf_epochs": record["elf_epochs_completed"] == 5
                and record["elf_updates_completed"] == 5 * math.ceil(record["owt_dataset_examples"] / 512)
                if record["owt_dataset_examples"] else False,
            "six_final_evaluations": result.get("status") == "completed"
                and len(samples) == 6 and {(r["seed"], r["configuration"]) for r in samples} == expected_samples
                and all(r["requested_count"] == r["generated_count"] == 1000 for r in samples),
        }
    hashes = [row["initial_elf_state_hash"] for row in training]
    checks["same_fresh_elf_initialization"] = bool(hashes[0]) and len(set(hashes)) == 1
    complete = checks["same_fresh_elf_initialization"] and all(all(checks[arm].values()) for arm in ("C", "D"))
    return complete, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--output", default=str(ROOT))
    args = parser.parse_args()
    artifacts, output = Path(args.artifacts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [PUBLISHED_A]
    status, evaluations = {}, {}
    for arm in ("C", "D"):
        path = artifacts / arm / "results.json"
        if path.exists():
            result = read_json(path)
            evaluations[arm] = result
            status[arm] = result["status"]
            for record in result["aggregates"]:
                samples = [item for item in result["per_seed"] if item["configuration"] == record["configuration"]]
                counts = {field: sum(item[field] for item in samples) for field in
                          ("requested_count", "generated_count", "nonempty_count", "ppl_scored_count", "entropy_scored_count")}
                rows.append({"arm": arm, "origin": "measured_this_project", "configuration": record["configuration"],
                             "status": result["status"], "gen_ppl_mean": record["gen_ppl"]["mean"],
                             "gen_ppl_se": record["gen_ppl"]["sem"], "gen_ppl_std": record["gen_ppl"]["sample_std"],
                             "entropy_mean": record["mean_sample_unigram_entropy"]["mean"],
                             "entropy_se": record["mean_sample_unigram_entropy"]["sem"],
                             "entropy_std": record["mean_sample_unigram_entropy"]["sample_std"],
                             "empty_fraction": record["empty_output_fraction"]["mean"],
                             **counts,
                             "uncertainty": "three inference seeds 0,1,2; one training seed",
                             "measured_here": True, "source": str(path)})
        else:
            stages = sorted((artifacts / arm).glob("stage-*.json"))
            status[arm] = {p.stem.removeprefix("stage-"): read_json(p)["state"] for p in stages} or "not_submitted"
            for configuration in ("primary", "secondary"):
                rows.append({"arm": arm, "origin": "measured_this_project", "configuration": configuration,
                             "status": "incomplete", "gen_ppl_mean": None, "entropy_mean": None,
                             "measured_here": False})
    compute = summarize_jobs(artifacts)
    training = summarize_training(artifacts)
    stage_compute = summarize_stage_compute(artifacts, compute)
    diagnostics = summarize_diagnostics(artifacts)
    complete, completion_checks = recorded_completion(artifacts, training, evaluations)
    atomic_json(output / "results.json", {"rows": rows, "arm_status": status,
                                          "B": "intentionally omitted", "complete": complete,
                                          "recorded_completion_checks": completion_checks})
    total_gpu_hours = sum(r["allocated_gpu_hours"] for r in compute)
    atomic_json(output / "compute.json", {"scheduler_runs": compute, "stage_attempts": stage_compute,
                                          "allocated_gpu_hours": total_gpu_hours,
                                          "accounting": "Scheduler allocation totals include idle/overhead time. Stage subtotals describe the same allocations and must not be added again."})
    atomic_json(output / "training.json", {"arms": training, "budget_convention": "512 nominal positions per source example, counted once"})
    atomic_json(output / "diagnostics.json", diagnostics)
    flat_diagnostics = [{key: value for key, value in row.items() if key != "covariance_eigenvalues"}
                        for row in diagnostics["representations"]]
    evaluation_rows = [{key: value for key, value in item.items() if not isinstance(value, (dict, list))}
                       for result in evaluations.values() for item in result["per_seed"]]
    for name, table in (("results.csv", rows), ("compute.csv", compute), ("stage_compute.csv", stage_compute),
                        ("training.csv", training), ("diagnostics.csv", flat_diagnostics), ("evaluation.csv", evaluation_rows)):
        write_csv(output / name, table)
    text = ["# C/D experiment results", "", "Experiment status: " + ("complete" if complete else "incomplete — training/evaluation requirements remain") + ".", "",
            "| Arm | Source | Configuration | GenPPL ± SE | Mean sample entropy ± SE | Status |",
            "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        ppl = "—" if row.get("gen_ppl_mean") is None else f"{row['gen_ppl_mean']:.4f}"
        entropy = "—" if row.get("entropy_mean") is None else f"{row['entropy_mean']:.4f}"
        if row.get("gen_ppl_se") is not None:
            ppl += f" ± {row['gen_ppl_se']:.4f}"
        if row.get("entropy_se") is not None:
            entropy += f" ± {row['entropy_se']:.4f}"
        text.append(f"| {row['arm']} | {row['origin']} | {row['configuration']} | {ppl} | {entropy} | {row['status']} |")
    text.extend(["", "Published A is 24.08 ± 0.16 GenPPL and 5.15 ± 0.002 entropy (mean ± SE over six inference seeds 0–5), from [ELF v2, Appendix E.4, Table 9](https://arxiv.org/html/2605.10938v2). It uses 32 SDE steps, gamma 1.5, and self-conditioning CFG 3. These are external published measurements.", "",
                 "C/D use one training seed each and inference seeds 0,1,2, with 1,000 samples per seed per configuration. Error estimates in the machine-readable table describe inference variation only. GenPPL is the repository's frozen GPT-2 Large generative perplexity, not ELF held-out likelihood; entropy is mean per-sample unigram entropy after GPT-2 retokenization.", "",
                 "B is intentionally omitted. C versus A includes continued pretraining, the C4 stream, SIGReg, and implementation differences; it cannot isolate SIGReg's causal effect. C versus D compares complete recipes. Lower perplexity with lower entropy is a quality/diversity trade-off, without a matched-entropy frontier claim.", ""])
    primary = {r["arm"]: r for r in rows if r["configuration"] == "primary"}
    for left, right in (("C", "A"), ("D", "A"), ("C", "D")):
        a, b = primary[left], primary[right]
        if a.get("gen_ppl_mean") is None or b.get("gen_ppl_mean") is None:
            text.append(f"{left} versus {right}: unavailable until required C/D measurements finish.")
        else:
            text.append(f"{left} minus {right}: GenPPL {a['gen_ppl_mean'] - b['gen_ppl_mean']:+.4f}, entropy {a['entropy_mean'] - b['entropy_mean']:+.4f}.")
    text.extend(["", "| Arm | Saved nominal positions | Saved T5 updates | Completed ELF epochs | Completed ELF updates |",
                 "| --- | --- | --- | --- | --- |"])
    display = lambda value: "—" if value is None else f"{value:,}"
    for record in training:
        values = [display(record[key]) for key in ("nominal_positions", "adaptation_saved_updates", "elf_epochs_completed", "elf_updates_completed")]
        text.append(f"| {record['arm']} | " + " | ".join(values) + " |")
    text.extend(["", "`training.json/csv` distinguish saved checkpoints from observed progress and retain actual source/target counts, encoder passes, corruption rate/span length, OWT normalization, initialization hashes, and full-epoch padding repeats."])
    measured = [row for row in rows if row["measured_here"]]
    if measured:
        text.extend(["", "| Arm | Configuration | Requested / generated | Nonempty | PPL / entropy scored | Empty fraction |",
                     "| --- | --- | --- | --- | --- | --- |"])
        for row in measured:
            text.append(f"| {row['arm']} | {row['configuration']} | {row['requested_count']} / {row['generated_count']} | {row['nonempty_count']} | {row['ppl_scored_count']} / {row['entropy_scored_count']} | {row['empty_fraction']:.4%} |")
        text.extend(["", "Counts sum the three inference seeds. `evaluation.csv` and the per-arm `results.json` retain per-seed scores, lengths, sample paths/hashes, and qualitative samples. Empty outputs are included in the requested/generated counts and explicitly excluded from scoring."])
    probe_results = diagnostics["readout"]
    probe_text = "The matched 500-update frozen-encoder readout diagnostic is complete for original/C/D; see `diagnostics.json`." if all(p["status"] == "completed" for p in probe_results.values()) else "The matched 500-update token-readout diagnostics are not all complete yet; see `diagnostics.json`."
    text.extend(["", probe_text + " Held-out representation summaries and covariance spectra appear in `diagnostics.json`; `diagnostics.csv` provides the scalar milestones at both 569 and 1024 positions. C's span CE and D's consistency are different objectives and are not comparable language-model losses.", "",
                 f"Measured scheduler allocation so far: **{total_gpu_hours:.3f} GPU hours**, including preparation, failed attempts, and pilots. Running-job totals are provisional; queued time is excluded. `compute.csv/json` provide allocation totals and `stage_compute.csv` separates adaptation, ELF, calibration, and evaluation attempts. Stage totals are part of the allocation total and must not be added a second time.", "",
                 f"Current arm state: C `{status['C']}`; D `{status['D']}`.", ""])
    (output / "RESULTS.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
