"""Audit completed study artifacts and export seed-level evidence without rescoring."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

from t5_adaptation.common import atomic_json, file_hash, read_json
from .common import METRICS, SAMPLERS, SCALES, summarize
from .submit import statuses


def csv_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def aggregate(rows):
    keys = sorted({(r["arm"], r["sampler"], r["stage"], r["sc_cfg"]) for r in rows})
    output = []
    for arm, sampler, stage, scale in keys:
        group = sorted([r for r in rows if (r["arm"], r["sampler"], r["stage"], r["sc_cfg"]) == (arm, sampler, stage, scale)], key=lambda r: r["seed"])
        result = {"arm": arm, "sampler": sampler, "stage": stage, "sc_cfg": scale,
                  "outside_trained_range": not .5 <= scale <= 5, "seeds": [r["seed"] for r in group],
                  "requested_count": sum(r["requested_count"] for r in group),
                  "generated_count": sum(r["generated_count"] for r in group),
                  "nonempty_count": sum(r["nonempty_count"] for r in group),
                  "scoring_failure_count": sum(len(r["scoring_failures"]) for r in group)}
        for metric in METRICS + ["t5_retokenized_length"]:
            result[metric] = summarize(r[metric] for r in group)
        output.append(result)
    return output


def paired_differences(rows):
    groups = {}
    for row in rows:
        if row["stage"] == "confirmation":
            groups.setdefault((row["arm"], row["sampler"], row["sc_cfg"]), {})[row["seed"]] = row
    comparisons = []
    for key in sorted(groups):
        arm, sampler, scale = key
        if arm in ("C", "D") and scale != 3:
            comparisons.append((key, (arm, sampler, 3), "within_arm_vs_sc3"))
        if arm in ("C", "D"):
            comparisons.append((key, ("A_measured_local", sampler, 3), "operating_point_vs_local_A"))
        if arm == "D" and ("C", sampler, scale) in groups:
            comparisons.append((key, ("C", sampler, scale), "same_scale_D_minus_C"))
    output = []
    for left, right, kind in comparisons:
        assert set(groups[left]) == set(groups[right]) == {20, 21, 22}
        row = {"comparison": kind, "sampler": left[1], "left_arm": left[0], "left_sc_cfg": left[2],
               "right_arm": right[0], "right_sc_cfg": right[2], "direction": "left minus right", "seeds": [20, 21, 22]}
        for metric in METRICS:
            values = [(groups[left][seed][metric] - groups[right][seed][metric])
                      if groups[left][seed][metric] is not None and groups[right][seed][metric] is not None else None for seed in (20, 21, 22)]
            row[metric] = summarize(values)
        output.append(row)
    return output


def flat_statistics(rows):
    output = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict) and "sample_sd" in value:
                flat.update({f"{key}_{stat}": number for stat, number in value.items()})
            else:
                flat[key] = value
        output.append(flat)
    return output


def audit_setting(path, inputs, selection):
    row = read_json(path)
    arm, sampler, scale, seed = [row[k] for k in ("arm", "sampler", "sc_cfg", "seed")]
    assert row["cfg"] == 1 and row["evaluator_hash"] == inputs["evaluator"]["content_hash"]
    assert row["identity"]["inputs_sha256"] == file_hash("guidance_sweep/inputs.json")
    assert row["identity"]["plan_sha256"] == file_hash("guidance_sweep/plan.json")
    for name, digest in row["artifact_hashes"].items():
        assert file_hash(path.parent / name) == digest
    samples = read_rows(path.parent / "samples.jsonl")
    scores = read_rows(path.parent / "per_sample_scores.jsonl")
    generation = read_json(path.parent / "generation.json")
    assert generation["identity"] == row["identity"]
    assert generation["checkpoint_sha256"] == inputs["arms"][arm]["checkpoint_sha256"]
    assert generation["ema_state_hash"] == inputs["arms"][arm]["ema_state_hash"]
    count = row["requested_count"]
    assert count == row["generated_count"] == len(samples) == len(scores)
    assert sum(generation["per_rank_requested_counts"]) == count and generation["extra_rank_padding_samples_trimmed"] == 0
    assert [s["id"] for s in samples] == [s["id"] for s in scores] == list(range(count))
    for sample, score in zip(samples, scores):
        assert score["text_sha256"] == hashlib.sha256(sample["generated"].encode()).hexdigest()
        assert score["nonempty"] == bool(sample["generated"].strip())
    assert row["nonempty_count"] == sum(s["nonempty"] for s in scores)
    assert row["ppl_scored_count"] == sum(s["ppl"] is not None for s in scores)
    assert row["entropy_scored_count"] == sum(s["entropy"] is not None for s in scores)
    assert row["scoring_failures"] == [s["id"] for s in scores if s["scoring_failure"]]
    assert row["empty_sample_ids"] == [s["id"] for s in scores if not s["nonempty"]]
    if row["gen_ppl"] is not None and not row["scoring_failures"]:
        eligible = [s for s in scores if s["ppl"] is not None]
        rebuilt = math.exp(math.fsum(math.log(s["ppl"]) * s["ppl_token_weight"] for s in eligible) / sum(s["ppl_token_weight"] for s in eligible))
        assert math.isclose(rebuilt, row["gen_ppl"], rel_tol=1e-9)
    entropy = [s["entropy"] for s in scores if s["entropy"] is not None]
    assert not entropy or abs(sum(entropy) / len(entropy) - row["entropy"]) < 2e-6
    if row["stage"] == "confirmation":
        assert seed in (20, 21, 22) and count == 1000
        assert row["started"] > datetime.fromisoformat(selection["frozen_at_utc"]).timestamp()
        if arm != "A_measured_local":
            assert scale in [r["sc_cfg"] for r in selection["arms"][arm][sampler]["selection"]["selected"]]
        else:
            assert scale == 3
    else:
        assert seed == 10 and count == 250
    row["metrics_file"] = str(path)
    row["metrics_sha256"] = file_hash(path)
    return row, samples, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default="guidance_sweep/results")
    args = parser.parse_args()
    inputs = read_json("guidance_sweep/inputs.json")
    selection = read_json("guidance_sweep/selection.json")
    root = Path(args.root or inputs["artifact_root"]).resolve()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    scheduler = statuses(root, read_json(root / "submission.json"))
    assert all(j["state"] not in {"RUNNING", "QUEUING", "QUEUED", "CANCELLING"} for j in scheduler["jobs"])
    assert all(not run["provisional"] for job in scheduler["jobs"] for run in job["runs"])
    compute = [{"arm": job["arm"], "phase": job["phase"], "job_id": job["id"], "account": job["account"],
                "job_state": job["state"], "gpus": job["resources"]["gpu"], "cpus": job["resources"]["cpu"],
                "ram_gb": job["resources"]["mem"], **run} for job in scheduler["jobs"] for run in job["runs"]]
    for arm in ("C", "D"):
        for phase in ("screening", "confirmation"):
            assert any(j["arm"] == arm and j["phase"] == phase and j["state"] == "SUCCEEDED" for j in scheduler["jobs"])
            assert read_json(root / arm / f"{phase}_job_completed.json")["state"] == "completed"
        assert read_json(root / arm / "guidance_audit.json")["ordinary_cfg_inactive"]
    for path, digest in inputs["original_tracked_files"].items():
        assert file_hash(path) == digest, path
    for path, digest in inputs["original_artifacts"].items():
        assert file_hash(path) == digest, path
    for arm, spec in inputs["arms"].items():
        assert file_hash(spec["checkpoint"]) == spec["checkpoint_sha256"]
        if arm == "A_measured_local":
            assert file_hash(spec["config"]) == spec["source"]["files"]["config.yml"]
        for phase in (["confirmation"] if arm == "A_measured_local" else ["screening", "confirmation"]):
            assert read_json(root / arm / f"{phase}_completed.json")["ema_unchanged"]

    rows, qualitative, artifacts = [], [], {}
    for arm in ("C", "D", "A_measured_local"):
        for path in sorted((root / arm).glob("*/*/sc_*/seed_*/metrics.json")):
            row, samples, scores = audit_setting(path, inputs, selection)
            rows.append(row)
            artifacts[str(path)] = row["metrics_sha256"]
            artifacts.update({str(path.parent / name): digest for name, digest in row["artifact_hashes"].items()})
            if row["stage"] == "confirmation" and row["seed"] == 20:
                qualitative.extend({"arm": arm, "sampler": row["sampler"], "sc_cfg": row["sc_cfg"], "seed": 20,
                                    **sample, "scores": score} for sample, score in zip(samples[:3], scores[:3]))
    for arm in ("C", "D"):
        for sampler in SAMPLERS:
            group = [r for r in rows if r["arm"] == arm and r["sampler"] == sampler]
            assert sorted(r["sc_cfg"] for r in group if r["stage"] == "screening") == SCALES
            assert sum(r["stage"] == "refinement" for r in group) <= 2
            low = next(r for r in group if r["stage"] == "screening" and r["sc_cfg"] == .5)
            extra = [r for r in group if r["stage"] == "extrapolation"]
            assert sorted(r["sc_cfg"] for r in extra) == ([0, .25] if low["entropy"] is not None and low["entropy"] < 5.10 else [])
            for stage in ("extrapolation", "refinement"):
                measured = [r for r in group if r["stage"] == stage]
                decision_path = root / arm / f"{stage}-{sampler}.json"
                if decision_path.exists():
                    decision = read_json(decision_path)
                    assert sorted(r["sc_cfg"] for r in measured) == sorted(decision["values"])
                    assert all(datetime.fromisoformat(decision["recorded_before_execution"]).timestamp() <= r["started"] for r in measured)
            expected = {(s["sc_cfg"], seed) for s in selection["arms"][arm][sampler]["selection"]["selected"] for seed in (20, 21, 22)}
            assert {(r["sc_cfg"], r["seed"]) for r in group if r["stage"] == "confirmation"} == expected
    assert len([r for r in rows if r["arm"] == "A_measured_local"]) == 6
    common_streams = []
    for sampler in SAMPLERS:
        for seed, count in [(10, 250), (20, 1000), (21, 1000), (22, 1000)]:
            name = f"{sampler}-seed{seed}-n{count}.json"
            reference = read_json(root / "C" / "randomness" / name)
            compare_arms = ["D"] + (["A_measured_local"] if seed != 10 else [])
            for arm in compare_arms:
                assert reference == read_json(root / arm / "randomness" / name), (arm, name)
            common_streams.append({"sampler": sampler, "seed": seed, "samples": count, "arms_equal": ["C", *compare_arms]})
    aggregates, pairs = aggregate(rows), paired_differences(rows)
    for name, data in [("per_seed", rows), ("aggregate", aggregates), ("paired_differences", pairs)]:
        atomic_json(destination / f"{name}.json", data)
        csv_rows(destination / f"{name}.csv", flat_statistics(data))
    # These fixed complete texts are kept with large artifacts, not in Git.
    atomic_json(root / "qualitative_samples.json", qualitative)
    artifacts[str(root / "qualitative_samples.json")] = file_hash(root / "qualitative_samples.json")
    cards = []
    for sample in qualitative:
        title = f"{sample['arm']} · {sample['sampler']} · SC-CFG {sample['sc_cfg']:g} · seed 20 · ID {sample['id']}"
        score = sample["scores"]
        details = {k: score[k] for k in ("ppl", "entropy", "repeated_4gram_fraction", "gpt2_length_before", "gpt2_length_after", "nonempty")}
        cards.append(f"<article><h2>{html.escape(title)}</h2><p>{html.escape(json.dumps(details))}</p>"
                     f"<pre>{html.escape(sample['generated']) or '[empty output]'}</pre></article>")
    (root / "qualitative_samples.html").write_text("<!doctype html><meta charset='utf-8'><title>Fixed guidance study samples</title>"
        "<style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui}h2{font-size:19px}"
        "article{border-top:1px solid #ccc;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 Georgia}p{overflow-wrap:anywhere}</style>"
        "<h1>Fixed qualitative samples</h1><p>Complete texts, without quality selection: seed 20, IDs 0, 1, 2 for every confirmed configuration. "
        "Auxiliary metrics do not establish coherence. SC-CFG 0 is an out-of-range scale token, not disabled self-conditioning.</p>" + "".join(cards))
    artifacts[str(root / "qualitative_samples.html")] = file_hash(root / "qualitative_samples.html")
    for arm in ("C", "D", "A_measured_local"):
        for path in sorted((root / arm).glob("*.json")):
            artifacts[str(path)] = file_hash(path)
        for path in sorted((root / arm).glob("*.log")):
            artifacts[str(path)] = file_hash(path)
    for name in ("submission.json",):
        artifacts[str(root / name)] = file_hash(root / name)
    for path in sorted(root.glob("*-job.yml")):
        artifacts[str(path)] = file_hash(path)
    atomic_json(destination / "guidance_audits.json", {arm: read_json(root / arm / "guidance_audit.json") for arm in ("C", "D")})
    atomic_json(destination / "scheduler.json", scheduler)
    # Pin the terminal snapshot; the live scheduler cache changes on status queries.
    artifacts[str((destination / "scheduler.json").resolve())] = file_hash(destination / "scheduler.json")
    csv_rows(destination / "compute.csv", compute)
    atomic_json(destination / "artifacts.json", {"root": str(root), "files_sha256": artifacts,
                "inputs_sha256": file_hash("guidance_sweep/inputs.json"), "selection_sha256": file_hash("guidance_sweep/selection.json")})
    audit = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "all_required_settings_completed": True,
             "original_tracked_files_unchanged": len(inputs["original_tracked_files"]), "original_artifacts_unchanged": len(inputs["original_artifacts"]),
             "all_ema_states_unchanged": True, "all_samples_scores_counts_and_hashes_checked": True,
             "common_random_streams": common_streams, "settings": len(rows),
             "requested_count": sum(r["requested_count"] for r in rows), "generated_count": sum(r["generated_count"] for r in rows),
             "empty_count": sum(r["generated_count"] - r["nonempty_count"] for r in rows),
             "scoring_failure_count": sum(len(r["scoring_failures"]) for r in rows),
             "stage_counts": {stage: {"settings": sum(r["stage"] == stage for r in rows),
                 "samples": sum(r["generated_count"] for r in rows if r["stage"] == stage)} for stage in sorted({r["stage"] for r in rows})},
             "allocated_gpu_hours": scheduler["allocated_gpu_hours"], "inference_variation_only": True}
    atomic_json(destination / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
