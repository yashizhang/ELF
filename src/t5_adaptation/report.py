"""Write factual result and compute tables from saved measurements and job state."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess

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
    paths = [artifacts / "shared/preparation-job.json", *sorted((artifacts / "preflight").rglob("job.json"))]
    job_ids = {}
    for path in paths:
        if path.exists():
            job = read_json(path)
            job_ids[job["id"]] = "shared_preparation" if "shared" in path.parts else "preflight"
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
                            "allocated_gpu_hours": seconds * job["resources"]["gpu"] / 3600,
                            "provisional": run["endedOn"] is None, "exit_code": run["exitCode"]})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--output", default=str(ROOT))
    args = parser.parse_args()
    artifacts, output = Path(args.artifacts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [PUBLISHED_A]
    status = {}
    for arm in ("C", "D"):
        path = artifacts / arm / "results.json"
        if path.exists():
            result = read_json(path)
            status[arm] = result["status"]
            for record in result["aggregates"]:
                rows.append({"arm": arm, "origin": "measured_this_project", "configuration": record["configuration"],
                             "status": result["status"], "gen_ppl_mean": record["gen_ppl"]["mean"],
                             "gen_ppl_se": record["gen_ppl"]["sem"], "gen_ppl_std": record["gen_ppl"]["sample_std"],
                             "entropy_mean": record["mean_sample_unigram_entropy"]["mean"],
                             "entropy_se": record["mean_sample_unigram_entropy"]["sem"],
                             "entropy_std": record["mean_sample_unigram_entropy"]["sample_std"],
                             "empty_fraction": record["empty_output_fraction"]["mean"],
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
    atomic_json(output / "results.json", {"rows": rows, "arm_status": status,
                                          "B": "intentionally omitted", "complete": all(status[a] == "completed" for a in ("C", "D"))})
    atomic_json(output / "compute.json", {"scheduler_runs": compute,
                                          "allocated_gpu_hours": sum(r["allocated_gpu_hours"] for r in compute)})
    for name, table in (("results.csv", rows), ("compute.csv", compute)):
        columns = list(dict.fromkeys(k for row in table for k in row))
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
    text = ["# C/D experiment results", "", "Experiment status: " + ("complete" if all(status[a] == "completed" for a in ("C", "D")) else "incomplete — training/evaluation requirements remain") + ".", "",
            "| Arm | Source | Configuration | GenPPL | Mean sample entropy | Status |",
            "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        ppl = "—" if row.get("gen_ppl_mean") is None else f"{row['gen_ppl_mean']:.4f}"
        entropy = "—" if row.get("entropy_mean") is None else f"{row['entropy_mean']:.4f}"
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
    probes = {"original": artifacts / "shared/original_probe.json", "C": artifacts / "C/readout.json", "D": artifacts / "D/readout.json"}
    probe_results = {key: read_json(path) if path.exists() else {"status": "not_run_yet"} for key, path in probes.items()}
    atomic_json(output / "diagnostics.json", probe_results)
    probe_text = "The matched 500-update frozen-encoder readout diagnostic is complete for original/C/D; see `diagnostics.json`." if all(p["status"] == "completed" for p in probe_results.values()) else "The matched 500-update token-readout diagnostics are not all complete yet; see `diagnostics.json`."
    text.extend(["", probe_text + " Primary held-out representation diagnostics run at initialization, 1B, 3B, and 10B nominal positions.", "",
                 "Compute includes preparation, failed attempts, and pilots. Running-job totals are provisional. See `compute.csv`, `compute.json`, and per-arm stage manifests for measured allocation time; queued time is excluded.", "",
                 f"Current arm state: C `{status['C']}`; D `{status['D']}`.", ""])
    (output / "RESULTS.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
