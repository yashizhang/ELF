"""Add the final C/D interpretation, length summaries, and fixed sample set.

Run after ``python -m t5_adaptation.report`` and the two generation audits.
This report-only entry point leaves the immutable execution source untouched.
"""
import json
import hashlib
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    results, compute = read("results.json"), read("compute.json")
    assert results["complete"]
    assert all(not row["provisional"] and row["ended_at"] for row in compute["scheduler_runs"])
    lengths, samples, counts = [], [], []
    for arm in ("C", "D"):
        path = ROOT / "artifacts" / arm / "results.json"
        measured = json.loads(path.read_text())
        audit = read(f"evidence/generation-{arm}.json")
        assert audit["aggregate_statistics_verified"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["results_sha256"]
        for config in ("primary", "secondary"):
            rows = sorted((r for r in measured["per_seed"] if r["configuration"] == config), key=lambda r: r["seed"])
            assert [r["seed"] for r in rows] == [0, 1, 2]
            entry = {"arm": arm, "configuration": config, "generated_count": sum(r["generated_count"] for r in rows)}
            for key in ("gpt2_lengths_before_truncation", "gpt2_lengths_after_truncation", "t5_retokenized_lengths"):
                values = [v for row in rows for v in row[key]]
                assert len(values) == 3000
                entry[key] = {"count": len(values), "min": min(values), "max": max(values),
                              "mean": statistics.mean(values), "median": statistics.median(values)}
            lengths.append(entry)
            counts.append({"arm": arm, "configuration": config,
                           **{k: sum(row[k] for row in rows) for k in
                              ("requested_count", "generated_count", "nonempty_count", "ppl_scored_count", "entropy_scored_count")}})
            seed_zero = rows[0]
            raw_path = Path(seed_zero["samples_file"])
            assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == seed_zero["samples_sha256"]
            raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            selected = [row for row in raw if row["id"] in (0, 1, 2)]
            assert [row["id"] for row in selected] == [0, 1, 2]
            samples.append({"arm": arm, "configuration": config, "seed": 0,
                            "samples_file": str(raw_path.relative_to(ROOT)),
                            "samples_sha256": seed_zero["samples_sha256"], "samples": selected})
    write("lengths.json", {"scope": "All 3000 saved texts per arm/configuration, including empty strings. Retokenized text lengths are not the original 1024-position latent length. GPT-2 scoring caps context at 1024.", "rows": lengths})
    write("qualitative_samples.json", {"selection": "Fixed seed 0, sample IDs 0, 1, 2 for each arm and configuration; no quality selection. Generated text is model output, not factual evidence.", "groups": samples})

    lines = ["# C/D experiment results", "", "Both fixed-budget pipelines completed. The results **do not establish an overall generation improvement**: lower GenPPL accompanies lower entropy, and the fixed qualitative samples contain substantial repetition and incoherence.", "",
             "| Arm | Source | Configuration | GenPPL ± SE | Mean sample entropy ± SE |",
             "| --- | --- | --- | --- | --- |",
             "| A | published_external | primary | 24.0800 ± 0.1600 | 5.1500 ± 0.0020 |"]
    measured_rows = []
    for arm in ("C", "D"):
        for row in read(f"artifacts/{arm}/results.json")["aggregates"]:
            measured_rows.append(row)
            p, e = row["gen_ppl"], row["mean_sample_unigram_entropy"]
            lines.append(f"| {arm} | measured_this_project | {row['configuration']} | {p['mean']:.4f} ± {p['sem']:.4f} | {e['mean']:.4f} ± {e['sem']:.4f} |")
    lines += ["", "Published A comes from [ELF paper v2, Appendix E.4, Table 9](https://arxiv.org/html/2605.10938v2#A5.T9), mean ± standard error over six inference seeds 0–5. It is an external reference, not a measurement from these jobs.", "",
              "C/D each use one training seed (42), inference seeds 0,1,2, and 1,000 sequences per seed/configuration. Primary: 32 SDE steps, gamma 1.5. Secondary: 64 steps, gamma 1.0. Both use SC-CFG 3, CFG 1, logit-normal time, and latent length 1024. Error bars describe inference variation only; [results.json](results.json) also reports sample standard deviations, and [evaluation.csv](evaluation.csv) gives every seed's scores.", "",
              "GenPPL uses the repository's frozen, pinned GPT-2 Large evaluator; it is not ELF held-out likelihood. Entropy is mean per-sample unigram entropy after GPT-2 retokenization, not corpus-global entropy.", ""]
    primary = {r["arm"]: r for r in measured_rows if r["configuration"] == "primary"}
    cp, dp = primary["C"], primary["D"]
    lines += [f"Primary C−A: GenPPL {cp['gen_ppl']['mean'] - 24.08:+.4f}, entropy {cp['mean_sample_unigram_entropy']['mean'] - 5.15:+.4f}; D−A: {dp['gen_ppl']['mean'] - 24.08:+.4f}, {dp['mean_sample_unigram_entropy']['mean'] - 5.15:+.4f}; C−D: {cp['gen_ppl']['mean'] - dp['gen_ppl']['mean']:+.4f}, {cp['mean_sample_unigram_entropy']['mean'] - dp['mean_sample_unigram_entropy']['mean']:+.4f}.", "",
              "B was intentionally omitted. C versus A includes continued pretraining, the C4 stream, implementation differences, and SIGReg; it cannot isolate SIGReg's causal effect. C versus D compares complete recipes. D has the lowest primary GenPPL and the lowest entropy. These observations support a quality/diversity trade-off, not matched-entropy frontier dominance. No guidance or checkpoint sweep was added.", "",
              "Each arm processed **10,000,007,168 nominal adaptation positions**, 19,531,264 source examples, and 11,093,757,952 packed source positions in 38,147 updates. Each fresh ELF-B then completed **five OWT epochs / 95,090 updates**. Five epochs cover 48,685,920 source-row visits plus 160 documented padding repeats. [training.json](training.json) retains actual token/pass counts, corruption statistics, normalization, initialization hashes, measured throughput, and memory.", "",
              "| Arm | Configuration | Requested / generated | Nonempty | PPL / entropy scored | Empty fraction |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in counts:
        empty_fraction = (row["generated_count"] - row["nonempty_count"]) / row["generated_count"]
        lines.append(f"| {row['arm']} | {row['configuration']} | {row['requested_count']} / {row['generated_count']} | {row['nonempty_count']} | {row['ppl_scored_count']} / {row['entropy_scored_count']} | {empty_fraction:.4%} |")
    lines += ["", "C primary's three empty outputs are seed 1 IDs 589/820 and seed 2 ID 895. They remain in the generated counts and are excluded from scoring. Every nonempty output was scored. [evaluation.csv](evaluation.csv) retains per-seed counts and scores.", "",
              "| Arm | Configuration | Mean GPT-2 tokens before / after cap | Mean T5 retokenized tokens |",
              "| --- | --- | --- | --- |"]
    for row in lengths:
        lines.append(f"| {row['arm']} | {row['configuration']} | {row['gpt2_lengths_before_truncation']['mean']:.2f} / {row['gpt2_lengths_after_truncation']['mean']:.2f} | {row['t5_retokenized_lengths']['mean']:.2f} |")
    lines += ["", "Lengths include all 3,000 texts per row, including empties. Decode/re-encode can change length; these are retokenized text lengths, not a different latent generation length. GPT-2 scoring caps context at 1024. [lengths.json](lengths.json) includes medians and ranges.", "",
              "The [fixed qualitative set](qualitative_samples.json) contains seed 0, IDs 0–2, for all four measured rows, without quality selection. Illustrative primary ID 0 excerpts:", ""]
    for group in samples:
        if group["configuration"] == "primary":
            excerpt = group["samples"][0]["generated"][:260].replace("\n", " ").rsplit(" ", 1)[0]
            lines.append(f"> **{group['arm']}:** {excerpt}…\n")
    lines += ["All original/C/D 500-update frozen-encoder readouts and the held-out representation diagnostics at 569/1024 positions completed; none were skipped. D's better token recovery did not establish better generation. [diagnostics.json](diagnostics.json) retains the probe and spectra; C's span CE and D's consistency are separate objectives.", "",
              f"Final scheduler allocation: **{compute['allocated_gpu_hours']:.3f} GPU hours**, including both full runs, both pilot attempts, and the one-GPU compile check. Shared CPU preparation contributes zero GPU hours. Every recorded allocation ended; queued time is excluded. [compute.json](compute.json) includes all attempts, and [stage_compute.csv](stage_compute.csv) subdivides those same allocations—do not add stage totals again.", "",
              "Both original eight-H100 jobs succeeded with exit 0 and no replacement runs: [scheduler evidence](evidence/final-scheduler.json). Deep final [C](evidence/elf-C-epoch-5.json) / [D](evidence/elf-D-epoch-5.json) checkpoint audits and final [C](evidence/generation-C.json) / [D](evidence/generation-D.json) generation audits passed. The latter reconstruct entropy and aggregate perplexity from saved per-sample PPL; they do not rerun GPT-2 forward passes. Exposure follows saved steps and the verified sampler rather than a per-example runtime trace.", "",
              "[ACCEPTANCE.md](ACCEPTANCE.md) maps every requirement to evidence and its limits. [EXPERIMENT.md](EXPERIMENT.md) documents reproduction and resume. Raw checkpoints, encoders, logs, all 12,000 generated texts, and per-seed metrics remain under `/home/toolkit/ELF/artifacts`; [artifact index](evidence/final-artifacts.json) records retained paths and hashes. Large artifacts are intentionally outside Git and were not uploaded to Hugging Face.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"report": "RESULTS.md", "length_rows": len(lengths), "qualitative_samples": sum(len(g['samples']) for g in samples)}))


if __name__ == "__main__":
    main()
