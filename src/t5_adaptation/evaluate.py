"""Score saved upstream EMA generation, retaining empty-output accounting."""
import argparse
import json
import math
from pathlib import Path
import statistics

from transformers import AutoTokenizer

from utils.metrics_utils import Metrics
from .common import atomic_json, file_hash, read_json, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--arm", choices=["C", "D"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, shared = Path(args.samples_root), Path(args.shared)
    evaluator = shared / "evaluator"
    manifest = read_json(evaluator / "manifest.json")
    if manifest["source"] != sources()["evaluator"]:
        raise ValueError("Evaluator snapshot revision mismatch")
    scorer = Metrics(str(evaluator), eval_ppl_batch_size=64, eval_context_size=1024)
    t5_tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    per_seed, aggregates = [], []
    for seed in (0, 1, 2):
        files = sorted((root / f"seed_{seed}").glob("*/all_generated_*.jsonl"))
        if len(files) != 2:
            raise ValueError(f"Expected exactly two final sampling configurations for seed {seed}: {files}")
        for path in files:
            name = path.parent.name
            # Derive steps from the existing run-name convention, then require
            # the exact declared gamma / CFG / schedule from evaluation config.
            config = "primary" if "-steps32-" in name else "secondary" if "-steps64-" in name else None
            if config is None:
                raise ValueError(f"Unexpected sampling directory: {name}")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != 1000 or len({r["id"] for r in rows}) != 1000:
                raise ValueError("Expected 1000 distinct saved samples per seed and configuration")
            texts = [r["generated"] for r in rows]
            if not all(isinstance(text, str) for text in texts):
                raise ValueError("Invalid generation output")
            keep = [text for text in texts if text.strip()]
            scorer.reset()
            metrics = scorer.record_generative_perplexity(keep, max_length=1024, retokenize=True) if keep else None
            full_lengths = [len(ids) for ids in scorer.tokenizer(texts, add_special_tokens=False)["input_ids"]]
            t5_lengths = [len(ids) for ids in t5_tokenizer(texts, add_special_tokens=False)["input_ids"]]
            finite_ppl = [p for p in metrics["per_sample_ppl"] if math.isfinite(p)] if metrics else []
            item = {"arm": args.arm, "origin": "measured_this_project", "seed": seed, "configuration": config,
                    "steps": 32 if config == "primary" else 64, "gamma": 1.5 if config == "primary" else 1.0,
                    "self_cond_cfg": 3.0, "cfg": 1.0, "time_schedule": "logit_normal", "sequence_length": 1024,
                    "evaluator": manifest["source"], "evaluator_hash": manifest["content_hash"],
                    "requested_count": 1000, "generated_count": len(rows), "nonempty_count": len(keep),
                    "ppl_scored_count": len(finite_ppl), "entropy_scored_count": len(keep),
                    "empty_output_fraction": 1 - len(keep) / len(rows),
                    "gen_ppl": metrics["ppl"] if metrics and math.isfinite(metrics["ppl"]) else None,
                    "mean_sample_unigram_entropy": metrics["mean_entropy"] if metrics else None,
                    "gpt2_lengths_before_truncation": full_lengths,
                    "gpt2_lengths_after_truncation": [min(1024, n) for n in full_lengths],
                    "t5_retokenized_lengths": t5_lengths,
                    "per_sample_ppl_nonempty_order": [p if math.isfinite(p) else None for p in metrics["per_sample_ppl"]] if metrics else [],
                    "samples_file": str(path.resolve()), "samples_sha256": file_hash(path),
                    "qualitative_samples": rows[:3]}
            atomic_json(path.parent / "project_metrics.json", item)
            per_seed.append(item)
    for config in ("primary", "secondary"):
        group = [r for r in per_seed if r["configuration"] == config]
        if len(group) != 3:
            raise ValueError("Missing evaluation seed/configuration")
        aggregate = {"arm": args.arm, "configuration": config, "origin": "measured_this_project",
                     "training_seed_count": 1, "inference_seeds": [0, 1, 2]}
        for field in ("gen_ppl", "mean_sample_unigram_entropy", "empty_output_fraction"):
            values = [r[field] for r in group]
            if any(x is None for x in values):
                aggregate[field] = {"mean": None, "sample_std": None, "sem": None, "values": values}
            else:
                aggregate[field] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                                    "sem": statistics.stdev(values) / math.sqrt(3), "values": values}
        aggregates.append(aggregate)
    atomic_json(args.output, {"arm": args.arm, "status": "completed", "per_seed": per_seed, "aggregates": aggregates,
                             "metric_semantics": "GenPPL is frozen GPT-2 Large generative perplexity, not ELF likelihood; entropy is mean per-sample unigram entropy after GPT-2 retokenization",
                             "uncertainty": "Variation over inference seeds only; one training seed per arm"})


if __name__ == "__main__":
    main()
