"""Run one independent eight-H100 inference study; never create an optimizer."""
import argparse
from datetime import datetime, timezone, timedelta
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from configs.config import SamplingConfig, load_config_from_yaml
from modules.model import ELF_models
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from utils.sampling_utils import _forward_sample, get_sampling_steps
from utils.metrics_utils import Metrics
from t5_adaptation.common import atomic_json, append_json, file_hash, object_hash, read_json, state_hash
from .common import SAMPLERS, SCALES, reset_rng, rng_hash, tensor_hash, score_texts, refinements, select_confirmation

STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def stop_signal(*_):
    global STOP
    STOP = True


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def broadcast(value):
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def load_model(spec, warm_shapes):
    assert file_hash(spec["checkpoint"]) == spec["checkpoint_sha256"]
    cfg = load_config_from_yaml(spec["config"])
    assert cfg.max_length == 1024 and cfg.num_self_cond_cfg_tokens == 4
    assert cfg.latent_mean == spec["latent_mean"] and cfg.latent_std == spec["latent_std"]
    tokenizer = AutoTokenizer.from_pretrained(spec["encoder"])
    assert len(tokenizer) == spec["vocab_size"]
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    assert payload["step"] == spec["step"] and payload["epoch"] == spec["epoch"]
    assert state_hash(payload["ema_params1"]) == spec["ema_state_hash"]
    model = ELF_models[cfg.model](text_encoder_dim=512, max_length=cfg.max_length, attn_drop=cfg.attn_dropout,
        proj_drop=cfg.proj_dropout, num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=len(tokenizer), num_model_mode_tokens=cfg.num_model_mode_tokens, bottleneck_dim=cfg.bottleneck_dim)
    model.load_state_dict(payload["ema_params1"], strict=True)
    model.requires_grad_(False).eval().cuda()
    del payload
    compiled = torch.compile(model)
    # Compile both forward modes and every study batch shape outside all study RNG streams.
    reset_rng(800001, dist.get_rank())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in warm_shapes:
            z = torch.randn(batch, 1024, 1024, device="cuda")
            t = torch.full((batch,), .5, device="cuda")
            scale = torch.full((batch,), 3., device="cuda")
            compiled(z, t, deterministic=True, self_cond_cfg_scale=scale)
            compiled(z, torch.ones_like(t), deterministic=True, self_cond_cfg_scale=scale, decoder_step_active=True)
    torch.cuda.synchronize()
    return model, compiled, cfg, tokenizer


@torch.no_grad()
def sample_batch(model, cfg, sampler, scale, generator, batch_size, ordinary_cfg=1):
    steps, gamma = SAMPLERS[sampler]
    trace = {"batch_size": batch_size, "rng_before": rng_hash(generator)}
    times = get_sampling_steps(steps, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std,
                               device=torch.device("cuda"), dtype=torch.float32)
    z = torch.randn((batch_size, cfg.max_length, 512), dtype=torch.float32, device="cuda") * cfg.denoiser_noise_scale
    trace.update(time_grid_sha256=tensor_hash(times), initial_latent_sha256=tensor_hash(z))
    latent = _generate_samples_single_batch(model, generator, z, times, None, None, cfg,
        SamplingConfig(sampling_method="sde", sde_gamma=gamma), ordinary_cfg, scale)
    ids = _dlm_decode_batch(latent, model, times[-1].item(), cfg, scale)
    trace["rng_after"] = rng_hash(generator)
    return latent, ids, trace


@torch.no_grad()
def audit_guidance(model, cfg):
    values = [0, .5, 1, 2, 3, 5]
    generator = reset_rng(900001)
    z = torch.randn(1, 1024, 512, device="cuda")
    previous = torch.randn_like(z)
    t = torch.full((1,), .37, device="cuda")
    conditioning, mask = torch.zeros_like(z), torch.zeros(1, 1024, device="cuda")
    forward = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for value in values:
            reset_rng(900002)
            v, x = _forward_sample(model, z, t, previous, cfg, value, 3., conditioning, mask)
            forward[value] = (v.cpu(), x.cpu())
    forward_differences = {str(value): max((a - b).abs().max().item() for a, b in zip(forward[value], forward[1])) for value in values}
    assert all(diff <= 1e-6 for diff in forward_differences.values())
    end_to_end = []
    for sampler in SAMPLERS:
        baseline = None
        results = {}
        for value in [1, 0, .5, 2, 3, 5]:
            generator = reset_rng(900003)
            latent, ids, trace = sample_batch(model, cfg, sampler, 3., generator, 1, value)
            latent, ids = latent.cpu(), ids.cpu()
            if baseline is None:
                baseline = (latent, ids, trace)
            difference = (latent - baseline[0]).abs().max().item()
            equal = torch.equal(ids, baseline[1])
            assert difference <= 1e-6 and equal and trace == baseline[2]
            results[str(value)] = {"max_latent_difference": difference, "generated_tokens_equal": equal, "rng_and_noise_equal": trace == baseline[2]}
        end_to_end.append({"sampler": sampler, "ordinary_cfg": results})
    # Use the real screening batch shape, with a separate audit seed and interleaved scale.
    first = sample_batch(model, cfg, "primary", 3., reset_rng(900004), 32)
    sample_batch(model, cfg, "primary", .5, reset_rng(900004), 32)
    replay = sample_batch(model, cfg, "primary", 3., reset_rng(900004), 32)
    assert torch.equal(first[0], replay[0]) and torch.equal(first[1], replay[1]) and first[2] == replay[2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = _forward_sample(model, z, t, previous, cfg, 1, .5, conditioning, mask)[1]
        b = _forward_sample(model, z, t, previous, cfg, 1, 5., conditioning, mask)[1]
    active_difference = (a - b).abs().max().item()
    assert active_difference > 1e-6
    class DecodeRecorder(torch.nn.Module):
        def forward(self, *args, **kwargs):
            self.recorded = kwargs["self_cond_cfg_scale"].detach().cpu().tolist()
            return model(*args, **kwargs)
    recorder = DecodeRecorder()
    _dlm_decode_batch(z, recorder, 1., cfg, .75)
    assert recorder.recorded == [.75]
    return {"ordinary_cfg_inactive": True, "fixed_ordinary_cfg": 1, "ordinary_cfg_values": values,
            "absolute_tolerance": 1e-6, "forward_max_differences": forward_differences,
            "forward_nonzero_previous_self_conditioning": True, "end_to_end": end_to_end,
            "sc_cfg_active_max_state_difference_0_5_vs_5": active_difference,
            "final_decoder_received_scale": recorder.recorded,
            "configuration_order_replay": {"order": [3, .5, 3], "batch_size": 32, "audit_seed": 900004,
                                           "latent_exact": True, "tokens_exact": True, "rng_exact": True},
            "audit_seeds_disjoint_from_study": [900001, 900002, 900003, 900004], "recorded_at_utc": now()}


class Study:
    def __init__(self, args, inputs, plan):
        self.args, self.inputs, self.plan = args, inputs, plan
        self.rank = dist.get_rank()
        self.root = Path(args.root).resolve()
        self.scorer = Metrics(inputs["evaluator"]["path"], eval_ppl_batch_size=64, eval_context_size=1024) if self.rank == 0 else None
        self.t5_tokenizer = AutoTokenizer.from_pretrained(inputs["t5"]["path"]) if self.rank == 0 else None
        self.identity = {"inputs_sha256": file_hash(args.inputs), "plan_sha256": file_hash(args.plan),
                         "execution_commit": os.environ["GUIDANCE_RELEASE"], "world_size": 8, "global_batch_size": 512,
                         "phase": args.phase}

    def setting(self, arm, sampler, scale, seed, count, stage, model, cfg, tokenizer):
        directory = self.root / arm / stage / sampler / f"sc_{scale:g}" / f"seed_{seed}"
        identity = self.identity | {"arm": arm, "sampler": sampler, "sc_cfg": scale, "cfg": 1,
                                  "seed": seed, "requested_count": count, "stage": stage}
        finished = None
        if self.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "metrics.json").exists():
                finished = read_json(directory / "metrics.json")
                assert finished["identity"] == identity
                for name, digest in finished["artifact_hashes"].items():
                    assert file_hash(directory / name) == digest
        finished = broadcast(finished)
        if finished:
            return finished
        if broadcast(STOP if self.rank == 0 else None):
            raise SystemExit(75)
        start = time.time()
        if self.rank == 0:
            atomic_json(self.root / self.args.arm / "status.json", identity | {"state": "running", "started": start})
            append_json(self.root / self.args.arm / "attempts.jsonl", {"event": "start", "time": start, "identity": identity})
        generation = None
        if self.rank == 0 and (directory / "generation.json").exists():
            generation = read_json(directory / "generation.json")
            assert generation["identity"] == identity
            assert file_hash(directory / "samples.jsonl") == generation["samples_sha256"]
        generation = broadcast(generation)
        if generation is None:
            generator = reset_rng(seed, self.rank)
            rank_block = (count + 7) // 8
            local_count = min(rank_block, max(0, count - self.rank * rank_block))
            texts, traces, token_hashes = [], [], []
            for offset in range(0, local_count, 64):
                batch_size = min(64, local_count - offset)
                _, ids, trace = sample_batch(model, cfg, sampler, scale, generator, batch_size)
                token_hashes.append(tensor_hash(ids))
                ids = mask_after_eos(ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
                texts.extend(tokenizer.decode(row.cpu().numpy(), skip_special_tokens=True) for row in ids)
                traces.append(trace)
            gathered = [None] * 8
            dist.all_gather_object(gathered, {"rank": self.rank, "texts": texts, "randomness": traces, "argmax_token_hashes": token_hashes})
            if self.rank == 0:
                merged = [text for part in gathered for text in part["texts"]]
                rows = [{"id": i, "generated": text} for i, text in enumerate(merged[:count])]
                write_rows(directory / "samples.jsonl", rows)
                generation = {"identity": identity, "samples_sha256": file_hash(directory / "samples.jsonl"),
                    "checkpoint_sha256": self.inputs["arms"][arm]["checkpoint_sha256"],
                    "ema_state_hash": self.inputs["arms"][arm]["ema_state_hash"],
                    "per_rank_requested_counts": [len(part["texts"]) for part in gathered],
                    "extra_rank_padding_samples_trimmed": len(merged) - count,
                    "rank_records": [{k: v for k, v in part.items() if k != "texts"} for part in gathered],
                    "generation_ended": time.time()}
                # Reuse one reference stream per seed/sampler/count across all scales and stages.
                rng_dir = self.root / arm / "randomness"
                rng_dir.mkdir(exist_ok=True)
                rng_path = rng_dir / f"{sampler}-seed{seed}-n{count}.json"
                stream = [{"rank": part["rank"], "randomness": part["randomness"]} for part in gathered]
                if rng_path.exists():
                    assert read_json(rng_path) == stream, "Configuration changed the paired random stream"
                else:
                    atomic_json(rng_path, stream)
                atomic_json(directory / "generation.json", generation)
        if self.rank == 0:
            rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
            summary, per_sample = score_texts(rows, self.scorer, self.t5_tokenizer)
            write_rows(directory / "per_sample_scores.jsonl", per_sample)
            finished = summary | {"identity": identity, "arm": arm, "sampler": sampler, "sc_cfg": scale, "cfg": 1,
                "seed": seed, "stage": stage, "outside_trained_range": not .5 <= scale <= 5,
                "samples_file": str(directory / "samples.jsonl"), "scores_file": str(directory / "per_sample_scores.jsonl"),
                "artifact_hashes": {name: file_hash(directory / name) for name in ["samples.jsonl", "per_sample_scores.jsonl", "generation.json"]},
                "started": start, "ended": time.time(), "entropy_and_repetition_scope": "GPT-2 tokens after 1024 cap",
                "evaluator_hash": self.inputs["evaluator"]["content_hash"]}
            atomic_json(directory / "metrics.json", finished)
            append_json(self.root / self.args.arm / "attempts.jsonl", {"event": "complete", "time": time.time(), "identity": identity,
                                                                      "metrics_sha256": file_hash(directory / "metrics.json")})
            print(json.dumps({k: finished[k] for k in ["arm", "stage", "sampler", "sc_cfg", "seed", "gen_ppl", "entropy", "empty_fraction"]}), flush=True)
        return broadcast(finished)

    def run_arm(self, arm, anchor=False):
        base_model, model, cfg, tokenizer = load_model(self.inputs["arms"][arm], (1, 32, 26) if self.args.phase == "screening" else (64, 61))
        directory = self.root / arm
        directory.mkdir(exist_ok=True)
        if self.args.phase == "screening":
            audit = audit_guidance(model, cfg) if self.rank == 0 else None
            audit = broadcast(audit)
            if self.rank == 0:
                atomic_json(directory / "guidance_audit.json", audit | {"identity": self.identity, "ema_state_hash": self.inputs["arms"][arm]["ema_state_hash"]})
            for sampler in SAMPLERS:
                rows = [self.setting(arm, sampler, scale, 10, 250, "screening", model, cfg, tokenizer) for scale in SCALES]
                low = next(r for r in rows if r["sc_cfg"] == .5)
                if low["entropy"] is not None and low["entropy"] < 5.10:
                    if self.rank == 0:
                        atomic_json(directory / f"extrapolation-{sampler}.json", {"values": [0, .25], "trigger_entropy_at_0_5": low["entropy"], "reason": "High-entropy end at 0.5 remains below 5.10; these scale-token values are outside training range and do not disable self-conditioning.", "recorded_before_execution": now()})
                    rows.extend(self.setting(arm, sampler, scale, 10, 250, "extrapolation", model, cfg, tokenizer) for scale in (0., .25))
                refine = refinements(rows)
                if self.rank == 0:
                    atomic_json(directory / f"refinement-{sampler}.json", refine | {"recorded_before_execution": now()})
                rows.extend(self.setting(arm, sampler, scale, 10, 250, "refinement", model, cfg, tokenizer) for scale in refine["values"])
                selection = select_confirmation(rows)
                selection_path = directory / f"selection-{sampler}.json"
                if self.rank == 0:
                    if selection_path.exists():
                        stored = read_json(selection_path)
                        assert stored["selection"] == selection
                    else:
                        atomic_json(selection_path, {"selection": selection, "frozen_at_utc": now(), "identity": self.identity,
                                                    "screening_metrics_sha256": [file_hash(Path(r["samples_file"]).parent / "metrics.json") for r in rows]})
        elif not anchor:
            frozen = read_json(self.args.selection)
            assert frozen["inputs_sha256"] == file_hash(self.args.inputs)
            assert set(frozen["arms"]) == {"C", "D"}
            for sampler in SAMPLERS:
                selection = frozen["arms"][arm][sampler]["selection"]
                for selected in selection["selected"]:
                    for seed in (20, 21, 22):
                        self.setting(arm, sampler, selected["sc_cfg"], seed, 1000, "confirmation", model, cfg, tokenizer)
        else:
            for sampler in SAMPLERS:
                for seed in (20, 21, 22):
                    self.setting(arm, sampler, 3., seed, 1000, "confirmation", model, cfg, tokenizer)
        final_hash = state_hash(base_model.state_dict())
        assert final_hash == self.inputs["arms"][arm]["ema_state_hash"]
        assert all(not p.requires_grad and p.grad is None for p in base_model.parameters())
        if self.rank == 0:
            atomic_json(directory / f"{self.args.phase}_completed.json", {"arm": arm, "state": "completed", "ended": time.time(),
                "ema_unchanged": True, "ema_state_hash": final_hash, "identity": self.identity})
        del base_model, model, cfg, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["C", "D"])
    parser.add_argument("--phase", required=True, choices=["screening", "confirmation"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--inputs", default="guidance_sweep/inputs.json")
    parser.add_argument("--plan", default="guidance_sweep/plan.json")
    parser.add_argument("--selection", default="guidance_sweep/selection.json")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() == 8 and all("H100" in torch.cuda.get_device_name(i) for i in range(8))
    dist.init_process_group("gloo", timeout=timedelta(hours=2))
    assert dist.get_world_size() == 8
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, stop_signal)
    root = Path(args.root).resolve()
    (root / args.arm).mkdir(parents=True, exist_ok=True)
    inputs, plan = read_json(args.inputs), read_json(args.plan)
    assert plan["account"] == "snow.research.er"
    if rank == 0:
        for path, expected in inputs["original_artifacts"].items():
            assert file_hash(path) == expected, path
        atomic_json(root / args.arm / f"{args.phase}-runtime-{time.time_ns()}.json", {"started": time.time(), "source_commit": os.environ["GUIDANCE_RELEASE"],
            "gpus": [torch.cuda.get_device_name(i) for i in range(8)], "gpu_memory": [torch.cuda.get_device_properties(i).total_memory for i in range(8)],
            "torch": torch.__version__, "cuda": torch.version.cuda, "pythonpath": os.environ["PYTHONPATH"],
            "inputs_sha256": file_hash(args.inputs), "plan_sha256": file_hash(args.plan)})
    dist.barrier()
    study = Study(args, inputs, plan)
    try:
        study.run_arm(args.arm)
        if args.arm == "C" and args.phase == "confirmation":
            study.run_arm("A_measured_local", anchor=True)
        if rank == 0:
            atomic_json(root / args.arm / f"{args.phase}_job_completed.json", {"state": "completed", "ended": time.time(),
                "arm": args.arm, "anchor_completed": args.arm == "C" and args.phase == "confirmation", "identity": study.identity})
    except BaseException as error:
        if rank == 0:
            atomic_json(root / args.arm / f"failure-{time.time_ns()}.json", {"error": repr(error), "traceback": traceback.format_exc(), "time": time.time()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
