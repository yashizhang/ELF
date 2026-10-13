"""One-GPU compiler and ELF local-batch check; no experimental checkpoint."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from configs.config import load_config_from_yaml
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from train_step import train_step
from utils.data_utils import get_dataloader, prepare_batch
from utils.train_utils import TrainState, get_optimizer
from .common import ROOT, atomic_json, read_json, runtime_info, seed_all, state_hash
from .elf_integration import assert_frozen, smoke_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shared, out = Path(args.shared), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("Compiler check requires one allocated H100")
    device = torch.device("cuda")
    seed_all(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    norm = read_json(shared / "c4_normalization.json")
    cfg = load_config_from_yaml(str(ROOT / "src/configs/training_configs/train_owt_ELF-B.yml"))
    cfg.encoder_model_name = str(shared / "t5")
    cfg.tokenizer_name = cfg.encoder_model_name
    cfg.latent_mean, cfg.latent_std = norm["mean"], norm["std"]
    cfg.use_wandb = False
    cfg.hf_repo_id = None
    cfg.output_dir = str(out)
    cfg.online_eval = False
    tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    encoder_cfg, encoder = get_encoder(cfg.encoder_model_name, torch.float32)
    encoder.to(device).eval().requires_grad_(False)
    encoder_hash = state_hash(encoder.model.state_dict())
    seed_all(42)
    model = ELF_models[cfg.model](text_encoder_dim=encoder_cfg.d_model, max_length=cfg.max_length,
                                 attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
                                 num_time_tokens=cfg.num_time_tokens, num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
                                 num_model_mode_tokens=cfg.num_model_mode_tokens, bottleneck_dim=cfg.bottleneck_dim,
                                 vocab_size=len(tokenizer), gradient_checkpointing=cfg.gradient_checkpointing).to(device)
    initial_hash = state_hash(model.state_dict())
    optimizer = get_optimizer(model, cfg, lr=0.002)
    state = TrainState(model, optimizer, ema_params1=TrainState.init_ema(model),
                       dropout_generator=torch.Generator().manual_seed(42))
    state = state.replace(model=torch.compile(model))
    tokens = np.memmap(shared / "c4/train/tokens.u16", dtype="<u2", mode="r", shape=(64, 1024))
    dataset = [{"input_ids": row.astype(np.int64)} for row in tokens]
    loader = get_dataloader(dataset, batch_size=64, shuffle=False, num_workers=0,
                            max_seq_length=1024, distributed=False)
    batch = prepare_batch(next(iter(loader)), cfg, state.dropout_generator)
    measurements = []
    for step in range(3):
        started = time.monotonic()
        state, metrics = train_step(state, encoder=encoder, batch=batch, config=cfg)
        torch.cuda.synchronize()
        assert all(torch.isfinite(value) for value in metrics.values())
        measurements.append({"step": step + 1, "seconds": time.monotonic() - started,
                             **{key: value.item() for key, value in metrics.items()}})
        print(measurements[-1], flush=True)
    assert_frozen(encoder, encoder_hash)
    smoke_generation(state, encoder, tokenizer, cfg, state.dropout_generator)
    atomic_json(out / "report.json", {"status": "passed", "runtime": runtime_info(),
                                      "purpose": "compiler/local-memory/sampling check with C4 fixture, not ELF training on OWT",
                                      "local_batch_size": 64, "max_length": 1024,
                                      "gradient_checkpointing": cfg.gradient_checkpointing,
                                      "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                                      "initial_elf_hash": initial_hash, "encoder_frozen": True,
                                      "measurements": measurements})


if __name__ == "__main__":
    main()
