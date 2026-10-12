"""Opt-in checks and artifacts around the unchanged upstream ELF model."""
import copy
import math
from pathlib import Path

import torch

from .common import (atomic_json, directory_manifest, object_hash, rank, read_json,
                     restore_rng, rng_state, state_hash)


def validate_inputs(config, encoder, dataset):
    expected = read_json(config.experiment_manifest)
    export = Path(config.encoder_model_name)
    manifest = read_json(export / "manifest.json")
    if directory_manifest(export)["content_hash"] != manifest["content_hash"]:
        raise ValueError("Encoder export content no longer matches its immutable manifest")
    if expected["encoder_hash"] != manifest["content_hash"]:
        raise ValueError("ELF requested the wrong adapted encoder")
    if expected["data_examples"] != len(dataset):
        raise ValueError("ELF dataset length changed")
    if expected["latent_mean"] != config.latent_mean or expected["latent_std"] != config.latent_std:
        raise ValueError("ELF normalization differs from the encoder-specific OWT calibration")
    if Path(config.tokenizer_name).resolve() != export.resolve():
        raise ValueError("Use the unchanged tokenizer in the encoder export")
    if any(p.requires_grad for p in encoder.parameters()) or encoder.training:
        raise ValueError("ELF encoder must be frozen in evaluation mode")
    state_digest = state_hash(encoder.model.state_dict())
    if state_digest != manifest["state_hash"]:
        raise ValueError("The ELF wrapper did not load the exported weights")
    return expected, state_digest


def record_initial(config, model, encoder_hash, identity):
    digest = state_hash(model.state_dict())
    record = {"initial_elf_state_hash": digest, "encoder_state_hash": encoder_hash,
              "seed": config.seed, "identity": identity, "fresh_initialization": True,
              "elf_parameters": sum(p.numel() for p in model.parameters())}
    if rank() == 0:
        path = Path(config.output_dir) / "initialization.json"
        if path.exists() and read_json(path) != record:
            raise ValueError("ELF initialization provenance changed")
        atomic_json(path, record)


def assert_frozen(encoder, state_digest=None):
    if encoder.training or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise ValueError("Adapted encoder received gradients during ELF training")
    if state_digest is not None and state_hash(encoder.model.state_dict()) != state_digest:
        raise ValueError("Adapted encoder weights changed during ELF training")


def smoke_generation(state, encoder, tokenizer, config, generator):
    from configs.config import SamplingConfig
    from generation import run_generation
    saved, saved_g = rng_state(), generator.get_state()
    smoke = copy.copy(config)
    smoke.sampling_configs = [SamplingConfig(sampling_method="sde", num_sampling_steps=[2],
                                             cfgs=[1], self_cond_cfg_scales=[3], sde_gamma=1.5,
                                             time_schedule="logit_normal")]
    smoke.num_samples, smoke.online_eval = 8, False
    smoke.output_dir = str(Path(config.output_dir) / "initial_smoke")
    smoke.use_compile = False
    smoke.use_wandb = False
    try:
        run_generation(state, encoder, None, tokenizer, smoke, generator, local_batch_size=1)
    finally:
        restore_rng(saved)
        generator.set_state(saved_g)
