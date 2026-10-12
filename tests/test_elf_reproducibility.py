"""Provenance and full-epoch accounting checks for experiment-specific hooks."""
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.data_utils import FullEpochSampler
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.train_utils import TrainState
from utils.muon_utils import muon_with_aux_adam


def test_full_epochs_include_all_rows_with_exact_global_batches():
    rows = list(range(1031))
    samplers = [FullEpochSampler(rows, 64, rank, 8, seed=42) for rank in range(8)]
    shards = [list(s) for s in samplers]
    assert all(len(s) == 192 for s in shards)
    merged = [shards[r][i] for i in range(192) for r in range(8)]
    assert set(merged) == set(rows)
    assert len(merged) == 3 * 512
    assert len(merged) - len(set(merged)) == 505
    for sampler in samplers:
        sampler.set_epoch(1)
    assert list(samplers[0]) != shards[0]
    samplers[0].set_epoch(0)
    assert list(samplers[0]) == shards[0]


def make_state(identity):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    return TrainState(model, opt, dropout_generator=torch.Generator().manual_seed(42),
                      experiment_identity=identity)


def test_elf_checkpoint_roundtrip_and_arm_rejection(tmp_path):
    state = make_state({"arm": "C", "encoder_hash": "encoder_c"})
    state.step, state.epoch = 10, 0.125
    state.model(torch.randn(2, 4)).sum().backward()
    state.optimizer.step()
    save_checkpoint(state, str(tmp_path), 10)
    expected_global_rng = torch.rand(5)
    expected_local_rng = torch.rand(5, generator=state.dropout_generator)
    loaded, step = load_checkpoint(str(tmp_path / "checkpoint_10"), state)
    assert step == 10 and loaded.epoch == 0.125
    torch.testing.assert_close(torch.rand(5), expected_global_rng)
    torch.testing.assert_close(torch.rand(5, generator=loaded.dropout_generator), expected_local_rng)
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(str(tmp_path / "checkpoint_10"), make_state({"arm": "D"}))
    (tmp_path / "checkpoint_999.tmp").touch()
    assert Path(find_latest_checkpoint(str(tmp_path))).name == "checkpoint_10"


def _muon_resume_worker(rank, init, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init}", world_size=2, rank=rank)
    try:
        torch.manual_seed(2026)
        base = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.Linear(4, 4))
        opt = muon_with_aux_adam(base, 0.002)
        model = DDP(base)
        state = TrainState(model, opt, dropout_generator=torch.Generator().manual_seed(42 + rank),
                           experiment_identity={"arm": "C", "purpose": "distributed_muon_resume"})
        def step():
            opt.zero_grad(set_to_none=True)
            model(torch.randn(3, 4)).square().mean().backward()
            opt.step()
        step()
        state.step = 1
        save_checkpoint(state, output, 1)
        dist.barrier()
        step()
        expected = {k: v.clone() for k, v in base.state_dict().items()}
        load_checkpoint(str(Path(output) / "checkpoint_1"), state)
        step()
        for k, v in base.state_dict().items():
            torch.testing.assert_close(v, expected[k], rtol=0, atol=0, msg=f"rank {rank}: {k}")
    finally:
        dist.destroy_process_group()


def test_distributed_muon_restores_each_optimizer_shard(tmp_path):
    mp.spawn(_muon_resume_worker, args=(str(tmp_path / "pg"), str(tmp_path)), nprocs=2, join=True)
