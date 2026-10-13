"""Audit an already committed ELF epoch checkpoint without loading a GPU model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import yaml

from t5_adaptation.common import atomic_json, file_hash, object_hash, state_hash


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            assert torch.isfinite(value).all().item(), "Non-finite checkpoint tensor"
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_tensors(v) for v in value)
    return 0


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['C', 'D'], required=True)
    parser.add_argument('--epoch', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent.parent
    output = args.output or root / 'artifacts' / f'elf-{args.arm}-epoch-{args.epoch}-audit.json'
    assert not output.exists(), f'Refusing to overwrite prior audit: {output}'
    directory = root / 'artifacts' / args.arm / 'elf'
    init = json.loads((directory / 'initialization.json').read_text())
    identity = init['identity']
    config = yaml.safe_load((directory / 'config.yml').read_text())
    assert identity['arm'] == args.arm
    assert identity['global_batch_size'] == 512 and identity['grad_accum_steps'] == 1
    assert identity['data_examples'] == 9737184
    assert identity['updates_per_epoch'] == 19018
    assert identity['padding_repeats_per_epoch'] == 32
    assert identity['epochs'] == config['epochs'] == 5
    assert config['full_epochs'] is True and config['eval_freq'] == 0
    assert config['save_freq'] == 1 and config['hf_repo_id'] is None
    assert init['fresh_initialization'] is True
    step = args.epoch * identity['updates_per_epoch']
    path = directory / f'checkpoint_{step}'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['step'] == step and payload['epoch'] == float(args.epoch)
    assert payload['experiment_identity'] == identity
    assert not payload['grad_accum_buffers']
    assert set(payload['ema_params1']) <= set(payload['params'])
    assert sum(v.numel() for v in payload['ema_params1'].values()) == init['elf_parameters']
    for key, value in payload['ema_params1'].items():
        assert value.shape == payload['params'][key].shape
    model_tensor_count = finite_tensors(payload['params'])
    ema_tensor_count = finite_tensors(payload['ema_params1'])
    model_hash = state_hash(payload['params'])
    ema_hash = state_hash(payload['ema_params1'])
    assert model_hash != init['initial_elf_state_hash']
    first_audit = json.loads((root / 'evidence' / f'elf-{args.arm}-checkpoint-500.json').read_text())
    assert model_hash != first_audit['model_state_hash']
    assert ema_hash != first_audit['ema_state_hash']
    assert payload['lr_scheduler']['last_epoch'] == step
    rngs = payload['rank_rngs']
    assert len(rngs) == 8
    files = [r['optimizer_file'] for r in rngs]
    assert len(set(files)) == 8 and files == payload['rank_optimizer_files']
    all_state_ids = set()
    all_parameter_ids = None
    reference_groups = None
    shards = []
    for rank, rng in enumerate(rngs):
        assert {'python', 'numpy', 'torch', 'cuda', 'dropout_generator'} <= rng.keys()
        assert rng['cuda'] and isinstance(rng['dropout_generator'], torch.Tensor)
        assert rng['dropout_generator'].numel() > 0
        name = rng['optimizer_file']
        assert Path(name).name == name and name.startswith(f'optimizer_{step}_rank{rank}_')
        shard_path = directory / name
        digest = file_hash(shard_path)
        assert digest == rng['optimizer_sha256']
        optimizer = torch.load(shard_path, map_location='cpu', weights_only=False)
        tensor_count = finite_tensors(optimizer)
        groups = optimizer['param_groups']
        if reference_groups is None:
            reference_groups = groups
            flat_ids = [p for g in groups for p in g['params']]
            assert len(set(flat_ids)) == len(flat_ids)
            all_parameter_ids = set(flat_ids)
        else:
            assert equal_tree(groups, reference_groups)
        expected = set()
        for group in groups:
            assert abs(group['lr'] - 0.002) < 1e-12
            assert group['weight_decay'] == 0
            expected.update(group['params'][rank::8] if group['use_muon'] else group['params'])
        assert set(optimizer['state']) == expected
        all_state_ids.update(expected)
        steps = [int(s['step']) for s in optimizer['state'].values()]
        assert steps and min(steps) == max(steps) == step
        if rank == 0:
            assert equal_tree(optimizer, payload['opt_state'])
        shards.append({
            'rank': rank, 'file': name, 'sha256': digest,
            'bytes': shard_path.stat().st_size,
            'optimizer_states': len(steps), 'optimizer_step_min': min(steps),
            'optimizer_step_max': max(steps), 'tensor_count': tensor_count,
            'all_optimizer_tensors_finite': True,
            'rank_sharding_matches_parameter_groups': True,
        })
    assert all_state_ids == all_parameter_ids
    result = {
        'arm': args.arm, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved full-epoch ELF checkpoint audit. Final generation and evaluation are outside this audit.',
        'checkpoint': str(path), 'sha256': file_hash(path),
        'step': step, 'epoch': payload['epoch'],
        'required_epochs': 5, 'required_updates': 5 * identity['updates_per_epoch'],
        'experiment_identity_matches': True, 'identity_sha256': object_hash(identity),
        'initial_elf_state_hash': init['initial_elf_state_hash'],
        'model_state_hash': model_hash, 'ema_state_hash': ema_hash,
        'model_and_ema_changed_since_audited_step_500': True,
        'model_and_ema_finite': True, 'model_tensor_count': model_tensor_count,
        'ema_tensor_count': ema_tensor_count,
        'rank_rng_count': len(rngs), 'optimizer_shards': shards,
        'all_optimizer_parameters_have_saved_state': True,
        'rank_zero_optimizer_matches_main_payload': True,
        'lr_scheduler_last_epoch': payload['lr_scheduler']['last_epoch'],
        'optimizer_learning_rate': 0.002, 'grad_accum_buffers_empty': True,
        'dataset_examples': identity['data_examples'],
        'updates_per_epoch': identity['updates_per_epoch'],
        'padding_repeats_per_epoch': identity['padding_repeats_per_epoch'],
        'examples_processed_including_padding_from_saved_step': step * 512,
        'sampling_policy': identity['full_epoch_policy'],
        'coverage_evidence_limit': 'Exposure count is inferred from saved steps and the separately verified full-epoch sampler, not a per-example runtime trace.',
        'automatic_intermediate_generation_disabled': config['eval_freq'] == 0,
        'auditor_sha256': file_hash(__file__),
    }
    atomic_json(output, result)
    print(json.dumps({'audit': str(output), 'arm': args.arm, 'epoch': args.epoch,
                      'step': step, 'checkpoint_sha256': result['sha256']}))


if __name__ == '__main__':
    main()
