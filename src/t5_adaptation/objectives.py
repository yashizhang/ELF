"""Exact additive objectives, entirely inside the module wrapped by DDP."""
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley

from .common import world_size


def global_token_mean(local_sum, local_count):
    count = torch.as_tensor(local_count, dtype=torch.long, device=local_sum.device).clone()
    logged_sum = local_sum.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(count)
        dist.all_reduce(logged_sum)
    if count.item() <= 0:
        raise ValueError("Objective has no valid tokens")
    # DDP averages parameter gradients; restore global sum/count weighting.
    return local_sum * world_size() / count, logged_sum / count, count


@contextmanager
def dropout_disabled(module):
    modes = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in modes:
            child.training = training


def make_sigreg(projection_seed=31415, slices=256):
    # Gloo lacks AVG. SUM/world is mathematically identical in forward AND
    # backward. Keep the upstream estimator and sample scaling unmodified.
    # Apply the same portable helper for both NCCL and Gloo, tested against a
    # single-process concatenation of the identical samples and projections.
    import lejepa.univariate.epps_pulley as ep
    from torch.distributed.nn.functional import all_reduce

    def distributed_average(x, op="AVG"):
        if not dist.is_initialized():
            return x
        if op != "AVG":
            raise ValueError(op)
        return all_reduce(x, op=dist.ReduceOp.SUM) / dist.get_world_size()

    ep.all_reduce = distributed_average
    regularizer = SlicingUnivariateTest(
        EppsPulley(t_max=3.0, n_points=17, integration="trapezoid"),
        num_slices=slices, reduction="mean")
    regularizer.global_step.fill_(projection_seed)
    return regularizer


class AdaptationObjective(torch.nn.Module):
    def __init__(self, model, arm, mu0, s0, projection_seed=31415, slices=256):
        super().__init__()
        if arm not in ("C", "D") or not s0 > 0:
            raise ValueError("Invalid arm or calibration")
        self.model, self.arm = model, arm
        self.register_buffer("mu0", torch.tensor(mu0, dtype=torch.float32))
        self.register_buffer("s0", torch.tensor(s0, dtype=torch.float32))
        self.regularizer = make_sigreg(projection_seed, slices)
        self.model.gradient_checkpointing_disable()
        for p in self.model.parameters():
            p.requires_grad_(True)

    @property
    def encoder(self):
        return self.model.get_encoder() if self.arm == "C" else self.model

    def clean_states(self, batch):
        with dropout_disabled(self.encoder):
            return self.encoder(input_ids=batch["clean"], attention_mask=batch["clean_mask"]).last_hidden_state

    def forward(self, batch):
        clean_h = self.clean_states(batch)
        # FP32 coordinates even though model forward uses BF16 autocast.
        clean_z = (clean_h.float() - self.mu0) / self.s0
        cs = batch["clean_sample"]
        clean_sample = clean_z[cs[:, 0], cs[:, 1]]
        if self.arm == "C":
            output = self.model(input_ids=batch["corrupt"], attention_mask=batch["corrupt_mask"],
                                labels=batch["labels"], use_cache=False)
            ce_sum = F.cross_entropy(output.logits.float().flatten(0, 1), batch["labels"].flatten(),
                                     ignore_index=-100, reduction="sum")
            task, task_logged, count = global_token_mean(ce_sum, batch["labels"].ne(-100).sum())
            samples = clean_sample
        else:
            with dropout_disabled(self.model):
                corrupt_h = self.model(input_ids=batch["corrupt"],
                                       attention_mask=batch["corrupt_mask"]).last_hidden_state
            corrupt_z = (corrupt_h.float() - self.mu0) / self.s0
            pairs = batch["pairs"]
            delta = clean_z[pairs[:, 0], pairs[:, 1]] - corrupt_z[pairs[:, 0], pairs[:, 2]]
            # Per-token mean over coordinates, then global mean over tokens.
            task, task_logged, count = global_token_mean(delta.square().mean(-1).sum(), len(pairs))
            ks = batch["corrupt_sample"]
            samples = torch.stack([clean_sample, corrupt_z[ks[:, 0], ks[:, 1]]])
        with torch.autocast(device_type=clean_h.device.type, enabled=False):
            regularizer = self.regularizer(samples.float())
        if regularizer.dtype != torch.float32:
            raise TypeError("SIGReg must use FP32 arithmetic")
        loss = task + 0.05 * regularizer
        return {"loss": loss, "task": task, "sigreg": regularizer,
                "task_global": task_logged, "valid_task_tokens": count,
                "clean_sample_raw": clean_h[cs[:, 0], cs[:, 1]].detach().float(),
                "clean_sample_normalized": clean_sample.detach()}
