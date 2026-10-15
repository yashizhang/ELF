# C/D inference-only guidance study

Status: implementation and CPU checks complete; GPU screening has not started.
The full requested scope is retained in [SPEC.md](guidance_sweep/SPEC.md).
The original experiment at `1d96853` and its reports/artifacts remain unchanged.

This study uses final C/D EMA checkpoints at update 95,090, their matching raw
encoders and stored normalization, and the same pinned GPT-2 Large evaluator.
[inputs.json](guidance_sweep/inputs.json) resolves and verifies their paths and
hashes. The compatible released original ELF-B checkpoint is pinned to Hugging
Face revision `146f84133c1389bfd4ef47f14ec7a955da22faa7`; its EMA loads strictly,
and its original normalization is mean 0 / standard deviation 0.2.

The [fixed plan](guidance_sweep/plan.json) records the two samplers, ten screening
scales, separate RNG streams, bounded refinement, and selection rules. The A
entropy neighborhood is defined before screening as **[5.10, 5.20]**.
Screening uses seed 10 and 250 samples; confirmation uses seeds 20/21/22 and
1,000 samples per selected configuration. All four selections will be frozen
before either confirmation job starts. Each phase has two independent
eight-H100 jobs under `snow.research.er`, with 64 CPUs / 512 GB each.
C's confirmation allocation also evaluates the released A anchor.

Source inspection finds ordinary CFG inactive for the zero unconditional
prefix. The GPU jobs must verify this numerically at CFG 0/.5/1/2/3/5, including
nonzero previous self-conditioning and full 32/64-step fixed-noise rollouts.
SC-CFG is a trained scale-token input (range .5–5), including final decoding.
Scale zero is not a no-self-conditioning ablation. No sampler, weights,
normalization, temperature, or gamma changes are made.

Metrics call the original GPT-2 Large implementation with batch 64 and a 1024
context cap. Its entropy is the mean sample unigram entropy **after the cap**.
Repeated 4-gram fraction is `1 - unique_4grams / total_4grams`, undefined below
four tokens. Adjacent repetition is equal neighboring token pairs divided by
`n-1`, undefined below two tokens. These diagnostics use post-cap GPT-2 tokens.
Duplicate-text rate is `1 - distinct exact nonempty strings / nonempty count`.
Empty strings remain in generated counts and are excluded from scoring.
Per-sample scores and all generated texts stay in the new artifact directory.

The sampler functions and final argmax/EOS decoder are reused unchanged.
Every configuration resets Python, NumPy, CPU and CUDA global RNGs and the
explicit CPU generator to `seed + rank*1000003`. Fixed rank blocks contain
32/32/32/32/32/32/32/26 screening samples; confirmation uses 125 per rank in
batches 64+61, matching the completed evaluator. No extra texts are discarded.
Compilation/warmup and numerical audits use separate streams. Time-grid,
initial-latent, and before/after RNG hashes verify common randomness.

Run from the existing tested environments; no setup or training is needed:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_guidance_sweep.py -q
.venv/bin/python -m guidance_sweep.submit --phase screening --dry-run
.venv/bin/python -m guidance_sweep.submit --phase screening
.venv/bin/python -m guidance_sweep.submit --status
# After both screening jobs succeed:
.venv/bin/python -m guidance_sweep.freeze
# Commit the frozen selection before creating the immutable confirmation release.
.venv/bin/python -m guidance_sweep.submit --phase confirmation --dry-run
.venv/bin/python -m guidance_sweep.submit --phase confirmation
.venv/bin/python -m guidance_sweep.submit --status
```

The submitter refuses duplicate live/completed jobs and always passes
`--account snow.research.er`. A verified terminal unsuccessful inference job can
be replaced using the same phase command with `--resume`; completed settings
are reused only when identities and sample/score hashes match. Partial writes
are atomic and each configuration has an independent reproducible RNG stream.
Logs, failures, settings, selections, and scheduler history are retained under
`/home/toolkit/ELF/artifacts/guidance_sweep_20261015`. Stage times are nested
within allocated GPU hours and must not be added again.
