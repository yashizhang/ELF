# T5-small SIGReg C/D experiment

The full preregistered scope is in [GOAL.md](GOAL.md). This branch starts at
official PyTorch ELF commit `b29d8833609e9ab7f67cd9da39435ac5cea04837`;
the experiment is delivered on `t5-sigreg-cd`. Before this goal began, the
fork's `main` had already advanced to `398ac406be5e7a0c61fbfdc3d854faf86d46ad96`,
an import of the same upstream tree preserving `5098bf2` as its parent.
This experiment leaves that branch intact; see
[repository evidence](evidence/final-repository-checks.json).
[RESULTS.md](RESULTS.md) now reports both completed pipelines.

## Environment and source pins

Use Python **3.11.15**, PyTorch **2.5.1+cu124**, CUDA runtime **12.4** for
H100 execution. CPU correctness checks use **2.5.1+cpu**. The actual H100
pilot records package versions, GPU model/memory and topology in
`artifacts/preflight/attempt2/report.json`. All transitive package versions
are pinned in `requirements-adaptation-gpu.lock` and
`requirements-adaptation.lock` (CPU). Setup does not modify a shared conda
environment.

The dataset reader uses **Datasets 4.0.0**, **PyArrow 17.0.0**, and
**NumPy 1.26.4**. The downloaded OWT release uses the
[Datasets `List` feature](https://huggingface.co/docs/datasets/v4.0.0/package_reference/main_classes#datasets.List),
which the initial 2.21.0 reader could not deserialize. The initial PyArrow 26
wheel also failed its NumPy runtime check. PyArrow 17's
[documented dependencies](https://arrow.apache.org/docs/17.0/python/install.html#dependencies)
permit the required NumPy 1.x environment. Both installed environments now
match their corrected locks, and the actual OWT/ELF loader passes, including
multiple worker processes and validity masks. The failed reader attempts are
retained in `artifacts/shared/owt-preparation*.log`; they ran within the existing
CPU allocation and never changed the downloaded token files.

The runtime image lacks a system C++ compiler. `compiler-linux-64.lock`
pins a shared conda-forge GCC/G++ 11.4.0 toolchain with glibc 2.17 sysroot;
the GPU setup installs it under `artifacts/compiler`. EAI launch configs set
`CC` and `CXX` explicitly so the upstream ELF `torch.compile` path works in
the Ubuntu 22.04 container without root package installation. A separate
one-H100 check uses 16 CPUs / 128 GB RAM and verifies compiled ELF updates,
freezing, memory at local batch 64 / length 1024, and a small generated batch.
It passed with 63.14 GB peak allocated memory and activation checkpointing off.
Its fixture uses original T5/C4 and is excluded from experimental OWT results.

```bash
git switch t5-sigreg-cd
git submodule update --init --recursive
bash scripts/setup_adaptation_env.sh cpu
bash scripts/setup_adaptation_env.sh gpu
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
```

The modifiable T5 source is
`third_party/transformers/src/transformers/models/t5/modeling_t5.py`.
Transformers `v4.44.2` resolves to
`174890280b340b89c5bfa092f6b4fb0e2dc2d7fc`; LeJEPA is pinned to
`c293d291ca87cd4fddee9d3fffe4e914c7272052`. Both install editable. Upstream
source and licenses are retained unchanged. Bytecode writes are disabled
because the LeJEPA revision itself tracks some Python bytecode files.
`src/configs/t5_adaptation/sources.json` pins T5, C4, OWT, GPT-2 Large and code
revisions. There is no FLAN/v1.1 substitution or randomly initialized T5
experiment.

## Data and exact objectives

`t5_adaptation.prepare` reads only the necessary lexicographic `en/c4-train`
shard prefix. It appends EOS at every document boundary, concatenates tokens
without discarding per-document remainders, and packs consecutive 568-position
sources. `documents.jsonl` records source shard, source row, text hash and
absolute token offsets, including a partial final document. Fixed packing
boundaries are multiples of 568. JSON source-shard hashes and output hashes
are retained. Resume truncates uncommitted shard tails and retains the
committed prefix. An OS preparation lock prevents concurrent writers.
Validation uses the official C4 validation shards, with a fixed prefix large
enough for diagnostics at both 569 and 1024 clean positions.

The boundary-free corruption exactly matches the pinned Transformers Flax
collator's NumPy methods, including sentinel order and 568→512/114 lengths.
For internal special-token boundaries, noise-token quotas use largest
remainders over eligible contiguous regions. Spans are drawn within each
region; singleton regions remain clean. No tokens or decoder labels are
truncated to canonical lengths. Occurrence-position maps align surviving
ordinary tokens. Separate example-ID-keyed RNGs control corruption (1729)
and per-view token sampling (2718); neither depends on decoder/dropout RNG
consumption. Projection seed/counter starts at 31415.

Calibration uses original frozen T5 on C4 training IDs 0–4095 in FP32,
excluding all special tokens, and merges FP64 population moments. The
calibration prefix hash is checked against the immutable packed data. Its
scalar mean/std remain fixed for both arms. ELF later computes a separate
per-encoder BF16 OWT normalization on the same original OWT rows 0–4095,
at length 1024, using the upstream validity-mask convention, including EOS.
Exported encoders always return raw T5 states.

OWT contains **9,737,184 examples** and **9,316,215,849 stored tokens**;
a complete scan found lengths 2–1024 and token IDs 1–31999, all within the
unchanged tokenizer. Each full ELF epoch has 19,018 updates and 32 prefix
duplicates to fill its last global batch: five epochs require **95,090 updates**.
See `evidence/owt-data-manifest.json` and `evidence/owt-loader-check.json`.
All OWT calibration batches pad to exactly 1024 positions, even when the
longest sequence in a batch is shorter; additional padding is excluded from
the population moments, while original EOS positions remain included.

The single DDP-wrapped project module implements:

- C: global valid-target CE + `0.05 * SIGReg(clean)`; ordinary train-mode
  span encoder/decoder pass, gradient-enabled dropout-disabled clean pass.
- D: global matched-token pairwise coordinate MSE +
  `0.05 * (SIGReg(clean) + SIGReg(corrupt)) / 2`; both encoder passes have
  gradients and no dropout. Shared embeddings and final RMS scale train.

Every rank samples four distinct ordinary tokens per default source example,
with deterministic spillover for exceptionally short examples: 256 samples
per rank, 2048 per view globally, 256 unit directions, 17 trapezoid points
through 3.0. D's views retain separate empirical characteristic functions.
SIGReg explicitly disables autocast and uses the upstream sample-scaled
Epps–Pulley statistic. A project-owned portable reduction helper uses
autograd-aware `SUM / world_size`, equivalent to AVG and supported by both
Gloo and NCCL. CE/consistency use global valid-token counts with DDP gradient
averaging accounted for. There is no convex loss mixture or division by N.

38,147 updates × 512 examples × 512 nominal positions = **10,000,007,168**
nominal positions, representing 19,531,264 source examples and 11,093,757,952
packed source positions. Counters separately report source tokens, ordinary
eligible tokens, removed tokens/spans, decoder targets, passes and compute.
The three-update GPU pilot is a disposable correctness check, excluded from
these training budgets and prohibited as an ELF initialization checkpoint.

## Scheduler commands

`cluster/eai.yml` records discovered EAI image, storage, account and GPU
spelling. **Every compute request explicitly selects `snow.research.er`.**
No AFM jobs are modified. Full pipelines request eight H100s, 64 CPUs and
512 GB RAM each. Shared CPU preparation uses 32 CPUs / 128 GB RAM.

Preparation already has a recorded job in
`artifacts/shared/preparation-job.json`. Its real entry point is
`bash scripts/prepare_cd.sh`; do not submit it again while that job is live.
The command used was:

```bash
eai job new --account snow.research.er --restartable \
  --name elf_cd_prepare_20261012 \
  --image docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755 \
  --cpu 32 --mem 128 --gpu 0 --max-run-time 0 \
  --data snow.home.yashi_zhang:/home/toolkit:rw --workdir /home/toolkit/ELF \
  -- bash scripts/prepare_cd.sh

PYTHONPATH=src .venv/bin/python -m t5_adaptation.prepare \
  --root artifacts/shared --stage evaluator
```

Calibration can read the already committed first C4 shard. The actual pilot
command inside an eight-H100 allocation is
`bash scripts/preflight_cd.sh artifacts/preflight/attempt2`. Its first
attempt failed on a missing `git` executable in the container, before any
adaptation training. The corrected logger reads provenance without requiring
git; the second attempt passed. Both job manifests and logs are retained.

After all shared preparation and the pilot have succeeded, the single
submission entry point creates two independent pipeline jobs:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src"
.venv/bin/python -m t5_adaptation.submit \
  --pilot artifacts/preflight/attempt2/report.json --dry-run
.venv/bin/python -m t5_adaptation.submit \
  --pilot artifacts/preflight/attempt2/report.json
.venv/bin/python -m t5_adaptation.submit \
  --pilot artifacts/preflight/attempt2/report.json --status
```

The live preparation controller can perform submission once the authoritative
preparation job succeeds:

```bash
.venv/bin/python -u -m t5_adaptation.await_preparation \
  --preparation-job 28b2739a-13f8-48cf-b4f9-e2239b524332 \
  --pilot artifacts/preflight/attempt2/report.json
```

Do not start duplicate controllers. The controller records its observations
in `artifacts/preparation-controller.json`; its process and scheduler job
must still be checked when deciding whether work is live.

Shared preparation completed successfully, and the controller submitted both
full pipelines on 2026-10-13 at 01:24 UTC under `snow.research.er`:

| Arm | Scheduler job | Initial node | Resources |
| --- | --- | --- | --- |
| C | `c185b27b-f543-4065-8b3c-d8fa3f419fc5` | `dgx-14` | 8 H100, 64 CPU, 512 GB RAM |
| D | `591cc3b5-32f9-4aef-8c17-2dfd0841fa7a` | `dgx-09` | 8 H100, 64 CPU, 512 GB RAM |

Both execute immutable source release
`1591a0a1719e3e7422439cf366f8229b63fcafd4`. The initial live audit in
[`evidence/full-pipeline-launch.json`](evidence/full-pipeline-launch.json)
verified eight physical H100s per arm, identical original encoder hashes,
shared data/calibration identity, finite initial diagnostics, and readable
recovery checkpoints with matching SHA-256 hashes and all eight RNG states.
The audited checkpoints were C update 2,000 and D update 2,500; these are
historical launch checks, not final experiment results. The complete C4
[training](evidence/c4-train-manifest.json) and
[validation](evidence/c4-validation-manifest.json) manifests retain exact
source shards, packing boundaries, document counts, and content hashes.
Refresh the report below for later progress; report-only commits do not
change the source release used by running jobs.

Both arms subsequently saved their first milestone at update 3,815
(1,000,079,360 nominal positions). The
[`first-billion-positions.json`](evidence/first-billion-positions.json) audit
records verified checkpoint hashes, identical data/corruption counters, and
finite held-out diagnostics at both sequence lengths. These diagnostics are
intermediate measurements; final ELF generation remains the primary outcome.

The subsequent [3B milestone audit](evidence/three-billion-positions.json)
verified both retained checkpoints at update 11,445 (3,000,238,080 nominal
positions), matching data/corruption counters, and finite diagnostics at both
lengths. These fixed milestones did not select the encoder used by ELF.

Both arms completed adaptation at update **38,147** (10,000,007,168 nominal
positions, 19,531,264 source examples, 11,093,757,952 source positions each).
The final [C](evidence/final-adaptation-C.json) and
[D](evidence/final-adaptation-D.json) checkpoint audits verified their digests,
finite FP32 model and optimizer tensors, final learning rates, projection
counters, and all eight RNG states. The
[paired audit](evidence/final-adaptation-paired.json) verified identical initial
encoder hashes, data/source/calibration identities, and final source,
corruption, and encoder-pass counters. C additionally processed 2,230,695,230
decoder target tokens; D used none.

Both [C](evidence/encoder-handoff-C.json) and
[D](evidence/encoder-handoff-D.json) encoder handoff audits verified that the
export exactly matches its final checkpoint, reloads through ELF with zero
measured FP32 output error, and preserves the original architecture and
tokenizer IDs. Separate OWT normalization used the same specified 4,096 rows:

| Arm | OWT latent mean | OWT latent standard deviation |
| --- | --- | --- |
| C | -0.0012828207528431412 | 0.17384837979065573 |
| D | -0.0009771881921668139 | 0.17378080905341894 |

The matched 500-update token readouts are complete:

| Frozen encoder | Clean token accuracy | Noisy token accuracy |
| --- | --- | --- |
| Original | 0.795288 | 0.795959 |
| C | 0.804993 | 0.804688 |
| D | 0.884460 | 0.884460 |

These are decodability diagnostics, not generation-quality results. Full probe
records and normalization identities are retained in
`evidence/readout-{original,C,D}.json` and `evidence/normalization_owt_{C,D}.json`.

Both arms then initialized fresh ELF-B models with the identical initial hash
`7ef18e4d2f653a4ac286726855b0ead1e86002e01ac18747169b04b2cd068357`,
generated their eight-sample smoke batches, and began the required five-epoch
runs with finite logged losses. The first ELF recovery checkpoint audits for
[C](evidence/elf-C-checkpoint-500.json) and
[D](evidence/elf-D-checkpoint-500.json) verified finite model/EMA states,
matching experiment identities, eight hashed optimizer shards, and eight RNG
records at update 500. C's audit also checked every saved optimizer tensor for
finiteness, all optimizer step counters, and the complete per-rank parameter
assignment. These are historical initialization checks; final completion
evidence appears below.

Both arms subsequently completed their first full OWT epoch at update 19,018.
The epoch checkpoint audits for [C](evidence/elf-C-epoch-1.json) and
[D](evidence/elf-D-epoch-1.json) verified epoch 1.0, finite model/EMA weights,
all eight optimizer-shard hashes, complete per-rank parameter assignments,
optimizer and scheduler counters at 19,018, and the configured learning rate
of 0.002. Their data-exposure counters match. Model and EMA hashes in each arm
advanced from its audited step-500 checkpoint. The retained log/status
observations confirm that both arms continued training into epoch two.

Both arms subsequently saved their second full-epoch checkpoints at update
38,036 and continued into epoch three. The milestone records for
[C](evidence/elf-C-epoch-2-progress.json) and [D](evidence/elf-D-epoch-2-progress.json)
verify the saved epoch/counters and experiment identities, the presence of all
eight referenced optimizer files per arm, the epoch-completion logs, and finite
continuation. Their epoch and data-exposure counters match. These progress
records cover checkpoint metadata, file presence, and observed continuation;
their scope is recorded explicitly.

Both arms subsequently saved their third full-epoch checkpoints at update
57,054 and continued into epoch four. The progress records for
[C](evidence/elf-C-epoch-3-progress.json) and [D](evidence/elf-D-epoch-3-progress.json)
verify saved epoch 3.0, matching experiment identities and scheduler counters,
eight referenced optimizer files per arm, the epoch-completion logs, and finite
continuation. Their epoch and data-exposure counters match: each represents
29,211,552 source-row visits and 29,211,648 example positions including 96
padding repeats under the verified sampler. These records have the same
metadata and progress scope as the epoch-two records above.

Both arms subsequently saved their fourth full-epoch checkpoints at update
76,072 and continued into epoch five. Their
[C](evidence/elf-C-epoch-4-progress.json) and
[D](evidence/elf-D-epoch-4-progress.json) progress records verify saved epoch
4.0, matching experiment identities and scheduler counters, eight referenced
optimizer files per arm, the epoch-completion logs, and finite continuation.
Their epoch and data-exposure counters match: each represents 38,948,736
source-row visits and 38,948,864 example positions including 128 padding
repeats under the verified sampler. These records have the same metadata and
progress scope as the preceding milestone records.

Both arms finished all five epochs at update **95,090**, then generated all
6,000 requested texts per arm. Scoring included 5,997 nonempty C texts and all
6,000 D texts; C's three empty outputs are disclosed in the report. The deep final checkpoint audits for
[C](evidence/elf-C-epoch-5.json) and [D](evidence/elf-D-epoch-5.json) verify finite
model/EMA and all eight optimizer shards, exact counters and identities, RNGs,
and matching parameter assignments. Final generation audits for
[C](evidence/generation-C.json) and [D](evidence/generation-D.json) verify the
protocol, provenance, samples, exclusions, tokenization, and aggregate metrics.
The original jobs succeeded with exit 0 at 2026-10-14 03:08:29 UTC (C) and
02:21:14 UTC (D), with no replacement attempts; see
[terminal scheduler evidence](evidence/final-scheduler.json).
The periodic ELF `status.json` files stop at update 95,000 and still say
`running`; their historical telemetry is superseded by the final checkpoint,
completion records, stage exits, and scheduler observations.

Audit an available full-epoch checkpoint in the CPU environment with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/audit_elf_epoch.py \
  --arm D --epoch 1 --output artifacts/elf-D-epoch-1-reaudit.json
```

Use `--arm C` for C and `--epoch 1` through `--epoch 5` for the corresponding
saved epoch. Choose an unused output path; the auditor preserves earlier audit
files. It checks the saved model, EMA, optimizer and RNG records without
loading a GPU model. Dataset exposure is inferred from the saved step and the
separately verified full-epoch sampler; it is not a per-example runtime trace.

The lightweight status recorder queries the job IDs in the submission manifest
under `snow.research.er` every 45 seconds:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/monitor_cd.py
```

Use `--once` for one observation before starting a persistent recorder. An
exclusive lock prevents duplicate recorders for the same artifacts directory.
`artifacts/monitor.json` holds the latest scheduler states, run history, stage
states, and saved/observed progress; `monitor-history.jsonl` retains observations.
The recorder follows updated job IDs after an authorized replacement and keeps
observing the other arm if one job fails. Scheduler query errors are recorded
as observation errors and retried against the same job IDs. It reads scheduler
state and writes observation files; job submission remains the submitter's job.
Check the recorded host/PID or query the scheduler directly before treating an
old observation as live. Scheduler success still requires the final scientific
acceptance audit described below.

Each pipeline stage retains an independent start/end record in
`stage_attempts/`. A restart cannot erase an unfinished attempt's start record.
Reports use the scheduler run's end time for an attempt terminated before it
could write its own end record, and label that timing source explicitly.
If a stage finishes successfully while a termination signal is being handled,
its verified output is retained and the pipeline exits before the next stage.

The submitter checks the explicit ER account, eight H100s, completed shared
data and GPU checks. It archives the committed project into an immutable
release directory and records job IDs, config/spec and release. Each arm
uses an independent torchrun `--standalone` rendezvous and the allocation's
visible devices. No four-GPU fallback exists. Re-running submission recovers
existing live jobs by authoritative scheduler state and name. For a verified
terminal unsuccessful job, use the same command with `--resume`; never infer
job death from a stale file or polling timeout.

The explicit recovery command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m t5_adaptation.submit \
  --pilot artifacts/preflight/attempt2/report.json --resume
```

Both jobs are already complete, so recovery is not needed for this result.
The submitter guards completed outputs and always supplies the ER account.

Pipeline order is adaptation → final-only raw encoder export/reload check →
OWT normalization → short frozen-encoder readout → fresh ELF-B → final EMA
generation → metrics. The first arm to reach the diagnostic stage also runs
the shared original-encoder readout; a nonblocking lock prevents duplicate
work without making the other arm wait. Each readout is capped at 500 updates.

For reproduction, these are the stage entry points for C (substitute D and
`arm_d.yml` for D). Training and generation run inside an eight-H100 ER
allocation obtained through the submission entry point above. The pipeline
wraps these commands with its ordered completion and identity checks; the
exact commands used by each completed stage are retained in `stage-*.json`.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src"
.venv-gpu/bin/python -m t5_adaptation.calibrate --shared artifacts/shared \
  --kind c4 --output artifacts/shared/c4_normalization.json
.venv-gpu/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m t5_adaptation.train --config src/configs/t5_adaptation/arm_c.yml \
  --shared artifacts/shared --output artifacts/C/adaptation
.venv-gpu/bin/python -m t5_adaptation.export \
  --checkpoint artifacts/C/adaptation/latest.pt --shared artifacts/shared \
  --output artifacts/C/encoder
.venv-gpu/bin/python -m t5_adaptation.calibrate --shared artifacts/shared \
  --kind owt --encoder artifacts/C/encoder --output artifacts/C/owt_normalization.json
.venv-gpu/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  src/train.py --config artifacts/C/elf.yml
.venv-gpu/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  src/eval.py --config artifacts/C/evaluation.yml \
  --checkpoint_path artifacts/C/elf/checkpoint_95090 --seeds 0,1,2
.venv-gpu/bin/python -m t5_adaptation.evaluate --samples-root artifacts/C/samples \
  --shared artifacts/shared --arm C --output artifacts/C/results.json
```

The pipeline creates `elf.yml` and `evaluation.yml` from the retained final
encoder and normalization. The `latest.pt` identity must mark the fixed-budget
final checkpoint before export is accepted. Running the adaptation command
again restores its verified latest state, including the exact data cursor.

Adaptation `latest.pt` contains model, optimizer, scheduler, per-rank RNGs,
SIGReg counter, fixed calibration, source/config identity and exact data
cursor. Milestones near 1B/3B/10B are retained. SIGTERM/SIGUSR1 checkpoints at
an update boundary. ELF checkpoints retain every rank's Muon momentum and
RNG, atomic main files, EMA, epoch/data position and encoder provenance;
each write uses new optimizer-shard filenames so an interrupted rewrite
cannot invalidate the previous committed checkpoint. Failed resume is fatal.
The asynchronous ELF prefetcher propagates loader exceptions to the trainer.
Training sample generation after the first ELF update
checks the generation path and restores RNGs. It does not select checkpoints.

ELF processes all OWT rows in each of five shuffled epochs. The last batch is
filled with documented deterministic prefix duplicates so effective batch
remains 512. Exact updates and padding repeats derive from the loaded dataset
and are recorded, rather than hardcoding the paper's approximate update count.
Both ELF models reset seed 42 immediately before fresh initialization and
record hashes. All upstream architecture, Muon hyperparameters, EMA and loss
settings remain those of the pinned ELF-B template; `eval_freq=0`, local
encoder/tokenizer paths, measured normalization, and disabled uploads are
explicit overrides.

## Evaluation and reporting

The pipeline writes resolved `artifacts/{C,D}/elf.yml` and `evaluation.yml`.
It calls the existing EMA generation entry point with
`--seeds 0,1,2`, 1,000 samples per configuration, and the original 32-step
gamma-1.5 / 64-step gamma-1.0 configurations (SC-CFG 3, CFG 1,
logit-normal schedule, length 1024). The pinned local GPT-2 Large snapshot is
scored through `utils.metrics_utils.Metrics`. Raw JSONL, hashes, per-seed
metrics, empty/scored counts, lengths and qualitative samples are retained.

After an arm's pipeline completes, audit its final checkpoint and generation
artifacts in the CPU environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/audit_elf_epoch.py \
  --arm D --epoch 5
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/audit_final_generation.py \
  --arm D
```

Use `--arm C` for C. Both commands preserve existing audit files; specify an
unused `--output` path for a new audit, and pass `--elf-audit` to the generation
auditor if the checkpoint audit used a nondefault path. The generation auditor
requires completed stages and checks the final checkpoint provenance, immutable
execution source, exact sampling settings, all 6,000 saved samples, snapshot
hashes, tokenization, exclusions, and inference-seed mean/std/SEM. It independently
reconstructs per-sample entropy and token-weighted aggregate perplexity from
saved per-sample PPL values using the repository's EOS/padding weights. It does
not rerun GPT-2 forward passes. Empty or unscorable outputs remain visible and
can be valid negative results. These audits support the full `GOAL.md`
acceptance review; their scope alone does not prove whole-experiment completion.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m t5_adaptation.report
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/finalize_cd_report.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/audit_cd_acceptance.py
```

The first command refreshes the base `RESULTS.md`, `results.json/csv`,
`compute.json/csv`, and `diagnostics.json/csv` from actual artifacts and scheduler
state. The second adds the final interpretation, pooled length summaries, and
twelve fixed-ID qualitative samples without changing the frozen execution
source. The third revalidates retained identities and artifacts and renders the
reviewed requirement mapping in [ACCEPTANCE.md](ACCEPTANCE.md). It does not
launch jobs or rerun model scoring. Its evidence limits are explicit.
`training.json/csv` retain saved and observed update counts, token accounting,
normalization, initialization hashes, logged throughput and rank-zero peak
memory. `evaluation.csv` records per-seed scores and explicit scored/empty
counts. `stage_compute.csv` provides stage subtotals within scheduler allocation
totals; these two views must not be added together. The reporting completion
flag requires both pipeline completion records, the fixed adaptation budgets,
five ELF epochs, matching fresh initialization hashes, and six evaluations per
arm. Final acceptance still includes the requirement-by-requirement audit in
`GOAL.md`. The final [artifact index](evidence/final-artifacts.json) gives paths,
sizes, and hashes for selected checkpoints, final optimizer shards, encoders,
logs, manifests, all raw sample files, and per-seed metrics. It separately labels
large shared-data hashes retained from preparation rather than freshly reread.
All earlier recovery checkpoints remain in the designated local storage; they
were not pruned or uploaded. Cloning the Git branch alone does not download
these large artifacts. Use `/home/toolkit/ELF/artifacts` on the experiment mount.

A remains
`published_external`; C/D become measurements only when their saved final
evaluations exist. B is omitted. A negative result still completes the
experiment; script completion, a queued job, or the pilot alone does not.
Constant held-out representations receive effective rank zero, so a poor
but finite diagnostic does not itself trigger a training failure.

## File changes

New project code lives in `src/t5_adaptation/`; new configs in
`src/configs/t5_adaptation/`, cluster config in `cluster/`, setup/preparation
scripts in `scripts/`, and focused checks in `tests/`. The final report and
acceptance scripts also live in `scripts/`; they do not modify training source.

Existing ELF changes are limited to provenance/freezing/initialization
checks and one smoke sample in `src/train.py`; full-epoch sampler support in
`data_utils.py`; atomic per-rank optimizer/RNG checkpointing in
`checkpoint_utils.py` and `train_utils.py`; propagation of background loader
failures in `train_utils.py`; finite-value gates in
`train_step.py`; detached optimizer communication buffers in `muon_utils.py`;
opt-in config fields; and matching the training tokenizer vocabulary size in
`eval.py`. No ELF architecture or T5 blocks were rewritten.
