# C/D inference-only guidance study

**Completed, 2026-10-15.** Guidance changes the C/D operating points, but no
screened or confirmed C/D setting reaches the predefined A entropy neighborhood
[5.10, 5.20]. D improves its 32-step GenPPL/entropy pair at SC-CFG=5, yet remains
highly repetitive. The released A checkpoint reproduces the neighborhood of the
published primary reference under the same local evaluation stack.

All 54,000 requested samples were generated: 10,000 broad-screening samples,
2,000 permitted extrapolation samples, 36,000 C/D confirmation samples, and
6,000 local A samples. All 53,974 nonempty outputs were scored; 26 empty outputs
remain in the counts. There were no scoring failures. Four independent
8-H100 inference jobs succeeded under `snow.research.er`, consuming **3.193333
allocated GPU-hours** including loading, compilation, audits, sampling and scoring.

The complete [request](guidance_sweep/SPEC.md), [pre-screening plan](guidance_sweep/plan.json),
[frozen selections](guidance_sweep/selection.json), [input hashes](guidance_sweep/inputs.json),
and [completion audit](guidance_sweep/results/audit.json) are retained. Work began
at the inspected revision `1d96853d8e9e78400ff1b3ac57500e058db30155` with no newer
changes. All 140 original tracked files and 31 pinned input artifacts remain
unchanged. No adaptation, training, normalization refit, weight update or
checkpoint selection was performed.

C/D use their final epoch-5, update-95,090 EMA weights. Their matching adapted
encoder exports and original stored normalization are verified unchanged; the
unconditional generation path uses the corresponding tokenizer without an
encoder forward pass. Exact checkpoint/EMA/export hashes and normalization
values are recorded in the input manifest.

Confirmation results below use 1,000 requested samples for each of seeds 20,
21 and 22. SD is the **sample** SD of three seed-level metrics; SE = SD/√3.
These describe inference variation only, with one training seed per model.
Sampler entries are SDE steps / gamma. `*` marks an out-of-trained-range scale
token; SC-CFG=0 still uses self-conditioning. Repetition is a mean sample fraction.

| Sampler | Arm | SC-CFG | GenPPL mean ± SD | Entropy mean ± SD | SE: PPL / entropy | Repeated 4-grams |
|---|---|---:|---:|---:|---:|---:|
| 32 / 1.5 | C | 1.5 | 15.146 ± 0.896 | 4.377 ± 0.059 | 0.518 / 0.034 | 20.35% |
| 32 / 1.5 | C | 2 | 14.194 ± 0.887 | 4.389 ± 0.062 | 0.512 / 0.036 | 21.49% |
| 32 / 1.5 | C | 3 | 12.887 ± 0.875 | 4.382 ± 0.060 | 0.505 / 0.035 | 23.13% |
| 32 / 1.5 | D | 3 | 7.313 ± 0.113 | 3.270 ± 0.021 | 0.065 / 0.012 | 46.28% |
| 32 / 1.5 | D | 4 | 6.859 ± 0.175 | 3.339 ± 0.032 | 0.101 / 0.019 | 47.23% |
| 32 / 1.5 | D | 5 | 6.712 ± 0.214 | 3.409 ± 0.040 | 0.123 / 0.023 | 46.67% |
| 32 / 1.5 | A local | 3 | 23.672 ± 0.433 | 5.150 ± 0.015 | 0.250 / 0.008 | 6.37% |
| 64 / 1.0 | C | 0* | 28.913 ± 0.661 | 4.734 ± 0.012 | 0.381 / 0.007 | 9.54% |
| 64 / 1.0 | C | 1.5 | 15.214 ± 0.561 | 4.568 ± 0.029 | 0.324 / 0.017 | 20.15% |
| 64 / 1.0 | C | 3 | 10.754 ± 0.446 | 4.398 ± 0.027 | 0.257 / 0.016 | 28.24% |
| 64 / 1.0 | D | 0* | 29.566 ± 0.779 | 4.286 ± 0.016 | 0.450 / 0.009 | 11.84% |
| 64 / 1.0 | D | 0.75 | 20.058 ± 1.034 | 4.105 ± 0.021 | 0.597 / 0.012 | 18.26% |
| 64 / 1.0 | D | 3 | 7.681 ± 0.294 | 3.834 ± 0.030 | 0.170 / 0.017 | 42.42% |
| 64 / 1.0 | A local | 3 | 19.196 ± 0.359 | 5.087 ± 0.009 | 0.207 / 0.005 | 12.67% |

For D at 32 steps, SC-CFG=5 versus 3 changes GenPPL by **−0.601 ± 0.122 SD
(SE 0.071)** and entropy by **+0.139 ± 0.019 SD (SE 0.011)**. Both directions
hold for all three paired seeds. Repeated 4-grams remain 46.67% versus 46.28%;
this numeric improvement does not demonstrate coherent text. At 32 steps,
C's screening entropy advantage at SC-CFG=1.5 does not persist: paired entropy
change is −0.0050 (SE 0.0097), while PPL rises 2.259 (SE 0.286). It does reduce
repeated 4-grams by 2.78 percentage points. C at SC-CFG=2 has essentially the
same entropy as its reference and higher PPL, with somewhat less repetition.

At 64 steps, reducing SC-CFG improves diversity and reduces repetition at a PPL
cost. C at the extrapolated scale 0 changes entropy by +0.336 (SE 0.0089),
PPL by +18.159 (SE 0.142), and repeated 4-grams by −18.70 percentage points
relative to scale 3. D at scale 0 changes entropy by +0.452 (SE 0.0106), PPL
by +21.885 (SE 0.281), and repeated 4-grams by −30.59 points. The selected
in-range settings show the same tradeoff. Maximum confirmed C/D entropies are
4.734 / 4.286, below both local A anchors. **No approximate matched-entropy
improvement over A is established.** Lower C/D PPL generally accompanies lower
entropy; higher unigram entropy alone does not establish coherence.

[All paired-seed differences](guidance_sweep/results/paired_differences.csv)
include within-arm comparisons, shared confirmed C/D scales, and each operating
point versus local A. Pairing uses the same recorded random streams and does
not supply training-seed uncertainty. Three inference seeds and one A scale per
sampler cannot establish a complete A frontier or broad statistical robustness.
Original C/D evaluations used seeds 0/1/2 and remain unchanged in [RESULTS.md](RESULTS.md);
they are not pooled with this study.

`A_measured_local` uses released checkpoint update 95,085 / epoch 5 from
[ELF-B-owt-torch](https://huggingface.co/embedded-language-flows/ELF-B-owt-torch/tree/146f84133c1389bfd4ef47f14ec7a955da22faa7),
strictly loaded EMA weights, and its original normalization 0 / 0.2. Its 32-step
GenPPL 23.672 and entropy 5.150 are close to `A_published_external` 24.08 and
5.15 ([paper Table 9](https://arxiv.org/html/2605.10938v2#A5.T9); external SEs
0.16 / 0.002 across six seeds 0–5). This is an evaluation-stack check, not an
exact replication or a measured A curve. The local 64-step anchor is 19.196 /
5.087; the published 32-step point is not treated as a 64-step observation.

The [32-step plot](guidance_sweep/results/primary.png) and
[64-step plot](guidance_sweep/results/secondary.png) show every screening point,
every confirmation seed, confirmation means ± sample SD, and repetition.
Labels identify SC-CFG, triangles identify extrapolation, and segments join only
empirical nondominated points as visual summaries. No unsampled interpolation
is claimed. SVG/PDF exports accompany the PNGs in the same directory.

Both actual C/D checkpoints passed the [GPU guidance audits](guidance_sweep/results/guidance_audits.json).
At ordinary CFG {0, .5, 1, 2, 3, 5}, the maximum forward and end-to-end latent
differences were exactly 0 (tolerance 1e-6), with identical decoded tokens and
RNG/noise records for both samplers. The forward test retained nonzero previous
self-conditioning and the real zero unconditional mask. Ordinary CFG is
therefore inactive and fixed to 1. SC-CFG is active: the .5-versus-5 predicted
state difference reaches 3.6604 for C and 9.3373 for D. The final decoder was
verified to receive the requested scale. Actual training YAML, pinned Config
defaults and sampling code establish the trained interval [.5, 5]; no per-update
scale trace is available.

The unchanged original sampling functions retain latent length 1024,
logit-normal times (P_mean −1.5, P_std .8), denoiser noise scale 2, t_epsilon .05,
FP32 parameters/latents, BF16 forward, final argmax and EOS masking. Each
configuration resets Python, NumPy, CPU/CUDA global RNGs and the explicit CPU
generator to `seed + rank*1000003`. Fixed rank blocks are 32/32/32/32/32/32/32/26
for screening and 125 per rank in batches 64+61 for confirmation. No padding
texts are generated and discarded. Compilation and audits use disjoint streams.
An interleaved [3, .5, 3] execution replay is bitwise identical; time-grid,
initial-latent and RNG hashes match across scales and arms, including local A.

All four groups screened [.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5] at seed 10,
250 samples each. Their .5 endpoints triggered the allowed 0/.25 explorations.
No in-range pair bracketed entropy 5.15, so the predefined rule added zero
refinements. Screening entropy maxima were C 4.378/4.790 and D 3.414/4.268
for 32/64 steps. The fixed rule retained scale 3, the highest-entropy frontier
point, and the best point in [5.10,5.20] or closest frontier point; deduplicated
slots used a distinct frontier point. All selections and their repetition,
length and empty diagnostics were frozen before either confirmation job was
submitted. Confirmation results did not initiate another search.

Scoring calls the original pinned GPT-2 Large `Metrics` implementation, batch
64, with the original preprocessing, padding/EOS weighting and 1024-token cap.
GenPPL is exp(total weighted NLL / total weights), not mean per-sample PPL or
ELF likelihood. Entropy is mean **per-sample** unigram entropy **after** the cap.
Repeated 4-gram fraction is `1 - unique_4grams/total_4grams`, undefined below
four tokens; adjacent repetition is equal neighboring pairs / (n−1), undefined
below two. Both use post-cap GPT-2 tokens. Duplicate rate is
`1 - distinct exact nonempty texts/nonempty count` within each seed/configuration.
Lengths before/after the cap, T5 retokenized lengths, undefined diagnostic
counts, empty IDs and scoring counts are preserved. The maximum exact-duplicate
rate was 0.1%; mostly distinct texts can still be highly repetitive internally.

Machine-readable [per-seed CSV](guidance_sweep/results/per_seed.csv) /
[JSON](guidance_sweep/results/per_seed.json), [aggregate CSV](guidance_sweep/results/aggregate.csv) /
[JSON](guidance_sweep/results/aggregate.json), and paired-difference CSV/JSON
retain all observed outcomes, including dominated points. Aggregate statistics
are means, sample SDs and SEs of seed metrics, not pooled-seed PPL.

Every generated text and its per-sample score is retained under
`/home/toolkit/ELF/artifacts/guidance_sweep_20261015`, outside Git.
[Fixed complete qualitative samples](/home/toolkit/ELF/artifacts/guidance_sweep_20261015/qualitative_samples.html)
([JSON](/home/toolkit/ELF/artifacts/guidance_sweep_20261015/qualitative_samples.json))
contain seed 20, IDs 0/1/2 for **every** confirmed configuration: 42 texts without
quality selection. The fixed C/D texts show broken prose, semantic contradictions
and repeated phrases/instructions even where auxiliary diversity improves.
[Artifact paths and SHA256s](guidance_sweep/results/artifacts.json) cover texts,
scores, generation records, logs and decisions; [input hashes](guidance_sweep/inputs.json)
include checkpoints, EMA state hashes, encoders, normalization and evaluator pins.

All jobs below exited 0 / SUCCEEDED, with no failed or restarted attempts. Each
requested 8 H100s, 64 CPUs and 512 GB RAM explicitly under `snow.research.er`.
C and D used separate allocations; C confirmation also ran local A. Queue time
is excluded. Stage times are contained in these allocations and are not added
again; prior experiment training costs are excluded.

| Arm / phase | Job ID | Node | GPU-hours |
|---|---|---|---:|
| C / screening | `7cdf5bab-c51c-4893-a80c-c0d69ccb9a58` | dgx-10 | 0.535556 |
| D / screening | `6f783014-f4c8-4841-9517-de70b635eb5a` | dgx-15 | 0.531111 |
| C / confirmation | `b5bf0c65-0278-451f-93ed-0c9850d33bbb` | dgx-15 | 1.188889 |
| D / confirmation | `aa0f57cd-f2b9-4df9-9551-e533c5fe5cf9` | dgx-10 | 0.937778 |

[Scheduler terminal evidence](guidance_sweep/results/scheduler.json) and
[compute CSV](guidance_sweep/results/compute.csv) retain allocation start/end
and node records. Screening release: `9d3ee65578632d58849df194d5d3c274b9c31d42`;
frozen confirmation release: `62010766741b6272a9204ddb41bf58bfb08333ab`. Exact EAI launch specifications and
immutable source archives are in the artifact root. [Checks](guidance_sweep/checks.json)
record seven passing focused tests, the unchanged tested GPU environment, and
successful numerical checks on both real checkpoints. Plotting used a separate
CPU-only environment pinned by [requirements-report.lock](guidance_sweep/requirements-report.lock).

The executed phase commands, from `/home/toolkit/ELF`, were:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD"
.venv/bin/python -m guidance_sweep.submit --phase screening --dry-run
.venv/bin/python -m guidance_sweep.submit --phase screening
.venv/bin/python -m guidance_sweep.submit --status
# After both screening jobs succeeded, before any confirmation:
.venv/bin/python -m guidance_sweep.freeze
git add guidance_sweep/selection.json
# The frozen manifest was committed in the confirmation release recorded above.
.venv/bin/python -m guidance_sweep.submit --phase confirmation --dry-run
.venv/bin/python -m guidance_sweep.submit --phase confirmation
.venv/bin/python -m guidance_sweep.submit --status
.venv/bin/python -m guidance_sweep.report
artifacts/guidance_sweep_20261015/report-env/bin/python guidance_sweep/plot.py
```

The submitter explicitly passes `--account snow.research.er` and verifies the
resource settings before every submission. Repeated commands skip live and
completed jobs. Only after a terminal unsuccessful attempt, resume with the
same phase plus `--resume`; the original immutable worker release and verified
completed settings are reused, even if later report commits exist. No replacement
is submitted for a live job. The completed study currently needs no resume.

```bash
.venv/bin/python -m guidance_sweep.submit --phase screening --resume
.venv/bin/python -m guidance_sweep.submit --phase confirmation --resume
```

For a **fresh full replay with the already frozen choices**, commit any local
study changes, use a new output root and job-name suffix, and run the following
phases sequentially. Keep the existing frozen manifest; this is reproduction,
not a new selection round. These extra jobs were not run for this report.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD"
STUDY_REPLAY="$PWD/artifacts/guidance_sweep_replay_20261015"
.venv/bin/python -m guidance_sweep.submit --root "$STUDY_REPLAY" --name-suffix replay_20261015 --phase screening --dry-run
.venv/bin/python -m guidance_sweep.submit --root "$STUDY_REPLAY" --name-suffix replay_20261015 --phase screening
.venv/bin/python -m guidance_sweep.submit --root "$STUDY_REPLAY" --status
# Wait for both screening jobs to succeed before confirmation.
.venv/bin/python -m guidance_sweep.submit --root "$STUDY_REPLAY" --name-suffix replay_20261015 --phase confirmation --dry-run
.venv/bin/python -m guidance_sweep.submit --root "$STUDY_REPLAY" --name-suffix replay_20261015 --phase confirmation
.venv/bin/python -m guidance_sweep.submit --root "$STUDY_REPLAY" --status
# After both confirmation jobs succeed:
.venv/bin/python -m guidance_sweep.report --root "$STUDY_REPLAY" --output "$STUDY_REPLAY/report"
artifacts/guidance_sweep_20261015/report-env/bin/python guidance_sweep/plot.py --directory "$STUDY_REPLAY/report"
```

To recreate only the plotting environment:

```bash
uv venv --python .venv/bin/python artifacts/guidance_sweep_20261015/report-env
uv pip install --python artifacts/guidance_sweep_20261015/report-env/bin/python -r guidance_sweep/requirements-report.lock
```

The GPU environment remains the original tested `.venv-gpu`; no setup jobs are needed.
