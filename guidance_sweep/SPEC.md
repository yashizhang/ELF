# Inference-only guidance sweep for completed ELF Arms C and D

Work in `https://github.com/yashizhang/ELF`, using the completed `t5-sigreg-cd` experiment and its retained artifacts. Implement and execute an inference-only guidance study to determine whether the completed C/D checkpoints admit better generative-perplexity/diversity operating points than their original SC-CFG=3 evaluations. Do not start encoder adaptation, ELF training, checkpoint selection sweeps, or new research methods.

## Existing experiment and immutable inputs

Read `EXPERIMENT.md`, `RESULTS.md`, the final artifact index, source pins, and final checkpoint/generation evidence. The inspected experiment revision is `1d96853d8e9e78400ff1b3ac57500e058db30155`; resolve the actual checkout and record any newer changes. Preserve existing user changes. Make narrowly scoped additions on a separate guidance-sweep branch; do not overwrite the original experiment reports or artifacts.

Use each arm's final five-epoch / 95,090-update ELF-B checkpoint, the same EMA weights used in the completed evaluation, its matching adapted encoder, its original stored normalization, and the pinned tokenizer/evaluator. Resolve exact paths and hashes from the existing manifests under `/home/toolkit/ELF/artifacts`; do not guess checkpoint filenames. No normalization refit, encoder replacement, training resume, or model-weight changes.

Reference results, not selection targets:

| Arm | Sampling | SC-CFG | CFG | GenPPL | Mean sample entropy |
|---|---|---:|---:|---:|---:|
| A, published external | 32 SDE steps, gamma=1.5 | 3 | 1 | 24.08 | 5.15 |
| C, measured | 32 SDE steps, gamma=1.5 | 3 | 1 | 13.3770 | 4.4051 |
| C, measured | 64 SDE steps, gamma=1.0 | 3 | 1 | 10.5249 | 4.3643 |
| D, measured | 32 SDE steps, gamma=1.5 | 3 | 1 | 7.5054 | 3.2489 |
| D, measured | 64 SDE steps, gamma=1.0 | 3 | 1 | 7.6361 | 3.8359 |

The published A point is not a measured local baseline, and the original measured C/D summaries use inference seeds 0,1,2. Keep those results unchanged and distinguish them from the new sweep.

## First audit which guidance controls actually do anything

Trace the actual project evaluation entry point through sampling and final token decoding. Do not assume YAML fields are consumed.

In the inspected source:

- `src/generation.py::test_generation_uncond` hardcodes `cfg_list = [1]`.
- `src/utils/generation_utils.py` creates zero conditioning tensors and a zero conditioning mask for unconditional generation.
- `src/utils/sampling_utils.py::_forward_sample` implements ordinary CFG by removing a conditioning prefix. With an all-zero conditioning mask, the two branches have identical inputs and ordinary CFG should be a no-op.
- `_forward_sample_self_cond` uses the requested SC-CFG scale as a learned conditioning input when `num_self_cond_cfg_tokens > 0`. This is the relevant trained guidance control; it is not a second, independent synonym for ordinary CFG.
- `_dlm_decode_batch` also receives the SC-CFG scale. Preserve this behavior.

Verify these statements against the checkout. Run a small numerical invariance test on both checkpoints using identical input latents, times, nonzero previous self-conditioning predictions, and RNG state. Compare ordinary CFG values `{0, 0.5, 1, 2, 3, 5}` at fixed SC-CFG. Include an end-to-end fixed-noise test and record numerical tolerance, maximum output differences, and generated-token equality.

If ordinary CFG is inactive, explicitly report it as inactive, fix it to 1, and omit the redundant CFG dimension. Do not invent prompts, alter the zero-prefix semantics, or change self-conditioning to manufacture an effect. A numerical or RNG discrepancy is not evidence of useful CFG. If the actual checkout contains a genuinely active conditioning mechanism, document it and use ordinary CFG `{0.5, 1, 2, 3}` in the screening stage; never silently drop an active requested control.

## Fixed sampler settings

Run both existing sampler settings separately:

- Primary: 32 SDE steps, gamma=1.5.
- Secondary: 64 SDE steps, gamma=1.0.

Keep latent length 1024, logit-normal time schedule and its parameters, noise scale, t-epsilon, precision, final argmax decoding, EOS behavior, and all other sampler/evaluation behavior unchanged. Do not change the SDE update, add temperature, add a no-self-conditioning ablation, or tune gamma in this study. Do not confuse sampler gamma with the denoiser's Gaussian noise scale.

Use the existing tested GPU environment. Run C and D as independent jobs, each requesting eight H100s; let the scheduler allocate or queue them. Never run both on the same allocation without explicit device separation. Reuse downloaded models and data read-only and avoid duplicate setup jobs. Record actual allocated GPU-hours, including failed attempts, without double counting stage subtotals.

## Stage 1: broad, inexpensive screening

For each arm and sampler setting, evaluate SC-CFG:

`[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]`

Use 250 requested sequences per setting, with screening seed 10. The expected ordinary-CFG-no-op case is 40 settings / 10,000 requested sequences across C/D and both samplers. These are exploratory estimates, not final results.

Audit the checkpoint's actual trained SC-CFG interval, expected to be `[0.5, 5.0]`. If the high-entropy end at 0.5 still fails to reach the neighborhood of the A reference, permit two additional exploratory values `{0.0, 0.25}` per affected arm/sampler. Label them outside the trained guidance range. SC-CFG=0 passed as a scale token is NOT automatically equivalent to disabling self-conditioning; never label it that way.

For paired comparisons, use common random numbers across arms/scales wherever the algorithms have the same randomness structure: initial Gaussian latents, time grids, and SDE noise. Reset every relevant RNG, including CUDA's global RNG: the inspected CUDA sampler does not rely solely on its passed CPU generator. Keep batch partitioning/rank-to-sample mapping fixed, or make randomness explicitly sample-ID keyed. Verify reproducibility and independence from configuration execution order. Compilation/warm-up must not consume the recorded evaluation stream.

Avoid recompiling/reloading unnecessarily, but do not trade away semantic equivalence to the completed evaluator for performance.

## Stage 2: bounded refinement and configuration selection

Construct each arm/sampler's empirical nondominated set: lower GenPPL and higher mean sample entropy are preferred. Keep all observations; do not hide low-entropy, repetitive, empty, or dominated outcomes.

Allow at most two additional in-range SC-CFG values per arm/sampler, chosen to refine a promising gap or bracket entropy near 5.15. Use the same screening seed and sample count. Do not run an unbounded adaptive search. Record the values and selection reasons before running them.

Before viewing confirmation results, select at most three configurations per arm/sampler for confirmation:

1. Always retain SC-CFG=3 as the reference setting.
2. Retain a high-entropy nondominated operating point.
3. Retain the best screening GenPPL point that reaches the A entropy neighborhood, or the closest nondominated point if the neighborhood is unreachable.

Deduplicate selections. A remaining slot may hold a distinct frontier point. Include repetition/length diagnostics in selection explanations; do not choose by GenPPL alone. Freeze the selected settings in a manifest. Do not use confirmation results for another search round.

## Stage 3: fresh-seed confirmation

For every selected configuration, generate 1,000 requested sequences for each fresh inference seed `{20, 21, 22}`. Keep confirmation samples disjoint from screening and original evaluations. With three selections in every arm/sampler group, this is at most 36,000 new C/D confirmation sequences.

Report per-seed results and mean +/- sample SD and SE across the three seed-level metrics. These error bars represent inference variation only; each model still has one training seed. Do not claim training-seed uncertainty. Use paired-seed differences when comparing confirmed settings, and describe their limitations.

Also attempt an inference-only reevaluation of the compatible released original ELF-B checkpoint at SC-CFG=3 under both sampler settings and the same confirmation seeds. Reuse the existing allocation rather than requesting another training job. Pin and verify its weights/configuration and use its original normalization, not C/D normalization. If the checkpoint is inaccessible or incompatible, document the exact limitation; do not train a replacement A. This local A anchor checks the evaluation stack, not a complete A frontier. Keep `A_published_external` separate from `A_measured_local`.

## Scoring and reporting

Reuse the pinned GPT-2 Large evaluator and the exact original GenPPL aggregation, text preprocessing, context cap, and mean per-sample unigram entropy. Inspect and document whether entropy is computed before or after the GPT-2 cap; do not silently change it. Do not replace mean sample entropy with corpus-global entropy or redefine GenPPL as ELF likelihood.

Retain all requested/generated counts, empty outputs, scoring failures, and pre/post-cap lengths. Empty text stays in the generated count and is excluded from PPL/entropy exactly as before. Score every eligible nonempty output. Save every generated text and its per-sample scores; never select or filter texts by quality before aggregation.

Add simple, explicitly defined diagnostics on GPT-2 tokens: repeated 4-gram fraction per sample (`1 - unique_4grams / total_4grams`, undefined for fewer than four tokens), adjacent-token repetition fraction, and duplicate-text rate across nonempty generations. Report original length and empty-rate diagnostics too. These are auxiliary checks, not new quality scores to optimize.

Produce separate GenPPL-versus-entropy plots for 32 and 64 steps. Label SC-CFG values, distinguish screening from confirmation and in-range from extrapolated settings, and mark published A as an external point. Show all measured points with enough clarity to see degeneracy. Plot empirical nondominated segments only as visual summaries; do not claim interpolation was sampled. Do not infer a full A curve from one point or assume identical guidance yields matched diversity.

Include fixed qualitative samples (seed 20, sample IDs 0,1,2) for each confirmed configuration, without selection. Conclusions must distinguish: lower PPL at lower entropy; improved diversity with persistent repetition; approximate matched-entropy improvements; and any uncertainty from unmatched evaluation or missing A frontier. Higher unigram entropy alone is not proof of coherent text.

## Deliverables and scope

Write a compact `GUIDANCE_SWEEP.md`, machine-readable per-seed and aggregate tables, the frozen selection manifest, plots, and artifact/checkpoint hashes. Put large texts and model outputs in a new guidance-sweep artifact directory, not Git. Keep original reports and artifacts immutable. Provide exact reproduction/resume commands and a concise compute summary with scheduler completion evidence.

Keep additions minimal and ordinary PyTorch, with concise comments/documentation and only essential progress logging. Add focused tests for active/inactive guidance, RNG replay, correct scale propagation through final decoding, aggregation, and counts. Do not add unrelated refactoring, dashboards, new training code, or Hugging Face uploads.

Completion means the requested inference study has run and its evidence is reported, not merely that scripts exist. Report inaccessible checkpoints, failed settings, and incomplete execution honestly without fabricated results or automatic training replacements.

## Inspected source references

- Experiment: https://github.com/yashizhang/ELF/tree/1d96853d8e9e78400ff1b3ac57500e058db30155
- Unconditional generation: https://github.com/yashizhang/ELF/blob/1d96853d8e9e78400ff1b3ac57500e058db30155/src/generation.py
- Guidance and SDE logic: https://github.com/yashizhang/ELF/blob/1d96853d8e9e78400ff1b3ac57500e058db30155/src/utils/sampling_utils.py
- Unconditional masks and final decoder: https://github.com/yashizhang/ELF/blob/1d96853d8e9e78400ff1b3ac57500e058db30155/src/utils/generation_utils.py
- Training scale range: https://arxiv.org/html/2605.10938v2