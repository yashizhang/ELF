# GOAL: Adapt T5-small with SIGReg and retrain ELF-B — Arms C and D

## 1. Objective and agreed scope

Implement and run two complete experiment pipelines in `https://github.com/yashizhang/ELF` to test whether adapting the pretrained T5-small representation improves ELF-B language generation:

- **Arm C:** continue pretraining the original T5-small encoder–decoder on C4 with its span-corruption cross-entropy loss plus SIGReg on clean encoder token representations; freeze the adapted encoder; train a fresh ELF-B on the released OpenWebText token dataset; evaluate generation.
- **Arm D:** adapt the same pretrained T5-small encoder on the same C4 examples with token-aligned consistency between clean and span-corrupted views plus SIGReg; freeze the adapted encoder; train a separate fresh ELF-B on the same OpenWebText data; run the same evaluation.

The user has selected **10B nominal encoder input positions per adaptation arm**, as defined in Section 4, and **two independent scheduler jobs requesting eight H100 GPUs each**. Submit both pipelines independently. They may overlap when the scheduler allocates 16 H100s and otherwise queue. Do not reduce either job to four GPUs to manufacture concurrency.

Use **SIGReg coefficient 0.05**, fixed throughout each adaptation run, with the exact additive loss definitions below. All T5 adaptation starts from pretrained weights. Each subsequent ELF-B starts from a fresh initialization.

**Arm A is the published original ELF-B result, used as an external reference. Arm B is intentionally omitted to conserve compute.** The working assumption that ordinary continued pretraining would resemble A remains untested. Do not fabricate a B row, copy A into B, or claim that a C-versus-A difference isolates the causal effect of SIGReg. C-versus-D compares the complete adaptation recipes.

Implement both pipelines, run the necessary correctness checks, execute the authorized training and evaluation, and produce a factual results report. A negative result is a completed experiment. Finishing launch scripts or submitting jobs alone is not completion. If the executing environment lacks cluster access or resources, finish the runnable implementation and state the exact blocked stage; never invent job IDs, training progress, or results.

Out of scope: training T5 from random initialization, training A or B, additional regularization coefficients or learning-rate sweeps, EMA-teacher masked-latent prediction, extra learned projection heads, supervised-task mixtures, conditional-generation datasets, and retraining ELF for multiple intermediate T5 checkpoints.

## 2. Repository starting point and T5 source integration

Repository inspection on 2026-10-12 found:

| Component | Inspected source |
| --- | --- |
| User fork | `yashizhang/ELF`, only `main`, commit `5098bf28b5e9b52c329970a7e4e1cc28251c76e6` |
| Existing fork implementation | JAX/TPU |
| Official GPU implementation | `lillian039/ELF`, branch `pytorch_elf`, commit `b29d8833609e9ab7f67cd9da39435ac5cea04837` |
| Compatible T5 implementation | Hugging Face Transformers tag `v4.44.2` |
| SIGReg implementation | `galilai-group/lejepa`, commit `c293d291ca87cd4fddee9d3fffe4e914c7272052` |

Recheck the working tree and applicable repository instructions before editing. Preserve unrelated user changes. Create an experiment branch named `t5-sigreg-cd` in the user's fork, based on the pinned official PyTorch implementation. Fetch/import that upstream commit if necessary; **do not assume `origin/pytorch_elf` exists**. Preserve the fork's existing `main`; do not create a new PyTorch port of ELF.

Provide modifiable T5 source through a pinned checkout or Git submodule at `third_party/transformers`, initially using `v4.44.2`, and install it editable. Resolve and record the tag's exact commit. The relevant file is `src/transformers/models/t5/modeling_t5.py` within that checkout. This satisfies the need for T5 source inside the project while using its maintained PyTorch implementation. The original Google T5 repository is the objective/preprocessing reference; its TensorFlow/Mesh training stack is not required for these GPU pipelines.

Similarly pin SIGReg source at `third_party/lejepa`. Preserve upstream licenses and attribution. Put new data preparation, losses, training, and experiment configuration in project-owned files. Avoid changing T5 blocks when the new objective can be implemented around ordinary model outputs.

The ELF PyTorch requirements specify `transformers>=4.41.2,<4.45.0`, `torch>=2.3.0`, and `numpy>=1.26.4,<2.0.0`. Lock a tested Python/PyTorch/CUDA/dependency environment compatible with these constraints. Record exact package versions and source commits; a version range alone is not the final environment specification.

## 3. Run matrix, checkpoint selection, and compute boundary

| Item | Arm C | Arm D |
| --- | --- | --- |
| Initial model | Original `google-t5/t5-small` encoder–decoder | Encoder from the identical checkpoint |
| Adaptation data | Same pinned C4 English stream | Same examples, order, and corruption seeds |
| Adaptation budget | 10B nominal positions | 10B nominal positions |
| SIGReg coefficient | 0.05 | 0.05 |
| GPU request | 8 H100s | 8 H100s |
| Adaptation training seed | 42 | 42 |
| Encoder selected for ELF | Final fixed-budget checkpoint | Final fixed-budget checkpoint |
| ELF training | Fresh ELF-B, five OWT epochs | Fresh ELF-B, five OWT epochs |
| ELF training seed | 42 | 42 |

Use one training seed per arm. Intermediate adaptation checkpoints near 1B and 3B positions are for diagnostics and recovery; the final checkpoint near 10B is the preregistered input to ELF. Do not select a checkpoint by final generation results or launch an unrequested checkpoint sweep.

An early numerical check is part of each run, not an additional independently trained arm. Continue healthy runs to the fixed budget even when an auxiliary metric is unpromising. Halt an affected stage for non-finite loss/gradients, corrupt data, an incorrect objective, or an invalid checkpoint, preserve evidence, and fix implementation errors. Do not silently change the coefficient, objective, training budget, or model variant to obtain a better result.

## 4. Data, span corruption, and token accounting

### 4.1 T5 adaptation data and initialization

Use `allenai/c4`, English configuration `en`, training split. Reserve the official validation split for evaluation. Resolve and record dataset revision, source shard list, preprocessing version, tokenizer revision, and deterministic data order. This is C4 continuation; it is not a bit-for-bit recreation of every task and dataset version used in the original T5 checkpoint.

Load the **original** `google-t5/t5-small` weights and tokenizer at pinned revisions. Keep vocabulary, token IDs, architecture, and final hidden width 512 unchanged. Do not substitute T5 v1.1, FLAN-T5, or an initialization constructed solely from a configuration.

Prepare a shared immutable tokenized C4 dataset for the required prefix, using streaming/shards where useful. Download only what the run needs. Both jobs reuse the same prepared data and calibration artifacts. Use a preparation lock or explicit preparation dependency to prevent concurrent writers. Preserve example IDs and packing boundaries in the manifest; never let each GPU independently stream the same examples.

Use deterministic token packing with documented EOS/document boundaries. Keep special tokens out of corruption and regularizer sampling. Derive corruption and token-sampling RNGs from separate, recorded seeds and stable example IDs so Arm C's decoder/dropout RNG consumption cannot change Arm D's masks or data order. The canonical fixed-length case below assumes an ordinary 568-token source segment; explicitly account for protected EOS boundaries when packing.

### 4.2 Canonical T5 corruption

Implement the original T5 random-span corruption with noise density `0.15` and mean noise-span length `3.0`. Each removed span is replaced by its unique sentinel. Decoder targets contain the removed spans with the corresponding sentinels and the prescribed EOS handling. Use the official preprocessing algorithm/reference collator, including its rounding and sentinel-order conventions.

For a canonical boundary-free source segment of 568 ordinary tokens, corruption produces 512 encoder input tokens and 114 decoder target tokens, including EOS. The auxiliary clean encoder input contains the source plus EOS, hence 569 positions. Validate this case against the pinned reference implementation. Protecting internal EOS/document boundaries is an explicitly documented extension of the reference collator: construct spans within eligible regions, permit the resulting input/target lengths to vary, and use padding and attention/validity masks. Never truncate tokens, sentinels, or targets merely to force the canonical 512/114 lengths. Record the actual corruption rate and span lengths for boundary-containing examples.

The collator must retain the original token-position map. For every surviving ordinary token, store the correspondence from its position in the clean source to its position in the compressed corrupted sequence. Repeated token identities must remain distinguishable by occurrence position. Sentinels do not stand in one-to-one correspondence with the tokens they replace.

For Arm C, decoder labels on padding are `-100`, and CE is the global mean over valid target tokens. If per-rank target counts differ, an unweighted average of Hugging Face's rank-local mean losses is incorrect. Aggregate valid-token counts and use the appropriate local loss scaling for DDP's gradient average (for example, `world_size * local_CE_sum / global_valid_target_count`), while logging the global sum/count. Verify this scaling in the distributed gradient check. For Arm D, no decoder targets enter the training objective.

### 4.3 Exact budget convention

The agreed 10B budget retains the earlier planning convention: **512 nominal corrupted-encoder input positions per source example, counted once**. Do not double-count the clean view or count decoder targets as extra distinct training data. Use this same counter for D even though D has no vocabulary decoder.

With global batch 512 and accumulation 1:

```text
nominal_positions_per_update = 512 examples * 512 positions = 262,144
updates = ceil(10,000,000,000 / 262,144) = 38,147
actual_nominal_positions = 10,000,007,168
source_examples = 19,531,264
canonical_raw_source_tokens = source_examples * 568 = 11,093,757,952
```

The final full-batch overshoot of 7,168 nominal positions is intentional. Log actual non-padding source tokens, source examples, nominal positions, decoder target tokens, encoder passes, optimizer updates, and GPU hours separately. If boundary handling changes actual lengths, preserve the nominal stopping counter and report the real counts. This is approximately 11.09B raw source tokens in the canonical case, **not 10B raw tokens**.

### 4.4 ELF data

Use the author-released `embedded-language-flows/openwebtext-t5` token dataset at a pinned revision. Keep the original tokenizer and existing 1024-position ELF training preprocessing. Reuse its token IDs; changed encoder weights do not require retokenization. Any latent cache must be tagged with the encoder hash and normalization, although the initial implementation should retain ELF's online encoding.

## 5. Shared representation and fixed SIGReg normalization

Let `h = E_theta(input_ids, attention_mask).last_hidden_state` denote the **final 512-dimensional token representation** that would feed ELF, including T5's final RMS normalization.

Before adaptation, run the original frozen encoder with dropout disabled on a deterministic calibration set of 4,096 C4 training source sequences. Compute a single scalar population mean `mu0` and standard deviation `s0` over all coordinates of valid ordinary clean token representations, using streaming accumulators. Exclude padding, EOS, sentinels, and other special tokens. Record sequence IDs, counts, estimator, and checkpoint hash.

For the adaptation losses define:

```text
z = (h - mu0) / s0
```

Hold `mu0` and `s0` fixed and share them across C and D. They are not trainable, not recomputed per minibatch, and not updated during training. This is a scalar coordinate convention for the complete token vector, not a whitening transform or an added representation head. It avoids making the initial regularizer primarily compensate for the original encoder's overall output scale. Apply the same convention to D's consistency loss.

Keep T5's architecture and final RMS normalization. Exact unrestricted Gaussianity is not an attainable acceptance requirement for these normalized vectors; treat SIGReg as a soft regularizer. Log both raw `h` statistics and normalized `z` statistics. Low SIGReg alone is not evidence that ELF's joint sequence distribution is easier to learn.

This shared C4 normalization is only for **adaptation**. Section 9 separately defines per-encoder OWT normalization for the subsequent ELF runs. Export ordinary raw T5 encoder outputs; do not accidentally normalize twice.

## 6. Exact objectives

### 6.1 Arm C: span CE plus clean-token SIGReg

```text
L_C = mean_span_target_cross_entropy + 0.05 * R(Z_clean)
```

`Z_clean` is the fixed-size sample of normalized final clean token vectors defined in Section 7. The span loss is produced by `T5ForConditionalGeneration` on the corrupted input and target spans.

Train all encoder and decoder parameters. Keep the ordinary checkpoint dropout behavior for the corrupted encoder–decoder span-loss pass. Add a **dropout-disabled clean encoder pass with gradients enabled** for SIGReg. The clean pass must not use `no_grad`, `inference_mode`, or a detached output. CE gradients and SIGReg gradients both reach the encoder.

For this small model, initially disable activation checkpointing. Temporarily switching to eval mode for the clean pass and restoring train mode before backward is unsafe if checkpoint recomputation uses the restored dropout mode; if checkpointing becomes necessary, explicitly preserve and verify the recomputation mode.

### 6.2 Arm D: token-aligned consistency plus per-view SIGReg

Use the same trainable `T5EncoderModel` for one clean view and one independently generated span-corrupted view of each source example. Both encoder passes have dropout disabled and gradients enabled. The span corruption supplies the view variation.

Let `V` be all ordinary source token positions surviving corruption, `pi(i)` their locations in the corrupted sequence, and `d = 512`:

```text
L_cons = sum_{examples, i in V, k in 1..d}
           (z_clean[i,k] - z_corrupt[pi(i),k])^2
         / (number_of_matched_tokens * d)

R_D = 0.5 * (R(Z_clean) + R(Z_corrupt))
L_D = L_cons + 0.05 * R_D
```

Average consistency across the global batch using valid-token counts; rank-local means are sufficient only when the counts are equal. Use all valid matched tokens for consistency. The smaller fixed sample is only for SIGReg.

Train every encoder parameter, including its shared input embedding and final RMS scale. Load the encoder directly; freezing the decoder of a full tied T5 can accidentally freeze the encoder's shared embedding. There is no vocabulary decoder, EMA teacher, stop-gradient branch, predictor, sentence pooling, or learned projection head in D.

Keep the two view distributions separate when evaluating SIGReg, then average their statistics. Do not flatten both views into a sample set twice as large. The clean branch is trainable and therefore is not a fixed information-preserving teacher.

### 6.3 Meaning of lambda

In both arms, `sigreg_lambda = 0.05` means the explicit **additive** coefficient above. Do not silently replace this with a convex mixture such as `0.95 * task_loss + 0.05 * R` or use LeJEPA's center-MSE normalization in place of the defined pairwise MSE. For two views, center MSE and pairwise MSE differ by a factor of four. D is a token-level LeJEPA-inspired adaptation, not a literal reproduction of the image recipe.

Keep lambda fixed from the first update; no automatic balancing, ramp, sweep, or fallback lambda. Log unweighted component losses, weighted contributions, and occasional encoder gradient norms from the task and SIGReg terms. A coefficient of 0.05 does not mean that SIGReg contributes 5% of the gradient.

## 7. SIGReg estimator and distributed semantics

Use the pinned upstream `SlicingUnivariateTest` with `EppsPulley` and the following fixed definition:

| Setting | Value |
| --- | --- |
| Representation width | 512 |
| Samples per GPU per view per loss evaluation | 256 |
| Global samples per view on eight GPUs | 2,048 |
| Random unit projection directions | 256 |
| Epps–Pulley quadrature | `t_max=3.0`, `n_points=17`, `integration="trapezoid"` |
| Reduction over projections | Mean |
| Statistic normalization | Upstream sample-scaled statistic, including global sample-count factor |
| Arithmetic | FP32, with autocast explicitly disabled for SIGReg |

At the default 64 source examples per GPU, sample four distinct eligible token positions per example, for 256 local samples. Sample clean and corrupted states separately for D, with the same per-view count. Sample across source sequences; exclude all padding and special tokens. If packing changes eligible counts, handle exceptional examples deterministically while keeping the exact per-rank sample count.

The upstream Epps–Pulley implementation multiplies its characteristic-function discrepancy by `N_local * world_size`. Consequently the sample count is part of the meaning of lambda. Do not change it silently, divide the statistic by N, or substitute a locally averaged statistic.

Initialize the process group before constructing state that captures `world_size`. All ranks use identical random projection directions for each loss evaluation. Aggregate characteristic-function sufficient statistics with correct autograd-aware distributed reductions before applying the nonlinear discrepancy. Averaging independent rank-local SIGReg losses is not equivalent.

A view axis such as `[2, N_local, 512]` is acceptable for D if reductions preserve separate view statistics before averaging. Restore SIGReg counters/RNGs on resume. For repeated gradient-accumulation microsteps, document that averaging microstep statistics is different from one statistic over their concatenated samples; the initial run uses accumulation 1 to avoid changing this definition.

Put all forward passes and losses of an arm inside a single project-owned module wrapped by DDP. Avoid bypassing DDP via `.module.encoder` for the auxiliary pass. Verify global loss and parameter-gradient scaling against a single-process reference before long training.

## 8. Initial adaptation training configuration

These are proposed fixed starting settings for this experiment, not a claimed published optimum:

| Setting | Both arms unless stated otherwise |
| --- | --- |
| Optimizer | AdamW |
| Peak learning rate | `1e-4` |
| Betas / epsilon | `(0.9, 0.999)` / `1e-8` |
| Weight decay | `0.01`; exclude bias and normalization parameters |
| Gradient clipping | Global parameter-gradient norm `1.0` |
| Global source-example batch | 512 |
| Local source-example batch | 64 on each of eight ranks |
| Gradient accumulation | 1 |
| Warmup | 500 optimizer updates |
| Schedule afterward | Cosine decay to `1e-5` at the final update |
| Precision | BF16 model-forward autocast; FP32 model/master parameters, optimizer states, and SIGReg |
| Seed | 42, with separate recorded data/corruption/sampling RNG streams |
| Activation checkpointing | Off initially |
| T5 encoder EMA | None |
| Checkpoints | Latest resumable state, approximately 1B, 3B, and final 10B nominal positions |
| Validation | Fixed held-out examples at initialization, saved diagnostic milestones, and final |

Measure complete-update throughput and memory after warmup on the actual eight-H100 allocation. Report the measured ETA for remaining adaptation and later ELF stages; do not present earlier planning ranges as measured results. Include data loading, clean passes, SIGReg, communication, and checkpoint/evaluation overhead in accounting.

If the initial microbatch does not fit, resolve the implementation or choose a documented memory configuration during the pilot, before committing both full runs. Preserve the agreed effective batch, data budget, lambda, sample-count definition, and global reductions; account explicitly for any accumulation semantics. Do not silently change the experiment mid-run.

Use separate output roots, manifests, logger identities, checkpoint names, and RNG states for C and D. Save model, optimizer, LR scheduler, data cursor/shard order, RNGs, SIGReg state, calibration constants, counters, and resolved config. Resume only a checkpoint whose arm, source hashes, objective, and manifest agree. An interrupted run must not repeat a large data prefix unnoticed.

Suggested project-owned additions are `src/t5_adaptation/`, arm YAMLs under `src/configs/t5_adaptation/`, a data preparation entry point, a calibration/export entry point, and scheduler launch scripts. Names may follow the existing repository style, but the delivered commands must be real and documented.

## 9. Export adapted encoders and train fresh ELF-B models

For each arm, export the final encoder and unchanged tokenizer to a self-contained Hugging Face directory. Record an immutable content hash and source adaptation checkpoint. For C, retain its full encoder–decoder training checkpoint separately, while exporting an encoder that loads through `T5EncoderModel.from_pretrained`. For D, export its encoder directly.

Test a reload with the actual ELF encoder wrapper: a fixed clean input must give matching final token outputs, allowing only the documented precision tolerance. Confirm that the output has one 512-dimensional vector per input position and that adapted weights differ from the original checkpoint.

For each final encoder, recompute scalar `latent_mean` and `latent_std` on the **same fixed 4,096 OWT training sequences** at ELF length 1024, with dropout disabled, BF16 forward behavior matching ELF, and stable accumulators. Use non-padding positions participating in the ELF objective, including EOS if the existing validity mask trains it. Record the mask convention, IDs, counts, mean, and population standard deviation. Keep the final exported encoder raw; ELF applies this normalization exactly once.

Use the pinned `src/configs/training_configs/train_owt_ELF-B.yml` as the training template:

| ELF setting | Required behavior |
| --- | --- |
| Model | Original ELF-B, approximately 105M parameters; unchanged architecture |
| Encoder | Final frozen C or D encoder, eval mode |
| Dataset / length | Released OWT tokens / 1024 |
| Training duration | Five full dataset epochs |
| Effective global batch | 512 |
| Optimizer | Muon |
| Base LR / actual LR | `blr=0.001`, yielding `lr=0.002` at effective batch 512 |
| Warmup / subsequent schedule | 0.5 epoch, then constant, matching the pinned configuration |
| Weight decay | 0 |
| ELF EMA decay | 0.9999 |
| Decoder CE probability | 0.2 |
| Self-conditioning probability | 0.5 |
| Denoiser / decoder noise scales | 2.0 / 5.0 |
| `t_eps` | 0.05 |
| Seed | 42 |
| Automatic intermediate sampling | `eval_freq: 0`; run the final evaluation stage explicitly |

Preserve all other architecture, denoising, conditioning, optimizer, and data settings from the pinned template unless a documented implementation correction is necessary. The explicit `eval_freq: 0` override avoids the template's otherwise automatic sampling after each epoch; retain recovery checkpoints. Derive exact updates from the actual dataset and effective batch; the paper's approximately 95K updates is not an exact hardcoded target. Changing gradient accumulation must retain both five-epoch data exposure and effective batch 512.

**Integration details that must be checked:**

- Set `encoder_model_name` to the arm's immutable local encoder export. In the inspected PyTorch train/eval path, `encoder_checkpoint` is only logged and is not sufficient to load adapted weights.
- Set `tokenizer_name` explicitly to the unchanged pinned tokenizer export.
- Freeze every encoder parameter for ELF training. Verify no encoder gradients or optimizer updates occur.
- In the existing ELF configuration, `global_batch_size` is the global microbatch; effective batch is `global_batch_size * grad_accum_steps`. If accumulation changes, adjust the microbatch and epoch/update accounting accordingly.
- Use a fresh arm-specific ELF output directory. The existing trainer can auto-resume checkpoints found in `output_dir` even with `resume: null`; verify provenance before allowing this behavior.
- Initialize both ELF models from the same seed and save initial weight hashes. Never initialize either from the published trained ELF checkpoint or from the other arm's trained ELF.
- Keep automatic model uploads disabled (`hf_repo_id: null`). Store outputs locally on the designated experiment storage.

Keep recovery checkpoints during training and retain the final EMA/model states. Full sampling evaluation is required at the end; routine checkpointing must not create an unrequested expensive checkpoint-selection sweep.

## 10. Scheduler and execution behavior

Discover the cluster's existing scheduler conventions, account, H100 resource spelling, partition/QoS, time limits, environment activation, and storage paths from the executing environment. Put these in a separate cluster configuration; do not invent working account or partition names.

Provide a single submission entry point that submits **two independent eight-H100 pipelines** and records their job IDs, configurations, and status files. Prefer one eight-GPU node per active training stage where available. The physical device check must show eight H100s for each full training allocation.

Within each arm, execute in dependency order:

1. Adapt T5 to the fixed budget, with resumable checkpoints.
2. Export its final encoder and compute the arm-specific OWT normalization.
3. Train its fresh ELF-B for five OWT epochs.
4. Evaluate the final ELF checkpoint and write the arm report.

Shared data preparation and original-T5 calibration run once before the dependent jobs. C and D do not wait for one another. If cluster walltime requires separate stage jobs, use explicit success dependencies and resumable continuations. Never run ELF for an arm before its own final encoder and normalization are complete and verified. At most two eight-GPU training allocations are active for this experiment at a time.

Use the scheduler allocation's visible devices and independent rendezvous settings; do not hardcode both jobs onto the same GPU IDs or port. The two jobs can run simultaneously only with 16 concurrently allocated H100s. Queueing is expected when fewer are available and is not a reason to change the model or batch semantics.

Avoid duplicate submissions on restart. A submission manifest must distinguish queued, running, failed, preempted, and completed stages. Monitor both pipelines to completion or a concrete external blocker, and report actual status. Honor walltime/preemption signals with safe checkpoint writes. Record GPU model, memory, topology, measured throughput, walltime, and GPU hours per stage.

## 11. Required checks and inexpensive diagnostics

Keep verification focused on risks that could invalidate the experiment:

1. **Corruption and correspondence:** compare sentinel order, input/target reconstruction, canonical lengths, protected special tokens, repeated-token handling, and position maps with the pinned T5 reference. On a toy example, every consistency pair must be the same original token occurrence.
2. **Gradient routing:** show that C's SIGReg alone produces encoder gradients, C's CE trains encoder and decoder, and D trains all intended encoder parameters. Show that clean passes are deterministic with gradients enabled. Verify SIGReg arithmetic is FP32.
3. **Distributed objective:** with identical synthetic inputs and projections, match an equivalent single-process loss and parameter gradients to the distributed implementation, including world-size and valid-token normalization. A small multi-process test is sufficient; one short eight-GPU update smoke check verifies the target launch path.
4. **Resume and export:** verify continuation restores counters/RNG/data position without arm mixing, and exported encoders reproduce their source checkpoint outputs in ELF.
5. **ELF initialization and freezing:** verify a fresh ELF model, the correct adapted encoder, correct latent normalization, and no encoder updates. Generate a small smoke sample batch before committing a full ELF run.

At fixed milestones, measure clean-view SIGReg using the same sample count, raw/normalized feature mean and variance, a covariance spectrum/effective-rank summary, and valid/finite output statistics on a fixed held-out C4 set. Evaluate clean representations at both adaptation length and ELF length 1024. Log C's held-out span CE and D's held-out consistency separately; they are different objectives and must not be plotted as comparable language-model losses.

Include one small, matched token-readout diagnostic for frozen original, C, and D encoders if feasible within the stage's diagnostic allowance: a fresh linear 512-to-vocabulary readout, fixed training/validation examples, identical initialization and optimizer, and a fixed short budget capped at 500 updates each. Test token CE/accuracy on clean latents and on latents with shared scalar normalization plus Gaussian noise of standard deviation 0.1. Reuse calibration data where appropriate and label this as a practical decodability probe, not an additional encoder-training arm. If omitted for budget or runtime reasons, report that explicitly and proceed to the primary ELF experiment.

Do not reject D merely because the original frozen T5 decoder no longer works: a coordinate change can impair that decoder without erasing information. Conversely, low consistency/SIGReg or good clean-token recovery is not sufficient to claim improved generation. Diagnostic quality is not an automatic early-stopping gate.

## 12. Final generation evaluation and comparison with Arm A

Use ELF's existing EMA generation path and metric implementation. Use identical C/D inference settings and inference seeds `0,1,2`, with 1,000 generated sequences per seed per configuration:

| Configuration | SDE steps | SDE gamma | Self-conditioning CFG | CFG | Time schedule |
| --- | --- | --- | --- | --- | --- |
| Primary | 32 | 1.5 | 3.0 | 1.0 | logit-normal |
| Secondary | 64 | 1.0 | 3.0 | 1.0 | logit-normal |

Match all remaining default sampling settings and sequence length. Pin the evaluator model/tokenizer revision. Report the repository's frozen GPT-2 Large generative perplexity and mean per-sample unigram entropy after its GPT-2 retokenization. This entropy is not corpus-global entropy; generative perplexity is not ELF's held-out likelihood.

Save all generated JSONL samples and per-seed metrics. Report the mean and variation across the three inference seeds, while explicitly noting there is only one training seed per arm. Include empty-output fraction, requested/generated/scored counts, lengths, and a small qualitative sample set. The existing evaluator drops empty strings before calculating metrics; make those exclusions visible rather than silently improving the reported scores.

For the primary 32-step configuration, the paper's Arm A reference is approximately **GenPPL 24.1 and entropy 5.15**. Preserve the exact source/table/figure, reported uncertainty, and inference protocol when entering the final A row. Paper v2, Appendix E.4, Table 9 reports GenPPL 24.08 +/- 0.16 and entropy 5.15 +/- 0.002, as mean +/- standard error over six evaluation seeds 0–5; those are published results, not measurements from these jobs. Verify the relevant paper version and configuration before filling any additional A rows.

Use a results table separating `published_external` A from `measured_this_project` C/D. A lower GenPPL accompanied by lower entropy is a quality/diversity trade-off, not automatically an overall improvement. Without an additional matched-entropy sweep, report that trade-off rather than claiming frontier dominance. Do not add a guidance sweep automatically.

There is no A/B training or required baseline rerun in this scope. An inference-only check with the official released ELF checkpoint can be useful if an implementation mismatch emerges, but it is optional and must be reported separately from the paper row.

## 13. Deliverables and completion criteria

Deliver the experiment branch with modifiable pinned T5/SIGReg source, focused project-owned training code, exact environment setup, arm configurations, data/calibration manifests, scheduler submission/resume commands, encoder export/integration, and final evaluation commands. Document which files are new, which existing ELF files changed, and why.

For each arm, retain:

- Resolved configs, code/dependency/data/model revisions, and job/stage manifests.
- Adaptation logs, fixed normalization, selected checkpoint, raw encoder export, tokenizer, and content hashes.
- OWT encoder normalization, initial ELF hash, final ELF/EMA checkpoint, training logs, and compute accounting.
- Generated samples, per-seed metrics, aggregate results, diagnostic summaries, and limitations.

Write a concise `RESULTS.md` and machine-readable results/compute tables. Summarize C versus published A, D versus published A, and C versus D. State that B was omitted and that any C-versus-A change includes the effect of continued pretraining, the chosen data stream, and implementation differences as well as SIGReg. Report negative results directly.

The task is complete when:

- [ ] Both objectives, the fixed lambda definition, token alignment, and distributed gradients have passed the focused checks.
- [ ] Both arms started from the same pinned pretrained T5-small and reached the agreed nominal data budget, with actual counts recorded.
- [ ] Each final encoder was exported, reloaded, and integrated through the field ELF actually uses.
- [ ] Each frozen adapted encoder has a separately trained fresh ELF-B that completed five OWT epochs.
- [ ] Both final ELF models were evaluated with the identical declared protocol and saved raw samples.
- [ ] Results distinguish external A from measured C/D and include compute, uncertainty limitations, and skipped diagnostics.
- [ ] Reproduction and resume commands work from the documented environment and manifests.

If a required item is externally blocked, leave it incomplete and identify the blocker and the last verified artifact. Completion does not require either arm to outperform A.

## 14. Primary references

- User fork: https://github.com/yashizhang/ELF
- Official PyTorch ELF base: https://github.com/lillian039/ELF/tree/b29d8833609e9ab7f67cd9da39435ac5cea04837
- ELF training configuration: https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/configs/training_configs/train_owt_ELF-B.yml
- ELF encoder loading: https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/train.py
- ELF normalization: https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/utils/encoder_utils.py
- ELF paper: https://arxiv.org/html/2605.10938v2
- PyTorch T5 source: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/t5/modeling_t5.py
- Reference T5 corruption collator: https://github.com/huggingface/transformers/blob/v4.44.2/examples/flax/language-modeling/run_t5_mlm_flax.py
- Original T5 objective and preprocessing: https://github.com/google-research/text-to-text-transfer-transformer
- T5 checkpoint: https://huggingface.co/google-t5/t5-small
- C4 English: https://huggingface.co/datasets/allenai/c4
- ELF OWT tokens: https://huggingface.co/datasets/embedded-language-flows/openwebtext-t5
- Pinned SIGReg source: https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052
- LeJEPA paper: https://arxiv.org/html/2511.08544v3
