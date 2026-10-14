# C/D experiment results

Both fixed-budget pipelines completed. The results **do not establish an overall generation improvement**: lower GenPPL accompanies lower entropy, and the fixed qualitative samples contain substantial repetition and incoherence.

| Arm | Source | Configuration | GenPPL ± SE | Mean sample entropy ± SE |
| --- | --- | --- | --- | --- |
| A | published_external | primary | 24.0800 ± 0.1600 | 5.1500 ± 0.0020 |
| C | measured_this_project | primary | 13.3770 ± 0.3397 | 4.4051 ± 0.0102 |
| C | measured_this_project | secondary | 10.5249 ± 0.2017 | 4.3643 ± 0.0135 |
| D | measured_this_project | primary | 7.5054 ± 0.1580 | 3.2489 ± 0.0221 |
| D | measured_this_project | secondary | 7.6361 ± 0.2821 | 3.8359 ± 0.0301 |

Published A comes from [ELF paper v2, Appendix E.4, Table 9](https://arxiv.org/html/2605.10938v2#A5.T9), mean ± standard error over six inference seeds 0–5. It is an external reference, not a measurement from these jobs.

C/D each use one training seed (42), inference seeds 0,1,2, and 1,000 sequences per seed/configuration. Primary: 32 SDE steps, gamma 1.5. Secondary: 64 steps, gamma 1.0. Both use SC-CFG 3, CFG 1, logit-normal time, and latent length 1024. Error bars describe inference variation only; [results.json](results.json) also reports sample standard deviations, and [evaluation.csv](evaluation.csv) gives every seed's scores.

GenPPL uses the repository's frozen, pinned GPT-2 Large evaluator; it is not ELF held-out likelihood. Entropy is mean per-sample unigram entropy after GPT-2 retokenization, not corpus-global entropy.

Primary C−A: GenPPL -10.7030, entropy -0.7449; D−A: -16.5746, -1.9011; C−D: +5.8716, +1.1562.

B was intentionally omitted. C versus A includes continued pretraining, the C4 stream, implementation differences, and SIGReg; it cannot isolate SIGReg's causal effect. C versus D compares complete recipes. D has the lowest primary GenPPL and the lowest entropy. These observations support a quality/diversity trade-off, not matched-entropy frontier dominance. No guidance or checkpoint sweep was added.

Each arm processed **10,000,007,168 nominal adaptation positions**, 19,531,264 source examples, and 11,093,757,952 packed source positions in 38,147 updates. Each fresh ELF-B then completed **five OWT epochs / 95,090 updates**. Five epochs cover 48,685,920 source-row visits plus 160 documented padding repeats. [training.json](training.json) retains actual token/pass counts, corruption statistics, normalization, initialization hashes, measured throughput, and memory.

| Arm | Configuration | Requested / generated | Nonempty | PPL / entropy scored | Empty fraction |
| --- | --- | --- | --- | --- | --- |
| C | primary | 3000 / 3000 | 2997 | 2997 / 2997 | 0.1000% |
| C | secondary | 3000 / 3000 | 3000 | 3000 / 3000 | 0.0000% |
| D | primary | 3000 / 3000 | 3000 | 3000 / 3000 | 0.0000% |
| D | secondary | 3000 / 3000 | 3000 | 3000 / 3000 | 0.0000% |

C primary's three empty outputs are seed 1 IDs 589/820 and seed 2 ID 895. They remain in the generated counts and are excluded from scoring. Every nonempty output was scored. [evaluation.csv](evaluation.csv) retains per-seed counts and scores.

| Arm | Configuration | Mean GPT-2 tokens before / after cap | Mean T5 retokenized tokens |
| --- | --- | --- | --- |
| C | primary | 953.94 / 950.63 | 981.30 |
| C | secondary | 969.72 / 962.18 | 999.18 |
| D | primary | 927.00 / 922.97 | 976.63 |
| D | secondary | 963.52 / 956.64 | 1003.22 |

Lengths include all 3,000 texts per row, including empties. Decode/re-encode can change length; these are retokenized text lengths, not a different latent generation length. GPT-2 scoring caps context at 1024. [lengths.json](lengths.json) includes medians and ranges.

The [fixed qualitative set](qualitative_samples.json) contains seed 0, IDs 0–2, for all four measured rows, without quality selection. Illustrative primary ID 0 excerpts:

> **C:** Your personality has a emotional character. There are many positive aspects to your lives: 1. You don’t describe yourself as a real person, you treated as a real person, you are treated as a real person and as a woman. Live as a person. Your life has a very…

> **D:** You’ll need to create a lamp in your new lamp To create a file You can create a new lamp. You’ll need to create the file if you don’t like the lamp in your lighting. You might need to create You can create a file in your new lamp The lamp You might need to…

All original/C/D 500-update frozen-encoder readouts and the held-out representation diagnostics at 569/1024 positions completed; none were skipped. D's better token recovery did not establish better generation. [diagnostics.json](diagnostics.json) retains the probe and spectra; C's span CE and D's consistency are separate objectives.

Final scheduler allocation: **405.747 GPU hours**, including both full runs, both pilot attempts, and the one-GPU compile check. Shared CPU preparation contributes zero GPU hours. Every recorded allocation ended; queued time is excluded. [compute.json](compute.json) includes all attempts, and [stage_compute.csv](stage_compute.csv) subdivides those same allocations—do not add stage totals again.

Both original eight-H100 jobs succeeded with exit 0 and no replacement runs: [scheduler evidence](evidence/final-scheduler.json). Deep final [C](evidence/elf-C-epoch-5.json) / [D](evidence/elf-D-epoch-5.json) checkpoint audits and final [C](evidence/generation-C.json) / [D](evidence/generation-D.json) generation audits passed. The latter reconstruct entropy and aggregate perplexity from saved per-sample PPL; they do not rerun GPT-2 forward passes. Exposure follows saved steps and the verified sampler rather than a per-example runtime trace.

[ACCEPTANCE.md](ACCEPTANCE.md) maps every requirement to evidence and its limits. [EXPERIMENT.md](EXPERIMENT.md) documents reproduction and resume. Raw checkpoints, encoders, logs, all 12,000 generated texts, and per-seed metrics remain under `/home/toolkit/ELF/artifacts`; [artifact index](evidence/final-artifacts.json) records retained paths and hashes. Large artifacts are intentionally outside Git and were not uploaded to Hugging Face.
