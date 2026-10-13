# C/D experiment results

Experiment status: incomplete — training/evaluation requirements remain.

| Arm | Source | Configuration | GenPPL ± SE | Mean sample entropy ± SE | Status |
| --- | --- | --- | --- | --- | --- |
| A | published_external | primary | 24.0800 ± 0.1600 | 5.1500 ± 0.0020 | published |
| C | measured_this_project | primary | — | — | incomplete |
| C | measured_this_project | secondary | — | — | incomplete |
| D | measured_this_project | primary | — | — | incomplete |
| D | measured_this_project | secondary | — | — | incomplete |

Published A is 24.08 ± 0.16 GenPPL and 5.15 ± 0.002 entropy (mean ± SE over six inference seeds 0–5), from [ELF v2, Appendix E.4, Table 9](https://arxiv.org/html/2605.10938v2). It uses 32 SDE steps, gamma 1.5, and self-conditioning CFG 3. These are external published measurements.

C/D use one training seed each and inference seeds 0,1,2, with 1,000 samples per seed per configuration. Error estimates in the machine-readable table describe inference variation only. GenPPL is the repository's frozen GPT-2 Large generative perplexity, not ELF held-out likelihood; entropy is mean per-sample unigram entropy after GPT-2 retokenization.

B is intentionally omitted. C versus A includes continued pretraining, the C4 stream, SIGReg, and implementation differences; it cannot isolate SIGReg's causal effect. C versus D compares complete recipes. Lower perplexity with lower entropy is a quality/diversity trade-off, without a matched-entropy frontier claim.

C versus A: unavailable until required C/D measurements finish.
D versus A: unavailable until required C/D measurements finish.
C versus D: unavailable until required C/D measurements finish.

| Arm | Saved nominal positions | Saved T5 updates | Completed ELF epochs | Completed ELF updates |
| --- | --- | --- | --- | --- |
| C | 10,000,007,168 | 38,147 | — | — |
| D | 10,000,007,168 | 38,147 | — | — |

`training.json/csv` distinguish saved checkpoints from observed progress and retain actual source/target counts, encoder passes, corruption rate/span length, OWT normalization, initialization hashes, and full-epoch padding repeats.

The matched 500-update frozen-encoder readout diagnostic is complete for original/C/D; see `diagnostics.json`. Held-out representation summaries and covariance spectra appear in `diagnostics.json`; `diagnostics.csv` provides the scalar milestones at both 569 and 1024 positions. C's span CE and D's consistency are different objectives and are not comparable language-model losses.

Measured scheduler allocation so far: **251.533 GPU hours**, including preparation, failed attempts, and pilots. Running-job totals are provisional; queued time is excluded. `compute.csv/json` provide allocation totals and `stage_compute.csv` separates adaptation, ELF, calibration, and evaluation attempts. Stage totals are part of the allocation total and must not be added a second time.

Current arm state: C `{'adaptation': 'completed', 'elf': 'running', 'export': 'completed', 'owt_normalization': 'completed', 'readout': 'completed'}`; D `{'adaptation': 'completed', 'elf': 'running', 'export': 'completed', 'original_readout': 'completed', 'owt_normalization': 'completed', 'readout': 'completed'}`.
