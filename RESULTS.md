# C/D experiment results

Experiment status: incomplete — training/evaluation requirements remain.

| Arm | Source | Configuration | GenPPL | Mean sample entropy | Status |
| --- | --- | --- | --- | --- | --- |
| A | published_external | primary | 24.0800 | 5.1500 | published |
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

The matched 500-update token-readout diagnostics are not all complete yet; see `diagnostics.json`. Primary held-out representation diagnostics run at initialization, 1B, 3B, and 10B nominal positions.

Compute includes preparation, failed attempts, and pilots. Running-job totals are provisional. See `compute.csv`, `compute.json`, and per-arm stage manifests for measured allocation time; queued time is excluded.

Current arm state: C `not_submitted`; D `not_submitted`.
