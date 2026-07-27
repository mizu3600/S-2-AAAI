# Extraction Protocol Small-Scale A/B

- Model: `deepseek-v4-flash`
- Temperature: `0.0`
- Cases: `6`
- Design: old and new prompts use the same current grounding and ID validator.

| Metric | Legacy v1 | OpenIE n-ary v2 | Delta |
|---|---:|---:|---:|
| Entity coverage | 40.0% | 92.5% | +52.5% |
| Fact coverage | 12.5% | 75.0% | +62.5% |
| Retained entities | 16.00 | 36.00 | +20.00 |
| Retained facts | 9.00 | 14.00 | +5.00 |
| Entity retention | 100.0% | 97.3% | -2.7% |
| Fact retention | 60.0% | 93.3% | +33.3% |
| N-ary fact rate | 11.1% | 50.0% | +38.9% |
| Mean fact arity | 2.11 | 3.14 | +1.03 |
| Raw evidence grounding | 100.0% | 100.0% | +0.0% |
| Unknown raw arguments | 8.00 | 0.00 | -8.00 |
| Prompt tokens | 2484.00 | 12808.00 | +10324.00 |
| Completion tokens | 2960.00 | 4043.00 | +1083.00 |
| Prompt cache hit tokens | 2048.00 | 6400.00 | +4352.00 |
| Latency | 22091 ms | 27554 ms | +5464 ms |

## Per Case

| Case | Legacy entity | V2 entity | Legacy fact | V2 fact |
|---|---:|---:|---:|---:|
| encyclopedia | 2/8 | 6/8 | 0/6 | 4/6 |
| technical_evaluation | 3/8 | 8/8 | 0/2 | 2/2 |
| clinical_trial | 4/7 | 7/7 | 0/1 | 0/1 |
| acquisition | 5/7 | 7/7 | 2/3 | 3/3 |
| no_proper_noun | 0/4 | 4/4 | 0/2 | 2/2 |
| correlation | 2/6 | 5/6 | 0/2 | 1/2 |

## Scope

This is a six-case directional smoke test, not a statistically powered benchmark.
Coverage uses hand-authored expected entity surfaces and relation argument sets; it does
not use an LLM judge. Inspect `comparison.json` for every raw and retained extraction.
