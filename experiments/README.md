# Experiments

Code for the Persona Memory Benchmark and every analysis in the paper. `run_all.sh` runs the whole pipeline. Each step is skipped when its output exists, and every LLM call is cached in `data/llm_cache.jsonl`, so an interrupted run resumes where it stopped.

## Layout

| Path | Purpose |
|---|---|
| `gen_scenarios.py` | Seeded benchmark generator: 12 personas over 11 attributes, with facts, distractors, updates and probes |
| `harness/intercept.js` | `fetch` wrapper: enforces temperature 0 and a fixed seed, logs token counts and latencies, caches responses |
| `harness/run_pmb.js` | `transcripts` → `ingest <cond>` → `probe <cond>` for every system (see below) |
| `harness/run_latency.js` | Live, uncached user-perceived latency (`v2`, `v2 5000` = 5 s user pause, `naive_full`) |
| `harness/bench_scan.js` | Retrieval scan cost versus store size |
| `baselines/run_mem0.py` | Mem0 baseline (`mem0ai` 2.2.1, default configuration) |
| `baselines/export_mem0.py` | Exports the complete Mem0 stores for the store statistics |
| `analysis/score.py` | Alias-regex scoring, Wilson intervals, exact McNemar tests, persona-cluster bootstrap |
| `analysis/judge_validate.py` | Agreement of the scorer with an independent Qwen3-8B judge (Cohen's κ) |
| `analysis/store_stats.py` | Store quality: coverage, contradictions, duplicates, misattribution, noise |
| `analysis/shapley.py` | Exact 2³ Shapley attribution with Harsanyi dividends, retrieval–generation decomposition, logistic model |
| `analysis/embed_geometry.py`, `geometry_extra.py` | Embedding-geometry validation of the merge-rule propositions |
| `analysis/make_paper_assets.py`, `make_figs2.py` | Write tables (`.tex`) and figures (`.pdf`) to `paper_assets/` |

## Conditions (`harness/run_pmb.js`)

| Condition | System |
|---|---|
| `v2` | Memory Engine (from `../source_code`) |
| `nomem`, `naive_trunc`, `naive_full` | No memory; history replay truncated to 4,096 tokens; full replay |
| `rag_turns` | Dense retrieval of the top-5 raw turns with the same embedder |
| `rsum` | Recursive summarisation (running summary rewritten every 4 turns) |
| `mem0` | Mem0; the write path runs in `baselines/run_mem0.py` |
| `v2_threshold`, `v2_v1retr`, `v2_v1prompt` | Ablations −C (cosine-threshold merge), −R (fixed top-3 retrieval), −P (generic verbose extraction prompt) |
| `fac_PCR` | Remaining configurations of the 2³ factorial (P, C, R ∈ {0, 1}) |
| `<cond>@8b` | Same system with Qwen3-8B as chat and analyzer model |

The −P prompt is the analyzer prompt of the original `memory-engine-node` 1.0.3 release. `run_all.sh` downloads that release from npm into `engine_v1/`. All systems answer with the same chat model, system prompt and prompt template, so only the memory differs.

## Notes

- **Mem0's context window.** Mem0's extraction prompt is about 8k tokens. Its Ollama client therefore runs with a 16k context window, a deployment setting that leaves its prompts and logic untouched. Run Mem0 on its own: mixing context sizes on one Ollama server forces model reloads.
- **Concurrency.** Set it with `PMB_CONC` (harness) and `CONC` (Mem0). The default of 2 matches `OLLAMA_NUM_PARALLEL=2`.
