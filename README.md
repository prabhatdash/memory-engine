# Memory Engine

**A versioned, analyzer-driven memory layer for LLM chatbots.**

Memory Engine (ME) gives an LLM application a memory that survives across sessions. It keeps a token-bounded **Working Memory** of recent turns, plus a local **Long-Term Memory** of atomic facts that an **Analyzer** model maintains. It needs no external database and works with OpenAI, Google Gemini, Anthropic and local Ollama models.

What makes it different from storing chat turns or summaries:

- **Contradiction-aware writes.** The analyzer sees the memories related to each exchange and labels every new fact `add`, `update(target)` or `noop(target)`.
- **Versioning instead of overwriting.** An update closes the old record with a validity interval and links it to its successor. Retrieval sees only current facts, and the history is kept.
- **Off the response path.** `chat()` returns as soon as the reply exists. Analysis and consolidation run on a background queue (`flush()` waits for it).
- **Relevance-gated, diverse retrieval.** Candidates must pass a relative similarity gate, and the top-k are chosen by maximal marginal relevance.
- **Compact local store.** It is held in memory, written atomically, and stores embeddings as Float32 in base64.

## Relation to the original work

This repository builds on the original **Memory Engine** library by MD Samiul Islam:
**[github.com/Samiul-Islam-123/memory-engine-node](https://github.com/Samiul-Islam-123/memory-engine-node)** (npm `memory-engine-node`, release 1.0.3). The original introduced the three-model design: a chat model, an embedding model and an analyzer model, with a token-bounded working memory and a local long-term store. This repository is a redesign of its memory pipeline (v2.0.0), together with the benchmark and experiments used to evaluate it.

### What was modified

| Area | Original `memory-engine-node` 1.0.3 | This repository (2.0.0) |
|---|---|---|
| Consolidation timing | `chat()` waits for `analyze()` + `resolveMemory()`, so the analyzer call sits on the response path | **Off-path consolidation**: `chat()` returns once the reply exists; analysis runs on a serialised background queue (`flush()`, `newSession()`) |
| What the analyzer sees | Only the new exchange, never the existing memories | **Contradiction-aware analysis**: up to six related memories (aliases M1–M6); every fact is labelled `add`, `update(target)` or `noop(target)` |
| Merge decision | Cosine similarity alone: update if 0.90 ≤ cos < 0.98, ignore if cos ≥ 0.98 | The analyzer's label decides, with an **asymmetric-loss fallback**: an `add` whose nearest active record has cos ≥ 0.92 supersedes it |
| Updates | Overwrite the record in place, with no validity data | **Temporal versioning**: the old record is closed (`validTo`, `status: superseded`) and linked to its successor; only active records are indexed; prompts show "(updated; previously: …)" |
| Extraction schema | Verbose 7-field schema with free-text fields; `store:false` and importance are ignored | **Compact schema**; `store:false` and importance < 0.3 are honoured (fewer malformed-JSON losses and noise records) |
| Embeddings | Records embedded as JSON blobs, without the embedder's task prefixes | Plain-text facts with task prefixes (`search_query:` / `search_document:` for nomic-embed-text) |
| Retrieval | Fixed threshold 0.55, top-3 | **Relevance gate** `cos ≥ max(0.35, best − 0.2)`, then top-5 by **maximal marginal relevance** (λ = 0.75) |
| Storage | Whole store re-read from disk on every query; pretty-printed float JSON (exceeds V8's string limit at about 18k records) | In-memory store, atomic writes, base64 Float32 vectors (about 4.5 KB per record, about 1.2 × 10⁵ records) |
| Tests | — | 8 offline unit tests (`npm test`) |

### Measured difference (same benchmark, models and transcripts)

| Metric (Qwen3-4B) | Original 1.0.3 | This repository |
|---|---:|---:|
| Probe accuracy (120 probes) | 55.8% | **97.5%** (McNemar p = 8×10⁻¹³) |
| Accuracy on updated facts | 38.9% | **100%** |
| Answers giving the outdated value (update probes) | 30.6% | **0%** |
| Stated facts present in the store | 57.3% | **97.9%** |
| Updated facts with old and new value both active | 11.1% | **0%** |
| Prompt tokens per probe | 429 | **305** |
| Mean user-perceived response time | 43.2 s (analyzer on path: 22.0 s) | **31.2 s** back-to-back, **16.5 s** with a 5 s user pause |

With **Qwen3-8B** as chat and analyzer model the accuracy gap closes (original 83.3%, this repository 80.0%, p = 0.58). The larger model writes valid JSON, which removes the original's main failure, but it still keeps both values active for 30.6% of updated facts; this repository keeps 0%. The 8B analyzer is more conservative under the compact prompt, which lowers extraction recall.

## Repository layout

| Path | Contents |
|---|---|
| [`source_code/`](source_code/) | The Memory Engine library (npm package `memory-engine-node`, v2.0.0), with its unit tests and package README |
| [`experiments/`](experiments/) | Benchmark generator, evaluation harness, baselines (RAG over turns, recursive summarisation, Mem0), statistics and figure scripts |

## Quick start

```bash
cd source_code
npm install
npm test            # offline unit tests
```

```javascript
const MemoryEngine = require("./source_code");

const chat     = { model: "qwen3:4b-instruct", provider: "ollama" };   // or openai / google / anthropic + apiKey
const embedder = { model: "nomic-embed-text",  provider: "ollama" };   // keep the embedding model fixed for a store

const mem = new MemoryEngine(chat, embedder, chat, "./memory-data", {
    asyncConsolidation: true,       // consolidate in the background (default)
    workingMemoryTokens: 2000,
    longTerm: { topK: 5, absThreshold: 0.35, relMargin: 0.2, mmrLambda: 0.75, minImportance: 0.3 }
});

await mem.chat("My budget for the side project is 1.5 lakh rupees.");
await mem.chat("Update: the budget went up to 4,700 dollars.");
await mem.flush();                  // wait for background consolidation
mem.newSession();                   // new conversation, same long-term memory
console.log(await mem.chat("What is my budget?"));   // answers with the current value
```

See [`source_code/README.md`](source_code/README.md) for every provider and option.

## Results

Persona Memory Benchmark: 12 seeded personas, 444 turns and 120 probes (recall, update and abstention). Every system uses the same Qwen3-4B chat model, the same transcripts and the same prompt template; only the memory differs.

| System | Accuracy | Prompt tokens / probe | Updated facts with old **and** new value active |
|---|---:|---:|---:|
| **Memory Engine** | **97.5%** | **305** | **0%** |
| RAG over turns | 95.0% | 634 | n/a (all turns kept) |
| Recursive summarisation | 95.0% | 765 | 11% |
| Mem0 (`mem0ai` 2.2.1) | 93.3% | 397 | 100% |
| Full history replay | 67.5% | 4,347 | n/a |

Memory Engine has the highest accuracy and the smallest prompts. Its accuracy lead over RAG, recursive summarisation and Mem0 is **not** statistically significant on 120 probes. Its clearest advantage is a consistent store: the outdated value reaches the context of 11% of update probes, marked as superseded, against 36–100% for the baselines.

## How the experiments were conducted

- **Benchmark.** The *Persona Memory Benchmark* is generated with a fixed seed by `experiments/gen_scenarios.py`. It has 12 personas over 11 attributes (city, budget, framework, weekly hours, diet, job, trip, dog, sister, allergy, instrument), with values too unusual to guess.
  - *Session 1* (24 turns) states 8 facts among 14 unrelated requests, plus 2 distractor facts about third parties.
  - *Session 2* (13 turns) updates 3 facts, restates 2 and adds 8 fillers.
  - *Session 3* asks 10 probes, each in a fresh session: 5 recall, 3 update and 2 abstention. In total: 444 turns and 120 probes.
- **Controlled replay.** Assistant replies for sessions 1–2 are generated once by a stateless chat model. Every system then ingests the *identical* transcript through its normal write path, so the memory method is the only variable. Probes go through each system's real read path on a pristine copy of its store.
- **Systems.**
  - Memory Engine (this repository).
  - Baselines: no memory; replay truncated to 4,096 tokens; full replay; RAG over turns (top-5 raw turns, same embedder); recursive summarisation (Wang et al., 2025); Mem0 (`mem0ai` 2.2.1).
  - The original release 1.0.3.
  - Ablations of Memory Engine (−P generic prompt, −C cosine-threshold merge, −R fixed top-3 retrieval) and the full 2³ factorial of these components.
  - Every system answers with the same chat model, system prompt and prompt template.
- **Models.** All inference is local through Ollama on an Apple M4 with 16 GB. Qwen3-4B-Instruct-2507 (Q4_K_M) is the chat and analyzer model, Qwen3-8B is used for the scale check, and nomic-embed-text produces the embeddings. Temperature is 0 with a fixed seed. A `fetch` wrapper logs token counts and latencies and caches every response, so all numbers can be regenerated exactly.
- **Scoring and statistics.**
  - Answers are scored by alias regular expressions as correct, stale, confused, abstained or wrong. Abstention probes are scored as abstained or hallucinated.
  - An independent Qwen3-8B judge re-graded 400 answers: 90.5% agreement, Cohen's κ = 0.79.
  - We report Wilson intervals, exact McNemar tests on paired probes and persona-cluster bootstrap intervals.
- **Interpretability.**
  - An exact retrieval–generation decomposition: acc = ρα + (1 − ρ)β.
  - Exact Shapley values with Harsanyi dividends over the 2³ component configurations, with 2,000 bootstrap replicates.
  - A ridge logistic model of correctness on context features.
- **Latency.** Live and uncached: 2 personas × 37 turns per system, measuring the user-perceived time until `chat()` resolves.

The original release can be evaluated with the same harness. `run_all.sh` places it in `experiments/engine_v1/`; then run `node harness/run_pmb.js ingest v1`, `node harness/run_pmb.js probe v1` and `node harness/run_latency.js v1`.

## Reproducing the experiments

All inference is local. The full pipeline takes several hours on an Apple M4 with 16 GB; most of that time goes to Mem0 ingestion and the 8B runs.

1. Install [Ollama](https://ollama.com) and pull the models:
   ```bash
   ollama pull qwen3:4b-instruct
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=2 ollama serve
   ```
2. Create the Python environment:
   ```bash
   cd experiments
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
3. Run everything:
   ```bash
   ./run_all.sh
   ```

Each step is skipped if its output already exists, so the script can be resumed. Tables and figures are written to `experiments/paper_assets/`. See [`experiments/README.md`](experiments/README.md) for the individual steps.

## Paper

*Memory Engine: A Versioned, Analyzer-Driven Memory Layer for LLM Chatbots.* MD Samiul Islam, Prabhat Das, Bodhi Chakraborty, Ayushman Bilas Thakur. Department of Computer Science and Engineering, Centre of Excellence in Artificial Intelligence, Adamas University.

## License

ISC (see [`source_code/package.json`](source_code/package.json)).
