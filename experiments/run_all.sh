#!/bin/zsh
# Reproduces every experiment in the paper, end to end.
# Prerequisites: Node >= 18, Python >= 3.10, and an Ollama server with the models
#   qwen3:4b-instruct, qwen3:8b and nomic-embed-text   (see README.md).
# Every step is skipped when its output already exists, so the script can be re-run after an interruption.
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PMB_CONC=${PMB_CONC:-2} MEM0_TELEMETRY=False
mkdir -p results data

# Setup: library dependencies, and the original memory-engine-node 1.0.3 release from npm,
# whose verbose analyzer prompt is the "generic extraction prompt" of the -P ablation.
(cd ../source_code && npm install --silent)
if [ ! -d engine_v1 ]; then
  npm pack memory-engine-node@1.0.3 --silent >/dev/null
  mkdir engine_v1 && tar -xzf memory-engine-node-1.0.3.tgz -C engine_v1 --strip-components=1 && rm memory-engine-node-1.0.3.tgz
  (cd engine_v1 && npm install --silent)
fi

# A. Benchmark (seeded) and fixed assistant replies
[ -f data/scenarios.json ]   || $PY gen_scenarios.py 12 data/scenarios.json
[ -f data/transcripts.json ] || node harness/run_pmb.js transcripts

# B. Write paths: Memory Engine (v2), its ablation stores, and the in-harness baselines
for c in v2 v2_threshold v2_v1prompt fac_000 rag_turns rsum v2@8b; do
  [ -f results/ingest_$c.json ] || node harness/run_pmb.js ingest $c
done
# Mem0 (mem0ai, default configuration); run alone: it needs a 16k context window
[ -f results/retrieved_mem0.json ] || CONC=2 $PY baselines/run_mem0.py 4b
$PY baselines/export_mem0.py 4b > /dev/null

# C. Probes (read path, fresh session per probe)
for c in nomem naive_trunc naive_full rag_turns rsum mem0 v2 v2_threshold v2_v1retr v2_v1prompt \
         fac_100 fac_010 fac_001 fac_000 naive_full@8b v2@8b; do
  [ -f results/probes_$c.json ] || node harness/run_pmb.js probe $c
done

# D. Live, uncached latency (sequential)
[ -f results/latency_v2.json ]           || node harness/run_latency.js v2
[ -f results/latency_v2_think5000.json ] || node harness/run_latency.js v2 5000
[ -f results/latency_naive_full.json ]   || node harness/run_latency.js naive_full

# E. Embedding geometry, scan cost, scoring, judge validation, attribution, paper assets
[ -f results/geom_v2_prefixed.npz ] || $PY analysis/embed_geometry.py
$PY analysis/geometry_extra.py > /dev/null
[ -f results/bench_scan.json ] || node harness/bench_scan.js
$PY analysis/score.py > results/score.out
[ -f results/judge_validation.json ] || $PY analysis/judge_validate.py > results/judge.out
$PY analysis/store_stats.py > /dev/null
$PY analysis/shapley.py > /dev/null
$PY analysis/make_paper_assets.py
$PY analysis/make_figs2.py
echo "ALL DONE: tables and figures are in experiments/paper_assets/"
