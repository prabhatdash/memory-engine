"""Mem0 baseline (mem0ai, default configuration) on the Persona Memory Benchmark.

Ingest: every turn of sessions 1-2 is added as a user/assistant message pair with
Mem0's own LLM extraction (infer=True), exactly as an application would call it.
Search: for every probe, Mem0's top-k memories are saved; the harness then answers
with the same chat model and prompt template as Memory Engine (run_pmb.js, cond "mem0").

    python baselines/run_mem0.py [4b|8b]
"""
import json, os, sys, time, shutil
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
SIZE = sys.argv[1] if len(sys.argv) > 1 else "4b"
MODEL = {"4b": "qwen3:4b-instruct", "8b": "qwen3:8b"}[SIZE]
TAG = "mem0" if SIZE == "4b" else "mem0@8b"
TOPK = 5  # same k as Memory Engine


def _patch_ctx():
    """Mem0's extraction prompt (~8k tokens) exceeds Ollama's default 4k context and
    Mem0 exposes no num_ctx option, so we raise the context window in the client.
    Deployment setting only; Mem0's prompts and logic are untouched."""
    import ollama
    if getattr(ollama.Client.chat, "_pmb", False):
        return
    orig = ollama.Client.chat

    def chat(self, *a, **k):
        opts = dict(k.get("options") or {})
        opts["num_ctx"] = 16384
        opts.setdefault("seed", 42)
        k["options"] = opts
        if MODEL.startswith("qwen3:8b"):
            k.setdefault("think", False)
        return orig(self, *a, **k)
    chat._pmb = True
    ollama.Client.chat = chat


def make(pid):
    _patch_ctx()
    from mem0 import Memory
    d = os.path.join(RES, "stores", TAG, pid)
    os.makedirs(d, exist_ok=True)
    return Memory.from_config({
        "llm": {"provider": "ollama", "config": {"model": MODEL, "temperature": 0.0, "max_tokens": 2000,
                                                 "ollama_base_url": "http://localhost:11434"}},
        "embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text", "embedding_dims": 768,
                                                      "ollama_base_url": "http://localhost:11434"}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": f"pmb_{pid}", "path": os.path.join(d, "qdrant"),
                                                          "embedding_model_dims": 768, "on_disk": True}},
        "history_db_path": os.path.join(d, "history.db"),
    })


def run(args):
    pid, turns, probes = args
    d = os.path.join(RES, "stores", TAG, pid)
    shutil.rmtree(d, ignore_errors=True)
    m = make(pid)
    log = []
    for ti, t in enumerate(turns):
        t0 = time.perf_counter()
        for attempt in range(4):
            try:
                r = m.add([{"role": "user", "content": t["text"]}, {"role": "assistant", "content": t["reply"]}], user_id=pid)
                break
            except Exception as e:  # transient Ollama errors
                r = {"error": str(e)[:200]}
                time.sleep(5 * (attempt + 1))
        log.append({"cond": TAG, "persona": pid, "turn": ti, "session": t["session"], "kind": t["kind"], "slot": t.get("slot"),
                    "result": r.get("results", r) if isinstance(r, dict) else r, "ms": (time.perf_counter() - t0) * 1000})
    ret = {}
    for pi, p in enumerate(probes):
        s = m.search(p["q"], top_k=TOPK, filters={"user_id": pid})
        ret[str(pi)] = [{"memory": x.get("memory"), "score": x.get("score"), "created_at": x.get("created_at"),
                         "updated_at": x.get("updated_at")} for x in s.get("results", [])]
    allm = m.get_all(filters={"user_id": pid}, top_k=100000)
    allm = allm.get("results", allm) if isinstance(allm, dict) else allm
    print("mem0", SIZE, pid, "done", len(allm), "memories", flush=True)
    return pid, log, ret, [{"memory": x.get("memory"), "created_at": x.get("created_at"), "updated_at": x.get("updated_at")} for x in allm]


if __name__ == "__main__":
    T = json.load(open(os.path.join(DATA, "transcripts.json")))
    S = json.load(open(os.path.join(DATA, "scenarios.json")))["scenarios"]
    jobs = [(s["id"], T[s["id"]], s["probes"]) for s in S]
    only = os.environ.get("ONLY")
    if only:
        jobs = [j for j in jobs if j[0] in only.split(",")]
    t0 = time.time()
    with Pool(int(os.environ.get("CONC", 4))) as pool:
        out = pool.map(run, jobs)
    log = [r for _, l, _, _ in out for r in l]
    json.dump(log, open(os.path.join(RES, f"ingest_{TAG}.json"), "w"))
    json.dump({pid: ret for pid, _, ret, _ in out}, open(os.path.join(RES, f"retrieved_{TAG}.json"), "w"), indent=1)
    json.dump({pid: allm for pid, _, _, allm in out}, open(os.path.join(RES, f"store_{TAG}.json"), "w"), indent=1)
    print(f"mem0 {SIZE} finished in {time.time() - t0:.0f}s", flush=True)
