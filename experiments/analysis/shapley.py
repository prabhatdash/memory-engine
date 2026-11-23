"""Exact Shapley attribution of ME-v2's gain to its three read/write components, plus an
interpretable retrieval-generation decomposition and a logistic model of answer correctness.

Players (1 = ME-v2 component on, 0 = ME-v1 behaviour):
  P  compact, contradiction-aware analyzer prompt   (write path)
  C  LLM-decided add/update/noop consolidation        (write path)
  R  gated, diversity-aware retrieval                 (read path)
All 2^3 configurations share the remaining ME-v2 infrastructure (versioning, hygiene, async).

Outputs results/shapley.json.
"""
import json, os, itertools, math, re, random
from collections import defaultdict
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(__file__))
from score import load, value_regex, norm, SC

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CONF = {"111": "v2", "101": "v2_threshold", "110": "v2_v1retr", "011": "v2_v1prompt",
        "100": "fac_100", "010": "fac_010", "001": "fac_001", "000": "fac_000"}
STORE = {"11": "v2", "10": "v2_threshold", "01": "v2_v1prompt", "00": "fac_000"}
PLAYERS = ["P", "C", "R"]


def shapley(v, n):
    """v: dict bitstring->value. Exact Shapley values and Harsanyi dividends."""
    N = range(n)
    phi = []
    for i in N:
        s = 0.0
        for k in range(n):
            for S in itertools.combinations([j for j in N if j != i], k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                a = "".join("1" if (j in S or j == i) else "0" for j in N)
                b = "".join("1" if j in S else "0" for j in N)
                s += w * (v[a] - v[b])
        phi.append(s)
    # Harsanyi dividends d(T) = sum_{S subset T} (-1)^{|T|-|S|} v(S)
    div = {}
    for T in itertools.product([0, 1], repeat=n):
        idx = [j for j in N if T[j]]
        d = 0.0
        for k in range(len(idx) + 1):
            for S in itertools.combinations(idx, k):
                key = "".join("1" if j in S else "0" for j in N)
                d += (-1) ** (len(idx) - k) * v[key]
        div["".join(map(str, T))] = d
    return phi, div


def store_metrics(cond):
    """per-persona store contradiction / coverage / noise for a stored condition."""
    out = {}
    for pid, s in SC.items():
        p = os.path.join(RES, "stores", cond, pid, "memory.json")
        if not os.path.exists(p):
            return None
        mems = [m for m in json.load(open(p))["memories"] if m.get("status", "active") == "active"]
        texts = [norm(m.get("summary", "")) for m in mems]
        cov, con, matched = [], [], [False] * len(texts)
        for slot, v in s["values"].items():
            cur = value_regex(v["value"])
            hits = [i for i, t in enumerate(texts) if cur.search(t)]
            for i in hits: matched[i] = True
            cov.append(bool(hits))
            if "old" in v:
                old = value_regex(v["old"])
                oh = [i for i, t in enumerate(texts) if old.search(t) and not cur.search(t)]
                for i in oh: matched[i] = True
                con.append(bool(oh) and bool(hits))
            if "distractor" in v:
                for i, t in enumerate(texts):
                    if value_regex(v["distractor"]).search(t): matched[i] = True
        out[pid] = {"coverage": (sum(cov), len(cov)), "contradiction": (sum(con), len(con)),
                    "noise": (sum(1 for x in matched if not x), 1)}
    return out


def ratio(per, personas, key):
    num = sum(per[p][key][0] for p in personas); den = sum(per[p][key][1] for p in personas)
    return num / den if den else 0.0


def probe_metric(rows, personas, typ=None):
    cnt = defaultdict(lambda: [0, 0])
    for r in rows:
        if typ and r["type"] != typ: continue
        cnt[r["persona"]][0] += r["ok"]; cnt[r["persona"]][1] += 1
    k = sum(cnt[p][0] for p in personas); n = sum(cnt[p][1] for p in personas)
    return k / n if n else 0.0


def boot_shapley(fn, keys, n, B=2000, seed=0):
    rng = random.Random(seed)
    ps = sorted(SC)
    point = shapley({k: fn(k, ps) for k in keys}, n)
    samples = []
    for _ in range(B):
        s = [rng.choice(ps) for _ in ps]
        phi, _ = shapley({k: fn(k, s) for k in keys}, n)
        samples.append(phi)
    a = np.array(samples)
    return {"phi": point[0], "dividends": point[1],
            "ci": [[float(np.percentile(a[:, i], 2.5)), float(np.percentile(a[:, i], 97.5))] for i in range(n)],
            "p_pos": [float((a[:, i] > 0).mean()) for i in range(n)]}


# ---------------------------------------------------------------- interpretability
def ctx_features(r):
    ctx = r.get("retrieved") or []
    cur = " ".join(norm(x.get("summary", "")) for x in ctx)
    g = r["answer_value"]
    vals = [norm(v["value"]) for v in SC[r["persona"]]["values"].values()]
    allvals = [value_regex(v["value"]) for v in SC[r["persona"]]["values"].values()] + \
              [value_regex(v["old"]) for v in SC[r["persona"]]["values"].values() if "old" in v]
    noise = sum(1 for x in ctx if not any(rx.search(norm(x.get("summary", ""))) for rx in allvals))
    gold = bool(g and value_regex(g).search(cur))
    stale = bool(r.get("stale") and value_regex(r["stale"]).search(cur) and not gold)
    topsim = max([x.get("sim") or 0 for x in ctx], default=0.0)
    return {"gold": gold, "stale": stale, "noise": noise, "n": len(ctx), "topsim": topsim}


def logit(X, y, iters=50, l2=1.0):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(len(w)); g = X.T @ (y - p) - l2 * w
        w += np.linalg.solve(H, g)
    p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    ll0 = len(y) * (y.mean() * np.log(y.mean()) + (1 - y.mean()) * np.log(1 - y.mean()))
    return w, se, 1 - ll / ll0


if __name__ == "__main__":
    out = {}
    probes = {k: load(c) for k, c in CONF.items()}
    have = [k for k, v in probes.items() if v]
    out["configs_available"] = have
    if len(have) == 8:
        res = {}
        for name, typ in (("overall", None), ("recall", "recall"), ("update", "update"), ("abstain", "abstain")):
            res[name] = boot_shapley(lambda k, ps, t=typ: probe_metric(probes[k], ps, t), list(CONF), 3)
            res[name]["v"] = {k: probe_metric(probes[k], sorted(SC), typ) for k in CONF}
        out["accuracy"] = res
    stores = {k: store_metrics(c) for k, c in STORE.items()}
    if all(stores.values()):
        sres = {}
        for key in ("contradiction", "coverage", "noise"):
            sres[key] = boot_shapley(lambda k, ps, kk=key: ratio(stores[k], ps, kk), list(STORE), 2)
            sres[key]["v"] = {k: ratio(stores[k], sorted(SC), key) for k in STORE}
        out["store"] = sres
    # retrieval-generation decomposition: acc = rho*alpha + (1-rho)*beta on recall+update probes
    dec = {}
    for c in ["v2", "v2_threshold", "v2_v1retr", "v2_v1prompt", "fac_100", "fac_010", "fac_001", "fac_000",
              "v2@8b", "naive_full", "naive_trunc", "naive_full@8b", "rag_turns", "rsum", "mem0",
              "rag_turns@8b", "rsum@8b", "mem0@8b"]:
        rows = load(c)
        if not rows: continue
        rs = [r for r in rows if r["type"] != "abstain"]
        if c.startswith("naive"):
            g = [True] * len(rs)  # every stated fact is present in the replayed prompt (budget kept >=34/37 turns)
        else:
            g = [ctx_features(r)["gold"] for r in rs]
        ok = [r["ok"] for r in rs]
        rho = float(np.mean(g))
        a = [o for o, gg in zip(ok, g) if gg]; b = [o for o, gg in zip(ok, g) if not gg]
        dec[c] = {"rho": rho, "alpha": float(np.mean(a)) if a else None, "beta": float(np.mean(b)) if b else None,
                  "acc": float(np.mean(ok)), "n": len(rs)}
    out["decomposition"] = dec
    # pooled logistic model over engine conditions (recall+update probes)
    X, y, names = [], [], ["gold_in_ctx", "stale_only_in_ctx", "noise_records", "top_similarity"]
    for c in ["v2", "v2_threshold", "v2_v1retr", "v2_v1prompt", "fac_100", "fac_010", "fac_001", "fac_000"]:
        rows = load(c)
        if not rows: continue
        for r in rows:
            if r["type"] == "abstain" or "retrieved" not in r: continue
            f = ctx_features(r)
            X.append([f["gold"], f["stale"], f["noise"], f["topsim"]]); y.append(r["ok"])
    if X:
        X = np.array(X, float); y = np.array(y, float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = X.copy(); Xs[:, 2:] = (X[:, 2:] - mu[2:]) / sd[2:]
        w, se, r2 = logit(Xs, y)
        out["logit"] = {"names": ["intercept"] + names, "coef": w.tolist(), "se": se.tolist(),
                        "odds_ratio": np.exp(w).tolist(), "mcfadden_r2": float(r2), "n": int(len(y)),
                        "note": "noise and top_similarity standardised (per SD)"}
    # abstention: hallucination vs top similarity of retrieved context (engine conds)
    ab = []
    for c in ["v2", "v2_threshold", "v2_v1retr", "v2_v1prompt"]:
        rows = load(c) or []
        for r in rows:
            if r["type"] == "abstain" and "retrieved" in r:
                f = ctx_features(r); ab.append((c, f["n"], f["topsim"], f["noise"], r["ok"]))
    out["abstain_ctx"] = ab
    json.dump(out, open(os.path.join(RES, "shapley.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("abstain_ctx",)}, indent=1)[:6000])
