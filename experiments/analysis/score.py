"""Deterministic scoring + statistics for the Persona Memory Benchmark."""
import json, re, os, sys, glob, unicodedata, math, random
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
SC = {s["id"]: s for s in json.load(open(os.path.join(HERE, "..", "data", "scenarios.json")))["scenarios"]}


def norm(s):
    s = (s or "").replace("ø", "o").replace("Ø", "O")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace("’", "'")


def value_regex(v):
    v = norm(v)
    m = re.match(r"([\d.,]+)\s*(lakh)?\s*(rupees|euros|dollars)?$", v)
    if m:
        num, lakh = m.group(1), m.group(2)
        if lakh:
            n = int(round(float(num) * 100000))
            west = f"{n:,}".replace(",", ",?")
            d = str(n)
            head, tail = d[:-3], d[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:]); head = head[:-2]
            if head:
                groups.insert(0, head)
            indian = ",?".join(groups + [tail])
            return re.compile("|".join([re.escape(num) + r"\s*lakh", west, indian]))
        digits = num.replace(",", "")
        if "." not in digits and len(digits) > 3:
            pat = digits[:-3] + ",?" + digits[-3:]
        else:
            pat = re.escape(digits)
        return re.compile(r"(?<![\d.,])" + pat + r"(?![\d])")
    special = {"solidjs": r"solid\s*\.?js|\bsolid\b", "gluten-free": r"gluten[- ]?free", "cluj-napoca": r"cluj",
               "air traffic controller": r"air[- ]traffic[- ]control"}
    if v in special:
        return re.compile(special[v])
    return re.compile(r"\b" + re.escape(v) + r"\b")


ABSTAIN = re.compile(r"(don'?t (know|have|recall|remember|see)|do not (know|have|recall)|didn'?t (mention|tell|share|say)|haven'?t (mentioned|told|shared|said|provided)|have not|not sure|no (record|information|idea|details|mention)|not (been )?(mentioned|told|shared|provided|specified)|unable to|i can'?t|i cannot|not aware|unknown to me|you haven'?t|could you (tell|share|let)|can you (tell|share|let)|isn'?t (mentioned|specified)|wasn'?t mentioned)")


def score_row(r):
    a = norm(r["answer"])
    if r["type"] == "abstain":
        return "correct" if ABSTAIN.search(a) else "hallucinated"
    cur = value_regex(r["answer_value"]) if "answer_value" in r else value_regex(r["answer"])
    hit = bool(cur.search(a))
    stale = bool(r.get("stale") and value_regex(r["stale"]).search(a))
    distr = bool(r.get("distractor") and value_regex(r["distractor"]).search(a))
    if hit:
        return "correct"
    if stale:
        return "stale"
    if distr:
        return "confused"
    if ABSTAIN.search(a):
        return "abstained"
    return "wrong"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load(cond):
    f = os.path.join(RES, f"probes_{cond}.json")
    if not os.path.exists(f):
        return None
    rows = json.load(open(f))
    for r in rows:
        # field collision: probe "answer" (gold) was overwritten by the model answer; recover gold
        gold = SC[r["persona"]]["probes"][r["probe"]]
        r["answer_value"] = gold.get("answer")
        r["outcome"] = score_row(r)
        r["ok"] = r["outcome"] == "correct"
    return rows


def retrieval_diag(rows):
    """Gold / stale presence in the retrieved long-term context (engine conditions)."""
    g = defaultdict(list)
    for r in rows:
        if r["type"] == "abstain" or "retrieved" not in r:
            continue
        ctx = " ".join(norm(x.get("summary", "")) + " " + norm(x.get("previously") or "") for x in r["retrieved"])
        cur_ctx = " ".join(norm(x.get("summary", "")) for x in r["retrieved"])
        g[r["type"] + "_gold_in_ctx"].append(bool(value_regex(r["answer_value"]).search(cur_ctx)))
        if r["type"] == "update":
            g["update_stale_as_current_in_ctx"].append(bool(value_regex(r["stale"]).search(cur_ctx)) and not value_regex(r["answer_value"]).search(cur_ctx))
            g["update_stale_anywhere_in_ctx"].append(bool(value_regex(r["stale"]).search(cur_ctx)))
        g["n_retrieved"].append(len(r["retrieved"]))
    return {k: float(np.mean(v)) for k, v in g.items()}


def summarize(rows):
    out = {}
    for t in ("recall", "update", "abstain"):
        rs = [r for r in rows if r["type"] == t]
        k = sum(r["ok"] for r in rs)
        p, lo, hi = wilson(k, len(rs))
        out[t] = {"acc": p, "ci": [lo, hi], "k": k, "n": len(rs), "outcomes": Counter(r["outcome"] for r in rs)}
    k = sum(r["ok"] for r in rows)
    p, lo, hi = wilson(k, len(rows))
    out["overall"] = {"acc": p, "ci": [lo, hi], "k": k, "n": len(rows)}
    out["prompt_tokens_mean"] = float(np.mean([r.get("promptTokens") or 0 for r in rows]))
    out["prompt_tokens_median"] = float(np.median([r.get("promptTokens") or 0 for r in rows]))
    return out


def mcnemar(a, b):
    """exact McNemar on paired probes; returns (b_only, a_only, p)."""
    ka = {(r["persona"], r["probe"]): r["ok"] for r in a}
    kb = {(r["persona"], r["probe"]): r["ok"] for r in b}
    keys = ka.keys() & kb.keys()
    n01 = sum(1 for k in keys if not ka[k] and kb[k])
    n10 = sum(1 for k in keys if ka[k] and not kb[k])
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def cluster_bootstrap_diff(a, b, B=5000, seed=0):
    """95% CI of acc(b)-acc(a), resampling personas."""
    rng = random.Random(seed)
    per = defaultdict(lambda: [[], []])
    for r in a:
        per[r["persona"]][0].append(r["ok"])
    for r in b:
        per[r["persona"]][1].append(r["ok"])
    ps = list(per)
    diffs = []
    for _ in range(B):
        s = [rng.choice(ps) for _ in ps]
        xa = sum(sum(per[p][0]) for p in s) / sum(len(per[p][0]) for p in s)
        xb = sum(sum(per[p][1]) for p in s) / sum(len(per[p][1]) for p in s)
        diffs.append(xb - xa)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


CONDS = ["nomem", "naive_trunc", "naive_full", "rag_turns", "rsum", "mem0", "v1", "v2", "v2_threshold", "v2_v1retr", "v2_v1prompt",
         "naive_full@8b", "rag_turns@8b", "rsum@8b", "mem0@8b", "v1@8b", "v2@8b"]
RETR = ("v", "rag_turns", "rsum", "mem0")

if __name__ == "__main__":
    data = {c: load(c) for c in CONDS}
    data = {c: v for c, v in data.items() if v}
    report = {"summary": {}, "tests": {}, "retrieval": {}}
    for c, rows in data.items():
        report["summary"][c] = summarize(rows)
        if c.startswith(RETR):
            report["retrieval"][c] = retrieval_diag(rows)
    if "v2" in data:
        for c, rows in data.items():
            if c == "v2" or "@" in c:
                continue
            n01, n10, p = mcnemar(rows, data["v2"])
            lo, hi = cluster_bootstrap_diff(rows, data["v2"])
            report["tests"][f"v2_vs_{c}"] = {"v2_only_correct": n01, "other_only_correct": n10, "mcnemar_p": p,
                                             "acc_diff": report["summary"]["v2"]["overall"]["acc"] - report["summary"][c]["overall"]["acc"],
                                             "diff_ci95_cluster_bootstrap": [lo, hi]}
            # per probe type
            for t in ("recall", "update", "abstain"):
                a = [r for r in rows if r["type"] == t]; b = [r for r in data["v2"] if r["type"] == t]
                n01, n10, p = mcnemar(a, b)
                report["tests"][f"v2_vs_{c}"][t] = {"v2_only": n01, "other_only": n10, "p": p}
    if "v2@8b" in data:
        for c in ("v1@8b", "naive_full@8b", "rag_turns@8b", "rsum@8b", "mem0@8b"):
            if c in data:
                n01, n10, p = mcnemar(data[c], data["v2@8b"])
                lo, hi = cluster_bootstrap_diff(data[c], data["v2@8b"])
                report["tests"][f"v2@8b_vs_{c}"] = {"v2_only_correct": n01, "other_only_correct": n10, "mcnemar_p": p,
                    "acc_diff": report["summary"]["v2@8b"]["overall"]["acc"] - report["summary"][c]["overall"]["acc"],
                    "diff_ci95_cluster_bootstrap": [lo, hi]}
    json.dump(report, open(os.path.join(RES, "score_report.json"), "w"), indent=1, default=lambda o: dict(o))
    json.dump({c: [{k: r[k] for k in ("persona", "probe", "type", "slot", "q", "answer", "answer_value", "outcome")} for r in rows]
               for c, rows in data.items()}, open(os.path.join(RES, "scored_answers.json"), "w"), indent=1)
    for c, s in report["summary"].items():
        print(f"{c:13s} recall {s['recall']['acc']:.3f}  update {s['update']['acc']:.3f}  abstain {s['abstain']['acc']:.3f}  "
              f"overall {s['overall']['acc']:.3f} [{s['overall']['ci'][0]:.2f},{s['overall']['ci'][1]:.2f}]  tok {s['prompt_tokens_mean']:.0f}")
        print("              ", dict(s["recall"]["outcomes"]), dict(s["update"]["outcomes"]), dict(s["abstain"]["outcomes"]))
    for k, v in report["retrieval"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    for k, v in report["tests"].items():
        print(k, f"diff {v['acc_diff']:+.3f} CI {v['diff_ci95_cluster_bootstrap'][0]:+.3f},{v['diff_ci95_cluster_bootstrap'][1]:+.3f} p={v['mcnemar_p']:.2g}")
