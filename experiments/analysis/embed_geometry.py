"""Embedding-geometry validation of the threshold merge rule (Sec. 4 of the paper).

Builds labelled pairs of memory records from the benchmark templates:
  PARA : same slot, same value, different wording      -> correct action NOOP (duplicate)
  UPD  : same slot, different value                    -> correct action UPDATE
  DIFF : different slots                               -> correct action ADD
and measures cosine similarity under v1's document encoding (JSON blob, no
prefix) and v2's (prefixed plain-text fact).  Outputs
  * per-class cosine distributions,
  * what v1's rule (update iff cos>=0.90, ignore iff cos>=0.98) does to each class,
  * ROC-AUC of cosine for UPD-vs-PARA and the best accuracy *any* threshold can reach,
  * P[cos(q,new) > cos(q,old)] for value-agnostic probe queries (exchangeability).
"""
import json, itertools, random, urllib.request, sys, os
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
slots = json.load(open(os.path.join(HERE, "..", "data", "scenarios.json")))["slots"]
TYPE = {"city": "fact", "budget": "constraint", "framework": "decision", "hours": "constraint", "diet": "preference",
        "job": "fact", "trip": "goal", "pet": "fact", "sibling": "relationship", "allergy": "fact", "instrument": "skill"}


def embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        req = urllib.request.Request("http://localhost:11434/api/embed",
                                     data=json.dumps({"model": "nomic-embed-text", "input": texts[i:i + bs]}).encode(),
                                     headers={"Content-Type": "application/json"})
        out += json.load(urllib.request.urlopen(req))["embeddings"]
    X = np.array(out, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def enc_v1(slot, s):
    return json.dumps({"entity": "user", "type": TYPE[slot], "category": slot, "summary": s}, separators=(",", ":"))


def enc_v2(slot, s):
    return "search_document: user: " + s


rng = random.Random(0)
records = []  # (slot, value, template_idx, summary)
for slot, d in slots.items():
    for v in d["values"]:
        for ti, t in enumerate(d["summary"]):
            records.append((slot, v, ti, t.format(v=v)))

pairs = {"PARA": [], "UPD": [], "DIFF": []}
for a, b in itertools.combinations(range(len(records)), 2):
    ra, rb = records[a], records[b]
    if ra[0] == rb[0] and ra[1] == rb[1]:
        pairs["PARA"].append((a, b))
    elif ra[0] == rb[0]:
        pairs["UPD"].append((a, b))
    else:
        pairs["DIFF"].append((a, b))
pairs["DIFF"] = rng.sample(pairs["DIFF"], 3000)


def auc(pos, neg):
    """P[pos > neg] (Mann-Whitney U / (n1 n2))."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold_acc(upd, para):
    """Best balanced accuracy of the rule 'UPDATE iff lo<=cos<hi, NOOP iff cos>=hi' for UPD vs PARA:
    since the rule only sees cos, the best it can do is one cut between the two classes."""
    cuts = np.unique(np.concatenate([upd, para]))
    best = (0.5, None)
    for c in cuts:
        # predict PARA if cos >= c else UPD  (PARA expected to be higher)
        acc = 0.5 * ((upd < c).mean() + (para >= c).mean())
        acc2 = 0.5 * ((upd >= c).mean() + (para < c).mean())
        for a in (acc, acc2):
            if a > best[0]:
                best = (a, float(c))
    return best


result = {}
for enc_name, enc in (("v1_json", enc_v1), ("v2_prefixed", enc_v2)):
    E = embed([enc(r[0], r[3]) for r in records])
    cos = {k: np.array([E[a] @ E[b] for a, b in v]) for k, v in pairs.items()}
    rule = {}
    for k, c in cos.items():
        rule[k] = {"insert": float((c < 0.90).mean()), "update": float(((c >= 0.90) & (c < 0.98)).mean()),
                   "ignore": float((c >= 0.98).mean())}
    acc, cut = best_threshold_acc(cos["UPD"], cos["PARA"])
    result[enc_name] = {
        "n_pairs": {k: len(v) for k, v in pairs.items()},
        "cos_stats": {k: {"mean": float(c.mean()), "std": float(c.std()), "p5": float(np.percentile(c, 5)),
                          "p50": float(np.median(c)), "p95": float(np.percentile(c, 95))} for k, c in cos.items()},
        "v1_rule_outcomes": rule,
        "auc_para_vs_upd": float(auc(cos["PARA"], cos["UPD"])),
        "auc_upd_vs_diff": float(auc(cos["UPD"], cos["DIFF"])),
        "best_any_threshold_balanced_acc_upd_vs_para": acc, "best_cut": cut,
    }
    np.savez(os.path.join(OUT, f"geom_{enc_name}.npz"), **{k: v for k, v in cos.items()})

# Exchangeability: does a value-agnostic query prefer the new or the old value?
Q = embed(["search_query: " + slots[s]["probe"] for s in slots])
qidx = {s: i for i, s in enumerate(slots)}
for enc_name, enc in (("v1_json", enc_v1), ("v2_prefixed", enc_v2)):
    E = embed([enc(r[0], r[3]) if enc_name == "v2_prefixed" else enc(r[0], r[3]) for r in records])
    Qe = Q if enc_name == "v2_prefixed" else embed([slots[s]["probe"] for s in slots])
    diffs = []
    for a, b in pairs["UPD"]:
        if records[a][2] != records[b][2]:
            continue  # same template, different value only
        q = Qe[qidx[records[a][0]]]
        diffs.append(float(q @ E[a] - q @ E[b]))
    d = np.abs(np.array(diffs))
    result[enc_name]["value_agnostic_query"] = {
        "n": len(diffs), "mean_abs_gap": float(d.mean()), "median_abs_gap": float(np.median(d)),
        "p_gap_below_0.02": float((d < 0.02).mean()),
        "note": "for a pair (old,new) the query prefers either with prob ~1/2 by symmetry; the gap is tiny"}

json.dump(result, open(os.path.join(OUT, "embed_geometry.json"), "w"), indent=1)
print(json.dumps(result, indent=1))
