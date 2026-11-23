"""Structural quality of the persisted long-term stores after ingestion.

For every engine condition and persona we inspect memory.json directly:
  coverage       stated facts whose *current* value is present in an active record
  duplicates     extra active records carrying the same fact value
  stale_only     updated facts where only the OLD value survives (update lost)
  contradiction  updated facts where OLD and NEW values are both active as separate records
  misattributed  distractor values (facts about other people) stored with entity "user"
  noise          active records that carry none of the persona's fact values
"""
import json, os, sys, re
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from score import value_regex, norm, SC

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def analyse(cond):
    agg = defaultdict(list)
    ops = defaultdict(int)
    f = os.path.join(RES, f"ingest_{cond}.json")
    if os.path.exists(f):
        for r in json.load(open(f)):
            for o in (r.get("ops") or []):
                ops[o["op"]] += 1
            for m in r.get("extracted", []):
                ops["extracted"] += 1
                if m.get("store") is False:
                    ops["extracted_store_false"] += 1
            if r.get("kind") == "filler":
                ops["filler_turns"] += 1
                ops["filler_extracted"] += len(r.get("extracted", []))
    for pid, s in SC.items():
        p = os.path.join(RES, "stores", cond, pid, "memory.json")
        if cond.startswith("mem0"):  # Mem0: final memory list exported by baselines/run_mem0.py
            f0 = os.path.join(RES, f"store_{cond}.json")
            if not os.path.exists(f0):
                continue
            mems = [{"summary": m["memory"], "status": "active"} for m in json.load(open(f0)).get(pid, [])]
        elif cond.startswith("rsum"):  # recursive summary: one fact per line
            f0 = os.path.join(RES, "stores", cond, f"{pid}.json")
            if not os.path.exists(f0):
                continue
            mems = [{"summary": l.lstrip("- ").strip(), "status": "active"} for l in json.load(open(f0))["summary"].split("\n") if l.strip()]
        elif not os.path.exists(p):
            continue
        else:
            mems = json.load(open(p))["memories"]
        active = [m for m in mems if m.get("status", "active") == "active"]
        texts = [norm(m.get("summary", "")) for m in active]
        ents = [norm(str(m.get("entity", ""))) for m in active]
        agg["n_records"].append(len(mems))
        agg["n_active"].append(len(active))
        stated = list(s["values"].items())
        matched_any = [False] * len(active)
        for slot, v in stated:
            cur = value_regex(v["value"])
            hits = [i for i, t in enumerate(texts) if cur.search(t)]
            for i in hits:
                matched_any[i] = True
            agg["coverage"].append(len(hits) > 0)
            agg["duplicates"].append(max(0, len(hits) - 1))
            if "old" in v:
                old = value_regex(v["old"])
                ohits = [i for i, t in enumerate(texts) if old.search(t) and not cur.search(t)]
                for i in ohits:
                    matched_any[i] = True
                agg["stale_only"].append(len(ohits) > 0 and len(hits) == 0)
                agg["contradiction"].append(len(ohits) > 0 and len(hits) > 0)
                agg["update_applied"].append(len(hits) > 0 and len(ohits) == 0)
            if "distractor" in v:
                d = value_regex(v["distractor"])
                dh = [i for i, t in enumerate(texts) if d.search(t)]
                for i in dh:
                    matched_any[i] = True
                # misattributed = the distractor value is stored as if it were the USER's own fact,
                # i.e. the record names neither the other person nor their relation to the user
                REL = re.compile(r"cousin|colleague|friend|neighbou?r|roommate|aunt|nephew|brother|sister-in-law|in-law|marco|arjun|lena|priya|tom\b|rosa|best friend")
                agg["misattributed"].append(any(not REL.search(texts[i]) for i in dh))
                agg["distractor_stored"].append(len(dh) > 0)
        agg["noise"].append(sum(1 for x in matched_any if not x))
    out = {k: float(np.mean(v)) if k not in ("n_records", "n_active", "noise", "duplicates") else
           {"mean": float(np.mean(v)), "sum": int(np.sum(v))} for k, v in agg.items()}
    out["ops"] = dict(ops)
    return out


if __name__ == "__main__":
    rep = {c: analyse(c) for c in ("v1", "v2", "v2_threshold", "v2_v1prompt", "fac_000", "v1@8b", "v2@8b", "rsum", "mem0", "rsum@8b", "mem0@8b") if os.path.isdir(os.path.join(RES, "stores", c))}
    json.dump(rep, open(os.path.join(RES, "store_stats.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
