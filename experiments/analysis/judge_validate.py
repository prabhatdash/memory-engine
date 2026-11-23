"""Validate the deterministic regex scorer against an independent LLM judge
(qwen3:8b, a larger model than the one under test) on a stratified sample.
Reports raw agreement and Cohen's kappa."""
import json, os, random, urllib.request, re
from collections import Counter

RES = os.path.join(os.path.dirname(__file__), "..", "results")
data = json.load(open(os.path.join(RES, "scored_answers.json")))
rng = random.Random(1)
sample = []
for cond, rows in data.items():
    sample += [dict(r, cond=cond) for r in rng.sample(rows, min(40, len(rows)))]

SYS = "You are a strict grader. Reply with exactly one word: CORRECT or INCORRECT."


def ask(prompt):
    body = {"model": "qwen3:8b", "prompt": prompt, "system": SYS, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 1, "num_predict": 8}}
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["response"]


cache_f = os.path.join(RES, "judge_cache.json")
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
agree, labels = [], []
for r in sample:
    key = f"{r['cond']}|{r['persona']}|{r['probe']}"
    if r["type"] == "abstain":
        crit = ("The user NEVER told the assistant this information. The response is CORRECT only if it says it does not know "
                "(or asks the user); it is INCORRECT if it states or guesses a specific answer.")
    else:
        crit = (f"The correct, current answer is: {r['answer_value']}. The response is CORRECT only if it gives this value "
                f"(paraphrases, units and number formats are fine). It is INCORRECT if it gives a different value, an outdated value, "
                f"or says it does not know.")
    prompt = f"Question asked by the user: {r['q']}\n{crit}\n\nAssistant response:\n{r['answer']}\n\nVerdict:"
    if key not in cache:
        cache[key] = ask(prompt)
    j = "INCORRECT" not in cache[key].upper() and "CORRECT" in cache[key].upper()
    s = r["outcome"] == "correct"
    labels.append((s, j, r["cond"], r["type"]))
json.dump(cache, open(cache_f, "w"))

n = len(labels)
po = sum(a == b for a, b, *_ in labels) / n
ps = sum(a for a, *_ in labels) / n
pj = sum(b for _, b, *_ in labels) / n
pe = ps * pj + (1 - ps) * (1 - pj)
kappa = (po - pe) / (1 - pe)
dis = [(c, t, s, j) for s, j, c, t in labels if s != j]
out = {"n": n, "agreement": po, "cohen_kappa": kappa, "scorer_pos_rate": ps, "judge_pos_rate": pj,
       "disagreements_by_cond": Counter(c for c, *_ in dis), "disagreement_direction": Counter(
        "scorer+ judge-" if s else "scorer- judge+" for *_, s, j in dis)}
json.dump(out, open(os.path.join(RES, "judge_validation.json"), "w"), indent=1, default=dict)
print(json.dumps(out, indent=1, default=dict))
