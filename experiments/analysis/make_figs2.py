"""Second figure/macro set: Shapley attribution, retrieval-generation decomposition,
cost crossover and latency ECDF. Writes final_paper/generated/numbers2.tex and figures/*.pdf."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
PAPER = os.path.join(HERE, "..", "paper_assets")  # tables (.tex) and figures (.pdf) for the paper
GEN = os.path.join(PAPER, "generated"); FIG = os.path.join(PAPER, "figures")
os.makedirs(GEN, exist_ok=True); os.makedirs(FIG, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 1.8, "pdf.fonttype": 42})
J = lambda f: json.load(open(os.path.join(RES, f))) if os.path.exists(os.path.join(RES, f)) else None
M = {}
pct = lambda x, d=1: f"{100 * x:.{d}f}"
sgn = lambda x, d=1: ("0.0" if abs(100 * x) < 0.05 else f"{{+}}{100 * x:.{d}f}" if x >= 0 else f"$-${100 * abs(x):.{d}f}")

sh = J("shapley.json") or {}
score = J("score_report.json") or {}
S = score.get("summary", {})
PCOL = {"P": C["blue"], "C": C["orange"], "R": C["aqua"], "I": C["yellow"]}
PNAME = {"P": "compact prompt", "C": "LLM consolidation", "R": "gated MMR retrieval", "I": "shared infrastructure"}

# ---------------------------------------------------------------- Shapley macros + figure
acc = sh.get("accuracy"); st = sh.get("store")
if acc:
    for name, key in (("overall", ""), ("recall", "Rec"), ("update", "Upd"), ("abstain", "Abs")):
        a = acc[name]
        for i, p in enumerate("PCR"):
            M[f"Phi{p}{key}"] = sgn(a["phi"][i])
            M[f"PhiCI{p}{key}"] = f"[{sgn(a['ci'][i][0])}, {sgn(a['ci'][i][1])}]"
        M[f"VNone{key}"] = pct(a["v"]["000"]); M[f"VAll{key}"] = pct(a["v"]["111"])
        d = a["dividends"]
        M[f"DivPC{key}"] = sgn(d["110"]); M[f"DivPR{key}"] = sgn(d["101"]); M[f"DivCR{key}"] = sgn(d["011"]); M[f"DivPCR{key}"] = sgn(d["111"])
    tot = sum(acc["overall"]["phi"])
    M["PhiShareP"] = f"{100 * acc['overall']['phi'][0] / tot:.0f}"; M["PhiTotal"] = sgn(tot)
if st:
    for k, key in (("contradiction", "Con"), ("coverage", "Cov"), ("noise", "Noise")):
        for i, p in enumerate("PC"):
            v = st[k]["phi"][i]
            M[f"Phi{p}{key}"] = (sgn(v) if k != "noise" else f"{v:+.1f}".replace("-", "$-$").replace("+", "{+}"))
            lo, hi = st[k]["ci"][i]
            M[f"PhiCI{p}{key}"] = (f"[{sgn(lo)}, {sgn(hi)}]" if k != "noise" else f"[{lo:+.1f}, {hi:+.1f}]".replace("-", "$-$").replace("+", "{+}"))
        M[f"VNone{key}"] = pct(st[k]["v"]["00"]) if k != "noise" else f"{st[k]['v']['00']:.1f}"


def fig_shapley():
    if not acc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.15), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    groups = ["overall", "recall", "update", "abstain"]
    w = 0.24
    for i, p in enumerate("PCR"):
        y = np.arange(len(groups)) + (i - 1) * w
        v = [100 * acc[g]["phi"][i] for g in groups]
        lo = [100 * (acc[g]["phi"][i] - acc[g]["ci"][i][0]) for g in groups]
        hi = [100 * (acc[g]["ci"][i][1] - acc[g]["phi"][i]) for g in groups]
        ax.barh(y, v, w * 0.9, color=PCOL[p], label=PNAME[p], xerr=[lo, hi],
                error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 1.5})
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups); ax.invert_yaxis()
    ax.set_xlabel("Shapley value $\\varphi_i$ (accuracy points)")
    ax.set_title("(a) probe accuracy (3 players)", fontsize=8, color=INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    ax = axes[1]
    if st:
        groups = ["coverage", "contradiction"]
        for i, p in enumerate("PC"):
            y = np.arange(len(groups)) + (i - 0.5) * 0.34
            v = [100 * st[g]["phi"][i] for g in groups]
            lo = [100 * (st[g]["phi"][i] - st[g]["ci"][i][0]) for g in groups]
            hi = [100 * (st[g]["ci"][i][1] - st[g]["phi"][i]) for g in groups]
            ax.barh(y, v, 0.3, color=PCOL[p], xerr=[lo, hi], error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 1.5})
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(["fact\ncoverage", "update\ncontradiction"]); ax.invert_yaxis()
        ax.set_xlabel("$\\varphi_i$ (percentage points)")
        ax.set_title("(b) store quality (2 players)", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(FIG, "shapley.pdf")); plt.close(fig)


# ---------------------------------------------------------------- decomposition
dec = sh.get("decomposition", {})
KEYS = {"v2": "VTwo", "naive_full": "NaiveFull", "naive_trunc": "NaiveTrunc", "v2_threshold": "Thr",
        "v2_v1retr": "OldRetr", "v2_v1prompt": "OldPrompt", "fac_000": "FacNone", "v2@8b": "VTwoEight",
        "naive_full@8b": "NaiveFullEight", "rag_turns": "Rag", "rsum": "Rsum", "mem0": "Memz",
        "rag_turns@8b": "RagEight", "rsum@8b": "RsumEight", "mem0@8b": "MemzEight"}
for c, d in dec.items():
    if c in KEYS:
        M[f"Rho{KEYS[c]}"] = f"{d['rho']:.2f}"
        M[f"Alpha{KEYS[c]}"] = f"{d['alpha']:.2f}" if d["alpha"] is not None else "--"
        M[f"Beta{KEYS[c]}"] = f"{d['beta']:.2f}" if d["beta"] is not None else "--"
# exact two-player Shapley split of an accuracy gap under g = rho*alpha (+ residual from beta)
for a_, b_, key in (("naive_full", "v2", "NaiveVTwo"), ("rag_turns", "v2", "RagVTwo"), ("rsum", "v2", "RsumVTwo"),
                    ("mem0", "v2", "MemzVTwo"), ("naive_full@8b", "v2@8b", "NaiveVTwoEight")):
    if a_ in dec and b_ in dec and dec[a_]["alpha"] is not None and dec[b_]["alpha"] is not None:
        r1, r2, a1, a2 = dec[a_]["rho"], dec[b_]["rho"], dec[a_]["alpha"], dec[b_]["alpha"]
        g1 = dec[a_]["acc"]; g2 = dec[b_]["acc"]
        phr = (r2 - r1) * (a1 + a2) / 2; pha = (a2 - a1) * (r1 + r2) / 2
        M[f"GapRho{key}"] = sgn(phr); M[f"GapAlpha{key}"] = sgn(pha)
        M[f"GapRes{key}"] = sgn((g2 - g1) - phr - pha); M[f"Gap{key}"] = sgn(g2 - g1)
        M[f"GapRhoShare{key}"] = f"{100 * phr / (g2 - g1):.0f}" if g2 != g1 else "--"
lg = sh.get("logit")
if lg:
    nm = lg["names"]
    for k, key in (("gold_in_ctx", "Gold"), ("stale_only_in_ctx", "Stale"), ("noise_records", "Noise"), ("top_similarity", "Sim")):
        i = nm.index(k)
        M[f"OR{key}"] = f"{lg['odds_ratio'][i]:.2f}" if lg["odds_ratio"][i] < 100 else f"{lg['odds_ratio'][i]:.0f}"
        M[f"Coef{key}"] = f"{lg['coef'][i]:.2f}".replace("-", "$-$"); M[f"SE{key}"] = f"{lg['se'][i]:.2f}"
    M["LogitRsq"] = f"{lg['mcfadden_r2']:.2f}"; M["LogitN"] = str(lg["n"])


def fig_decomp():
    if not dec:
        return
    retr = score.get("retrieval", {}) if isinstance(score, dict) else {}
    systems = [("naive_full", "Replay (full)"), ("rsum", "Recursive sum."), ("rag_turns", "RAG over turns"),
               ("mem0", "Mem0"), ("v2", "ME (ours)")]
    systems = [(c, l) for c, l in systems if c in dec and dec[c]["alpha"] is not None]
    stale = {c: (1.0 if c.startswith("naive") else retr.get(c, {}).get("update_stale_anywhere_in_ctx")) for c, _ in systems}
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    y = np.arange(len(systems)); h = 0.26
    series = [("$\\rho$: fact in context", [dec[c]["rho"] for c, _ in systems], C["blue"]),
              ("$\\alpha$: correct | fact in context", [dec[c]["alpha"] for c, _ in systems], C["aqua"]),
              ("stale value in context (updates)", [stale[c] if stale[c] is not None else 0 for c, _ in systems], C["orange"])]
    for i, (lab_, v, col) in enumerate(series):
        ax.barh(y + (i - 1) * h, v, h * 0.92, color=col, label=lab_)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in systems]); ax.invert_yaxis()
    ax.set_xlim(0, 1.0); ax.set_xlabel("proportion of recall+update probes")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.38, -0.30), ncol=1, handlelength=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "decomp.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


# ---------------------------------------------------------------- cost crossover + latency ECDF
lat = {k: J(f"latency_{k}.json") for k in ("v1", "v2", "v2_think5000", "naive_full")}
nums = {}
nf = os.path.join(GEN, "numbers.tex")
if os.path.exists(nf):
    import re
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(nf).read()):
        nums[m.group(1)] = m.group(2)


def fig_cost():
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.95))
    ax = axes[0]
    if all(k in nums for k in ("TurnTok", "NaiveIntercept", "ChatTokME", "AnaTokME")):
        cb, s0, pm, am = (float(nums[k].replace(",", "")) for k in ("TurnTok", "NaiveIntercept", "ChatTokME", "AnaTokME"))
        T = np.arange(1, 121)
        naive = s0 * T + cb * T * (T - 1) / 2
        eng = (pm + am) * T
        ax.plot(T, naive / 1e3, color=C["orange"], label="replay (fit, $\\Theta(T^2)$)")
        ax.plot(T, eng / 1e3, color=C["blue"], label="ME incl. analyzer ($\\Theta(T)$)")
        ts = 1 + 2 * (pm + am - s0) / cb
        ax.axvline(ts, color=INK2, linestyle="--", linewidth=0.8)
        ax.text(ts - 3, 0.45 * naive.max() / 1e3, f"$T^*\\approx{ts:.0f}$", fontsize=7, color=INK2, ha="right")
        for k, col in (("naive_full", C["orange"]), ("v2", C["blue"])):
            if lat.get(k):
                rows = lat[k]["rows"]
                for pid in sorted({r["persona"] for r in rows})[:1]:
                    rr = sorted([r for r in rows if r["persona"] == pid], key=lambda r: r["turn"])
                    cum = np.cumsum([r["chatPromptTokens"] for r in rr])
                    if k == "v2":
                        cum = cum + float(nums["AnaTokME"]) * np.arange(1, len(cum) + 1)
                    ax.scatter(np.arange(1, len(cum) + 1)[::3], cum[::3] / 1e3, s=8, color=col, edgecolor="white", linewidth=0.5, zorder=3)
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.30, 1.02), ncol=1, handlelength=1.2)
    ax.set_xlabel("conversation length $T$ (turns)"); ax.set_ylabel("cumulative tokens ($\\times10^3$)")
    ax.set_title("(a) cost crossover (Prop. 1)", fontsize=8, color=INK)
    ax = axes[1]
    for k, col, lab in (("v2", C["blue"], "ME, no pause"),
                        ("v2_think5000", C["aqua"], "ME, 5 s pause"), ("naive_full", C["orange"], "replay")):
        if not lat.get(k):
            continue
        r = np.sort([x["responseMs"] / 1000 for x in lat[k]["rows"]])
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("response time (s, log)"); ax.set_ylabel("ECDF")
    ax.set_title("(b) user-perceived latency", fontsize=8, color=INK)
    ax.set_xlim(1.5, 600)
    ax.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.02, 0.0), ncol=1, handlelength=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cost_latency.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


for f in (fig_shapley, fig_decomp, fig_cost):
    f()
with open(os.path.join(GEN, "numbers2.tex"), "w") as fh:
    fh.write("% auto-generated by experiments/analysis/make_figs2.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fh.write(f"\\providecommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros"); print(M)
