"""Turn experiment outputs into LaTeX macros, tables and PDF figures for the paper.
Every number in the paper's results section is produced here from results/*.json."""
import json, os, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
PAPER = os.path.join(HERE, "..", "paper_assets")  # tables (.tex) and figures (.pdf) for the paper
GEN = os.path.join(PAPER, "generated"); FIG = os.path.join(PAPER, "figures")
os.makedirs(GEN, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# Validated categorical palette (light mode) + text inks
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "pdf.fonttype": 42})

J = lambda f: json.load(open(os.path.join(RES, f))) if os.path.exists(os.path.join(RES, f)) else None
M = {}  # macros


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def tex_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    return f"{p / 10 ** e:.1f}\\times10^{{{e}}}"


score = J("score_report.json")
store = J("store_stats.json")
judge = J("judge_validation.json")
scan = J("bench_scan.json")
lat = {k: J(f"latency_{k}.json") for k in ("v1", "v2", "v2_think5000", "naive_full")}

NAMES = {"nomem": "No memory", "naive_trunc": "Replay, 4k budget", "naive_full": "Replay, full history",
         "rag_turns": "RAG over turns", "rsum": "Recursive summary~\\cite{wang2025recursive}", "mem0": "Mem0~\\cite{chhikara2025mem0}",
         "v2": "\\ME{} (ours)", "v2_threshold": "\\quad $-$C: cosine-threshold merge",
         "v2_v1retr": "\\quad $-$R: fixed top-3 retrieval",
         "v2_v1prompt": "\\quad $-$P: generic extraction prompt",
         "naive_full@8b": "Replay, full history", "rag_turns@8b": "RAG over turns", "rsum@8b": "Recursive summary",
         "mem0@8b": "Mem0", "v2@8b": "\\ME{} (ours)"}
ORDER = ["nomem", "naive_trunc", "naive_full", "rag_turns", "rsum", "mem0", "v2", "v2_threshold", "v2_v1retr", "v2_v1prompt"]
KEYS = {"v2": "VTwo", "naive_full": "NaiveFull", "naive_trunc": "NaiveTrunc", "nomem": "NoMem", "rag_turns": "Rag",
        "rsum": "Rsum", "mem0": "Memz", "v2_threshold": "Thr", "v2_v1retr": "OldRetr", "v2_v1prompt": "OldPrompt",
        "naive_full@8b": "NaiveFullEight", "rag_turns@8b": "RagEight", "rsum@8b": "RsumEight", "mem0@8b": "MemzEight",
        "v2@8b": "VTwoEight", "v1": "VOne", "v1@8b": "VOneEight"}

# ------------------------------------------------------------- main table
if score:
    S = score["summary"]
    rows = []
    for c in ORDER:
        if c not in S:
            continue
        s = S[c]
        upd_n = s["update"]["n"]
        stale = s["update"]["outcomes"].get("stale", 0) / upd_n if upd_n else 0
        conf = sum(s[t]["outcomes"].get("confused", 0) for t in ("recall",)) / max(1, s["recall"]["n"])
        ov = s["overall"]
        if c == "naive_full@8b":
            rows.append("\\midrule\n\\multicolumn{7}{@{}l}{\\emph{Model-scale check: Qwen3-8B as chat and analyzer model}} \\\\")
        if c == "v2":
            rows.append("\\midrule")
        if c == "v2_threshold":
            rows.append("\\multicolumn{7}{@{}l}{\\emph{Ablations of \\ME}} \\\\")
        best = c in ("v2", "v2@8b") and S[c]["overall"]["acc"] >= max(S[x]["overall"]["acc"] for x in ORDER if x in S and (("@8b" in x) == ("@8b" in c)))
        b = (lambda x: f"\\textbf{{{x}}}") if best else (lambda x: x)
        rows.append(f"{NAMES[c]} & {pct(s['recall']['acc'])} & {pct(s['update']['acc'])} & {pct(stale)} & "
                    f"{pct(s['abstain']['acc'])} & {b(pct(ov['acc']))} {{\\scriptsize[{pct(ov['ci'][0],0)},{pct(ov['ci'][1],0)}]}} & "
                    f"{s['prompt_tokens_mean']:,.0f} \\\\" + "")
        if c == "v2_v1prompt" or (c == "v2_v1retr" and "v2_v1prompt" not in S):
            pass
    t = r"""\begin{table}[t]
\caption{Probe accuracy (\%) on the Persona Memory Benchmark (12 personas, 60 recall, 36 update and 24 abstention probes). \emph{Stale}: share of update probes answered with the outdated value. Overall accuracy is given with a 95\% Wilson interval. \emph{Tokens}: mean chat-model prompt tokens per probe (Ollama tokenizer counts).}\label{tab:main}
\centering\scriptsize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
System & Recall & Update & Stale$\downarrow$ & Abstain & Overall [95\% CI] & Tokens \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(GEN, "table_main.tex"), "w").write(t)
    for c, key in KEYS.items():
        if c in S:
            M[f"Acc{key}"] = pct(S[c]["overall"]["acc"])
            M[f"Rec{key}"] = pct(S[c]["recall"]["acc"])
            M[f"Upd{key}"] = pct(S[c]["update"]["acc"])
            M[f"Abs{key}"] = pct(S[c]["abstain"]["acc"])
            M[f"Tok{key}"] = f"{S[c]['prompt_tokens_mean']:,.0f}"
            M[f"Stale{key}"] = pct(S[c]["update"]["outcomes"].get("stale", 0) / max(1, S[c]["update"]["n"]))
    T = score["tests"]
    for c, key in KEYS.items():
        if c in ("v2", "v2@8b"):
            continue
        k = f"v2@8b_vs_{c}" if "@" in c else f"v2_vs_{c}"
        if k in T:
            d = T[k]["acc_diff"]
            M[f"DiffVTwo{key}"] = f"{'+' if d >= 0 else '$-$'}{abs(100 * d):.1f}"
            M[f"PvTwo{key}"] = tex_p(T[k]["mcnemar_p"])
            lo, hi = T[k]["diff_ci95_cluster_bootstrap"]
            M[f"CIVTwo{key}"] = f"[{100 * lo:+.1f}, {100 * hi:+.1f}]".replace("+", "{+}").replace("-", "$-$")
            if "update" in T[k]: M[f"UpdPvTwo{key}"] = tex_p(T[k]["update"]["p"])
            M[f"DiscVTwo{key}"] = f"{T[k]['v2_only_correct']}/{T[k]['other_only_correct']}"
    if "v2" in S and "naive_full" in S:
        M["TokSaveVsFull"] = pct(1 - S["v2"]["prompt_tokens_mean"] / S["naive_full"]["prompt_tokens_mean"], 0)
    R = score.get("retrieval", {})
    for c, key in (("v1", "VOne"), ("v2", "VTwo"), ("v2_v1retr", "OldRetr"), ("v2_threshold", "Thr")):
        if c in R:
            M[f"GoldCtxRec{key}"] = pct(R[c].get("recall_gold_in_ctx", 0))
            M[f"GoldCtxUpd{key}"] = pct(R[c].get("update_gold_in_ctx", 0))
            M[f"StaleCtx{key}"] = pct(R[c].get("update_stale_as_current_in_ctx", 0))
    # outcome breakdown table
    rows = []
    for c in ORDER:
        if c not in S:
            continue
        o = {}
        for t_ in ("recall", "update"):
            for k_, v_ in S[c][t_]["outcomes"].items():
                o[k_] = o.get(k_, 0) + v_
        n = S[c]["recall"]["n"] + S[c]["update"]["n"]
        rows.append(f"{NAMES[c]} & " + " & ".join(pct(o.get(k_, 0) / n) for k_ in ("correct", "stale", "confused", "abstained", "wrong"))
                    + f" & {pct(S[c]['abstain']['outcomes'].get('hallucinated', 0) / max(1, S[c]['abstain']['n']))} \\\\")
    open(os.path.join(GEN, "table_errors.tex"), "w").write(r"""\begin{table}[t]
\caption{Error anatomy (\%). Columns 2--6 cover the 96 recall+update probes: \emph{stale} = old value given, \emph{confused} = a third party's value given, \emph{abstained} = said it did not know, \emph{wrong} = any other value. The last column is the hallucination rate on the 24 never-stated facts.}\label{tab:errors}
\centering\scriptsize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
System & Correct & Stale & Confused & Abstained & Wrong & Halluc. \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------- model-scale table
if score and "v2@8b" in score["summary"]:
    S = score["summary"]
    rows = []
    for size, sfx in (("4B", ""), ("8B", "@8b")):
        for c, nm in (("naive_full", "Naive replay, full"), ("v1", "\\vone"), ("v2", "\\vtwo")):
            k = c + sfx
            if k not in S:
                continue
            s_ = S[k]
            rows.append(f"{size} & {nm} & {pct(s_['recall']['acc'])} & {pct(s_['update']['acc'])} & {pct(s_['abstain']['acc'])} & "
                        f"{pct(s_['overall']['acc'])} {{\\scriptsize[{pct(s_['overall']['ci'][0],0)},{pct(s_['overall']['ci'][1],0)}]}} \\\\")
        rows.append("\\midrule")
    open(os.path.join(GEN, "table_scale.tex"), "w").write(r"""\begin{table}[t]
\caption{Model-scale check: the same benchmark with Qwen3-8B as chat and analyzer model (embedding model unchanged).}\label{tab:scale}
\centering\scriptsize
\begin{tabular}{@{}llrrrr@{}}
\toprule
Model & System & Recall & Update & Abstain & Overall [95\% CI] \\
\midrule
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------- store table
if store:
    rows = []
    for c, nm in (("v2", "\\ME{} (ours)"), ("mem0", "Mem0"), ("rsum", "Recursive summary"),
                  ("v2_threshold", "\\ME{} $-$C"), ("v2_v1prompt", "\\ME{} $-$P"), ("fac_000", "\\ME{} $-$P$-$C$-$R"),
                  ("v2@8b", "\\ME{} (8B)")):
        if c not in store or "n_active" not in store[c]:
            continue
        s = store[c]
        if c in ("v2_threshold",):
            rows.append("\\midrule")
        rows.append(f"{nm} & {s['n_active']['mean']:.1f} & {pct(s['coverage'])} & {pct(s['update_applied'])} & "
                    f"{pct(s['stale_only'])} & {pct(s['contradiction'])} & {s['duplicates']['mean']:.2f} & "
                    f"{pct(s['misattributed'])} & {s['noise']['mean']:.1f} \\\\")
        key = {"v2": "VTwo", "v2_threshold": "Thr", "v2_v1prompt": "OldPrompt", "fac_000": "FacNone", "v2@8b": "VTwoEight", "mem0": "Memz", "rsum": "Rsum"}[c]
        M[f"Cov{key}"] = pct(s["coverage"]); M[f"UpdApp{key}"] = pct(s["update_applied"])
        M[f"Contra{key}"] = pct(s["contradiction"]); M[f"StaleOnly{key}"] = pct(s["stale_only"])
        M[f"Noise{key}"] = f"{s['noise']['mean']:.1f}"; M[f"Active{key}"] = f"{s['n_active']['mean']:.1f}"
        M[f"Misattr{key}"] = pct(s["misattributed"]); M[f"Dup{key}"] = f"{s['duplicates']['mean']:.2f}"
        ops = s.get("ops", {})
        if ops.get("filler_turns"):
            M[f"FillerExtr{key}"] = f"{ops.get('filler_extracted', 0) / ops['filler_turns']:.2f}"
        M[f"StoreFalse{key}"] = str(ops.get("extracted_store_false", 0))
    open(os.path.join(GEN, "table_store.tex"), "w").write(r"""\begin{table}[t]
\caption{Quality of the persisted long-term store after sessions 1--2, averaged over 12 personas. \emph{Active}: active records per persona. \emph{Coverage}: stated facts whose current value is stored. \emph{Upd.\ ok}: updated facts with only the new value active. \emph{Lost}: only the old value survives. \emph{Contra.}: old and new values both active. \emph{Dup.}: extra records per fact. \emph{Misattr.}: a third party's fact stored as the user's. \emph{Noise}: records unrelated to any persona fact.}\label{tab:store}
\centering\scriptsize
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Store & Active & Coverage & Upd.\ ok & Lost & Contra. & Dup. & Misattr. & Noise \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------- tokens / cost
def usage(f):
    u = J(f)
    return u or []

ing = {c: usage(f"usage_ingest_{c}.json") for c in ("v1", "v2", "v2_threshold")}
for c, key in (("v1", "VOne"), ("v2", "VTwo")):
    a = [x for x in ing[c] if x.get("kind") == "analyzer"]
    turns = len({(x["persona"], x["turn"]) for x in a}) or 1
    if a:
        M[f"AnaPrompt{key}"] = f"{sum(x['promptTokens'] for x in a) / turns:.0f}"
        M[f"AnaOut{key}"] = f"{sum(x['evalTokens'] for x in a) / turns:.0f}"
        M[f"AnaTok{key}"] = f"{sum(x['promptTokens'] + x['evalTokens'] for x in a) / turns:.0f}"
        emb = [x for x in ing[c] if x.get("kind") == "embed"]
        M[f"EmbPerTurn{key}"] = f"{len(emb) / turns:.2f}"

# ------------------------------------------------------------- latency + Prop 1 fit
lat_rows = []
LBL = {"v1": "\\vone{} (consolidation on path)", "v2": "\\vtwo{}, back-to-back turns", "v2_think5000": "\\vtwo{}, 5\\,s user think time",
       "naive_full": "Naive replay, full"}
for k, L in lat.items():
    if not L:
        continue
    r = np.array([x["responseMs"] for x in L["rows"]]) / 1000
    ana = np.array([x["analyzerMsInResponse"] for x in L["rows"]]) / 1000
    chat = np.array([x["chatMs"] for x in L["rows"]]) / 1000
    bgm = np.array([b["analyzeMs"] + b["consolidateMs"] for b in L.get("bg", [])]) / 1000 if L.get("bg") else np.array([])
    boot = [np.mean(np.random.default_rng(i).choice(r, len(r))) for i in range(2000)]
    lat_rows.append(f"{LBL[k]} & {r.mean():.2f} {{\\scriptsize[{np.percentile(boot, 2.5):.2f},{np.percentile(boot, 97.5):.2f}]}} & {np.median(r):.2f} & "
                    f"{np.percentile(r, 95):.2f} & {chat.mean():.2f} & {ana.mean():.2f} & "
                    f"{(f'{bgm.mean():.2f}' if len(bgm) else '--')} \\\\")
    key = {"v1": "VOne", "v2": "VTwo", "v2_think5000": "VTwoThink", "naive_full": "Naive"}[k]
    M[f"Lat{key}"] = f"{r.mean():.2f}"
    M[f"LatMed{key}"] = f"{np.median(r):.2f}"
    M[f"LatP{key}"] = f"{np.percentile(r, 95):.2f}"
    M[f"LatAna{key}"] = f"{ana.mean():.2f}"
    if len(bgm):
        M[f"Bg{key}"] = f"{bgm.mean():.2f}"
    if k.startswith("v2"):
        M[f"MaxPending{key}"] = str(max(x["pendingAfter"] for x in L["rows"]))
    M[f"ChatPromptLat{key}"] = f"{np.mean([x['chatPromptTokens'] for x in L['rows']]):.0f}"
if lat_rows:
    open(os.path.join(GEN, "table_latency.tex"), "w").write(r"""\begin{table}[t]
\caption{Live, uncached user-perceived response time per turn in seconds (2 personas $\times$ 37 turns per system, run sequentially on an otherwise idle Apple M4). The mean is given with a 95\% bootstrap interval. \emph{Chat}: time spent in the chat-model call. \emph{Analyzer on path}: analyzer time inside the response. \emph{Background}: analysis plus consolidation time per turn moved off the path.}\label{tab:latency}
\centering\scriptsize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
System & Mean [95\% CI] & Median & p95 & Chat & Analyzer on path & Background \\
\midrule
""" + "\n".join(lat_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    if lat["v1"] and lat["v2"]:
        M["LatSpeedup"] = f"{np.mean([x['responseMs'] for x in lat['v1']['rows']]) / np.mean([x['responseMs'] for x in lat['v2']['rows']]):.1f}"
        M["LatReduction"] = pct(1 - np.mean([x['responseMs'] for x in lat['v2']['rows']]) / np.mean([x['responseMs'] for x in lat['v1']['rows']]), 0)

# Proposition 1: linear fit of naive prompt tokens vs turn index
if lat.get("naive_full"):
    xs, ys = [], []
    for x in lat["naive_full"]["rows"]:
        xs.append(x["turn"]); ys.append(x["chatPromptTokens"])
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    s0, cbar = coef
    M["TurnTok"] = f"{cbar:.0f}"; M["NaiveIntercept"] = f"{s0:.0f}"; M["FitRsq"] = f"{r2:.3f}"
    M["Horizon"] = f"{4096 / cbar:.0f}"
    if lat.get("v2") and "AnaTokVTwo" in M:
        pm = np.mean([x["chatPromptTokens"] for x in lat["v2"]["rows"]])
        am = float(M["AnaTokVTwo"])
        M["ChatTokME"] = f"{pm:.0f}"; M["AnaTokME"] = f"{am:.0f}"
        M["Tstar"] = f"{1 + 2 * (pm + am - s0) / cbar:.0f}"
        # slope of v2 prompt tokens vs turn (should be ~0 once WM saturates)
        x2 = np.array([x["turn"] for x in lat["v2"]["rows"]]); y2 = np.array([x["chatPromptTokens"] for x in lat["v2"]["rows"]])
        M["MESlope"] = f"{np.polyfit(x2[x2 >= 15], y2[x2 >= 15], 1)[0]:.1f}" if (x2 >= 15).any() else "--"
        M["NaiveFinalTok"] = f"{ys[xs == xs.max()].mean():,.0f}"
        M["MEFinalTok"] = f"{y2[x2 == x2.max()].mean():,.0f}"

# WM window in turns (engine's own token estimator)
tr = J(os.path.join("..", "data", "transcripts.json"))
if tr:
    est = lambda s: math.ceil(len(s.split()) * 1.3) if s else 0
    per = [est(t["text"]) + est(t["reply"]) for v in tr.values() for t in v]
    M["WMturns"] = f"{2000 / np.mean(per):.0f}"
    M["TurnTokEst"] = f"{np.mean(per):.0f}"
if "MaxPendingVTwo" in M:
    M["MaxLag"] = M["MaxPendingVTwo"]

# ------------------------------------------------------------- judge / scan
if judge:
    M["Kappa"] = f"{judge['cohen_kappa']:.2f}"; M["Agree"] = pct(judge["agreement"]); M["JudgeN"] = str(judge["n"])
if scan:
    for r in scan:
        if r["N"] in (1000, 10000, 100000):
            n = {1000: "OneK", 10000: "TenK", 100000: "HundredK"}[r["N"]]
            M[f"ScanVOne{n}"] = f"{r['v1_ms']:.1f}"; M[f"ScanVTwo{n}"] = f"{r['v2_ms']:.1f}"
            if r.get("v1_read_parse_ms"):
                M[f"ParseVOne{n}"] = f"{r['v1_read_parse_ms']:.0f}"

# ------------------------------------------------------------- figures
def fig_tokens():
    if not (lat.get("naive_full") and lat.get("v2")):
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for k, col, lab in (("naive_full", C["orange"], "Naive replay (full)"), ("v1", C["yellow"], "ME-v1"), ("v2", C["blue"], "ME-v2")):
        if not lat.get(k):
            continue
        rows = lat[k]["rows"]
        t = sorted({x["turn"] for x in rows})
        y = [np.mean([x["chatPromptTokens"] for x in rows if x["turn"] == i]) for i in t]
        ax.plot(np.array(t) + 1, y, color=col, label=lab)
        ax.annotate(lab, (t[-1] + 1, y[-1]), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Turn"); ax.set_ylabel("Chat prompt tokens")
    ax.set_xlim(1, 50); ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tokens_per_turn.pdf")); plt.close(fig)


def fig_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.75), sharey=True)
    for ax, enc, title in zip(axes, ("v1_json", "v2_prefixed"), ("JSON-blob encoding", "ME encoding (prefixed plain text)")):
        f = os.path.join(RES, f"geom_{enc}.npz")
        if not os.path.exists(f):
            return
        z = np.load(f)
        bins = np.linspace(0.4, 1.0, 61)
        for k, col, lab in (("DIFF", C["aqua"], "different fact"), ("UPD", C["orange"], "value update"), ("PARA", C["blue"], "paraphrase")):
            ax.hist(z[k], bins=bins, density=True, histtype="stepfilled", alpha=0.65, facecolor=col, edgecolor=col, linewidth=1.4, label=lab)
        for th in (0.90, 0.98):
            ax.axvline(th, color=INK, linewidth=0.9, linestyle="--")
        ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("cosine similarity")
    axes[0].set_ylabel("density")
    axes[1].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "geometry.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def fig_scan():
    if not scan:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    N = [r["N"] for r in scan]
    ax.loglog(N, [r["v1_ms"] for r in scan], color=C["yellow"], marker="o", markersize=4, label="v1 scan")
    ax.loglog(N, [r["v2_ms"] for r in scan], color=C["blue"], marker="o", markersize=4, label="v2 scan + MMR")
    pp = [(r["N"], r["v1_read_parse_ms"]) for r in scan if r.get("v1_read_parse_ms")]
    ax.loglog([a for a, _ in pp], [b for _, b in pp], color=C["orange"], marker="s", markersize=4, label="v1 per-turn file read+parse")
    fails = [r["N"] for r in scan if not r.get("v1_persist_ok", True)]
    if fails:
        ax.axvline(17983, color=INK2, linestyle="--", linewidth=0.8)
        ax.text(17983 * 1.1, ax.get_ylim()[0] * 3, "v1 store\ncannot persist", fontsize=6.5, color=INK2)
    ax.set_xlabel("records in store (N)"); ax.set_ylabel("ms per query")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scan.pdf")); plt.close(fig)


def fig_accuracy():
    if not score:
        return
    S = score["summary"]
    conds = [c for c in ("naive_full", "rsum", "rag_turns", "mem0", "v2") if c in S]
    lab = {"naive_full": "Replay", "rsum": "RecSum", "rag_turns": "RAG", "mem0": "Mem0", "v2": "ME"}
    types = ("recall", "update", "abstain")
    cols = (C["blue"], C["orange"], C["aqua"])
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    w = 0.26
    x = np.arange(len(conds))
    for i, (t, col) in enumerate(zip(types, cols)):
        v = [100 * S[c][t]["acc"] for c in conds]
        lo = [max(0.0, 100 * (S[c][t]["acc"] - S[c][t]["ci"][0])) for c in conds]
        hi = [max(0.0, 100 * (S[c][t]["ci"][1] - S[c][t]["acc"])) for c in conds]
        ax.bar(x + (i - 1) * w, v, w * 0.92, color=col, label=t, yerr=[lo, hi], error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 1.5})
    ax.set_xticks(x); ax.set_xticklabels([lab[c] for c in conds])
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "accuracy.pdf")); plt.close(fig)


for f in (fig_tokens, fig_geometry, fig_scan, fig_accuracy):
    f()

# fact-bearing turns for which the analyzer extracted nothing (parse failure or omission)
for c, key in (("v1", "VOne"), ("v2", "VTwo"), ("v2_threshold", "Thr"), ("v2_v1prompt", "OldPrompt"), ("v1@8b", "VOneEight"), ("v2@8b", "VTwoEight")):
    L = J(f"ingest_{c}.json")
    if not L:
        continue
    for kind, kk in (("fact", "Fact"), ("update", "Update")):
        rows_ = [r for r in L if r["kind"] == kind]
        empty = sum(1 for r in rows_ if not r["extracted"])
        M[f"Empty{kk}{key}"] = f"{empty}/{len(rows_)}"
        M[f"EmptyPct{kk}{key}"] = pct(empty / len(rows_), 0)

# stale value anywhere in the context on update probes, per system
if score:
    for c, key in KEYS.items():
        r = score.get("retrieval", {}).get(c)
        if r:
            M[f"StaleAny{key}"] = pct(r["update_stale_anywhere_in_ctx"], 0)
            M[f"GoldCtxRec{key}"] = pct(r["recall_gold_in_ctx"]); M[f"NRet{key}"] = f"{r['n_retrieved']:.1f}"
    M["StaleAnyNaiveFull"] = "100"

with open(os.path.join(GEN, "numbers.tex"), "w") as fh:
    fh.write("% auto-generated by experiments/analysis/make_paper_assets.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(M)} macros"); print(json.dumps(M, indent=0)[:3000])
