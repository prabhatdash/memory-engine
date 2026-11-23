/**
 * Persona Memory Benchmark (PMB) runner.
 *
 *   node harness/run_pmb.js transcripts      # A: generate fixed assistant replies (stateless model)
 *   node harness/run_pmb.js ingest <cond>    # B: replay sessions 1-2 into an engine's write path
 *   node harness/run_pmb.js probe <cond>     # C: answer probes in a fresh session (read path)
 *
 * Conditions
 *   nomem          stateless model (lower bound)
 *   naive_trunc    full persisted history, most recent turns that fit 4,096 tokens
 *   naive_full     full persisted history, no truncation (accuracy upper bound, linear cost)
 *   v1             original memory-engine-node 1.0.3 (experiments/engine_v1, unmodified)
 *   v2             improved engine (source_code)
 *   v2_threshold   v2 with v1's cosine-threshold merge rule (ablation: no LLM consolidation)
 *   v2_v1retr      v2 store, v1 retrieval settings: top-3, abs 0.55, no MMR/relative gate (ablation)
 */
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const IC = require("./intercept");
const { estimateTokens } = require("../../source_code/utils/UtilityFunctions");

const ROOT = path.join(__dirname, "..");
const DATA = path.join(ROOT, "data");
const RES = path.join(ROOT, "results");
const STORES = path.join(RES, "stores");
const SC = JSON.parse(fs.readFileSync(path.join(DATA, "scenarios.json"), "utf8")).scenarios;
const TRANSCRIPTS = path.join(DATA, "transcripts.json");

const MODEL = process.env.PMB_MODEL || "qwen3:4b-instruct";
const SYSTEM = "You are a helpful personal assistant. Answer concisely. If you do not know a fact about the user, say that you don't know instead of guessing.";
const BASE = { model: MODEL, provider: "ollama", system_prompt: SYSTEM };
const EMB = { model: "nomic-embed-text", provider: "ollama" };
const CONCURRENCY = Number(process.env.PMB_CONC || 4);

async function pool(items, n, fn) {
    const out = new Array(items.length);
    let i = 0;
    await Promise.all(Array.from({ length: n }, async () => {
        while (i < items.length) { const k = i++; out[k] = await fn(items[k], k); }
    }));
    return out;
}

// ------------------------------------------------------------ A: transcripts
async function transcripts() {
    const out = {};
    await pool(SC, CONCURRENCY, async (s) => {
        out[s.id] = [];
        for (const [si, sess] of s.sessions.entries()) {
            for (const turn of sess) {
                const r = await IC.run({ tags: { cond: "transcript", persona: s.id }, numPredict: 160 },
                    () => IC.generate(turn.text, SYSTEM, { model: MODEL }));
                out[s.id].push({ session: si + 1, ...turn, reply: r.text });
            }
        }
        console.log("transcript", s.id, out[s.id].length);
    });
    fs.writeFileSync(TRANSCRIPTS, JSON.stringify(out, null, 1));
}

// ------------------------------------------------------------ engine factory
// Condition names may carry a model suffix, e.g. "v2@8b" = v2 with qwen3:8b as chat+analyzer model.
const MODELS = { "8b": "qwen3:8b" };
const baseCond = (c) => c.split("@")[0];
const condModel = (c) => (c.includes("@") ? MODELS[c.split("@")[1]] : MODEL);
const baseCfg = (c) => ({ ...BASE, model: condModel(c) });

// Generic verbose extraction prompt for the -P ablation: the analyzer prompt of the original
// memory-engine-node 1.0.3 release, read verbatim from experiments/engine_v1 (fetched by run_all.sh).
let _genericPrompt = null;
function genericExtractionPrompt() {
    if (_genericPrompt) return _genericPrompt;
    const f = path.join(ROOT, "engine_v1", "StorageProvider", "LongTermStorage.js");
    if (!fs.existsSync(f)) throw new Error("experiments/engine_v1 not found: run `npm pack memory-engine-node@1.0.3` and unpack it there (run_all.sh does this).");
    return (_genericPrompt = fs.readFileSync(f, "utf8").match(/const system_prompt = `([\s\S]*?)`;/)[1]);
}

function makeEngine(cond, dir) {
    const b = baseCfg(cond);
    const bc = baseCond(cond);
    if (bc === "v1") {
        const V1 = require("../engine_v1/index.js");
        return new V1(b, EMB, b, dir);
    }
    const V2 = require("../../source_code/index.js");
    const LTS = require("../../source_code/StorageProvider/LongTermStorage.js");
    const longTerm = {};
    // full-factorial conditions fac_PCR (1 = v2 component on): P compact prompt, C LLM consolidation, R gated MMR
    const fac = bc.startsWith("fac_") ? bc.slice(4).split("").map(Number) : null;
    if (bc === "v2_threshold" || (fac && !fac[1])) longTerm.consolidation = "threshold";
    if (bc === "v2_v1retr" || (fac && !fac[2])) Object.assign(longTerm, { topK: 3, absThreshold: 0.55, relMargin: 1, mmrLambda: 1 });
    const eng = new V2(b, EMB, b, dir, { longTerm });
    if (bc === "v2_v1prompt" || (fac && !fac[0])) {
        const store = eng.long_term_mem;
        const orig = store.analyzer_model;
        store.analyzer_model = { generate: (prompt) => orig.generate(prompt, genericExtractionPrompt()) };
    }
    return eng;
}

const FAC_STORE = { "11": "v2", "10": "v2_threshold", "01": "v2_v1prompt", "00": "fac_000" };
const storeCond = (c) => {
    const bc = baseCond(c);
    if (bc === "v2_v1retr") return c.replace("v2_v1retr", "v2");
    if (bc.startsWith("fac_")) return FAC_STORE[bc.slice(4, 6)];
    return c;
};

// ------------------------------------------------------------ B: ingestion
async function ingest(cond) {
    const T = JSON.parse(fs.readFileSync(TRANSCRIPTS, "utf8"));
    const opsLog = [];
    await pool(SC, CONCURRENCY, async (s) => {
        const dir = path.join(STORES, cond, s.id);
        await fsp.rm(dir, { recursive: true, force: true });
        await fsp.mkdir(dir, { recursive: true });
        const eng = makeEngine(cond, dir);
        for (const [ti, turn] of T[s.id].entries()) {
            const ex = { user: turn.text, model: turn.reply };
            await IC.run({ tags: { cond, persona: s.id, phase: "ingest", turn: ti } }, async () => {
                const t0 = performance.now();
                let ops = null, analysis;
                if (baseCond(cond) === "v1") {
                    analysis = await eng.long_term_mem.analyze(ex);
                    await eng.long_term_mem.resolveMemory(analysis);
                } else {
                    analysis = await eng.long_term_mem.analyze(ex);
                    ops = await eng.long_term_mem.resolveMemory(analysis);
                }
                opsLog.push({ cond, persona: s.id, turn: ti, session: turn.session, kind: turn.kind, slot: turn.slot,
                              extracted: (analysis.memories || []).map(m => ({ action: m.action, target: m.target, summary: m.summary, store: m.store, importance: m.importance, entity: m.entity })),
                              ops, ms: performance.now() - t0 });
            });
        }
        console.log("ingest", cond, s.id, "done");
    });
    fs.writeFileSync(path.join(RES, `ingest_${cond}.json`), JSON.stringify(opsLog));
    dumpUsage(`usage_ingest_${cond}.json`);
}


// ------------------------------------------------------------ external baselines
// rag_turns : dense retrieval over raw conversation turns (top-5, same embedder) -- RAG-over-history
// rsum      : recursive summarisation memory (Wang et al.): an LLM-maintained running summary, updated every 4 turns
// mem0      : Mem0 (mem0ai 2.2.1, default config); write path runs in baselines/run_mem0.py, retrieved top-5 read here
// All three answer with the SAME chat model, system prompt and prompt template as Memory Engine.
const BASELINES = ["rag_turns", "rsum", "mem0"];
const isBaseline = (c) => BASELINES.includes(baseCond(c));
const MEPROMPT = require("../../source_code/Memory.js").prototype.buildPrompt;
function baselinePrompt(lines, q) {
    const items = lines.map(l => ({ memory: { summary: l.text, since: l.date || "" } }));
    return MEPROMPT.call({}, q, [], items).replace(/^- \[\] /gm, "- ");
}
const cosine = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb); };
let _ve = null;
const vecEngine = () => _ve || (_ve = new (require("../../source_code/StorageProvider/VectorEngine.js"))({ embedding_model_config: EMB }));
const RSUM_SYS = "You maintain the long-term memory of a personal assistant about one user.";
const rsumPrompt = (mem, chunk) => `Previous memory:
${mem || "(empty)"}

New dialogue:
${chunk.map(t => `User: ${t.text}\nAssistant: ${t.reply}`).join("\n\n")}

Rewrite the memory so that it contains every durable fact about the user (identity, preferences, plans, constraints, relationships, decisions) from the previous memory and the new dialogue. Keep exact names, numbers and units. If the user changed a fact, keep only the newest value. Ignore general advice given by the assistant. Output one fact per line starting with "- " and nothing else.`;

async function ingestBaseline(cond) {
    const T = JSON.parse(fs.readFileSync(TRANSCRIPTS, "utf8"));
    const dir = path.join(STORES, cond); fs.mkdirSync(dir, { recursive: true });
    const opsLog = [];
    await pool(SC, CONCURRENCY, async (s) => {
        const turns = T[s.id];
        if (baseCond(cond) === "rag_turns") {
            const docs = [];
            for (const [ti, t] of turns.entries()) {
                const text = `User: ${t.text}\nAssistant: ${t.reply}`;
                docs.push({ turn: ti, text, emb: Array.from(await vecEngine().embedDocument(text)) });
            }
            fs.writeFileSync(path.join(dir, `${s.id}.json`), JSON.stringify(docs));
        } else if (baseCond(cond) === "rsum") {
            let mem = "";
            for (let i = 0; i < turns.length; i += 4) {
                const chunk = turns.slice(i, i + 4);
                await IC.run({ tags: { cond, persona: s.id, phase: "ingest", turn: i }, numPredict: 1500 }, async () => {
                    const t0 = performance.now();
                    const r = await IC.generate(rsumPrompt(mem, chunk), RSUM_SYS, { model: condModel(cond) });
                    mem = r.text.split("\n").filter(l => l.trim().startsWith("-")).join("\n") || mem;
                    opsLog.push({ cond, persona: s.id, turn: i, promptTokens: r.promptTokens, ms: performance.now() - t0 });
                });
            }
            fs.writeFileSync(path.join(dir, `${s.id}.json`), JSON.stringify({ summary: mem }));
        }
        console.log("ingest", cond, s.id, "done");
    });
    fs.writeFileSync(path.join(RES, `ingest_${cond}.json`), JSON.stringify(opsLog));
}

async function baselineContext(cond, pid, pi, q) {
    const bc = baseCond(cond);
    if (bc === "rag_turns") {
        const docs = JSON.parse(fs.readFileSync(path.join(STORES, cond, `${pid}.json`), "utf8"));
        const qe = await vecEngine().embedQuery(q);
        const top = docs.map(d => ({ ...d, sim: cosine(qe, d.emb) })).sort((a, b) => b.sim - a.sim).slice(0, 5)
            .sort((a, b) => a.turn - b.turn);  // chronological order, as in RAG-over-history systems
        return top.map(d => ({ text: d.text, sim: d.sim }));
    }
    if (bc === "rsum") {
        const { summary } = JSON.parse(fs.readFileSync(path.join(STORES, cond, `${pid}.json`), "utf8"));
        return summary.split("\n").map(l => ({ text: l.replace(/^\s*-\s*/, ""), sim: null }));
    }
    if (bc === "mem0") {
        const R = JSON.parse(fs.readFileSync(path.join(RES, `retrieved_${cond}.json`), "utf8"));
        return (R[pid][String(pi)] || []).map(m => ({ text: m.memory, sim: m.score, date: (m.updated_at || m.created_at || "").slice(0, 10) }));
    }
}

// ------------------------------------------------------------ C: probes
function historyPrompt(history, q, budget) {
    let turns = history.map(t => `User: ${t.text}\nAssistant: ${t.reply}`);
    if (budget) {
        let tok = turns.reduce((a, t) => a + estimateTokens(t), 0);
        while (turns.length && tok > budget) tok -= estimateTokens(turns.shift());
    }
    const hist = turns.length ? `Conversation so far:\n${turns.join("\n\n")}\n\n` : "";
    return { prompt: `${hist}User: ${q}\nAssistant:`, keptTurns: turns.length };
}

async function probe(cond) {
    const T = fs.existsSync(TRANSCRIPTS) ? JSON.parse(fs.readFileSync(TRANSCRIPTS, "utf8")) : {};
    const rows = [];
    await pool(SC, CONCURRENCY, async (s) => {
        for (const [pi, p] of s.probes.entries()) {
            const ctx = { tags: { cond, persona: s.id, phase: "probe", probe: pi }, stubAnalyzer: true,
                          numPredict: 150, numCtx: baseCond(cond) === "naive_full" ? 16384 : 8192 };
            const row = await IC.run(ctx, async () => {
                const t0 = performance.now();
                if (baseCond(cond) === "nomem" || cond.startsWith("naive")) {
                    const budget = baseCond(cond) === "naive_trunc" ? 4096 : baseCond(cond) === "naive_full" ? null : 0;
                    const hist = baseCond(cond) === "nomem" ? [] : T[s.id];
                    const { prompt, keptTurns } = historyPrompt(hist, p.q, budget);
                    const r = await IC.generate(prompt, SYSTEM, { model: condModel(cond) });
                    return { answer: r.text, promptTokens: r.promptTokens, keptTurns, ms: performance.now() - t0 };
                }
                if (isBaseline(cond)) {
                    const lines = await baselineContext(cond, s.id, pi, p.q);
                    const r = await IC.generate(baselinePrompt(lines, p.q), SYSTEM, { model: condModel(cond) });
                    return { answer: r.text, promptTokens: r.promptTokens, ms: performance.now() - t0,
                             retrieved: lines.map(l => ({ sim: l.sim, summary: l.text })) };
                }
                // engine conditions: fresh session over a pristine copy of the store
                const src = path.join(STORES, storeCond(cond), s.id);
                const dir = path.join(RES, "tmp", cond, s.id, String(pi));
                await fsp.rm(dir, { recursive: true, force: true });
                await fsp.cp(src, dir, { recursive: true });
                const eng = makeEngine(cond, dir);
                const before = IC.usageLog.length;
                const answer = await eng.chat(p.q);
                if (eng.flush) await eng.flush();
                const calls = IC.usageLog.slice(before).filter(u => u.kind === "chat" && u.persona === s.id && u.probe === pi && u.cond === cond);
                let retrieved = [];
                if (eng.lastTurn) retrieved = eng.lastTurn.retrieved.map(r => ({ sim: r.similarity, summary: r.memory.summary, previously: r.memory.previously }));
                else {
                    const lt = await eng.long_term_mem.RetrieveLongTermMemory(p.q);
                    retrieved = lt.map(r => ({ sim: r.similarity, summary: r.memory.summary }));
                }
                await fsp.rm(dir, { recursive: true, force: true });
                return { answer: answer.replace(/<think>[\s\S]*?<\/think>/gi, "").trim(),
                         promptTokens: calls.reduce((a, c) => a + c.promptTokens, 0), retrieved, ms: performance.now() - t0 };
            });
            rows.push({ cond, persona: s.id, probe: pi, ...p, gold: p.answer, ...row });
        }
        console.log("probe", cond, s.id, "done");
    });
    rows.sort((a, b) => a.persona.localeCompare(b.persona) || a.probe - b.probe);
    fs.writeFileSync(path.join(RES, `probes_${cond}.json`), JSON.stringify(rows, null, 1));
    dumpUsage(`usage_probe_${cond}.json`);
}

function dumpUsage(name) {
    fs.writeFileSync(path.join(RES, name), JSON.stringify(IC.usageLog));
}

(async () => {
    const [phase, cond] = process.argv.slice(2);
    fs.mkdirSync(RES, { recursive: true });
    const t0 = Date.now();
    if (phase === "transcripts") await transcripts();
    else if (phase === "ingest") await (isBaseline(cond) ? ingestBaseline(cond) : ingest(cond));
    else if (phase === "probe") await probe(cond);
    else throw new Error("unknown phase");
    console.log(`${phase} ${cond || ""} finished in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
