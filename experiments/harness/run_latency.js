/**
 * Live end-to-end latency experiment (no cache, no stubs, sequential, one
 * condition at a time on an otherwise idle machine).
 *
 *   node harness/run_latency.js <cond> [thinkMs]
 *   cond: v1 | v2 | naive_full
 *
 * For every turn of sessions 1+2 of personas P01..P0n the user message is sent
 * through the real chat() API. We record the user-perceived response time
 * (until chat() resolves) and, from the instrumented Ollama calls, how much of
 * it was spent in embedding, chat generation and analysis.
 */
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const IC = require("./intercept");

const ROOT = path.join(__dirname, "..");
const RES = path.join(ROOT, "results");
const SC = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "scenarios.json"), "utf8")).scenarios;
const MODEL = "qwen3:4b-instruct";
const SYSTEM = "You are a helpful personal assistant. Answer concisely. If you do not know a fact about the user, say that you don't know instead of guessing.";
const BASE = { model: MODEL, provider: "ollama", system_prompt: SYSTEM };
const EMB = { model: "nomic-embed-text", provider: "ollama" };
const N_PERSONAS = Number(process.env.LAT_PERSONAS || 2);
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
    const [cond, thinkArg] = process.argv.slice(2);
    const thinkMs = Number(thinkArg || 0);
    const label = `${cond}${thinkMs ? `_think${thinkMs}` : ""}`;
    const rows = [];
    const bg = [];
    for (const s of SC.slice(0, N_PERSONAS)) {
        const dir = path.join(RES, "tmp", "latency", label, s.id);
        await fsp.rm(dir, { recursive: true, force: true });
        await fsp.mkdir(dir, { recursive: true });
        let eng = null;
        if (cond === "v1") eng = new (require("../engine_v1/index.js"))(BASE, EMB, BASE, dir);
        if (cond === "v2") {
            eng = new (require("../../source_code/index.js"))(BASE, EMB, BASE, dir);
            eng.onConsolidated = (ops, t) => bg.push({ persona: s.id, ...t });
        }
        const history = [];
        const turns = [...s.sessions[0], ...s.sessions[1]];
        for (const [ti, turn] of turns.entries()) {
            const tags = { cond: label, persona: s.id, turn: ti };
            const before = IC.usageLog.length;
            const t0 = performance.now();
            let reply;
            await IC.run({ tags, noCache: true, numPredict: 160, numCtx: cond === "naive_full" ? 16384 : 8192 }, async () => {
                if (cond === "naive_full") {
                    const hist = history.map(t => `User: ${t.u}\nAssistant: ${t.a}`).join("\n\n");
                    const prompt = (hist ? `Conversation so far:\n${hist}\n\n` : "") + `User: ${turn.text}\nAssistant:`;
                    reply = (await IC.generate(prompt, SYSTEM, { model: MODEL })).text;
                } else {
                    reply = await eng.chat(turn.text);
                }
            });
            const responseMs = performance.now() - t0;
            history.push({ u: turn.text, a: reply });
            const calls = IC.usageLog.slice(before).filter(u => u.persona === s.id && u.turn === ti);
            const sum = k => calls.filter(c => c.kind === k).reduce((a, c) => a + c.ms, 0);
            const ptok = calls.filter(c => c.kind === "chat").reduce((a, c) => a + c.promptTokens, 0);
            rows.push({ cond: label, persona: s.id, turn: ti, kind: turn.kind, responseMs,
                        chatMs: sum("chat"), embedMs: sum("embed"), analyzerMsInResponse: sum("analyzer"),
                        chatPromptTokens: ptok, pendingAfter: eng && eng.pendingConsolidations !== undefined ? eng.pendingConsolidations : 0 });
            process.stdout.write(`${label} ${s.id} t${ti} ${responseMs.toFixed(0)}ms\n`);
            if (thinkMs) await sleep(thinkMs);
        }
        if (eng && eng.flush) await eng.flush();
    }
    // background analyzer time for v2 (from all analyzer calls tagged to this condition)
    const analyzerCalls = IC.usageLog.filter(u => u.cond === label && u.kind === "analyzer");
    fs.writeFileSync(path.join(RES, `latency_${label}.json`), JSON.stringify({ rows, bg,
        analyzerCallMs: analyzerCalls.map(c => c.ms) }, null, 1));
    process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
