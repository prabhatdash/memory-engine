/**
 * Transparent instrumentation layer for Ollama HTTP calls.
 *
 * Both engine versions talk to Ollama through global fetch, so wrapping fetch
 * lets the harness run the *unmodified* engine code while
 *   - enforcing identical decoding settings (temperature 0, fixed seed, no thinking),
 *   - recording true tokenizer counts (prompt_eval_count / eval_count) and latency,
 *   - caching responses on disk so every experiment is exactly re-playable,
 *   - optionally stubbing analyzer calls (used during read-only probing).
 * Per-call context (condition, phase, persona) is carried with AsyncLocalStorage
 * so concurrent personas do not mix their accounting.
 */
const { AsyncLocalStorage } = require("node:async_hooks");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const als = new AsyncLocalStorage();
const realFetch = globalThis.fetch;

const CACHE_FILE = path.join(__dirname, "..", "data", "llm_cache.jsonl");
const cache = new Map();
if (fs.existsSync(CACHE_FILE)) {
    for (const line of fs.readFileSync(CACHE_FILE, "utf8").split("\n")) {
        if (!line.trim()) continue;
        try { const r = JSON.parse(line); cache.set(r.k, r.v); } catch {}
    }
}
fs.mkdirSync(path.dirname(CACHE_FILE), { recursive: true });
const cacheStream = fs.createWriteStream(CACHE_FILE, { flags: "a" });

const usageLog = [];
const settings = { useCache: true, seed: 42 };

function isAnalyzer(body) {
    return typeof body.system === "string" && body.system.includes("You are the Memory Analyzer");
}

globalThis.fetch = async function (url, init = {}) {
    const u = String(url);
    if (!u.includes(":11434/api/")) return realFetch(url, init);
    const ctx = als.getStore() || {};
    let body = JSON.parse(init.body);
    const kind = u.endsWith("/api/embed") ? "embed" : "generate";

    if (kind === "generate") {
        const analyzer = isAnalyzer(body);
        if (analyzer && ctx.stubAnalyzer) {
            usageLog.push({ ...ctx.tags, kind: "analyzer-stub", promptTokens: 0, evalTokens: 0, ms: 0 });
            return new Response(JSON.stringify({ response: '{"memories":[]}' }), { status: 200 });
        }
        body = {
            ...body,
            think: false,
            options: {
                ...(body.options || {}),
                temperature: 0,
                seed: settings.seed,
                num_ctx: ctx.numCtx || 8192,
                num_predict: analyzer ? 700 : (ctx.numPredict || 200)
            }
        };
        body._role = analyzer ? "analyzer" : "chat";
    }

    const key = crypto.createHash("sha256").update(kind + JSON.stringify(body)).digest("hex");
    let rec = settings.useCache && !ctx.noCache ? cache.get(key) : undefined;
    let cached = true;
    if (!rec) {
        cached = false;
        const sendBody = { ...body };
        delete sendBody._role;
        const t0 = performance.now();
        // retry transient failures (model swapping / queue timeouts under memory pressure)
        let res;
        for (let attempt = 1; ; attempt++) {
            try {
                res = await realFetch(url, { ...init, body: JSON.stringify(sendBody) });
                if (res.ok || attempt >= 4 || res.status < 500) break;
            } catch (e) {
                if (attempt >= 4) throw e;
            }
            await new Promise(r => setTimeout(r, 5000 * attempt));
        }
        const ms = performance.now() - t0;
        if (!res.ok) return res;
        const data = await res.json();
        rec = kind === "embed"
            ? { embeddings: data.embeddings, ms }
            : { response: data.response, promptTokens: data.prompt_eval_count, evalTokens: data.eval_count, ms };
        cache.set(key, rec);
        cacheStream.write(JSON.stringify({ k: key, v: rec }) + "\n");
    }
    usageLog.push({
        ...ctx.tags, kind: kind === "embed" ? "embed" : body._role,
        promptTokens: rec.promptTokens || 0, evalTokens: rec.evalTokens || 0, ms: rec.ms, cached
    });
    const payload = kind === "embed" ? { embeddings: rec.embeddings } : { response: rec.response,
        prompt_eval_count: rec.promptTokens, eval_count: rec.evalTokens };
    return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
};

/** Direct generate call (used for baselines that are not the engine). */
async function generate(prompt, system, ctxOverrides = {}) {
    const res = await globalThis.fetch("http://localhost:11434/api/generate", {
        method: "POST",
        body: JSON.stringify({ model: ctxOverrides.model || "qwen3:4b-instruct", prompt, system, stream: false })
    });
    const d = await res.json();
    return { text: (d.response || "").replace(/<think>[\s\S]*?<\/think>/gi, "").trim(),
             promptTokens: d.prompt_eval_count, evalTokens: d.eval_count };
}

function run(ctx, fn) { return als.run(ctx, fn); }

module.exports = { run, generate, usageLog, settings, als };
