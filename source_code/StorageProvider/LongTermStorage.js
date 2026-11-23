const LLMProvider = require("../LLMProviders/LLMProvider");
const { extractJSON } = require("../utils/UtilityFunctions");
const VectorEngine = require("./VectorEngine");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const crypto = require("crypto");

/**
 * Long-term memory store (v2).
 *
 * Differences from v1 (see paper, Sec. "Root-cause analysis"):
 *  1. Contradiction-aware consolidation: the analyzer is shown the existing
 *     memories most related to the current exchange and must label every
 *     extracted fact as ADD / UPDATE(target) / NOOP(target). A pure cosine
 *     threshold cannot separate "same fact, new value" from "same fact,
 *     paraphrased" (both have near-identical embeddings), so the decision is
 *     delegated to the model that can read the values.
 *  2. Temporal versioning: an UPDATE never overwrites. The old record is
 *     closed (validTo, status="superseded") and linked to its successor, so
 *     only the current value is indexed for retrieval while history is kept.
 *  3. Hygiene: honours the analyzer's store flag and an importance floor,
 *     de-duplicates within a batch (index is updated immediately), and
 *     embeds a plain-text fact instead of a JSON blob.
 *  4. The store is held in memory and persisted atomically (tmp + rename)
 *     instead of being re-read and re-parsed on every operation.
 */
class LongTermStorage {

    constructor({
        storagePath,
        analyzer_model_config,
        embedding_model_config,
        options = {}
    }) {
        this.analyzer_model = new LLMProvider({
            model: analyzer_model_config.model,
            provider: analyzer_model_config.provider,
            apiKey: analyzer_model_config.apiKey,
            baseURL: analyzer_model_config.baseURL,
            options: analyzer_model_config.options,
            think: analyzer_model_config.think
        });

        this.vector_engine = new VectorEngine({ embedding_model_config });

        this.opts = {
            consolidation: "llm",       // "llm" | "threshold"
            versioning: true,           // false => in-place overwrite (v1 behaviour)
            minImportance: 0.3,
            duplicateThreshold: 0.92,   // safety net for exact restatements
            updateThreshold: 0.90,      // used only by consolidation="threshold"
            neighbourCount: 6,
            topK: 5,
            absThreshold: 0.35,
            relMargin: 0.2,
            mmrLambda: 0.75,
            ...options
        };

        this.storagePath = storagePath || path.join(process.cwd(), ".memory-engine");
        if (!fs.existsSync(this.storagePath)) {
            fs.mkdirSync(this.storagePath, { recursive: true });
        }
        this.memoryFile = path.join(this.storagePath, "memory.json");
        this.data = this.#load();
    }

    // ------------------------------------------------------------------ io

    #load() {
        if (!fs.existsSync(this.memoryFile)) return { version: 2, memories: [] };
        try {
            const d = JSON.parse(fs.readFileSync(this.memoryFile, "utf8"));
            d.memories = (d.memories || []).map(m => {
                const rec = {
                    status: "active", validTo: null, supersedes: null, supersededBy: null,
                    validFrom: m.createdAt, ...m
                };
                if (typeof rec.embedding === "string") rec.embedding = LongTermStorage.decodeVector(rec.embedding);
                return rec;
            });
            d.version = 2;
            return d;
        } catch (e) {
            console.warn("[MemoryEngine] Corrupted memory file; starting with an empty store.");
            return { version: 2, memories: [] };
        }
    }

    /**
     * Embeddings are stored as base64-packed Float32 (4 KB per 768-d vector
     * instead of ~16-30 KB of decimal text). Float32 rounding changes cosine
     * scores by < 1e-6. Files written by v1 (plain arrays) still load.
     */
    static encodeVector(v) {
        return Buffer.from(new Float32Array(v).buffer).toString("base64");
    }

    static decodeVector(s) {
        const b = Buffer.from(s, "base64");
        return Array.from(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4));
    }

    async persist() {
        const tmp = this.memoryFile + ".tmp";
        const out = {
            ...this.data,
            memories: this.data.memories.map(m => ({ ...m, embedding: LongTermStorage.encodeVector(m.embedding) }))
        };
        await fsp.writeFile(tmp, JSON.stringify(out), "utf8");
        await fsp.rename(tmp, this.memoryFile);
    }

    active() {
        return this.data.memories.filter(m => m.status === "active");
    }

    // ----------------------------------------------------------- retrieval

    async RetrieveLongTermMemory(user_message, queryEmbedding) {
        const q = queryEmbedding || await this.vector_engine.embedQuery(user_message);
        const hits = this.vector_engine.search({
            queryEmbedding: q,
            memories: this.active(),
            topK: this.opts.topK,
            absThreshold: this.opts.absThreshold,
            relMargin: this.opts.relMargin,
            mmrLambda: this.opts.mmrLambda
        });
        const byId = new Map(this.data.memories.map(m => [m.id, m]));
        return hits.map(h => {
            h.memory.accessCount = (h.memory.accessCount || 0) + 1;
            const prev = h.memory.supersedes ? byId.get(h.memory.supersedes) : null;
            return {
                similarity: h.similarity,
                memory: {
                    id: h.memory.id,
                    entity: h.memory.entity,
                    type: h.memory.type,
                    summary: h.memory.summary,
                    since: h.memory.validFrom || h.memory.createdAt,
                    previously: prev ? prev.summary : undefined
                }
            };
        });
    }

    // ------------------------------------------------------------ analysis

    static SYSTEM_PROMPT = `You are the Memory Analyzer of an AI memory engine. You observe one exchange between a User and an Assistant as a neutral third party and decide what durable knowledge it creates or changes.

STORE: personal facts, identity, preferences, goals, plans, ongoing projects, decisions, instructions, skills, relationships, technical choices, constraints (budget, schedule, location, health), and anything the user would expect to be remembered next week.
DO NOT STORE: greetings, small talk, one-off questions, generic explanations given by the assistant, or the assistant's suggestions that the user did not adopt.

You are also given EXISTING MEMORIES related to this exchange. For every fact you extract, choose an action:
- "add": a genuinely new fact.
- "update": the exchange changes or corrects the value of an existing memory (e.g. a new budget, a new city, a changed preference). Set "target" to that memory's id. The new summary must state the NEW value.
- "noop": the exchange only repeats an existing memory without changing it. Set "target" to its id.

Rules:
- Summaries are short, self-contained, third-person sentences ("The user's budget is 2 lakh rupees.") and must keep exact names, numbers and units.
- Facts about other people (friends, colleagues, family) must use that person as the entity, never "user".
- Never merge two different facts into one memory.
- importance is between 0 and 1.

Return ONLY JSON:
{"memories":[{"action":"add|update|noop","target":null,"store":true,"importance":0.9,"entity":"user","type":"fact|preference|goal|project|decision|instruction|relationship|skill|event|constraint|other","summary":"..."}]}
If nothing is worth storing return {"memories":[]}`;

    async analyze(conversation, contextEmbedding) {
        // Existing memories related to this exchange (ids shortened to M1..Mn so
        // the model copies them reliably).
        let related = [];
        if (this.opts.consolidation === "llm" && this.active().length) {
            const e = contextEmbedding ||
                await this.vector_engine.embedQuery(conversation.user);
            related = this.vector_engine
                .neighbours(e, this.active(), this.opts.neighbourCount, 0.3)
                .map((n, i) => ({ alias: `M${i + 1}`, memory: n.memory }));
        }
        const existingBlock = related.length
            ? related.map(r => `${r.alias}: [${r.memory.entity}] ${r.memory.summary}`).join("\n")
            : "(none)";

        const prompt = `EXISTING MEMORIES:
${existingBlock}

EXCHANGE:
User: ${conversation.user}
Assistant: ${conversation.model}`;

        const response = await this.analyzer_model.generate(prompt, LongTermStorage.SYSTEM_PROMPT);
        const parsed = extractJSON(response);
        if (!parsed || !Array.isArray(parsed.memories)) return { memories: [], related };
        const aliasMap = new Map(related.map(r => [r.alias, r.memory.id]));
        for (const m of parsed.memories) {
            if (m && m.target && aliasMap.has(m.target)) m.targetId = aliasMap.get(m.target);
        }
        return { memories: parsed.memories, related };
    }

    // -------------------------------------------------------- consolidation

    /**
     * Apply an analysis. Returns a log of the operations performed, which the
     * evaluation harness uses to compute consolidation statistics.
     */
    async resolveMemory(analysis) {
        const ops = [];
        const now = () => new Date().toISOString();

        for (const fact of analysis.memories || []) {
            if (!fact || typeof fact.summary !== "string" || !fact.summary.trim()) continue;
            if (fact.store === false) { ops.push({ op: "skip:store=false", summary: fact.summary }); continue; }
            const importance = typeof fact.importance === "number" ? fact.importance : 0.5;
            if (importance < this.opts.minImportance) { ops.push({ op: "skip:low-importance", summary: fact.summary }); continue; }

            const text = `${fact.entity || "user"}: ${fact.summary}`;
            const embedding = await this.vector_engine.embedDocument(text);
            const active = this.active();

            let action = (fact.action || "add").toLowerCase();
            let target = fact.targetId ? active.find(m => m.id === fact.targetId) : null;
            const nn = this.vector_engine.nearest(embedding, active);

            if (this.opts.consolidation === "threshold") {
                // v1-style decision rule, kept for ablation.
                target = nn.memory;
                action = nn.score >= 0.98 ? "noop" : nn.score >= this.opts.updateThreshold ? "update" : "add";
            } else {
                if ((action === "update" || action === "noop") && !target) action = "add";
                // Geometric safety net for facts the analyzer labelled "add" although a
                // near-identical record exists. Cosine cannot tell a paraphrase from a
                // value change, but it does not need to: superseding a paraphrase loses
                // nothing, superseding a changed value is exactly right. The only error
                // is superseding a *different* fact, which is rare above this threshold
                // (see paper, Proposition 2).
                if (action === "add" && nn.memory && nn.score >= this.opts.duplicateThreshold) {
                    action = "update"; target = nn.memory; fact._auto = true;
                }
            }

            if (action === "noop" && target) {
                target.reinforcement = (target.reinforcement || 0) + 1;
                target.updatedAt = now();
                ops.push({ op: "noop", summary: fact.summary, target: target.summary, score: nn.score });
                continue;
            }

            const record = {
                id: crypto.randomUUID(),
                createdAt: now(),
                updatedAt: now(),
                validFrom: now(),
                validTo: null,
                status: "active",
                supersedes: null,
                supersededBy: null,
                accessCount: 0,
                reinforcement: 0,
                entity: fact.entity || "user",
                type: fact.type || "other",
                category: fact.category,
                importance,
                summary: fact.summary.trim(),
                embedding
            };

            if (action === "update" && target) {
                if (this.opts.versioning) {
                    target.status = "superseded";
                    target.validTo = record.validFrom;
                    target.supersededBy = record.id;
                    record.supersedes = target.id;
                    this.data.memories.push(record);
                } else {
                    Object.assign(target, { ...record, id: target.id, createdAt: target.createdAt });
                }
                ops.push({ op: fact._auto ? "update:auto" : "update", summary: record.summary, target: target.summary, score: nn.score });
            } else {
                this.data.memories.push(record);
                ops.push({ op: "add", summary: record.summary, score: nn.score });
            }
        }

        await this.persist();
        return ops;
    }
}

module.exports = LongTermStorage;
