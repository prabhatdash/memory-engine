const LLMProvider = require("./LLMProviders/LLMProvider");
const LongTermStorage = require("./StorageProvider/LongTermStorage");
const WorkingMemory = require("./StorageProvider/WorkingMemory");

/**
 * MemoryEngine v2.
 *
 * chat() critical path:   embed(query) -> retrieve -> generate -> return
 * background (queued):    analyze(exchange, related memories) -> consolidate -> persist
 *
 * In v1 the analysis and consolidation were awaited before chat() returned,
 * so the user paid for two extra LLM/embedding round-trips on every turn.
 * Here they run on a serialized background queue. A fact stated at turn t is
 * still visible at turn t+1 through Working Memory while it is being
 * consolidated, so deferring consolidation never makes a fact invisible as
 * long as the consolidation lag is shorter than the Working Memory window.
 */
class MemoryEngine {

    /**
     * @param {object} base_model_config       {model, provider, apiKey, baseURL, system_prompt, options, think}
     * @param {object} embedding_model_config  {model, provider, apiKey, baseURL, queryPrefix, documentPrefix}
     * @param {object} [analyzer_model_config] defaults to base_model_config
     * @param {string} [storagePath]
     * @param {object} [options]
     * @param {boolean} [options.asyncConsolidation=true]
     * @param {number}  [options.workingMemoryTokens=2000]
     * @param {object}  [options.longTerm]    passed to LongTermStorage (topK, consolidation, versioning, ...)
     */
    constructor(base_model_config, embedding_model_config, analyzer_model_config = base_model_config, storagePath, options = {}) {
        if (!base_model_config || !embedding_model_config) {
            throw new Error("[MemoryEngine] Missing model configurations. Ensure base_model and embedding_model configs are provided.");
        }
        analyzer_model_config = analyzer_model_config || base_model_config;

        this.base_model_config = base_model_config;
        this.embedding_model_config = embedding_model_config;
        this.analyzer_model_config = analyzer_model_config;
        this.options = { asyncConsolidation: true, workingMemoryTokens: 2000, ...options };

        this.llm = new LLMProvider({
            model: base_model_config.model,
            provider: base_model_config.provider,
            apiKey: base_model_config.apiKey,
            baseURL: base_model_config.baseURL,
            options: base_model_config.options,
            think: base_model_config.think
        });

        this.working_mem = new WorkingMemory({ maxTokens: this.options.workingMemoryTokens });

        this.long_term_mem = new LongTermStorage({
            storagePath,
            analyzer_model_config,
            embedding_model_config,
            options: this.options.longTerm || {}
        });

        this._queue = Promise.resolve();
        this._pending = 0;
        this.lastTurn = null;      // instrumentation for the evaluation harness
        this.onConsolidated = null; // optional callback(ops, timings)
    }

    static formatMemories(results) {
        if (!results.length) return "(no relevant long-term memories)";
        return results.map(r => {
            const m = r.memory;
            const date = (m.since || "").slice(0, 10);
            const prev = m.previously ? ` (updated; previously: ${m.previously})` : "";
            return `- [${date}] ${m.summary}${prev}`;
        }).join("\n");
    }

    static formatWorkingMemory(turns) {
        if (!turns.length) return "(this is the start of the conversation)";
        return turns.map(t => `User: ${t.user}\nAssistant: ${t.model}`).join("\n\n");
    }

    buildPrompt(user_message, workingMemory, longTermMemory) {
        return `LONG-TERM MEMORY (facts remembered from earlier conversations; each line is the CURRENT value, dated when it was learned):
${MemoryEngine.formatMemories(longTermMemory)}

RECENT CONVERSATION:
${MemoryEngine.formatWorkingMemory(workingMemory)}

CURRENT USER MESSAGE:
${user_message}

Reply to the current user message. Use the memories only if they are relevant, and speak as if you naturally remember them (never say "according to my memory"). If a memory is marked as updated, the newest value is the true one.`;
    }

    async chat(user_message) {
        const t0 = performance.now();
        const workingMemory = this.working_mem.readWorkingMemory().slice();

        let longTermMemory = [];
        let queryEmbedding;
        const tR0 = performance.now();
        try {
            queryEmbedding = await this.long_term_mem.vector_engine.embedQuery(user_message);
            longTermMemory = await this.long_term_mem.RetrieveLongTermMemory(user_message, queryEmbedding);
        } catch (e) {
            console.warn("[MemoryEngine] Long-term memory retrieval failed. Proceeding with working memory only.", e.message);
        }
        const tR1 = performance.now();

        const prompt = this.buildPrompt(user_message, workingMemory, longTermMemory);
        const response = await this.llm.generate(prompt, this.base_model_config.system_prompt);
        const tG1 = performance.now();

        const exchange = { user: user_message, model: response };
        try {
            this.working_mem.writeWorkingMemory(exchange);
        } catch (e) {
            console.warn("[MemoryEngine] Failed to write to working memory.", e.message);
        }

        const consolidation = this.#enqueue(exchange, queryEmbedding);
        if (!this.options.asyncConsolidation) await consolidation;

        this.lastTurn = {
            prompt,
            retrieved: longTermMemory,
            timings: {
                retrievalMs: tR1 - tR0,
                generationMs: tG1 - tR1,
                responseMs: tG1 - t0,
                totalSyncMs: performance.now() - t0
            }
        };
        return response;
    }

    #enqueue(exchange, contextEmbedding) {
        this._pending++;
        const job = this._queue.then(async () => {
            const tA0 = performance.now();
            try {
                const analysis = await this.long_term_mem.analyze(exchange, contextEmbedding);
                const tA1 = performance.now();
                const ops = await this.long_term_mem.resolveMemory(analysis);
                const tA2 = performance.now();
                if (this.onConsolidated) {
                    this.onConsolidated(ops, { analyzeMs: tA1 - tA0, consolidateMs: tA2 - tA1 }, analysis);
                }
            } catch (e) {
                console.warn("[MemoryEngine] Background analysis and storage failed. Memory may not be updated.", e.message);
            } finally {
                this._pending--;
            }
        });
        this._queue = job;
        return job;
    }

    /** Ingest an exchange that did not go through chat() (e.g. imported history). */
    async observe(user_message, model_response) {
        const exchange = { user: user_message, model: model_response };
        this.working_mem.writeWorkingMemory(exchange);
        return this.#enqueue(exchange);
    }

    /** Wait until all queued consolidation work has been persisted. */
    async flush() {
        await this._queue;
    }

    /** Start a new session: clears Working Memory, keeps Long-Term Memory. */
    newSession() {
        this.working_mem.reset();
    }

    get pendingConsolidations() {
        return this._pending;
    }
}

module.exports = MemoryEngine;
