const LLMProvider = require("../LLMProviders/LLMProvider");
const { LRUCache } = require("../utils/UtilityFunctions");

/**
 * Asymmetric-retrieval prefixes required by some embedding families.
 * nomic-embed-text, for example, is trained with "search_query:" /
 * "search_document:" task prefixes and degrades noticeably without them.
 */
function defaultPrefixes(model = "") {
    const m = model.toLowerCase();
    if (m.includes("nomic-embed")) return { query: "search_query: ", document: "search_document: " };
    if (m.includes("e5")) return { query: "query: ", document: "passage: " };
    return { query: "", document: "" };
}

class VectorEngine {
    constructor({ embedding_model_config }) {

        this.embedding_model = new LLMProvider({
            model: embedding_model_config.model,
            provider: embedding_model_config.provider,
            apiKey: embedding_model_config.apiKey,
            baseURL: embedding_model_config.baseURL
        });

        const d = defaultPrefixes(embedding_model_config.model);
        this.queryPrefix = embedding_model_config.queryPrefix ?? d.query;
        this.documentPrefix = embedding_model_config.documentPrefix ?? d.document;

        this.cache = new LRUCache(1024);
        this.stats = { embedCalls: 0, cacheHits: 0 };
    }

    /** Raw embedding with LRU caching. */
    async generateEmbeddings(text) {
        const hit = this.cache.get(text);
        if (hit) { this.stats.cacheHits++; return hit; }
        this.stats.embedCalls++;
        const v = await this.embedding_model.generateEmbeddings(text);
        this.cache.set(text, v);
        return v;
    }

    embedQuery(text) { return this.generateEmbeddings(this.queryPrefix + text); }
    embedDocument(text) { return this.generateEmbeddings(this.documentPrefix + text); }

    cosineSimilarity(vectorA, vectorB) {

        if (!Array.isArray(vectorA) || !Array.isArray(vectorB)) {
            throw new Error("Both inputs must be arrays.");
        }

        if (vectorA.length !== vectorB.length) {
            throw new Error("Embedding dimensions do not match.");
        }

        let dot = 0, a2 = 0, b2 = 0;
        for (let i = 0; i < vectorA.length; i++) {
            dot += vectorA[i] * vectorB[i];
            a2 += vectorA[i] * vectorA[i];
            b2 += vectorB[i] * vectorB[i];
        }
        if (a2 === 0 || b2 === 0) return 0;
        return dot / (Math.sqrt(a2) * Math.sqrt(b2));
    }

    /**
     * Relevance-gated, diversity-aware top-k search.
     *
     *   s_i   = cos(q, m_i)
     *   keep  i  iff  s_i >= max(absThreshold, s_max - relMargin)
     *   rank  by  s_i + wImportance * importance_i
     *   select k items greedily by MMR:
     *         argmax  lambda * score_i - (1 - lambda) * max_{j in S} cos(m_i, m_j)
     *
     * The relative gate adapts to the score scale of the embedding model
     * (absolute cosine values are not comparable across models), and MMR
     * prevents k near-duplicate records from crowding out other facts.
     */
    search({
        queryEmbedding,
        memories,
        topK = 5,
        absThreshold = 0.35,
        relMargin = 0.2,
        wImportance = 0.05,
        mmrLambda = 0.75
    }) {
        if (!memories.length) return [];

        // One pass of dot products with cached norms: O(N d).
        const qn = this.#norm(queryEmbedding);
        let sMax = -Infinity;
        const scored = new Array(memories.length);
        for (let i = 0; i < memories.length; i++) {
            const m = memories[i];
            const sim = qn === 0 ? 0 : this.#dot(queryEmbedding, m.embedding) / (qn * this.#norm(m.embedding) || 1);
            scored[i] = { memory: m, similarity: sim };
            if (sim > sMax) sMax = sim;
        }
        const gate = Math.max(absThreshold, sMax - relMargin);

        // MMR only over a bounded candidate list: O(C log C + k^2 C d), independent of N.
        const candidateLimit = Math.max(4 * topK, 20);
        const pool = scored
            .filter(s => s.similarity >= gate)
            .map(s => ({ ...s, score: s.similarity + wImportance * (s.memory.importance ?? 0.5) }))
            .sort((a, b) => b.score - a.score)
            .slice(0, candidateLimit);

        const selected = [];
        const redundancy = new Array(pool.length).fill(0);
        while (selected.length < topK && pool.length) {
            let bestIdx = 0, bestVal = -Infinity;
            for (let i = 0; i < pool.length; i++) {
                const val = mmrLambda * pool[i].score - (1 - mmrLambda) * redundancy[i];
                if (val > bestVal) { bestVal = val; bestIdx = i; }
            }
            const chosen = pool.splice(bestIdx, 1)[0];
            redundancy.splice(bestIdx, 1);
            selected.push(chosen);
            if (mmrLambda < 1) {
                for (let i = 0; i < pool.length; i++) {
                    redundancy[i] = Math.max(redundancy[i],
                        this.cosineSimilarity(pool[i].memory.embedding, chosen.memory.embedding));
                }
            }
        }
        return selected;
    }

    #dot(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    #norms = new WeakMap();
    #norm(v) {
        let n = this.#norms.get(v);
        if (n === undefined) { n = Math.sqrt(this.#dot(v, v)); this.#norms.set(v, n); }
        return n;
    }

    /** Nearest active neighbour of `embedding` among `memories`. */
    nearest(embedding, memories) {
        let best = { score: -1, memory: null };
        for (const m of memories) {
            const s = this.cosineSimilarity(embedding, m.embedding);
            if (s > best.score) best = { score: s, memory: m };
        }
        return best;
    }

    /** Top-n neighbours (used to show the analyzer existing, related memories). */
    neighbours(embedding, memories, n = 6, minScore = 0.3) {
        return memories
            .map(m => ({ memory: m, score: this.cosineSimilarity(embedding, m.embedding) }))
            .filter(x => x.score >= minScore)
            .sort((a, b) => b.score - a.score)
            .slice(0, n);
    }
}

module.exports = VectorEngine;
