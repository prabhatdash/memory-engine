/**
 * Robustly extract a JSON object from an LLM response.
 * Handles: ```json fences, <think>...</think> blocks, leading/trailing prose.
 */
function extractJSON(text) {

    if (!text || typeof text !== "string") {
        return { memories: [] };
    }

    let cleaned = text
        .replace(/<think>[\s\S]*?<\/think>/gi, "")
        .replace(/```(?:json)?/gi, "")
        .trim();

    try {
        return JSON.parse(cleaned);
    } catch (_) {
        // Fall back to the outermost {...} span.
        const start = cleaned.indexOf("{");
        const end = cleaned.lastIndexOf("}");
        if (start !== -1 && end > start) {
            try {
                return JSON.parse(cleaned.slice(start, end + 1));
            } catch (_) { /* fall through */ }
        }
        console.warn("[MemoryEngine] JSON Parse Error in extractJSON. Returning empty memories.");
        return { memories: [] };
    }
}

function estimateTokens(text) {
    if (!text) return 0;
    return Math.ceil(text.toString().trim().split(/\s+/).length * 1.3);
}

/** Tiny LRU cache used for embeddings. */
class LRUCache {
    constructor(limit = 512) {
        this.limit = limit;
        this.map = new Map();
    }
    get(key) {
        if (!this.map.has(key)) return undefined;
        const v = this.map.get(key);
        this.map.delete(key);
        this.map.set(key, v);
        return v;
    }
    set(key, value) {
        if (this.map.has(key)) this.map.delete(key);
        this.map.set(key, value);
        if (this.map.size > this.limit) {
            this.map.delete(this.map.keys().next().value);
        }
    }
}

module.exports = { extractJSON, estimateTokens, LRUCache };
