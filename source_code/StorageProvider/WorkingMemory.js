const { estimateTokens } = require('../utils/UtilityFunctions');

/** Token-bounded FIFO window of recent turns. */
class WorkingMemory {
    constructor({ maxTokens = 2000 } = {}) {
        this.maxTokens = maxTokens;
        this.currentTokens = 0;
        this.memory = [];
    }

    readWorkingMemory() {
        return this.memory;
    }

    writeWorkingMemory(data) {
        if (!data) return;

        const turn = {
            user: data.user || "",
            model: data.model || ""
        };
        turn.totalTokens = estimateTokens(turn.user) + estimateTokens(turn.model);

        this.memory.push(turn);
        this.currentTokens += turn.totalTokens;

        // Evict oldest turns, but always keep the most recent one.
        while (this.currentTokens > this.maxTokens && this.memory.length > 1) {
            const removed = this.memory.shift();
            this.currentTokens -= removed.totalTokens;
        }
    }

    reset() {
        this.memory = [];
        this.currentTokens = 0;
    }
}

module.exports = WorkingMemory;
