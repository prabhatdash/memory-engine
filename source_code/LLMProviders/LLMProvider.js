// Providers are loaded lazily so that an application only needs the SDK of the
// provider it actually uses (e.g. an Ollama-only deployment needs no cloud SDKs).
// NOTE: file names are case-sensitive on Linux; they must match exactly.
const PROVIDERS = {
    openai: () => require("./OpenAiProvider"),
    google: () => require("./GeminiProvider"),
    anthropic: () => require("./AnthropicProvider"),
    ollama: () => require("./OllamaProvider")
};

class LLMProvider {

    constructor(config) {

        const load = PROVIDERS[config.provider];

        if (!load) {
            throw new Error(`Unsupported provider: ${config.provider}`);
        }

        const Provider = load();
        this.config = config;
        this.llm = new Provider(config);
    }

    async generate(prompt, systemPrompt) {
        return await this.llm.generate({
            prompt,
            systemPrompt
        });
    }

    async generateEmbeddings(text) {
        if (typeof this.llm.embed !== "function") {
            throw new Error(`Provider "${this.config.provider}" does not support embeddings.`);
        }
        return await this.llm.embed(text);
    }

}

module.exports = LLMProvider;
