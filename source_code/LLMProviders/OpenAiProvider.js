const OpenAI = require("openai");

class OpenAIProvider {

    constructor({ model, apiKey, baseURL }) {

        if (!model) {
            throw new Error("OpenAI model is required.");
        }

        if (!apiKey) {
            throw new Error("OpenAI API key is required.");
        }

        this.model = model;

        this.client = new OpenAI({
            apiKey,
            ...(baseURL ? { baseURL } : {})
        });
    }

    async generate({prompt, systemPrompt}) {

        try {

            if (!prompt || typeof prompt !== "string") {
                throw new Error("Prompt must be a non-empty string.");
            }

            const response = await this.client.chat.completions.create({

                model: this.model,

                messages: [

                    {
                        role: "system",
                        content: systemPrompt
                    },

                    {
                        role: "user",
                        content: prompt
                    }

                ]

            });

            return response.choices?.[0]?.message?.content || "";

        } catch (error) {

            if (error.status === 401) {
                throw new Error("Invalid OpenAI API Key.");
            }

            if (error.status === 429) {
                throw new Error("OpenAI rate limit exceeded.");
            }

            if (error.status === 404) {
                throw new Error(`Model "${this.model}" not found.`);
            }

            throw new Error(
                `OpenAI Error: ${error.message}`
            );
        }

    }

    async embed(text) {

        try {

            if (!text || typeof text !== "string") {
                throw new Error("Text must be a non-empty string.");
            }

            const response = await this.client.embeddings.create({
                model: this.model,
                input: text
            });

            return response.data[0].embedding;

        } catch (error) {

            if (error.status === 401) {
                throw new Error("Invalid OpenAI API Key.");
            }

            if (error.status === 429) {
                throw new Error("OpenAI rate limit exceeded.");
            }

            throw new Error(`OpenAI Embedding Error: ${error.message}`);
        }

    }

}

module.exports = OpenAIProvider;