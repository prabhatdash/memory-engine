// Offline unit tests (no LLM / network): `npm test`
const test = require("node:test");
const assert = require("node:assert");
const os = require("os");
const fs = require("fs");
const path = require("path");
const LongTermStorage = require("../StorageProvider/LongTermStorage");
const VectorEngine = require("../StorageProvider/VectorEngine");
const WorkingMemory = require("../StorageProvider/WorkingMemory");
const MemoryEngine = require("../Memory");

const cfg = { model: "mock", provider: "ollama" };
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "me-test-"));

// Deterministic bag-of-words embedding so tests do not need a model.
function bow(text) {
    const v = new Array(64).fill(0);
    for (const w of text.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean)) {
        let h = 0; for (const c of w) h = (h * 31 + c.charCodeAt(0)) >>> 0;
        v[h % 64] += 1;
    }
    return v;
}
function mockStore(dir, options = {}, analyses = []) {
    const s = new LongTermStorage({ storagePath: dir, analyzer_model_config: cfg, embedding_model_config: cfg, options });
    s.vector_engine.embedding_model = { generateEmbeddings: async t => bow(t.replace(/^search_(query|document): /, "")) };
    s.analyzer_model = { generate: async () => JSON.stringify(analyses.shift() || { memories: [] }) };
    return s;
}

test("vector round-trip through base64 Float32 is lossless for cosine", () => {
    const v = Array.from({ length: 768 }, () => Math.random() - 0.5);
    const w = LongTermStorage.decodeVector(LongTermStorage.encodeVector(v));
    const ve = new VectorEngine({ embedding_model_config: cfg });
    assert.ok(Math.abs(ve.cosineSimilarity(v, w) - 1) < 1e-6);
});

test("update supersedes instead of overwriting, and only the current value is retrievable", async () => {
    const dir = tmp();
    const s = mockStore(dir, {}, [
        { memories: [{ action: "add", entity: "user", type: "fact", summary: "The user lives in Hobart.", importance: 0.9 }] },
    ]);
    await s.resolveMemory(await s.analyze({ user: "I live in Hobart", model: "nice" }));
    const old = s.active()[0];
    await s.resolveMemory({ memories: [{ action: "update", targetId: old.id, entity: "user", summary: "The user lives in Tromso.", importance: 0.9 }] });
    assert.equal(s.active().length, 1);
    assert.equal(s.data.memories.length, 2);
    assert.equal(old.status, "superseded");
    const r = await s.RetrieveLongTermMemory("where does the user live");
    assert.equal(r[0].memory.summary, "The user lives in Tromso.");
    assert.equal(r[0].memory.previously, "The user lives in Hobart.");
    // persisted and reloadable
    const s2 = mockStore(dir);
    assert.equal(s2.active()[0].summary, "The user lives in Tromso.");
});

test("store=false and low-importance facts are not stored", async () => {
    const s = mockStore(tmp());
    const ops = await s.resolveMemory({ memories: [
        { action: "add", summary: "User said hi.", store: false, importance: 0.9 },
        { action: "add", summary: "User asked about recursion.", importance: 0.1 }] });
    assert.equal(s.active().length, 0);
    assert.deepEqual(ops.map(o => o.op), ["skip:store=false", "skip:low-importance"]);
});

test("near-identical 'add' supersedes the existing record (no duplicates within a batch)", async () => {
    const s = mockStore(tmp());
    await s.resolveMemory({ memories: [
        { action: "add", entity: "user", summary: "The user has a dog named Biscotti.", importance: 0.8 },
        { action: "add", entity: "user", summary: "The user has a dog named Biscotti.", importance: 0.8 }] });
    assert.equal(s.active().length, 1);
});

test("threshold ablation reproduces v1 decision rule", async () => {
    const s = mockStore(tmp(), { consolidation: "threshold" });
    await s.resolveMemory({ memories: [{ summary: "The user lives in Hobart.", entity: "user", importance: 0.9 }] });
    const ops = await s.resolveMemory({ memories: [{ summary: "The user lives in Hobart.", entity: "user", importance: 0.9 }] });
    assert.equal(ops[0].op, "noop");
});

test("MMR search returns at most topK and avoids exact duplicates", () => {
    const ve = new VectorEngine({ embedding_model_config: cfg });
    const mk = (t) => ({ summary: t, embedding: bow(t), importance: 0.5 });
    const mems = [mk("user lives in hobart"), mk("user lives in hobart"), mk("user lives in hobart city"), mk("user likes cello")];
    const r = ve.search({ queryEmbedding: bow("where does user live"), memories: mems, topK: 2, mmrLambda: 0.5, absThreshold: 0 });
    assert.equal(r.length, 2);
    assert.notDeepEqual(r[0].memory.embedding, r[1].memory.embedding);
});

test("working memory is token-bounded and does not mutate its input", () => {
    const wm = new WorkingMemory({ maxTokens: 30 });
    const turn = { user: "one two three four five", model: "six seven eight nine ten" };
    for (let i = 0; i < 10; i++) wm.writeWorkingMemory(turn);
    assert.ok(wm.currentTokens <= 30);
    assert.equal(turn.totalTokens, undefined);
    wm.reset();
    assert.equal(wm.readWorkingMemory().length, 0);
});

test("chat() returns before background consolidation finishes; flush() completes it", async () => {
    const dir = tmp();
    const eng = new MemoryEngine(cfg, cfg, cfg, dir);
    const s = eng.long_term_mem;
    s.vector_engine.embedding_model = { generateEmbeddings: async t => bow(t) };
    let release;
    const gate = new Promise(r => (release = r));
    s.analyzer_model = { generate: async () => { await gate; return JSON.stringify({ memories: [{ action: "add", entity: "user", summary: "The user plays the cello.", importance: 0.9 }] }); } };
    eng.llm = { generate: async () => "Nice!" };
    const reply = await eng.chat("I play the cello");
    assert.equal(reply, "Nice!");
    assert.equal(eng.pendingConsolidations, 1);
    assert.equal(s.active().length, 0);
    release();
    await eng.flush();
    assert.equal(s.active().length, 1);
});
