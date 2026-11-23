/**
 * Exact-scan retrieval cost of the local vector store as a function of store
 * size N (d = 768, nomic-embed-text dimensionality). Uses the engines' own
 * VectorEngine code paths:
 *   v1: VectorEngine.search  (threshold filter + sort)
 *   v2: VectorEngine.search  (relative gate + MMR re-ranking)
 * The embedding call is excluded (measured separately); this isolates the
 * O(N d) scan the paper's complexity analysis predicts.
 */
const path = require("path");
const fs = require("fs");
const V1 = require("../engine_v1/StorageProvider/VectorEngine.js");
const V2 = require("../../source_code/StorageProvider/VectorEngine.js");

const d = 768;
function rnd(n) { // vectors with a shared mean direction, like real embeddings (cos ~ 0.6)
    const base = Array.from({ length: d }, () => Math.random() - 0.5);
    return Array.from({ length: n }, () => base.map(b => b + (Math.random() - 0.5) * 1.1));
}
const cfg = { embedding_model_config: { model: "nomic-embed-text", provider: "ollama" } };
const e1 = new V1(cfg), e2 = new V2(cfg);
(async () => {
const out = [];
for (const N of [100, 300, 1000, 3000, 10000, 30000, 100000]) {
    const vecs = rnd(N + 1);
    const q = vecs.pop();
    const mems = vecs.map((v, i) => ({ id: i, embedding: v, importance: 0.5 }));
    const reps = N <= 3000 ? 30 : N <= 30000 ? 8 : 3;
    // v1 search embeds the query internally; stub it to return the vector.
    e1.generateEmbeddings = async () => q;
    const t = { v1: [], v2: [] };
    for (let r = 0; r < reps; r++) {
        let t0 = performance.now(); await e1.search({ query: "x", memories: mems }); t.v1.push(performance.now() - t0);
        t0 = performance.now(); e2.search({ queryEmbedding: q, memories: mems, topK: 5 }); t.v2.push(performance.now() - t0);
    }
    // v1 re-reads and re-parses the whole JSON store on every retrieval and rewrites it on every insert.
    let ioMs = null, v1PersistOk = true;
    try { if (N >= 30000) JSON.stringify({ memories: mems.map(m => ({ summary: "The user lives in Hobart.", entity: "user", embedding: m.embedding })) }, null, 4); }
    catch (e) { v1PersistOk = false; }
    if (N <= 10000) {
        const f = path.join(__dirname, "..", "results", "tmp_bench.json");
        fs.writeFileSync(f, JSON.stringify({ memories: mems.map(m => ({ summary: "The user lives in Hobart.", entity: "user", embedding: m.embedding })) }, null, 4));
        const io = [];
        for (let r = 0; r < Math.min(reps, 5); r++) { const t0 = performance.now(); JSON.parse(fs.readFileSync(f, "utf8")); io.push(performance.now() - t0); }
        ioMs = io.sort((x, y) => x - y)[Math.floor(io.length / 2)];
        fs.unlinkSync(f);
    }
    const med = a => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
    const row = { N, v1_ms: med(t.v1), v2_ms: med(t.v2), v1_read_parse_ms: ioMs, v1_persist_ok: v1PersistOk,
                  json_mb: Buffer.byteLength(JSON.stringify(mems.slice(0, 100))) * N / 100 / 1e6 };
    out.push(row);
    console.log(JSON.stringify(row));
}
fs.writeFileSync(path.join(__dirname, "..", "results", "bench_scan.json"), JSON.stringify(out, null, 1));
})();
