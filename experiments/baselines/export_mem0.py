"""Re-export the complete Mem0 stores (get_all defaults to top_k=20)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "4b"]
import run_mem0 as R
out = {}
for pid in sorted(os.listdir(os.path.join(R.RES, "stores", R.TAG))):
    m = R.make(pid)
    a = m.get_all(filters={"user_id": pid}, top_k=100000)
    a = a.get("results", a) if isinstance(a, dict) else a
    out[pid] = [{"memory": x.get("memory"), "created_at": x.get("created_at"), "updated_at": x.get("updated_at")} for x in a]
    print(pid, len(out[pid]))
json.dump(out, open(os.path.join(R.RES, f"store_{R.TAG}.json"), "w"), indent=1)
