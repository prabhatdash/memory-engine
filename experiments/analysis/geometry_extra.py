"""Extra statistics for Propositions 2-3: tail masses at tau_s and the relevance margin."""
import json, os, urllib.request, numpy as np
HERE=os.path.dirname(__file__); RES=os.path.join(HERE,"..","results")
out={}
for enc in ("v1_json","v2_prefixed"):
    z=np.load(os.path.join(RES,f"geom_{enc}.npz"))
    o={}
    for tau in (0.85,0.90,0.92,0.95,0.98):
        o[f"tau_{tau}"]={k:float((z[k]>=tau).mean()) for k in ("PARA","UPD","DIFF")}
    # Bayes-optimal SAME-vs-DIFF balanced accuracy over all cuts
    same=np.concatenate([z["PARA"],z["UPD"]]); diff=z["DIFF"]
    cuts=np.unique(np.concatenate([same,diff]))
    o["best_same_vs_diff_bal_acc"]=float(max(0.5*((same>=c).mean()+(diff<c).mean()) for c in cuts[::7]))
    out[enc]=o
# relevance margin: query vs same-slot record minus query vs best other-slot record (v2 encoding)
slots=json.load(open(os.path.join(HERE,"..","data","scenarios.json")))["slots"]
def embed(t):
    r=urllib.request.Request("http://localhost:11434/api/embed",data=json.dumps({"model":"nomic-embed-text","input":t}).encode(),headers={"Content-Type":"application/json"})
    X=np.array(json.load(urllib.request.urlopen(r))["embeddings"]); return X/np.linalg.norm(X,axis=1,keepdims=True)
recs=[(s,v,t.format(v=v)) for s,d in slots.items() for v in d["values"] for t in d["summary"]]
for enc,qp,dp in (("v2_prefixed","search_query: ","search_document: user: "),("v1_raw","",None)):
    if dp is None:
        TYPE={"city":"fact","budget":"constraint","framework":"decision","hours":"constraint","diet":"preference","job":"fact","trip":"goal","pet":"fact","sibling":"relationship","allergy":"fact","instrument":"skill"}
        E=embed([json.dumps({"entity":"user","type":TYPE[s],"category":s,"summary":t},separators=(",",":")) for s,v,t in recs])
    else:
        E=embed([dp+t for s,v,t in recs])
    Q=embed([qp+slots[s]["probe"] for s in slots]); qi={s:i for i,s in enumerate(slots)}
    margins=[];top1=[];sims=[]
    for s in slots:
        sc=E@Q[qi[s]]; own=np.array([r[0]==s for r in recs])
        margins.append(float(sc[own].mean()-sc[~own].max())); top1.append(bool(sc.argmax() in np.where(own)[0])); sims.append(float(sc[own].mean()))
    out[f"relevance_{enc}"]={"mean_margin_own_minus_best_other":float(np.mean(margins)),"top1_correct_slot":float(np.mean(top1)),
        "mean_query_sim_to_own_slot":float(np.mean(sims)),"frac_own_slot_sim_below_0.55":None}
    sc_all=[float(x) for s in slots for x,r in zip(E@Q[qi[s]],recs) if r[0]==s]
    out[f"relevance_{enc}"]["frac_own_slot_sim_below_0.55"]=float(np.mean(np.array(sc_all)<0.55))
json.dump(out,open(os.path.join(RES,"embed_geometry_extra.json"),"w"),indent=1); print(json.dumps(out,indent=1))
