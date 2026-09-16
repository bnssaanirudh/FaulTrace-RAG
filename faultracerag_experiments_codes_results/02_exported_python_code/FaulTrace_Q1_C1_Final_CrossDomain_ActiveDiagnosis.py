"""Exported code cells from FaulTrace_Q1_C1_Final_CrossDomain_ActiveDiagnosis.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
%%capture
!pip -q install -U "beir>=2.0.0" "sentence-transformers>=3.0" "rank-bm25>=0.2.2" "scikit-learn>=1.4" "pandas>=2.0" "pyarrow>=15" "scipy>=1.11" "tqdm>=4.66" "matplotlib>=3.8"

# ============================================================
# NOTEBOOK CELL 3 / CODE CELL 2
# ============================================================
from __future__ import annotations
import os, sys, json, time, random, platform, subprocess, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats

FAST_MODE = True
PAPER_MODE = False
FORCE_RERUN = False
RUN_HEAVY = False       # True adds TREC-COVID + SCIDOCS
TOP_K = 10

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVERS = ["bm25","dense","hybrid"]
BASE_DATASETS = ["scifact","nfcorpus","arguana","fiqa"]
HEAVY_DATASETS = ["trec-covid","scidocs"]
BENCHMARKS = BASE_DATASETS + (HEAVY_DATASETS if RUN_HEAVY else [])

N_QUERIES = 40 if FAST_MODE else 300
SEEDS = [42] if FAST_MODE else [13,42,87,2026,31415]
SEVERITIES = [0.35] if FAST_MODE else [0.20,0.35,0.50]
PLAYERS = ("R","E","A")
FAULTS = [
    ("R",),("E",),("A",),
    ("R","E"),("R","A"),("E","A"),
    ("R","E","A")
]
STAGE_COST = {"R":1.0,"E":2.5,"A":1.0}

print({
    "FAST_MODE":FAST_MODE, "PAPER_MODE":PAPER_MODE,
    "datasets":BENCHMARKS, "seeds":SEEDS, "severities":SEVERITIES
})

# ============================================================
# NOTEBOOK CELL 5 / CODE CELL 3
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/FaulTrace_RAG_Experiments")
except Exception:
    DRIVE = Path("/content/FaulTrace_RAG_Experiments")

ROOT = DRIVE/"C1_FINAL_CROSS_DOMAIN"
CHUNKS=ROOT/"chunks"; CACHE=ROOT/"cache"; TABLES=ROOT/"paper_tables"; PLOTS=ROOT/"plots"; FAILS=ROOT/"failure_cases"; EXPORT=ROOT/"github_export"
for p in [ROOT,CHUNKS,CACHE,TABLES,PLOTS,FAILS,EXPORT]: p.mkdir(parents=True,exist_ok=True)

REPO=Path("/content/FaulTrace-RAG")
if REPO.exists():
    subprocess.run(["git","-C",str(REPO),"fetch","origin"],check=False)
    subprocess.run(["git","-C",str(REPO),"pull","--ff-only"],check=False)
else:
    subprocess.run(["git","clone","--depth","1","https://github.com/bnssaanirudh/FaulTrace-RAG.git",str(REPO)],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-e",str(REPO)],check=True)
COMMIT=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()

CP=ROOT/"checkpoint.json"
def read_cp():
    if CP.exists():
        try:return json.loads(CP.read_text())
        except Exception: pass
    return {"commit":COMMIT,"completed":{}}
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(tmp,path)
def mark_done(key,meta):
    cp=read_cp(); cp["completed"][key]=meta; cp["updated_utc"]=datetime.now(timezone.utc).isoformat(); atomic_json(CP,cp)
def done(key,path):
    return (not FORCE_RERUN) and key in read_cp().get("completed",{}) and path.exists()

(ROOT/"environment.txt").write_text(
    f"utc={datetime.now(timezone.utc).isoformat()}\ncommit={COMMIT}\npython={sys.version}\nplatform={platform.platform()}\n"
)
print("Commit:",COMMIT)

# ============================================================
# NOTEBOOK CELL 7 / CODE CELL 4
# ============================================================
import itertools, math, hashlib
from collections import defaultdict

def powerset(players):
    players = tuple(players)
    for r in range(len(players)+1):
        for c in itertools.combinations(players, r):
            yield frozenset(c)

def exact_shapley(values, players):
    players = tuple(players)
    n = len(players)
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [x for x in players if x != p]
        for S in powerset(others):
            w = math.factorial(len(S))*math.factorial(n-len(S)-1)/math.factorial(n)
            phi[p] += w*(values[S|{p}] - values[S])
    return phi

def harsanyi(values, players):
    out = {}
    for S in powerset(players):
        out[S] = sum(((-1)**(len(S)-len(T)))*values[T] for T in powerset(S))
    return out

def minimal_repair_sets(losses, players, tol):
    valid = [S for S,l in losses.items() if l <= tol]
    if not valid:
        return []
    m = min(len(S) for S in valid)
    return [S for S in valid if len(S)==m]

def min_cost_repair(losses, players, costs, tol):
    valid = [S for S,l in losses.items() if l <= tol]
    if not valid:
        return None, float("inf")
    ranked = sorted(
        ((sum(costs[p] for p in S), len(S), tuple(sorted(S)), S) for S in valid),
        key=lambda x: (x[0],x[1],x[2])
    )
    c,_,_,S = ranked[0]
    return S, float(c)

def stable_seed(*parts):
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

def set_f1(pred, truth):
    pred, truth = set(pred), set(truth)
    if not pred and not truth: return 1.0
    if not pred or not truth: return 0.0
    p = len(pred & truth)/len(pred)
    r = len(pred & truth)/len(truth)
    return 2*p*r/(p+r) if (p+r) else 0.0

def jaccard(pred, truth):
    pred, truth=set(pred),set(truth)
    return len(pred&truth)/len(pred|truth) if (pred|truth) else 1.0

def ndcg_at_k(ranking, qrels, k=10):
    gains=[float(qrels.get(d,0)) for d in ranking[:k]]
    dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gains))
    ideal=sorted([float(v) for v in qrels.values() if float(v)>0], reverse=True)[:k]
    idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

def recall_at_k(ranking, qrels, k=10):
    rel={d for d,g in qrels.items() if float(g)>0}
    return len(rel & set(ranking[:k]))/len(rel) if rel else float("nan")

def mrr_at_k(ranking, qrels, k=10):
    for i,d in enumerate(ranking[:k],1):
        if float(qrels.get(d,0))>0:
            return 1/i
    return 0.0

def cliffs_delta(x, y):
    x=list(x); y=list(y)
    if not x or not y: return float("nan")
    gt=sum(a>b for a in x for b in y)
    lt=sum(a<b for a in x for b in y)
    return (gt-lt)/(len(x)*len(y))

# ============================================================
# NOTEBOOK CELL 8 / CODE CELL 5
# ============================================================
# Mathematical invariants.
toy={
    frozenset():0.0,
    frozenset({"R"}):.2,frozenset({"E"}):.3,frozenset({"A"}):.1,
    frozenset({"R","E"}):.65,frozenset({"R","A"}):.36,frozenset({"E","A"}):.47,
    frozenset({"R","E","A"}):.85,
}
phi=exact_shapley(toy,PLAYERS)
assert abs(sum(phi.values())-(toy[frozenset(PLAYERS)]-toy[frozenset()]))<1e-10
div=harsanyi(toy,PLAYERS)
assert abs(div[frozenset({"R","E"})]-(.65-.2-.3))<1e-10
print("Attribution self-tests passed.")

# ============================================================
# NOTEBOOK CELL 10 / CODE CELL 6
# ============================================================
from beir import util
from beir.datasets.data_loader import GenericDataLoader

BEIR_BASE="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

def load_beir(name):
    folder=CACHE/"beir"; folder.mkdir(parents=True,exist_ok=True)
    path=folder/name
    if not path.exists():
        util.download_and_unzip(f"{BEIR_BASE}/{name}.zip",str(folder))
    corpus,queries,qrels=GenericDataLoader(data_folder=str(path)).load(split="test")
    qids=sorted(set(queries)&set(qrels))
    rg=np.random.default_rng(20260914)
    if len(qids)>N_QUERIES:
        qids=sorted(rg.choice(qids,size=N_QUERIES,replace=False).tolist())
    return corpus,{q:queries[q] for q in qids},{q:qrels[q] for q in qids}

# ============================================================
# NOTEBOOK CELL 12 / CODE CELL 7
# ============================================================
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def tok(x): return re.findall(r"[A-Za-z0-9]+",str(x).lower())
def doctext(d): return (str(d.get("title",""))+" "+str(d.get("text",""))).strip()
def topidx(scores,k):
    k=min(k,len(scores))
    if k==0:return np.array([],dtype=int)
    idx=np.argpartition(scores,-k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]
def rrf(a,b,k=10,c=60):
    s=defaultdict(float)
    for i,d in enumerate(a,1):s[d]+=1/(c+i)
    for i,d in enumerate(b,1):s[d]+=1/(c+i)
    return [d for d,_ in sorted(s.items(),key=lambda z:(-z[1],z[0]))[:k]]

_dense=None
def dense_model():
    global _dense
    if _dense is None:_dense=SentenceTransformer(DENSE_MODEL)
    return _dense

def build_rankings(name,corpus,queries):
    ids=list(corpus)
    texts=[doctext(corpus[d]) for d in ids]

    bm=BM25Okapi([tok(t) for t in tqdm(texts,desc=f"{name} BM25 tokenize",leave=False)])
    bmrank={}
    for qid,q in tqdm(queries.items(),desc=f"{name} BM25",leave=False):
        sc=np.asarray(bm.get_scores(tok(q)),dtype=np.float32)
        bmrank[qid]=[ids[i] for i in topidx(sc,TOP_K)]

    model=dense_model()
    tag=DENSE_MODEL.split("/")[-1].replace("-","_")
    ef=CACHE/f"{name}_{tag}_emb.npy"; idf=CACHE/f"{name}_{tag}_ids.json"
    if ef.exists() and idf.exists() and json.loads(idf.read_text())==ids:
        emb=np.load(ef,mmap_mode="r")
    else:
        emb=model.encode(texts,batch_size=128,show_progress_bar=True,normalize_embeddings=True).astype("float32")
        np.save(ef,emb); idf.write_text(json.dumps(ids))
    qids=list(queries)
    qe=model.encode([queries[q] for q in qids],batch_size=128,show_progress_bar=False,normalize_embeddings=True).astype("float32")
    drank={}
    for i,qid in enumerate(qids):
        sc=np.asarray(emb@qe[i],dtype=np.float32)
        drank[qid]=[ids[j] for j in topidx(sc,TOP_K)]
    hrank={q:rrf(bmrank[q],drank[q],TOP_K) for q in qids}
    return {"bm25":bmrank,"dense":drank,"hybrid":hrank},ids

# ============================================================
# NOTEBOOK CELL 14 / CODE CELL 8
# ============================================================
def fault_R(clean_rank,pool,qrels,severity,rng):
    out=list(clean_rank); forbidden=set(out)
    pos=[i for i,d in enumerate(out) if float(qrels.get(d,0))>0]
    n=max(1,int(round(max(1,len(pos))*severity)))
    targets=pos[:n] if pos else list(range(min(n,len(out))))
    for i in targets:
        replacement=None
        for _ in range(200):
            d=pool[int(rng.integers(0,len(pool)))]
            if d not in forbidden and float(qrels.get(d,0))<=0:
                replacement=d;break
        if replacement is None:return list(reversed(out))
        out[i]=replacement;forbidden.add(replacement)
    return out

def extract_clean(rank,qrels):
    return [(d,float(qrels.get(d,0))) for d in rank]

def fault_E(ev,severity,rng):
    out=[]
    for d,g in ev:
        if g>0 and rng.random()<severity: out.append((d,0.0))
        elif g<=0 and rng.random()<0.10*severity: out.append((d,1.0))
        else: out.append((d,g))
    return out

def aggregate_clean(ev,qrels):
    # Aggregate extracted relevance values. This is essential: an E-stage
    # corruption must propagate into the downstream answer.
    gains=[float(g) for _,g in ev[:TOP_K]]
    dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gains))
    ideal=sorted([float(v) for v in qrels.values() if float(v)>0],reverse=True)[:TOP_K]
    idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

def fault_A(ev,qrels,severity):
    clean=aggregate_clean(ev,qrels)
    # Plausible reducer bug: use an undiscounted mean instead of NDCG.
    wrong=float(np.mean([g for _,g in ev])) if ev else 0.0
    return (1-severity)*clean+severity*wrong

def execute3(clean_rank,pool,qrels,faults,repairs,severity,seed,qid):
    faults,repairs=set(faults),set(repairs)
    r=list(clean_rank)
    if "R" in faults and "R" not in repairs:
        r=fault_R(clean_rank,pool,qrels,severity,np.random.default_rng(stable_seed(seed,qid,"R",severity)))
    e=extract_clean(r,qrels)
    if "E" in faults and "E" not in repairs:
        e=fault_E(e,severity,np.random.default_rng(stable_seed(seed,qid,"E",severity)))
    y=aggregate_clean(e,qrels)
    if "A" in faults and "A" not in repairs:
        y=fault_A(e,qrels,severity)
    return float(y)

def diagnose3(clean_rank,pool,qrels,faults,severity,seed,qid):
    target=execute3(clean_rank,pool,qrels,(),PLAYERS,severity,seed,qid)
    losses={}
    for S in powerset(PLAYERS):
        y=execute3(clean_rank,pool,qrels,faults,S,severity,seed,qid)
        losses[S]=abs(y-target)
    base=losses[frozenset()]
    values={S:base-l for S,l in losses.items()}
    phi=exact_shapley(values,PLAYERS)
    div=harsanyi(values,PLAYERS)
    tol=max(1e-8,0.02*base)
    mcr=minimal_repair_sets(losses,PLAYERS,tol)
    cmcr,cost=min_cost_repair(losses,PLAYERS,STAGE_COST,tol)
    return target,base,losses,values,phi,div,mcr,cmcr,cost,tol

# Regression: a pure retrieval fault must be fully repaired by R.
_q={"d1":1,"d2":0,"d3":0,"d4":0}; _r=["d1","d2","d3"]
_,b,L,*_=diagnose3(_r,list(_q),_q,("R",),0.9,42,"toy")
assert L[frozenset({"R"})] <= 1e-8

# Extraction fault must be observable and repairable.
_,bE,LE,*_=diagnose3(_r,list(_q),_q,("E",),1.0,42,"toy-E")
assert bE > 0, "Injected extraction fault is not observable."
assert LE[frozenset({"E"})] <= 1e-8, "Repairing E did not restore clean output."
print("Counterfactual semantic self-tests passed for R and E.")

# ============================================================
# NOTEBOOK CELL 16 / CODE CELL 9
# ============================================================
def greedy_active(losses, players, costs, tol, budget=None):
    """
    Budgeted active diagnosis using measured intervention outcomes.
    1) observe baseline
    2) probe each singleton
    3) greedily add the stage with best loss reduction per added cost
    Stops at recovery or budget.
    Returns repair set and interventions used.
    """
    budget = budget or (2*len(players)-1)
    observed={frozenset():losses[frozenset()]}
    interventions=1

    # singleton probes
    for p in players:
        if interventions>=budget:break
        S=frozenset({p}); observed[S]=losses[S]; interventions+=1

    current=frozenset()
    current_loss=losses[current]
    chosen=set()

    while current_loss>tol and interventions<budget and len(chosen)<len(players):
        candidates=[]
        for p in players:
            if p in chosen: continue
            S=frozenset(chosen|{p})
            if S not in observed:
                observed[S]=losses[S]; interventions+=1
            gain=current_loss-observed[S]
            score=gain/max(costs[p],1e-9)
            candidates.append((score,gain,-costs[p],p,S))
            if interventions>=budget: break
        if not candidates: break
        _,_,_,p,S=max(candidates,key=lambda z:(z[0],z[1],z[2],z[3]))
        chosen.add(p); current=S; current_loss=observed[S]

    return current, interventions, current_loss

# Active-search unit check.
_toyL={
    frozenset():1.0,
    frozenset({"R"}):0.0,frozenset({"E"}):.8,frozenset({"A"}):.9,
    frozenset({"R","E"}):0.0,frozenset({"R","A"}):0.0,frozenset({"E","A"}):.7,
    frozenset({"R","E","A"}):0.0
}
S,n,l=greedy_active(_toyL,PLAYERS,STAGE_COST,1e-9,budget=5)
assert S==frozenset({"R"}) and l<=1e-9
print("Active-diagnosis self-test passed.")

# ============================================================
# NOTEBOOK CELL 18 / CODE CELL 10
# ============================================================
def run_block(dataset,retriever,rankings,pool,qrels_all,seed,severity):
    rows=[]
    for qid,rank in tqdm(rankings.items(),desc=f"{dataset}/{retriever}/s{seed}/v{severity}",leave=False):
        qrels=qrels_all[qid]
        ir={"ndcg10":ndcg_at_k(rank,qrels,10),"recall10":recall_at_k(rank,qrels,10),"mrr10":mrr_at_k(rank,qrels,10)}
        for truth in FAULTS:
            target,base,L,V,phi,div,mcr,cmcr,cost,tol=diagnose3(rank,pool,qrels,truth,severity,seed,qid)
            identifiable=base>1e-10
            k=len(truth)
            order=sorted(PLAYERS,key=lambda p:(-phi[p],p)); shap=set(order[:k])
            direct={p:V[frozenset({p})] for p in PLAYERS}; dord=sorted(PLAYERS,key=lambda p:(-direct[p],p)); delta=set(dord[:k])
            rr=np.random.default_rng(stable_seed(seed,qid,"random",truth,severity))
            rand=set(rr.choice(list(PLAYERS),size=k,replace=False).tolist())
            active_S,active_n,active_loss=greedy_active(L,PLAYERS,STAGE_COST,tol,budget=5)
            pred_mcr=set(next(iter(mcr),frozenset()))
            cmcr_set=set(cmcr or frozenset())
            rows.append({
                "dataset":dataset,"retriever":retriever,"qid":qid,"seed":seed,"severity":severity,
                "truth":"+".join(truth),"fault_k":k,"identifiable":identifiable,"baseline_loss":base,
                **ir,
                **{f"phi_{p}":phi[p] for p in PLAYERS},
                "shapley_exact":int(shap==set(truth)),"shapley_f1":set_f1(shap,truth),
                "delta_exact":int(delta==set(truth)),"delta_f1":set_f1(delta,truth),
                "random_exact":int(rand==set(truth)),"random_f1":set_f1(rand,truth),
                "mcr_exact":int(pred_mcr==set(truth)),"mcr_residual":L.get(frozenset(pred_mcr),np.nan),
                "cost_mcr_exact":int(cmcr_set==set(truth)),"cost_mcr_cost":cost,"cost_mcr_residual":L.get(frozenset(cmcr_set),np.nan),
                "active_exact":int(set(active_S)==set(truth)),"active_f1":set_f1(active_S,truth),
                "active_interventions":active_n,"active_residual":active_loss,
                "exhaustive_interventions":2**len(PLAYERS),
                "pair_interaction":sum(abs(v) for S,v in div.items() if len(S)==2),
                "triple_interaction":abs(div.get(frozenset(PLAYERS),0.0)),
            })
    return pd.DataFrame(rows)

for dataset in BENCHMARKS:
    print("\n===",dataset,"===")
    corpus,queries,qrels=load_beir(dataset)
    rankings,pool=build_rankings(dataset,corpus,queries)
    for retriever in RETRIEVERS:
        for seed in SEEDS:
            for sev in SEVERITIES:
                key=f"{dataset}__{retriever}__s{seed}__v{str(sev).replace('.','p')}"
                path=CHUNKS/f"{key}.parquet"
                if done(key,path):
                    print("skip",key);continue
                t=time.time()
                df=run_block(dataset,retriever,rankings[retriever],pool,qrels,seed,sev)
                tmp=path.with_suffix(".tmp.parquet");df.to_parquet(tmp,index=False);os.replace(tmp,path)
                mark_done(key,{"rows":len(df),"seconds":round(time.time()-t,2),"commit":COMMIT})
                print("saved",key,len(df))

# ============================================================
# NOTEBOOK CELL 20 / CODE CELL 11
# ============================================================
files=sorted(CHUNKS.glob("*.parquet"))
if not files: raise RuntimeError("No result chunks found.")
res=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
res=res.drop_duplicates(subset=["dataset","retriever","qid","seed","severity","truth"],keep="last")
res.to_parquet(ROOT/"all_results.parquet",index=False);res.to_csv(ROOT/"all_results.csv",index=False)
ri=res[res.identifiable].copy()

summary=ri.groupby(["dataset","retriever","severity"],as_index=False).agg(
    n=("qid","size"),
    ndcg10=("ndcg10","mean"),recall10=("recall10","mean"),mrr10=("mrr10","mean"),
    shapley_exact=("shapley_exact","mean"),shapley_f1=("shapley_f1","mean"),
    delta_exact=("delta_exact","mean"),delta_f1=("delta_f1","mean"),
    random_exact=("random_exact","mean"),random_f1=("random_f1","mean"),
    mcr_exact=("mcr_exact","mean"),cost_mcr_exact=("cost_mcr_exact","mean"),
    active_exact=("active_exact","mean"),active_f1=("active_f1","mean"),
    active_interventions=("active_interventions","mean"),exhaustive_interventions=("exhaustive_interventions","mean"),
    active_residual=("active_residual","mean"),pair_interaction=("pair_interaction","mean"),
)
summary.to_csv(TABLES/"C1_main_results.csv",index=False)
display(summary)

# Dataset-macro table avoids large datasets dominating conclusions.
macro=summary.groupby(["retriever","severity"],as_index=False).mean(numeric_only=True)
macro.to_csv(TABLES/"C1_macro_results.csv",index=False)
display(macro)

# Paired Wilcoxon + effect size.
pairs=ri[["shapley_f1","delta_f1","active_f1"]].dropna()
def paired_test(a,b,name):
    d=a-b
    if np.allclose(d,0): W,p=0.0,1.0
    else: W,p=stats.wilcoxon(a,b,zero_method="zsplit")
    return {"comparison":name,"n":len(a),"mean_a":a.mean(),"mean_b":b.mean(),"mean_diff":d.mean(),
            "wilcoxon_W":W,"p_value":p,"cliffs_delta":cliffs_delta(a,b)}
tests=pd.DataFrame([
    paired_test(pairs.shapley_f1,pairs.delta_f1,"Shapley vs singleton-delta"),
    paired_test(pairs.active_f1,pairs.delta_f1,"Active vs singleton-delta"),
])
tests.to_csv(TABLES/"C1_significance.csv",index=False)
display(tests)

# ============================================================
# NOTEBOOK CELL 22 / CODE CELL 12
# ============================================================
rng=np.random.default_rng(20260914)
def bootstrap_ci(df,col,B=2000):
    vals=[]
    groups=[g for _,g in df.groupby("dataset")]
    for _ in range(B):
        dsmeans=[]
        for g in groups:
            idx=rng.integers(0,len(g),len(g))
            dsmeans.append(g.iloc[idx][col].mean())
        vals.append(np.mean(dsmeans))
    return np.mean(vals),np.quantile(vals,.025),np.quantile(vals,.975)

rows=[]
for col in ["shapley_f1","delta_f1","random_f1","active_f1","mcr_exact","cost_mcr_exact"]:
    m,lo,hi=bootstrap_ci(ri,col)
    rows.append({"metric":col,"mean":m,"ci95_low":lo,"ci95_high":hi})
ci=pd.DataFrame(rows);ci.to_csv(TABLES/"C1_bootstrap_ci.csv",index=False);display(ci)

import matplotlib.pyplot as plt
curve=ri.groupby("active_interventions",as_index=False).agg(f1=("active_f1","mean"),exact=("active_exact","mean"))
plt.figure(figsize=(6,4));plt.plot(curve.active_interventions,curve.f1,marker="o")
plt.axvline(2**len(PLAYERS),linestyle="--")
plt.xlabel("Counterfactual interventions");plt.ylabel("Fault-set F1");plt.title("Active diagnosis: accuracy vs intervention budget")
plt.tight_layout();plt.savefig(PLOTS/"C1_accuracy_cost.png",dpi=220);plt.show()

# ============================================================
# NOTEBOOK CELL 24 / CODE CELL 13
# ============================================================
interesting=ri[
    ((ri.shapley_f1<1)&(ri.active_f1==1)) |
    ((ri.shapley_f1==1)&(ri.mcr_exact==0)) |
    (ri.active_residual>1e-8) |
    (ri.pair_interaction>ri.pair_interaction.quantile(.95))
].copy()
interesting=interesting.sort_values(["baseline_loss","pair_interaction"],ascending=False).head(100)
interesting.to_csv(FAILS/"C1_top_failure_cases.csv",index=False)
display(interesting.head(20))

# ============================================================
# NOTEBOOK CELL 26 / CODE CELL 14
# ============================================================
import shutil
for src in [TABLES,PLOTS,FAILS]:
    dst=EXPORT/src.name
    if dst.exists():shutil.rmtree(dst)
    shutil.copytree(src,dst)
for f in [ROOT/"all_results.csv",ROOT/"all_results.parquet",ROOT/"environment.txt",CP]:
    if f.exists():shutil.copy2(f,EXPORT/f.name)
readme=f"""# FaulTrace C1 Final Cross-Domain Evidence

Source commit: `{COMMIT}`

This folder contains PAPER_MODE/FAST_MODE causal fault-localization results,
active-diagnosis comparisons, minimum-cost repair results, statistics,
confidence intervals, and failure analysis.

Do not describe FAST_MODE results as final paper evidence.
"""
(EXPORT/"README.md").write_text(readme)
archive=ROOT/"FaulTrace_C1_GitHub_Export.zip"
if archive.exists():archive.unlink()
shutil.make_archive(str(archive.with_suffix("")),"zip",EXPORT)
print("Export:",archive)
