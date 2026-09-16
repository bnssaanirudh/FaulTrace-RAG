"""Exported code cells from FaulTrace_Q1_MASTER_All_Benchmarks_FIXED_V2.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
# ============================================================
# MASTER CONFIGURATION
# ============================================================

FAST_MODE = True
PAPER_MODE = False

# Run switches
RUN_CROSS_DOMAIN = True
RUN_MULTIHOP = True
RUN_RAGTRUTH = True

# Extra heavy BEIR datasets
RUN_HEAVY_BEIR = False

# Force regeneration even if Drive checkpoints exist
FORCE_RERUN = False

print({
    "FAST_MODE": FAST_MODE,
    "PAPER_MODE": PAPER_MODE,
    "RUN_CROSS_DOMAIN": RUN_CROSS_DOMAIN,
    "RUN_MULTIHOP": RUN_MULTIHOP,
    "RUN_RAGTRUTH": RUN_RAGTRUTH,
    "RUN_HEAVY_BEIR": RUN_HEAVY_BEIR,
})

# ============================================================
# NOTEBOOK CELL 3 / CODE CELL 2
# ============================================================
%%capture
!pip -q install -U \
  "beir>=2.0.0" \
  "datasets>=3.0" \
  "sentence-transformers>=3.0" \
  "transformers>=4.45" \
  "rank-bm25>=0.2.2" \
  "scikit-learn>=1.4" \
  "pandas>=2.0" \
  "pyarrow>=15" \
  "scipy>=1.11" \
  "tqdm>=4.66" \
  "matplotlib>=3.8"

# ============================================================
# NOTEBOOK CELL 4 / CODE CELL 3
# ============================================================
# ============================================================
# SHARED IMPORTS — RUN THIS BEFORE ALL EXPERIMENT SECTIONS
# ============================================================

from __future__ import annotations

import os
import sys
import re
import json
import time
import math
import random
import hashlib
import platform
import subprocess
import itertools
import shutil

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy import stats

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("Shared imports loaded successfully.")

# ============================================================
# NOTEBOOK CELL 5 / CODE CELL 4
# ============================================================
# ============================================================
# MASTER SANITY CHECK
# ============================================================

_required_names = [
    "Path",
    "np",
    "pd",
    "tqdm",
    "stats",
    "subprocess",
    "datetime",
    "timezone",
    "defaultdict",
]

_missing = [name for name in _required_names if name not in globals()]

if _missing:
    raise RuntimeError(
        "Required imports are missing: "
        + ", ".join(_missing)
        + ". Re-run the Shared Imports cell above."
    )

print("Master dependency sanity check: PASS")

# ============================================================
# NOTEBOOK CELL 8 / CODE CELL 5
# ============================================================
# ============================================================
# C1 CONFIGURATION
# ============================================================

if RUN_CROSS_DOMAIN:
    TOP_K = 10
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RETRIEVERS = ["bm25", "dense", "hybrid"]

    BASE_DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]
    HEAVY_DATASETS = ["trec-covid", "scidocs"]

    BENCHMARKS = BASE_DATASETS + (HEAVY_DATASETS if RUN_HEAVY_BEIR else [])

    N_QUERIES = 40 if FAST_MODE else 300
    SEEDS = [42] if FAST_MODE else [13, 42, 87, 2026, 31415]
    SEVERITIES = [0.35] if FAST_MODE else [0.20, 0.35, 0.50]

    PLAYERS = ("R", "E", "A")
    FAULTS = [
        ("R",), ("E",), ("A",),
        ("R", "E"), ("R", "A"), ("E", "A"),
        ("R", "E", "A"),
    ]

    STAGE_COST = {"R": 1.0, "E": 2.5, "A": 1.0}

    print("Cross-domain datasets:", BENCHMARKS)

# ============================================================
# NOTEBOOK CELL 11 / CODE CELL 6
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/FaulTrace_RAG_Experiments")
except Exception:
    DRIVE = Path("/content/FaulTrace_RAG_Experiments")

ROOT = DRIVE/"MASTER_Q1"/"C1_CROSS_DOMAIN"
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
# NOTEBOOK CELL 13 / CODE CELL 7
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
# NOTEBOOK CELL 14 / CODE CELL 8
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
# NOTEBOOK CELL 16 / CODE CELL 9
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
# NOTEBOOK CELL 18 / CODE CELL 10
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
# NOTEBOOK CELL 20 / CODE CELL 11
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
# NOTEBOOK CELL 22 / CODE CELL 12
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
# NOTEBOOK CELL 24 / CODE CELL 13
# ============================================================
if RUN_CROSS_DOMAIN:
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
# NOTEBOOK CELL 26 / CODE CELL 14
# ============================================================
if RUN_CROSS_DOMAIN:
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
# NOTEBOOK CELL 28 / CODE CELL 15
# ============================================================
if RUN_CROSS_DOMAIN:
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
# NOTEBOOK CELL 30 / CODE CELL 16
# ============================================================
if RUN_CROSS_DOMAIN:
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
# NOTEBOOK CELL 32 / CODE CELL 17
# ============================================================
if RUN_CROSS_DOMAIN:
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

# ============================================================
# NOTEBOOK CELL 34 / CODE CELL 18
# ============================================================
# ============================================================
# C2 CONFIGURATION
# ============================================================

if RUN_MULTIHOP or RUN_RAGTRUTH:
    N_MULTIHOP = 80 if FAST_MODE else 300

    SEEDS = [42] if FAST_MODE else [13, 42, 87, 2026, 31415]
    SEVERITIES = [0.40] if FAST_MODE else [0.25, 0.40, 0.55]

    TOP_SENTENCES = 16

    PLAYERS = ("S", "R", "E", "A", "G")

    FAULTS = [
        ("S",), ("R",), ("E",), ("A",), ("G",),
        ("S", "R"), ("R", "E"), ("E", "A"), ("A", "G"),
        ("S", "R", "E"), ("R", "E", "A"),
        ("E", "A", "G"), ("R", "E", "A", "G"),
    ]

    STAGE_COST = {
        "S": 1.0,
        "R": 2.0,
        "E": 4.0,
        "A": 1.0,
        "G": 8.0,
    }

    ACTIVE_BUDGET = 9

    RAG_TRAIN = 400 if FAST_MODE else 3000
    RAG_CAL = 240 if FAST_MODE else 1800
    RAG_TEST = 400 if FAST_MODE else 4000

    NLI_MODEL = (
        "cross-encoder/nli-MiniLM2-L6-H768"
        if FAST_MODE
        else "cross-encoder/nli-deberta-v3-base"
    )

    EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    ALPHA = 0.10
    DELTA = 0.05

# ============================================================
# NOTEBOOK CELL 37 / CODE CELL 19
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive",force_remount=False)
    DRIVE=Path("/content/drive/MyDrive/FaulTrace_RAG_Experiments")
except Exception:
    DRIVE=Path("/content/FaulTrace_RAG_Experiments")
ROOT=DRIVE/"MASTER_Q1"/"C2_MULTIHOP_RAGTRUTH"
CHUNKS=ROOT/"chunks";CACHE=ROOT/"cache";TABLES=ROOT/"paper_tables";PLOTS=ROOT/"plots";FAILS=ROOT/"failure_cases";SCORES=ROOT/"ragtruth_scores";EXPORT=ROOT/"github_export"
for p in [ROOT,CHUNKS,CACHE,TABLES,PLOTS,FAILS,SCORES,EXPORT]:p.mkdir(parents=True,exist_ok=True)

REPO=Path("/content/FaulTrace-RAG")
if REPO.exists():
    subprocess.run(["git","-C",str(REPO),"fetch","origin"],check=False);subprocess.run(["git","-C",str(REPO),"pull","--ff-only"],check=False)
else: subprocess.run(["git","clone","--depth","1","https://github.com/bnssaanirudh/FaulTrace-RAG.git",str(REPO)],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-e",str(REPO)],check=True)
COMMIT=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()

CP=ROOT/"checkpoint.json"
def read_cp():
    if CP.exists():
        try:return json.loads(CP.read_text())
        except:pass
    return {"commit":COMMIT,"completed":{}}
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,default=str));os.replace(tmp,path)
def mark(k,m):
    cp=read_cp();cp["completed"][k]=m;cp["updated_utc"]=datetime.now(timezone.utc).isoformat();atomic_json(CP,cp)
def done(k,p):return (not FORCE_RERUN) and k in read_cp().get("completed",{}) and p.exists()
(ROOT/"environment.txt").write_text(f"commit={COMMIT}\npython={sys.version}\nplatform={platform.platform()}\n")
print("Commit:",COMMIT)

# ============================================================
# NOTEBOOK CELL 39 / CODE CELL 20
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
# NOTEBOOK CELL 41 / CODE CELL 21
# ============================================================
if RUN_MULTIHOP:
    from datasets import load_dataset

    def take(ds,n,seed):
        n=min(n,len(ds));rg=np.random.default_rng(seed);idx=sorted(rg.choice(len(ds),n,replace=False).tolist());return ds.select(idx)

    def norm_hotpot(ex):
        ctx=ex["context"]
        docs={str(t):[str(s) for s in ss] for t,ss in zip(ctx["title"],ctx["sentences"])}
        sf=ex["supporting_facts"];supports={(str(t),int(i)) for t,i in zip(sf["title"],sf["sent_id"])}
        return {"id":str(ex["id"]),"question":str(ex["question"]),"answers":[str(ex.get("answer",""))],"docs":docs,"supports":supports}

    def norm_2wiki(ex):
        m=ex.get("metadata",ex);ctx=m["context"];sf=m.get("supporting_facts",ex.get("supporting_facts",{}))
        sent=ctx.get("content")
        if sent is None:sent=ctx.get("sentences")
        if sent is None:raise KeyError(f"Unsupported 2Wiki context keys: {list(ctx)}")
        docs={str(t):[str(s) for s in ss] for t,ss in zip(ctx["title"],sent)}
        supports={(str(t),int(i)) for t,i in zip(sf.get("title",[]),sf.get("sent_id",[]))}
        ans=ex.get("golden_answers")
        if ans is None:
            a=ex.get("answer","");ans=a if isinstance(a,list) else [a]
        return {"id":str(ex["id"]),"question":str(ex["question"]),"answers":[str(x) for x in ans] or [""],"docs":docs,"supports":supports}

    def load_multihop():
        hp=take(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation"),N_MULTIHOP,20260914)
        tw=take(load_dataset("cmriat/2wikimultihopqa",split="validation"),N_MULTIHOP,20260915)
        return {"hotpotqa":[norm_hotpot(x) for x in hp],"2wikimultihopqa":[norm_2wiki(x) for x in tw]}

    MULTIHOP=load_multihop()
    for k,v in MULTIHOP.items():print(k,len(v),len(v[0]["supports"]))

    # Schema regression tests
    a=norm_2wiki({"id":"x","question":"q","golden_answers":["a"],"metadata":{"context":{"title":["T"],"content":[["s"]]},"supporting_facts":{"title":["T"],"sent_id":[0]}}})
    b=norm_2wiki({"id":"y","question":"q","answer":"a","context":{"title":["T"],"sentences":[["s"]]},"supporting_facts":{"title":["T"],"sent_id":[0]}})
    assert ("T",0) in a["supports"] and ("T",0) in b["supports"]
    print("2Wiki schema tests passed.")

# ============================================================
# NOTEBOOK CELL 43 / CODE CELL 22
# ============================================================
from rank_bm25 import BM25Okapi

def tokens(x):return re.findall(r"[A-Za-z0-9]+",str(x).lower())
def flatten(ex):
    return [{"title":t,"sent_id":i,"text":str(s),"key":(t,i)} for t,ss in ex["docs"].items() for i,s in enumerate(ss)]

RANK_CACHE={}
def full_rank(ex,scope):
    key=(ex["id"],tuple(sorted(scope)))
    if key in RANK_CACHE:return RANK_CACHE[key]
    rows=[r for r in flatten(ex) if r["title"] in scope]
    if not rows:RANK_CACHE[key]=[];return []
    bm=BM25Okapi([tokens(r["text"]) for r in rows]);sc=np.asarray(bm.get_scores(tokens(ex["question"])))
    rank=[rows[i] for i in np.argsort(sc)[::-1]]
    RANK_CACHE[key]=rank
    return rank

def scope_clean(ex):return set(ex["docs"])
def scope_fault(ex,severity):
    scope=set(ex["docs"]);sup=sorted({t for t,_ in ex["supports"]})
    n=max(1,int(round(max(1,len(sup))*severity)))
    for t in sup[:n]:scope.discard(t)
    return scope

def retrieve_fault(rank,ex,severity):
    top=list(rank[:TOP_SENTENCES]);tail=list(rank[TOP_SENTENCES:]);sup=ex["supports"]
    targets=[i for i,r in enumerate(top) if r["key"] in sup]
    n=max(1,int(round(max(1,len(targets))*severity)))
    repl=[r for r in tail if r["key"] not in sup]
    for i,r in zip(targets[:n],repl):top[i]=r
    return top

def extract_clean(rows,ex):
    return [{"key":r["key"],"support":int(r["key"] in ex["supports"]),"text":r["text"]} for r in rows]
def extract_fault(ev,severity,seed):
    rg=np.random.default_rng(seed);out=[]
    for x in ev:
        y=dict(x)
        if y["support"] and rg.random()<severity:y["support"]=0
        elif not y["support"] and rg.random()<0.1*severity:y["support"]=1
        out.append(y)
    return out
def aggregate_clean(ev,ex):
    need=max(1,len(ex["supports"]))
    found=len({x["key"] for x in ev if x["support"] and x["key"] in ex["supports"]})
    cov=min(1.0,found/need);return {"coverage":cov,"ready":cov>=1.0}
def aggregate_fault(ev,ex,severity):
    c=aggregate_clean(ev,ex);bad=max(0.0,c["coverage"]*(1-severity));return {"coverage":bad,"ready":bad>=.999}
def generate_clean(agg,ex):return ex["answers"][0] if agg["ready"] else "<INSUFFICIENT_EVIDENCE>"
def generate_fault(agg,ex):return "<HALLUCINATED_ANSWER>" if agg["ready"] else ex["answers"][0]

def execute5(ex,faults,repairs,severity,seed):
    faults,repairs=set(faults),set(repairs)
    scope=scope_clean(ex) if ("S" not in faults or "S" in repairs) else scope_fault(ex,severity)
    rank=full_rank(ex,scope)
    rows=rank[:TOP_SENTENCES] if ("R" not in faults or "R" in repairs) else retrieve_fault(rank,ex,severity)
    ev=extract_clean(rows,ex)
    if "E" in faults and "E" not in repairs:ev=extract_fault(ev,severity,stable_seed(seed,ex["id"],"E",severity))
    agg=aggregate_clean(ev,ex) if ("A" not in faults or "A" in repairs) else aggregate_fault(ev,ex,severity)
    ans=generate_clean(agg,ex) if ("G" not in faults or "G" in repairs) else generate_fault(agg,ex)
    return {"coverage":float(agg["coverage"]),"answer":ans}

def outloss(y,t):
    return .55*abs(y["coverage"]-t["coverage"])+.45*float(y["answer"]!=t["answer"])

def diagnose5(ex,faults,severity,seed):
    target=execute5(ex,(),PLAYERS,severity,seed)
    L={}
    for S in powerset(PLAYERS):L[S]=outloss(execute5(ex,faults,S,severity,seed),target)
    base=L[frozenset()];V={S:base-l for S,l in L.items()};phi=exact_shapley(V,PLAYERS);div=harsanyi(V,PLAYERS)
    tol=max(1e-8,.02*base);mcr=minimal_repair_sets(L,PLAYERS,tol);cmcr,cost=min_cost_repair(L,PLAYERS,STAGE_COST,tol)
    return target,base,L,V,phi,div,mcr,cmcr,cost,tol

# ============================================================
# NOTEBOOK CELL 45 / CODE CELL 23
# ============================================================
def greedy_active(losses,players,costs,tol,budget=ACTIVE_BUDGET):
    observed={frozenset():losses[frozenset()]};n=1;chosen=set();current=frozenset();cur=losses[current]
    # seed with all singleton probes while budget allows
    for p in players:
        if n>=budget:break
        S=frozenset({p});observed[S]=losses[S];n+=1
    while cur>tol and n<budget and len(chosen)<len(players):
        cand=[]
        for p in players:
            if p in chosen:continue
            S=frozenset(chosen|{p})
            if S not in observed:
                if n>=budget:break
                observed[S]=losses[S];n+=1
            gain=cur-observed[S];cand.append((gain/max(costs[p],1e-9),gain,-costs[p],p,S))
        if not cand:break
        *_,p,S=max(cand,key=lambda z:(z[0],z[1],z[2],z[3]))
        chosen.add(p);current=S;cur=observed[S]
    return current,n,cur

# ============================================================
# NOTEBOOK CELL 47 / CODE CELL 24
# ============================================================
if RUN_MULTIHOP:
    def run_block(dataset,examples,seed,sev):
        rows=[]
        for ex in tqdm(examples,desc=f"{dataset}/s{seed}/v{sev}",leave=False):
            for truth in FAULTS:
                target,base,L,V,phi,div,mcr,cmcr,cost,tol=diagnose5(ex,truth,sev,seed)
                k=len(truth);order=sorted(PLAYERS,key=lambda p:(-phi[p],p));shap=set(order[:k])
                direct={p:V[frozenset({p})] for p in PLAYERS};dord=sorted(PLAYERS,key=lambda p:(-direct[p],p));delta=set(dord[:k])
                rg=np.random.default_rng(stable_seed(seed,ex["id"],"random",truth,sev));rand=set(rg.choice(list(PLAYERS),size=k,replace=False).tolist())
                active,nprobe,aloss=greedy_active(L,PLAYERS,STAGE_COST,tol)
                pmcr=set(next(iter(mcr),frozenset()));cm=set(cmcr or frozenset())
                rows.append({
                    "dataset":dataset,"id":ex["id"],"seed":seed,"severity":sev,"truth":"+".join(truth),"k":k,
                    "identifiable":base>1e-10,"baseline_loss":base,
                    "shapley_exact":int(shap==set(truth)),"shapley_f1":set_f1(shap,truth),
                    "delta_exact":int(delta==set(truth)),"delta_f1":set_f1(delta,truth),
                    "random_exact":int(rand==set(truth)),"random_f1":set_f1(rand,truth),
                    "mcr_exact":int(pmcr==set(truth)),"mcr_residual":L.get(frozenset(pmcr),np.nan),
                    "cost_mcr_exact":int(cm==set(truth)),"cost_mcr_cost":cost,"cost_mcr_residual":L.get(frozenset(cm),np.nan),
                    "active_exact":int(set(active)==set(truth)),"active_f1":set_f1(active,truth),"active_probes":nprobe,"active_residual":aloss,
                    "exhaustive_probes":2**len(PLAYERS),
                    "pair_interaction":sum(abs(v) for S,v in div.items() if len(S)==2),
                    "higher_interaction":sum(abs(v) for S,v in div.items() if len(S)>=3),
                })
        return pd.DataFrame(rows)

    for dataset,examples in MULTIHOP.items():
        for seed in SEEDS:
            for sev in SEVERITIES:
                key=f"mh__{dataset}__s{seed}__v{str(sev).replace('.','p')}";path=CHUNKS/f"{key}.parquet"
                if done(key,path):print("skip",key);continue
                t=time.time();df=run_block(dataset,examples,seed,sev)
                tmp=path.with_suffix(".tmp.parquet");df.to_parquet(tmp,index=False);os.replace(tmp,path)
                mark(key,{"rows":len(df),"seconds":round(time.time()-t,2),"commit":COMMIT});print("saved",key,len(df))

# ============================================================
# NOTEBOOK CELL 49 / CODE CELL 25
# ============================================================
if RUN_MULTIHOP:
    files=sorted(CHUNKS.glob("mh__*.parquet"))
    if not files:raise RuntimeError("No multi-hop chunks.")
    mh=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    mh=mh.drop_duplicates(subset=["dataset","id","seed","severity","truth"],keep="last")
    mh.to_parquet(ROOT/"multihop_all.parquet",index=False);mh.to_csv(ROOT/"multihop_all.csv",index=False)
    mi=mh[mh.identifiable].copy()

    main=mi.groupby(["dataset","severity"],as_index=False).agg(
        n=("id","size"),shapley_f1=("shapley_f1","mean"),delta_f1=("delta_f1","mean"),random_f1=("random_f1","mean"),
        mcr_exact=("mcr_exact","mean"),cost_mcr_exact=("cost_mcr_exact","mean"),
        active_f1=("active_f1","mean"),active_probes=("active_probes","mean"),
        exhaustive_probes=("exhaustive_probes","mean"),active_residual=("active_residual","mean"),
        pair_interaction=("pair_interaction","mean"),higher_interaction=("higher_interaction","mean")
    )
    main.to_csv(TABLES/"C2_multihop_main.csv",index=False);display(main)

    fault=mi.groupby(["dataset","truth"],as_index=False).agg(
        n=("id","size"),shapley_f1=("shapley_f1","mean"),delta_f1=("delta_f1","mean"),
        active_f1=("active_f1","mean"),mcr_exact=("mcr_exact","mean"),cost_mcr_exact=("cost_mcr_exact","mean")
    )
    fault.to_csv(TABLES/"C2_fault_ablation.csv",index=False)

    def ptest(a,b,name):
        d=a-b
        if np.allclose(d,0):W,p=0.0,1.0
        else:W,p=stats.wilcoxon(a,b,zero_method="zsplit")
        return {"comparison":name,"n":len(a),"mean_a":a.mean(),"mean_b":b.mean(),"diff":d.mean(),"W":W,"p":p,"cliffs_delta":cliffs_delta(a,b)}
    tests=pd.DataFrame([
        ptest(mi.shapley_f1,mi.delta_f1,"Shapley vs singleton"),
        ptest(mi.active_f1,mi.delta_f1,"Active vs singleton"),
    ])
    tests.to_csv(TABLES/"C2_multihop_significance.csv",index=False);display(tests)

# ============================================================
# NOTEBOOK CELL 51 / CODE CELL 26
# ============================================================
if RUN_RAGTRUTH:
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    from scipy.special import softmax

    trainpool=load_dataset("wandb/RAGTruth-processed",split="train")
    testds=load_dataset("wandb/RAGTruth-processed",split="test")

    # Strict disjoint split: model fitting != certification calibration.
    need=min(len(trainpool),RAG_TRAIN+RAG_CAL)
    rg=np.random.default_rng(20260914)
    perm=rg.permutation(len(trainpool))[:need]
    train_idx=sorted(perm[:min(RAG_TRAIN,len(perm))].tolist())
    cal_start=len(train_idx)
    cal_idx=sorted(perm[cal_start:cal_start+min(RAG_CAL,max(0,len(perm)-cal_start))].tolist())
    ragtrain=trainpool.select(train_idx)
    cal=trainpool.select(cal_idx)
    test=take(testds,RAG_TEST,20260915)

    assert set(train_idx).isdisjoint(set(cal_idx))
    print("RAGTruth splits:",len(ragtrain),"train |",len(cal),"calibration |",len(test),"test")

    emb=SentenceTransformer(EMB_MODEL)
    nli=CrossEncoder(NLI_MODEL,max_length=512)

    def hallucinated(ex):
        x=ex.get("hallucination_labels_processed",ex.get("hallucination_labels"))
        if isinstance(x,dict):return int(any(int(v or 0)>0 for v in x.values()))
        if isinstance(x,list):return int(len(x)>0)
        if isinstance(x,str):
            try:
                z=json.loads(x)
                if isinstance(z,dict):return int(any(bool(v) for v in z.values()))
                if isinstance(z,list):return int(len(z)>0)
            except:pass
            return int(x.strip().lower() not in ("","[]","{}","none","null","0","false"))
        return int(bool(x))

    def sents(text,n=6):
        x=[z.strip() for z in re.split(r"(?<=[.!?])\s+",str(text)) if z.strip()]
        return x[:n] or [str(text)[:500]]
    def chunks(text,n=5,chars=1200):
        t=str(text);x=[t[i:i+chars] for i in range(0,len(t),chars)]
        return x[:n] or [""]

    def lexical_overlap(a,b):
        A=set(re.findall(r"[A-Za-z0-9]+",a.lower()));B=set(re.findall(r"[A-Za-z0-9]+",b.lower()))
        return len(A&B)/len(A) if A else 0.0

    def features(ex):
        ss=sents(ex["output"],4 if FAST_MODE else 6);cc=chunks(ex["context"],3 if FAST_MODE else 5)
        cemb=emb.encode(cc,normalize_embeddings=True);semb=emb.encode(ss,normalize_embeddings=True)
        sim=np.asarray(semb)@np.asarray(cemb).T
        maxsim=sim.max(axis=1)
        pairs=[(c,s) for s in ss for c in cc]
        logits=np.asarray(nli.predict(pairs,batch_size=32,show_progress_bar=False));pr=softmax(logits,axis=1)
        # SentenceTransformers cross-encoder NLI convention for these models is checked from model labels when available.
        labels={str(v).lower():int(k) for k,v in getattr(nli.model.config,"id2label",{}).items()}
        entail=labels.get("entailment",1);contr=labels.get("contradiction",0)
        ent=[];con=[];j=0
        for _ in ss:
            block=pr[j:j+len(cc)];j+=len(cc)
            ent.append(float(block[:,entail].max()));con.append(float(block[:,contr].max()))
        lex=[max(lexical_overlap(s,c) for c in cc) for s in ss]
        return {
            "sim_min":float(np.min(maxsim)),"sim_mean":float(np.mean(maxsim)),
            "entail_min":float(np.min(ent)),"entail_mean":float(np.mean(ent)),
            "contr_max":float(np.max(con)),"contr_mean":float(np.mean(con)),
            "lex_min":float(np.min(lex)),"lex_mean":float(np.mean(lex)),
            "n_sent":len(ss),"out_chars":len(str(ex["output"])),"ctx_chars":len(str(ex["context"]))
        }

    def score_dataset(ds,split):
        rows=[]
        for ex in tqdm(ds,desc=f"RAGTruth {split}"):
            uid=hashlib.sha256((str(ex["id"])+"||"+str(ex.get("model",""))+"||"+str(ex.get("output",""))).encode()).hexdigest()[:20]
            f=SCORES/f"{split}_{uid}.json"
            if f.exists() and not FORCE_RERUN:
                rows.append(json.loads(f.read_text()));continue
            row={"uid":uid,"id":str(ex["id"]),"task_type":str(ex.get("task_type","")),"y":hallucinated(ex),**features(ex)}
            atomic_json(f,row);rows.append(row)
        return pd.DataFrame(rows)

    # Fit features on a dedicated training split.
    # Calibration remains untouched for certificate selection.
    # Test remains untouched for final held-out reporting.
    traindf=score_dataset(ragtrain,"train")
    caldf=score_dataset(cal,"cal")
    testdf=score_dataset(test,"test")

    assert len(traindf) > 0, "RAGTruth training feature table is empty."
    assert len(caldf) > 0, "RAGTruth calibration feature table is empty."
    assert len(testdf) > 0, "RAGTruth test feature table is empty."

    traindf.to_parquet(
        ROOT/"ragtruth_train_features.parquet",
        index=False
    )
    caldf.to_parquet(
        ROOT/"ragtruth_cal_features.parquet",
        index=False
    )
    testdf.to_parquet(
        ROOT/"ragtruth_test_features.parquet",
        index=False
    )

    print(
        "RAGTruth feature tables ready:",
        len(traindf), "train |",
        len(caldf), "calibration |",
        len(testdf), "test"
    )

# ============================================================
# NOTEBOOK CELL 53 / CODE CELL 27
# ============================================================
if RUN_RAGTRUTH:
    required_frames=["traindf","caldf","testdf"]
    missing_frames=[
        name for name in required_frames
        if name not in globals()
    ]

    if missing_frames:
        raise RuntimeError(
            "Missing RAGTruth feature tables: "
            + ", ".join(missing_frames)
            + ". Run the RAGTruth feature-extraction cell immediately above first."
        )

    FEATURE_SETS={
        "lexical":["lex_min","lex_mean","n_sent","out_chars","ctx_chars"],
        "similarity":["sim_min","sim_mean","n_sent","out_chars","ctx_chars"],
        "nli":["entail_min","entail_mean","contr_max","contr_mean","n_sent"],
        "full":["sim_min","sim_mean","entail_min","entail_mean","contr_max","contr_mean","lex_min","lex_mean","n_sent","out_chars","ctx_chars"]
    }
    rows=[];models={}
    for name,cols in FEATURE_SETS.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))
        model.fit(traindf[cols],traindf.y)
        prob=model.predict_proba(testdf[cols])[:,1]
        pred=(prob>=.5).astype(int);y=testdf.y.values
        row={"model":name,"n_test":len(y),"auroc":roc_auc_score(y,prob),"auprc":average_precision_score(y,prob),
             "f1":f1_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0)}
        rows.append(row);models[name]=(model,cols)
    ab=pd.DataFrame(rows);ab.to_csv(TABLES/"C2_ragtruth_feature_ablation.csv",index=False);display(ab)
    BEST=max(rows,key=lambda r:r["auprc"])["model"]
    print("Best by held-out AUPRC (reported only; certification threshold still calibrated on calibration split):",BEST)

# ============================================================
# NOTEBOOK CELL 55 / CODE CELL 28
# ============================================================
if RUN_RAGTRUTH:
    # Certification model is fixed to FULL a priori to avoid selecting a certification model on test.
    model,cols=models["full"]
    calprob=model.predict_proba(caldf[cols])[:,1]  # hallucination probability
    testprob=model.predict_proba(testdf[cols])[:,1]

    def binom_upper(err,n,delta):
        if n<=0:return 1.0
        if err>=n:return 1.0
        return float(stats.beta.ppf(1-delta,err+1,n-err))

    def calibrate_threshold(prob,y,alpha=ALPHA,delta=DELTA,min_n=30):
        # certify if hallucination probability <= tau
        cand=np.unique(np.quantile(prob,np.linspace(.01,.95,120)))
        de=delta/max(1,len(cand))  # family-wise correction for threshold search
        feasible=[]
        for tau in cand:
            mask=prob<=tau;n=int(mask.sum())
            if n<min_n:continue
            err=int(np.asarray(y)[mask].sum());ub=binom_upper(err,n,de)
            if ub<=alpha:feasible.append((tau,n/len(prob),ub,n))
        return sorted(feasible,key=lambda x:(-x[1],x[2],x[0]))[0] if feasible else None

    cert=calibrate_threshold(calprob,caldf.y.values)
    if cert is None:
        certrow={"tau":np.nan,"cal_coverage":0.0,"cal_upper_bound":np.nan,"test_coverage":0.0,"test_risk":np.nan,"alpha":ALPHA,"delta":DELTA}
        print("No non-zero-coverage certificate satisfies the requested bound. This remains a valid negative result.")
    else:
        tau,ccov,cub,cn=cert
        mask=testprob<=tau
        certrow={"tau":tau,"cal_coverage":ccov,"cal_upper_bound":cub,"cal_n":cn,
                 "test_coverage":float(mask.mean()),"test_risk":float(testdf.y.values[mask].mean()) if mask.any() else np.nan,
                 "test_n":int(mask.sum()),"alpha":ALPHA,"delta":DELTA}
    certtab=pd.DataFrame([certrow]);certtab.to_csv(TABLES/"C2_risk_certificate.csv",index=False);display(certtab)

# ============================================================
# NOTEBOOK CELL 57 / CODE CELL 29
# ============================================================
import matplotlib.pyplot as plt

# ============================================================
# MULTI-HOP FAILURE ANALYSIS
# ============================================================

if RUN_MULTIHOP:

    if "mi" not in globals() or "main" not in globals():
        raise RuntimeError(
            "Multi-hop result tables are missing. "
            "Run the multi-hop merge/statistics cell first."
        )

    fails=mi[
        ((mi.shapley_f1<1)&(mi.active_f1==1)) |
        ((mi.shapley_f1==1)&(mi.mcr_exact==0)) |
        (mi.active_residual>1e-8) |
        (
            mi.higher_interaction >
            mi.higher_interaction.quantile(.95)
        )
    ].sort_values(
        ["baseline_loss","higher_interaction"],
        ascending=False
    ).head(150)

    fails.to_csv(
        FAILS/"C2_multihop_failures.csv",
        index=False
    )

    p=main.groupby("dataset")[
        [
            "shapley_f1",
            "delta_f1",
            "active_f1"
        ]
    ].mean()

    ax=p.plot(
        kind="bar",
        figsize=(8,4)
    )

    ax.set_ylim(0,1)
    ax.set_ylabel("Fault-set F1")
    ax.set_title(
        "Multi-hop diagnosis ablation"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS/"C2_multihop_ablation.png",
        dpi=220
    )

    plt.show()


# ============================================================
# RAGTRUTH FAILURE ANALYSIS
# ============================================================

if RUN_RAGTRUTH:

    required=[
        "testdf",
        "testprob",
        "ab"
    ]

    missing=[
        name
        for name in required
        if name not in globals()
    ]

    if missing:
        raise RuntimeError(
            "Missing RAGTruth result objects: "
            + ", ".join(missing)
            + ". Run feature extraction, ablation, and certification first."
        )

    testout=testdf.copy()

    testout[
        "p_hallucination"
    ]=testprob

    testout[
        "error"
    ]=np.abs(
        testout.y
        -
        testout.p_hallucination
    )

    testout.sort_values(
        "error",
        ascending=False
    ).head(150).to_csv(
        FAILS/
        "C2_ragtruth_hard_cases.csv",
        index=False
    )

    ax=ab.set_index(
        "model"
    )[
        [
            "auroc",
            "auprc",
            "f1"
        ]
    ].plot(
        kind="bar",
        figsize=(8,4)
    )

    ax.set_ylim(0,1)

    ax.set_title(
        "RAGTruth grounding feature ablation"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS/
        "C2_ragtruth_ablation.png",
        dpi=220
    )

    plt.show()


print(
    "Failure-analysis section complete."
)

# ============================================================
# NOTEBOOK CELL 59 / CODE CELL 30
# ============================================================
import shutil
for src in [TABLES,PLOTS,FAILS]:
    dst=EXPORT/src.name
    if dst.exists():shutil.rmtree(dst)
    shutil.copytree(src,dst)
for f in [ROOT/"multihop_all.csv",ROOT/"multihop_all.parquet",ROOT/"ragtruth_train_features.parquet",ROOT/"ragtruth_cal_features.parquet",ROOT/"ragtruth_test_features.parquet",ROOT/"environment.txt",CP]:
    if f.exists():shutil.copy2(f,EXPORT/f.name)
(EXPORT/"README.md").write_text(f"""# FaulTrace C2 Final Multi-Hop and RAGTruth Evidence

Source commit: `{COMMIT}`

Contains multi-hop causal diagnosis, active diagnosis, cost-aware MCR,
RAGTruth grounding feature ablations, selective-risk certification,
statistics and failure analysis.

Negative certification results are intentionally preserved.
FAST_MODE outputs are not final paper evidence.
""")
archive=ROOT/"FaulTrace_C2_GitHub_Export.zip"
if archive.exists():archive.unlink()
shutil.make_archive(str(archive.with_suffix("")),"zip",EXPORT)
print("Export:",archive)

# ============================================================
# NOTEBOOK CELL 61 / CODE CELL 31
# ============================================================
# ============================================================
# MASTER EXPORT
# ============================================================

import shutil
from pathlib import Path

try:
    MASTER_ROOT = DRIVE / "MASTER_Q1"
except NameError:
    MASTER_ROOT = Path("/content/FaulTrace_RAG_Experiments/MASTER_Q1")

MASTER_ROOT.mkdir(parents=True, exist_ok=True)

master_readme = f"""
# FaulTrace-RAG MASTER Q1 Experimental Evidence

This directory contains the complete final experimental program:

- C1_CROSS_DOMAIN
- C2_MULTIHOP_RAGTRUTH

Mode:
- FAST_MODE = {FAST_MODE}
- PAPER_MODE = {PAPER_MODE}

Run switches:
- RUN_CROSS_DOMAIN = {RUN_CROSS_DOMAIN}
- RUN_MULTIHOP = {RUN_MULTIHOP}
- RUN_RAGTRUTH = {RUN_RAGTRUTH}

Do not report FAST_MODE outputs as final manuscript results.
"""

(MASTER_ROOT / "README.md").write_text(master_readme)

archive = MASTER_ROOT / "FaulTrace_MASTER_Q1_GitHub_Results.zip"

if archive.exists():
    archive.unlink()

# Avoid recursively including the zip itself.
tmp_export = MASTER_ROOT / "_master_export"
if tmp_export.exists():
    shutil.rmtree(tmp_export)

tmp_export.mkdir()

for folder_name in ["C1_CROSS_DOMAIN", "C2_MULTIHOP_RAGTRUTH"]:
    src = MASTER_ROOT / folder_name
    if src.exists():
        shutil.copytree(
            src,
            tmp_export / folder_name,
            ignore=shutil.ignore_patterns(
                "cache",
                "*.tmp",
                "*.tmp.parquet",
                "FaulTrace_*_GitHub_Export.zip"
            )
        )

shutil.copy2(MASTER_ROOT / "README.md", tmp_export / "README.md")

shutil.make_archive(
    str(archive.with_suffix("")),
    "zip",
    tmp_export
)

shutil.rmtree(tmp_export)

print("Master archive:", archive)
