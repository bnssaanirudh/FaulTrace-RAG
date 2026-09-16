"""Exported code cells from FaulTrace_Q1_FINAL_CLOSURE.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
# ============================================================
# 0. FINAL RUN CONFIGURATION
# ============================================================

FAST_MODE = False
PAPER_MODE = True

RUN_CROSS_DOMAIN = True
RUN_ACTIVE_DIAGNOSIS = True
RUN_COST_AWARE_MCR = True
RUN_CERTIFICATION_SWEEP = True

RUN_HEAVY_BEIR = True

FORCE_RERUN = False

print({
    "FAST_MODE": FAST_MODE,
    "PAPER_MODE": PAPER_MODE,
    "RUN_CROSS_DOMAIN": RUN_CROSS_DOMAIN,
    "RUN_ACTIVE_DIAGNOSIS": RUN_ACTIVE_DIAGNOSIS,
    "RUN_COST_AWARE_MCR": RUN_COST_AWARE_MCR,
    "RUN_CERTIFICATION_SWEEP": RUN_CERTIFICATION_SWEEP,
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
# NOTEBOOK CELL 5 / CODE CELL 3
# ============================================================
from __future__ import annotations

import os, sys, re, json, time, math, random, hashlib, itertools
import platform, subprocess, shutil, zipfile
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

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/FaulTrace_RAG_Experiments")
except Exception:
    DRIVE = Path("/content/FaulTrace_RAG_Experiments")

ROOT = DRIVE / "FINAL_CLOSURE"
C1_ROOT = ROOT / "cross_domain"
ACTIVE_ROOT = ROOT / "active_diagnosis"
COST_ROOT = ROOT / "cost_mcr"
CERT_ROOT = ROOT / "certification"
TABLES = ROOT / "paper_tables"
PLOTS = ROOT / "plots"
FAILS = ROOT / "failure_cases"
EXPORT = ROOT / "github_export"

for p in [ROOT,C1_ROOT,ACTIVE_ROOT,COST_ROOT,CERT_ROOT,TABLES,PLOTS,FAILS,EXPORT]:
    p.mkdir(parents=True, exist_ok=True)

REPO = Path("/content/FaulTrace-RAG")
if REPO.exists():
    subprocess.run(["git","-C",str(REPO),"fetch","origin"],check=False)
    subprocess.run(["git","-C",str(REPO),"pull","--ff-only"],check=False)
else:
    subprocess.run(
        ["git","clone","--depth","1","https://github.com/bnssaanirudh/FaulTrace-RAG.git",str(REPO)],
        check=True
    )

subprocess.run([sys.executable,"-m","pip","install","-q","-e",str(REPO)],check=True)
COMMIT = subprocess.check_output(
    ["git","-C",str(REPO),"rev-parse","HEAD"], text=True
).strip()

CHECKPOINT = ROOT / "checkpoint.json"

def read_checkpoint():
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    return {"commit":COMMIT,"completed":{}}

def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(tmp,path)

def mark_done(key, meta):
    cp = read_checkpoint()
    cp["completed"][key] = meta
    cp["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(CHECKPOINT,cp)

def done(key, path):
    return (
        (not FORCE_RERUN)
        and key in read_checkpoint().get("completed",{})
        and path.exists()
    )

(ROOT/"environment.txt").write_text(
    f"utc={datetime.now(timezone.utc).isoformat()}\n"
    f"commit={COMMIT}\n"
    f"python={sys.version}\n"
    f"platform={platform.platform()}\n"
)

print("Repository commit:",COMMIT)
print("Results root:",ROOT)

# ============================================================
# NOTEBOOK CELL 7 / CODE CELL 4
# ============================================================
def powerset(players):
    players=tuple(players)
    for r in range(len(players)+1):
        for c in itertools.combinations(players,r):
            yield frozenset(c)

def exact_shapley(values, players):
    players=tuple(players)
    n=len(players)
    phi={p:0.0 for p in players}
    for p in players:
        others=[x for x in players if x!=p]
        for S in powerset(others):
            w=math.factorial(len(S))*math.factorial(n-len(S)-1)/math.factorial(n)
            phi[p]+=w*(values[S|{p}]-values[S])
    return phi

def harsanyi(values, players):
    out={}
    for S in powerset(players):
        out[S]=sum(((-1)**(len(S)-len(T)))*values[T] for T in powerset(S))
    return out

def stable_seed(*parts):
    h=hashlib.sha256("||".join(map(str,parts)).encode()).hexdigest()
    return int(h[:8],16)

def set_f1(pred,truth):
    pred,truth=set(pred),set(truth)
    if not pred and not truth:return 1.0
    if not pred or not truth:return 0.0
    p=len(pred&truth)/len(pred)
    r=len(pred&truth)/len(truth)
    return 2*p*r/(p+r) if p+r else 0.0

def ndcg_from_extracted(ev,qrels,k=10):
    gains=[float(g) for _,g in ev[:k]]
    dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gains))
    ideal=sorted([float(v) for v in qrels.values() if float(v)>0],reverse=True)[:k]
    idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

def ndcg_at_k(rank,qrels,k=10):
    return ndcg_from_extracted([(d,float(qrels.get(d,0))) for d in rank],qrels,k)

def recall_at_k(rank,qrels,k=10):
    rel={d for d,g in qrels.items() if float(g)>0}
    return len(rel&set(rank[:k]))/len(rel) if rel else float("nan")

def mrr_at_k(rank,qrels,k=10):
    for i,d in enumerate(rank[:k],1):
        if float(qrels.get(d,0))>0:return 1/i
    return 0.0

# Mathematical invariants
toy={
    frozenset():0.0,
    frozenset({"R"}):.2,
    frozenset({"E"}):.3,
    frozenset({"A"}):.1,
    frozenset({"R","E"}):.65,
    frozenset({"R","A"}):.36,
    frozenset({"E","A"}):.47,
    frozenset({"R","E","A"}):.85,
}
phi=exact_shapley(toy,("R","E","A"))
assert abs(sum(phi.values())-.85)<1e-10
print("Causal math self-test: PASS")

# ============================================================
# NOTEBOOK CELL 9 / CODE CELL 5
# ============================================================
# ============================================================
# A1. Cross-domain configuration
# ============================================================

TOP_K=10
DENSE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RETRIEVERS=["bm25","dense","hybrid"]

BEIR_BASE_DATASETS=["scifact","nfcorpus","arguana","fiqa"]
BEIR_HEAVY_DATASETS=["trec-covid","scidocs"]

BEIR_DATASETS = (
    BEIR_BASE_DATASETS + BEIR_HEAVY_DATASETS
    if RUN_HEAVY_BEIR
    else BEIR_BASE_DATASETS
)

BEIR_QUERIES = 50 if FAST_MODE else 300
BEIR_SEEDS = [42] if FAST_MODE else [13,42,87,2026,31415]
BEIR_SEVERITIES = [0.35] if FAST_MODE else [0.20,0.35,0.50]

C1_PLAYERS=("R","E","A")
C1_FAULTS=[
    ("R",),("E",),("A",),
    ("R","E"),("R","A"),("E","A"),
    ("R","E","A"),
]

print("Final BEIR datasets:",BEIR_DATASETS)

# ============================================================
# NOTEBOOK CELL 10 / CODE CELL 6
# ============================================================
if RUN_CROSS_DOMAIN:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    BEIR_CACHE=C1_ROOT/"cache"
    BEIR_CHUNKS=C1_ROOT/"chunks"
    BEIR_CACHE.mkdir(parents=True,exist_ok=True)
    BEIR_CHUNKS.mkdir(parents=True,exist_ok=True)

    def tokenize(x):
        return re.findall(r"[A-Za-z0-9]+",str(x).lower())

    def doc_text(doc):
        return (str(doc.get("title",""))+" "+str(doc.get("text",""))).strip()

    def top_indices(scores,k):
        k=min(k,len(scores))
        if k==0:return np.array([],dtype=int)
        idx=np.argpartition(scores,-k)[-k:]
        return idx[np.argsort(scores[idx])[::-1]]

    def rrf(a,b,k=10,c=60):
        s=defaultdict(float)
        for i,d in enumerate(a,1):s[d]+=1/(c+i)
        for i,d in enumerate(b,1):s[d]+=1/(c+i)
        return [d for d,_ in sorted(s.items(),key=lambda z:(-z[1],z[0]))[:k]]

    _dense_model=None

    def get_dense():
        global _dense_model
        if _dense_model is None:
            _dense_model=SentenceTransformer(DENSE_MODEL)
        return _dense_model

    def load_beir(name):
        base=BEIR_CACHE/"datasets"
        base.mkdir(parents=True,exist_ok=True)
        path=base/name
        if not path.exists():
            util.download_and_unzip(
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
                str(base)
            )
        corpus,queries,qrels=GenericDataLoader(data_folder=str(path)).load(split="test")
        qids=sorted(set(queries)&set(qrels))
        rg=np.random.default_rng(20260914)
        if len(qids)>BEIR_QUERIES:
            qids=sorted(rg.choice(qids,size=BEIR_QUERIES,replace=False).tolist())
        return corpus,{q:queries[q] for q in qids},{q:qrels[q] for q in qids}

    def build_rankings(name,corpus,queries):
        ids=list(corpus)
        texts=[doc_text(corpus[d]) for d in ids]

        bm=BM25Okapi([tokenize(t) for t in tqdm(texts,desc=f"{name} BM25 tokenize",leave=False)])
        bmrank={}
        for qid,q in tqdm(queries.items(),desc=f"{name} BM25",leave=False):
            sc=np.asarray(bm.get_scores(tokenize(q)),dtype=np.float32)
            bmrank[qid]=[ids[i] for i in top_indices(sc,TOP_K)]

        model=get_dense()
        tag=DENSE_MODEL.split("/")[-1].replace("-","_")
        ef=BEIR_CACHE/f"{name}_{tag}_emb.npy"
        idf=BEIR_CACHE/f"{name}_{tag}_ids.json"

        if ef.exists() and idf.exists() and json.loads(idf.read_text())==ids:
            emb=np.load(ef,mmap_mode="r")
        else:
            emb=model.encode(
                texts,batch_size=128,show_progress_bar=True,normalize_embeddings=True
            ).astype("float32")
            np.save(ef,emb)
            idf.write_text(json.dumps(ids))

        qids=list(queries)
        qe=model.encode(
            [queries[q] for q in qids],
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype("float32")

        drank={}
        for i,qid in enumerate(qids):
            sc=np.asarray(emb@qe[i],dtype=np.float32)
            drank[qid]=[ids[j] for j in top_indices(sc,TOP_K)]

        hrank={qid:rrf(bmrank[qid],drank[qid],TOP_K) for qid in qids}
        return {"bm25":bmrank,"dense":drank,"hybrid":hrank},ids

# ============================================================
# NOTEBOOK CELL 11 / CODE CELL 7
# ============================================================
if RUN_CROSS_DOMAIN:
    def c1_fault_R(clean_rank,pool,qrels,severity,rng):
        out=list(clean_rank)
        forbidden=set(out)
        positions=[i for i,d in enumerate(out) if float(qrels.get(d,0))>0]
        n=max(1,int(round(max(1,len(positions))*severity)))
        targets=positions[:n] if positions else list(range(min(n,len(out))))
        for i in targets:
            replacement=None
            for _ in range(200):
                d=pool[int(rng.integers(0,len(pool)))]
                if d not in forbidden and float(qrels.get(d,0))<=0:
                    replacement=d
                    break
            if replacement is None:
                return list(reversed(out))
            out[i]=replacement
            forbidden.add(replacement)
        return out

    def c1_extract(rank,qrels):
        return [(d,float(qrels.get(d,0))) for d in rank]

    def c1_fault_E(ev,severity,rng):
        out=[]
        for d,g in ev:
            if g>0 and rng.random()<severity:
                out.append((d,0.0))
            elif g<=0 and rng.random()<0.10*severity:
                out.append((d,1.0))
            else:
                out.append((d,g))
        return out

    def c1_fault_A(ev,qrels,severity):
        clean=ndcg_from_extracted(ev,qrels,TOP_K)
        wrong=float(np.mean([g for _,g in ev])) if ev else 0.0
        return (1-severity)*clean+severity*wrong

    def c1_execute(clean_rank,pool,qrels,faults,repairs,severity,seed,qid):
        faults,repairs=set(faults),set(repairs)
        rank=list(clean_rank)
        if "R" in faults and "R" not in repairs:
            rank=c1_fault_R(
                clean_rank,pool,qrels,severity,
                np.random.default_rng(stable_seed(seed,qid,"R",severity))
            )
        ev=c1_extract(rank,qrels)
        if "E" in faults and "E" not in repairs:
            ev=c1_fault_E(
                ev,severity,
                np.random.default_rng(stable_seed(seed,qid,"E",severity))
            )
        y=ndcg_from_extracted(ev,qrels,TOP_K)
        if "A" in faults and "A" not in repairs:
            y=c1_fault_A(ev,qrels,severity)
        return float(y)

    def c1_diagnose(rank,pool,qrels,truth,severity,seed,qid):
        target=c1_execute(rank,pool,qrels,(),C1_PLAYERS,severity,seed,qid)
        losses={}
        for S in powerset(C1_PLAYERS):
            y=c1_execute(rank,pool,qrels,truth,S,severity,seed,qid)
            losses[S]=abs(y-target)
        base=losses[frozenset()]
        values={S:base-l for S,l in losses.items()}
        phi=exact_shapley(values,C1_PLAYERS)
        div=harsanyi(values,C1_PLAYERS)
        tol=max(1e-8,.02*base)
        valid=[S for S,l in losses.items() if l<=tol]
        mcr=min(valid,key=lambda S:(len(S),tuple(sorted(S)))) if valid else frozenset()
        return base,losses,values,phi,div,mcr

    # Semantic regression
    q={"d1":1,"d2":0,"d3":0,"d4":0}
    r=["d1","d2","d3"]
    base,L,*_=c1_diagnose(r,list(q),q,("E",),1.0,42,"toy")
    assert base>0
    assert L[frozenset({"E"})]<=1e-8
    print("Cross-domain causal semantic test: PASS")

# ============================================================
# NOTEBOOK CELL 12 / CODE CELL 8
# ============================================================
if RUN_CROSS_DOMAIN:
    def c1_run_block(dataset,retriever,rankings,pool,qrels_all,seed,severity):
        rows=[]
        for qid,rank in tqdm(
            rankings.items(),
            desc=f"{dataset}/{retriever}/s{seed}/v{severity}",
            leave=False
        ):
            qrels=qrels_all[qid]
            ir={
                "ndcg10":ndcg_at_k(rank,qrels,10),
                "recall10":recall_at_k(rank,qrels,10),
                "mrr10":mrr_at_k(rank,qrels,10),
            }

            for truth in C1_FAULTS:
                base,L,V,phi,div,mcr=c1_diagnose(
                    rank,pool,qrels,truth,severity,seed,qid
                )

                k=len(truth)
                order=sorted(C1_PLAYERS,key=lambda p:(-phi[p],p))
                pred=set(order[:k])

                direct={p:V[frozenset({p})] for p in C1_PLAYERS}
                dorder=sorted(C1_PLAYERS,key=lambda p:(-direct[p],p))
                delta=set(dorder[:k])

                rg=np.random.default_rng(
                    stable_seed(seed,qid,"random",truth,severity)
                )
                rand=set(
                    rg.choice(list(C1_PLAYERS),size=k,replace=False).tolist()
                )

                rows.append({
                    "dataset":dataset,
                    "retriever":retriever,
                    "qid":qid,
                    "seed":seed,
                    "severity":severity,
                    "truth":"+".join(truth),
                    "k":k,
                    "identifiable":base>1e-10,
                    "baseline_loss":base,
                    **ir,
                    "shapley_exact":int(pred==set(truth)),
                    "shapley_f1":set_f1(pred,truth),
                    "delta_exact":int(delta==set(truth)),
                    "delta_f1":set_f1(delta,truth),
                    "random_exact":int(rand==set(truth)),
                    "random_f1":set_f1(rand,truth),
                    "mcr_exact":int(set(mcr)==set(truth)),
                    "mcr_residual":L.get(mcr,np.nan),
                    "pair_interaction":sum(
                        abs(v) for S,v in div.items() if len(S)==2
                    ),
                    "triple_interaction":abs(
                        div.get(frozenset(C1_PLAYERS),0.0)
                    ),
                })

        return pd.DataFrame(rows)

    for dataset in BEIR_DATASETS:
        print("\n===",dataset,"===")
        corpus,queries,qrels=load_beir(dataset)
        rankings,pool=build_rankings(dataset,corpus,queries)

        for retriever in RETRIEVERS:
            for seed in BEIR_SEEDS:
                for sev in BEIR_SEVERITIES:
                    key=f"c1__{dataset}__{retriever}__s{seed}__v{str(sev).replace('.','p')}"
                    out=C1_ROOT/f"{key}.parquet"

                    if done(key,out):
                        print("skip",key)
                        continue

                    t=time.time()

                    df=c1_run_block(
                        dataset,retriever,rankings[retriever],pool,qrels,seed,sev
                    )

                    tmp=out.with_suffix(".tmp.parquet")
                    df.to_parquet(tmp,index=False)
                    os.replace(tmp,out)

                    mark_done(
                        key,
                        {
                            "rows":len(df),
                            "seconds":round(time.time()-t,2),
                            "commit":COMMIT,
                        }
                    )

                    print("saved",key,len(df))

# ============================================================
# NOTEBOOK CELL 13 / CODE CELL 9
# ============================================================
if RUN_CROSS_DOMAIN:
    files=sorted(C1_ROOT.glob("c1__*.parquet"))
    if not files:
        raise RuntimeError("No cross-domain result chunks were created.")

    c1=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    c1=c1.drop_duplicates(
        subset=["dataset","retriever","qid","seed","severity","truth"],
        keep="last"
    )

    c1.to_parquet(ROOT/"C1_FINAL_all.parquet",index=False)
    c1.to_csv(ROOT/"C1_FINAL_all.csv",index=False)

    c1i=c1[c1.identifiable].copy()

    c1_summary=c1i.groupby(
        ["dataset","retriever","severity"],as_index=False
    ).agg(
        n=("qid","size"),
        ndcg10=("ndcg10","mean"),
        recall10=("recall10","mean"),
        mrr10=("mrr10","mean"),
        shapley_exact=("shapley_exact","mean"),
        shapley_f1=("shapley_f1","mean"),
        delta_exact=("delta_exact","mean"),
        delta_f1=("delta_f1","mean"),
        random_exact=("random_exact","mean"),
        random_f1=("random_f1","mean"),
        mcr_exact=("mcr_exact","mean"),
        residual=("mcr_residual","mean"),
        pair_interaction=("pair_interaction","mean"),
        triple_interaction=("triple_interaction","mean"),
    )

    c1_summary.to_csv(TABLES/"FINAL_C1_cross_domain.csv",index=False)
    display(c1_summary)

# ============================================================
# NOTEBOOK CELL 15 / CODE CELL 10
# ============================================================
# ============================================================
# B1. Multi-hop configuration
# ============================================================

MH_N = 50 if FAST_MODE else 200
MH_SEEDS = [42] if FAST_MODE else [13,42,87,2026,31415]
MH_SEVERITIES = [0.40] if FAST_MODE else [0.25,0.40,0.55]

MH_PLAYERS=("S","R","E","A","G")
MH_FAULTS=[
    ("S",),("R",),("E",),("A",),("G",),
    ("S","R"),("R","E"),("E","A"),("A","G"),
    ("S","R","E"),("R","E","A"),("E","A","G"),
    ("R","E","A","G"),
]

TOP_SENTENCES=16

ACTIVE_BUDGETS=[4,6,8,10,12] if not FAST_MODE else [4,6]
ACTIVE_CONFIDENCE=0.90

INTERVENTION_COST={
    "S":1.0,
    "R":2.0,
    "E":4.0,
    "A":1.0,
    "G":8.0,
}

# ============================================================
# NOTEBOOK CELL 16 / CODE CELL 11
# ============================================================
if RUN_ACTIVE_DIAGNOSIS or RUN_COST_AWARE_MCR:
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    def deterministic_take(ds,n,seed):
        n=min(n,len(ds))
        rg=np.random.default_rng(seed)
        idx=sorted(rg.choice(len(ds),size=n,replace=False).tolist())
        return ds.select(idx)

    def norm_hotpot(ex):
        ctx=ex["context"]
        docs={
            str(t):[str(s) for s in ss]
            for t,ss in zip(ctx["title"],ctx["sentences"])
        }
        sf=ex["supporting_facts"]
        supports={
            (str(t),int(i))
            for t,i in zip(sf["title"],sf["sent_id"])
        }
        return {
            "id":str(ex["id"]),
            "question":str(ex["question"]),
            "answers":[str(ex.get("answer",""))],
            "docs":docs,
            "supports":supports,
        }

    def norm_2wiki(ex):
        m=ex.get("metadata",ex)
        ctx=m["context"]
        sf=m.get("supporting_facts",ex.get("supporting_facts",{}))

        sent=ctx.get("content")
        if sent is None:
            sent=ctx.get("sentences")

        if sent is None:
            raise KeyError(
                f"Unsupported 2Wiki context keys: {list(ctx.keys())}"
            )

        docs={
            str(t):[str(s) for s in ss]
            for t,ss in zip(ctx["title"],sent)
        }

        supports={
            (str(t),int(i))
            for t,i in zip(sf.get("title",[]),sf.get("sent_id",[]))
        }

        ans=ex.get("golden_answers")
        if ans is None:
            a=ex.get("answer","")
            ans=a if isinstance(a,list) else [a]

        return {
            "id":str(ex["id"]),
            "question":str(ex["question"]),
            "answers":[str(x) for x in ans] or [""],
            "docs":docs,
            "supports":supports,
        }

    hp=deterministic_take(
        load_dataset("hotpotqa/hotpot_qa","distractor",split="validation"),
        MH_N,
        20260914
    )

    tw=deterministic_take(
        load_dataset("cmriat/2wikimultihopqa",split="validation"),
        MH_N,
        20260915
    )

    MULTIHOP={
        "hotpotqa":[norm_hotpot(x) for x in hp],
        "2wikimultihopqa":[norm_2wiki(x) for x in tw],
    }

    for k,v in MULTIHOP.items():
        print(k,len(v))

# ============================================================
# NOTEBOOK CELL 17 / CODE CELL 12
# ============================================================
if RUN_ACTIVE_DIAGNOSIS or RUN_COST_AWARE_MCR:
    def mh_tokens(x):
        return re.findall(r"[A-Za-z0-9]+",str(x).lower())

    def mh_flatten(ex):
        return [
            {
                "title":t,
                "sent_id":i,
                "text":str(s),
                "key":(t,i),
            }
            for t,ss in ex["docs"].items()
            for i,s in enumerate(ss)
        ]

    RANK_CACHE={}

    def mh_full_rank(ex,scope):
        key=(ex["id"],tuple(sorted(scope)))

        if key in RANK_CACHE:
            return RANK_CACHE[key]

        rows=[
            r for r in mh_flatten(ex)
            if r["title"] in scope
        ]

        if not rows:
            RANK_CACHE[key]=[]
            return []

        bm=BM25Okapi([mh_tokens(r["text"]) for r in rows])
        sc=np.asarray(bm.get_scores(mh_tokens(ex["question"])))
        rank=[rows[i] for i in np.argsort(sc)[::-1]]

        RANK_CACHE[key]=rank
        return rank

    def mh_scope_clean(ex):
        return set(ex["docs"])

    def mh_scope_fault(ex,severity):
        scope=set(ex["docs"])
        support_titles=sorted({t for t,_ in ex["supports"]})
        n=max(1,int(round(max(1,len(support_titles))*severity)))
        for t in support_titles[:n]:
            scope.discard(t)
        return scope

    def mh_retrieve_fault(full_rank,ex,severity):
        top=list(full_rank[:TOP_SENTENCES])
        tail=list(full_rank[TOP_SENTENCES:])
        sup=ex["supports"]

        targets=[i for i,r in enumerate(top) if r["key"] in sup]
        n=max(1,int(round(max(1,len(targets))*severity)))

        replacements=[r for r in tail if r["key"] not in sup]

        for i,repl in zip(targets[:n],replacements):
            top[i]=repl

        return top

    def mh_extract_clean(rows,ex):
        return [
            {
                "key":r["key"],
                "text":r["text"],
                "support":int(r["key"] in ex["supports"]),
            }
            for r in rows
        ]

    def mh_extract_fault(ev,severity,seed):
        rg=np.random.default_rng(seed)
        out=[]

        for x in ev:
            y=dict(x)
            if y["support"] and rg.random()<severity:
                y["support"]=0
            elif not y["support"] and rg.random()<.10*severity:
                y["support"]=1
            out.append(y)

        return out

    def mh_aggregate_clean(ev,ex):
        need=max(1,len(ex["supports"]))
        found=len({
            x["key"]
            for x in ev
            if x["support"] and x["key"] in ex["supports"]
        })
        coverage=min(1.0,found/need)
        return {
            "coverage":coverage,
            "ready":coverage>=1.0,
        }

    def mh_aggregate_fault(ev,ex,severity):
        c=mh_aggregate_clean(ev,ex)
        bad=max(0.0,c["coverage"]*(1-severity))
        return {
            "coverage":bad,
            "ready":bad>=.999,
        }

    def mh_generate_clean(agg,ex):
        return (
            ex["answers"][0]
            if agg["ready"]
            else "<INSUFFICIENT_EVIDENCE>"
        )

    def mh_generate_fault(agg,ex):
        return (
            "<HALLUCINATED_ANSWER>"
            if agg["ready"]
            else ex["answers"][0]
        )

    def mh_execute(ex,faults,repairs,severity,seed):
        faults,repairs=set(faults),set(repairs)

        scope=(
            mh_scope_clean(ex)
            if ("S" not in faults or "S" in repairs)
            else mh_scope_fault(ex,severity)
        )

        full=mh_full_rank(ex,scope)

        rows=(
            full[:TOP_SENTENCES]
            if ("R" not in faults or "R" in repairs)
            else mh_retrieve_fault(full,ex,severity)
        )

        ev=mh_extract_clean(rows,ex)

        if "E" in faults and "E" not in repairs:
            ev=mh_extract_fault(
                ev,
                severity,
                stable_seed(seed,ex["id"],"E",severity)
            )

        agg=(
            mh_aggregate_clean(ev,ex)
            if ("A" not in faults or "A" in repairs)
            else mh_aggregate_fault(ev,ex,severity)
        )

        ans=(
            mh_generate_clean(agg,ex)
            if ("G" not in faults or "G" in repairs)
            else mh_generate_fault(agg,ex)
        )

        return {
            "coverage":float(agg["coverage"]),
            "answer":ans,
        }

    def mh_loss(y,target):
        return (
            .55*abs(y["coverage"]-target["coverage"])
            +
            .45*float(y["answer"]!=target["answer"])
        )

    MH_INTERVENTIONS=list(powerset(MH_PLAYERS))
    MH_NAMES=[
        "none" if not S else "+".join(sorted(S))
        for S in MH_INTERVENTIONS
    ]

    def mh_signature(ex,truth,severity,seed):
        target=mh_execute(ex,(),MH_PLAYERS,severity,seed)

        losses={}
        for S in MH_INTERVENTIONS:
            losses[S]=mh_loss(
                mh_execute(ex,truth,S,severity,seed),
                target
            )

        base=losses[frozenset()]

        norm={
            S:(losses[S]/base if base>1e-12 else 0.0)
            for S in MH_INTERVENTIONS
        }

        return base,losses,norm

# ============================================================
# NOTEBOOK CELL 18 / CODE CELL 13
# ============================================================
if RUN_ACTIVE_DIAGNOSIS or RUN_COST_AWARE_MCR:
    SIG_DIR=ACTIVE_ROOT/"signatures"
    SIG_DIR.mkdir(parents=True,exist_ok=True)

    rows=[]

    for dataset,examples in MULTIHOP.items():

        for seed in MH_SEEDS:

            for sev in MH_SEVERITIES:

                key=f"sig__{dataset}__s{seed}__v{str(sev).replace('.','p')}"
                out=SIG_DIR/f"{key}.parquet"

                if done(key,out):
                    rows.append(pd.read_parquet(out))
                    print("skip",key)
                    continue

                block=[]

                for ex in tqdm(
                    examples,
                    desc=f"signatures {dataset}/s{seed}/v{sev}",
                    leave=False
                ):

                    for truth in MH_FAULTS:

                        base,L,N=mh_signature(
                            ex,
                            truth,
                            sev,
                            seed
                        )

                        row={
                            "dataset":dataset,
                            "id":ex["id"],
                            "seed":seed,
                            "severity":sev,
                            "truth":"+".join(truth),
                            "truth_set":json.dumps(sorted(truth)),
                            "identifiable":base>1e-10,
                            "baseline_loss":base,
                        }

                        for S,name in zip(MH_INTERVENTIONS,MH_NAMES):
                            row[f"loss__{name}"]=float(L[S])
                            row[f"norm__{name}"]=float(N[S])

                        block.append(row)

                df=pd.DataFrame(block)

                tmp=out.with_suffix(".tmp.parquet")
                df.to_parquet(tmp,index=False)
                os.replace(tmp,out)

                mark_done(
                    key,
                    {
                        "rows":len(df),
                        "commit":COMMIT,
                    }
                )

                rows.append(df)
                print("saved",key,len(df))

    sig=pd.concat(rows,ignore_index=True)
    sig=sig[sig.identifiable].copy()

    # Group-disjoint train/test split by dataset + question id.
    def is_train(dataset,qid):
        h=stable_seed("split",dataset,qid)
        return (h % 1000) < 700

    sig["split"]=[
        "train" if is_train(d,q) else "test"
        for d,q in zip(sig.dataset,sig.id)
    ]

    sig.to_parquet(ACTIVE_ROOT/"all_signatures.parquet",index=False)

    print(sig.split.value_counts())

# ============================================================
# NOTEBOOK CELL 20 / CODE CELL 14
# ============================================================
if RUN_ACTIVE_DIAGNOSIS:
    train_sig=sig[sig.split=="train"].copy()
    test_sig=sig[sig.split=="test"].copy()

    norm_cols=[f"norm__{n}" for n in MH_NAMES]

    hypotheses=sorted(train_sig.truth.unique().tolist())

    template_mean={}
    template_std={}

    for h in hypotheses:
        g=train_sig[train_sig.truth==h]
        template_mean[h]=g[norm_cols].mean().values.astype(float)
        template_std[h]=np.maximum(
            g[norm_cols].std(ddof=0).fillna(0).values.astype(float),
            0.08
        )

    intervention_sets=MH_INTERVENTIONS
    intervention_names=MH_NAMES

    # Do not actively choose baseline/no-intervention; it is already known.
    candidate_indices=list(range(1,len(intervention_sets)))

    def intervention_cost(S):
        return .25+sum(INTERVENTION_COST[p] for p in S)

    def posterior_from_observations(obs):
        # obs = {intervention_index: normalized_loss}
        logp=np.zeros(len(hypotheses),dtype=float)

        for hi,h in enumerate(hypotheses):
            mu=template_mean[h]
            sd=template_std[h]

            for j,y in obs.items():
                z=(y-mu[j])/sd[j]
                logp[hi]+=(-.5*z*z)-math.log(sd[j])

        logp-=np.max(logp)
        p=np.exp(logp)
        return p/p.sum()

    def choose_next_intervention(posterior,observed):
        # Expected class separation / intervention cost.
        choices=[]

        for j in candidate_indices:
            if j in observed:
                continue

            means=np.array(
                [template_mean[h][j] for h in hypotheses],
                dtype=float
            )

            weighted_mean=float(np.sum(posterior*means))
            disagreement=float(
                np.sum(
                    posterior*(means-weighted_mean)**2
                )
            )

            S=intervention_sets[j]
            score=disagreement/max(intervention_cost(S),1e-9)

            choices.append(
                (
                    score,
                    disagreement,
                    -intervention_cost(S),
                    j
                )
            )

        return max(choices)[-1] if choices else None

    def bacd(row,budget,confidence=ACTIVE_CONFIDENCE):
        observed={}
        posterior=np.ones(len(hypotheses))/len(hypotheses)

        for step in range(budget):

            j=choose_next_intervention(
                posterior,
                observed
            )

            if j is None:
                break

            y=float(row[norm_cols[j]])
            observed[j]=y

            posterior=posterior_from_observations(observed)

            if (
                len(observed)>=3
                and posterior.max()>=confidence
            ):
                break

        pred=hypotheses[int(np.argmax(posterior))]

        return {
            "pred":pred,
            "n_probes":len(observed),
            "confidence":float(posterior.max()),
            "observed":[intervention_names[j] for j in observed],
        }

    def random_probe(row,budget,seed):
        rg=np.random.default_rng(seed)
        js=rg.choice(
            candidate_indices,
            size=min(budget,len(candidate_indices)),
            replace=False
        ).tolist()

        obs={
            j:float(row[norm_cols[j]])
            for j in js
        }

        posterior=posterior_from_observations(obs)
        pred=hypotheses[int(np.argmax(posterior))]

        return {
            "pred":pred,
            "n_probes":len(obs),
            "confidence":float(posterior.max()),
        }

    print(
        "Learned templates:",
        len(hypotheses),
        "fault hypotheses from",
        len(train_sig),
        "training cases."
    )

# ============================================================
# NOTEBOOK CELL 21 / CODE CELL 15
# ============================================================
if RUN_ACTIVE_DIAGNOSIS:
    active_rows=[]

    for budget in ACTIVE_BUDGETS:

        for idx,row in tqdm(
            test_sig.iterrows(),
            total=len(test_sig),
            desc=f"BACD budget={budget}"
        ):

            truth=set(row.truth.split("+"))

            bac=bacd(row,budget)

            rnd=random_probe(
                row,
                budget,
                stable_seed(
                    row.dataset,
                    row.id,
                    row.seed,
                    row.severity,
                    row.truth,
                    budget
                )
            )

            bpred=set(bac["pred"].split("+"))
            rpred=set(rnd["pred"].split("+"))

            # Exhaustive Shapley baseline from full loss lattice.
            base=float(row["loss__none"])
            values={}
            for S,name in zip(MH_INTERVENTIONS,MH_NAMES):
                loss=float(row[f"loss__{name}"])
                values[S]=base-loss

            phi=exact_shapley(values,MH_PLAYERS)
            k=len(truth)
            shap=set(
                sorted(
                    MH_PLAYERS,
                    key=lambda p:(-phi[p],p)
                )[:k]
            )

            active_rows.append({
                "dataset":row.dataset,
                "id":row.id,
                "seed":row.seed,
                "severity":row.severity,
                "truth":row.truth,
                "fault_k":k,
                "budget":budget,
                "bacd_exact":int(bpred==truth),
                "bacd_f1":set_f1(bpred,truth),
                "bacd_probes":bac["n_probes"],
                "bacd_confidence":bac["confidence"],
                "random_probe_exact":int(rpred==truth),
                "random_probe_f1":set_f1(rpred,truth),
                "shapley_exact":int(shap==truth),
                "shapley_f1":set_f1(shap,truth),
                "exhaustive_probes":len(MH_INTERVENTIONS),
            })

    active=pd.DataFrame(active_rows)
    active.to_parquet(ACTIVE_ROOT/"bacd_results.parquet",index=False)
    active.to_csv(ACTIVE_ROOT/"bacd_results.csv",index=False)

    active_summary=active.groupby(
        ["budget"],as_index=False
    ).agg(
        n=("id","size"),
        bacd_exact=("bacd_exact","mean"),
        bacd_f1=("bacd_f1","mean"),
        bacd_probes=("bacd_probes","mean"),
        random_probe_exact=("random_probe_exact","mean"),
        random_probe_f1=("random_probe_f1","mean"),
        shapley_exact=("shapley_exact","mean"),
        shapley_f1=("shapley_f1","mean"),
        exhaustive_probes=("exhaustive_probes","mean"),
    )

    active_summary.to_csv(
        TABLES/"FINAL_BACD_accuracy_cost.csv",
        index=False
    )

    display(active_summary)

# ============================================================
# NOTEBOOK CELL 23 / CODE CELL 16
# ============================================================
if RUN_ACTIVE_DIAGNOSIS:
    test_rows=[]

    for budget in ACTIVE_BUDGETS:
        g=active[active.budget==budget]

        def wilcox(a,b):
            d=np.asarray(a)-np.asarray(b)
            if np.allclose(d,0):
                return 0.0,1.0
            return stats.wilcoxon(a,b,zero_method="zsplit")

        W1,p1=wilcox(g.bacd_f1,g.random_probe_f1)
        W2,p2=wilcox(g.bacd_f1,g.shapley_f1)

        test_rows.extend([
            {
                "budget":budget,
                "comparison":"BACD vs random probes",
                "mean_a":g.bacd_f1.mean(),
                "mean_b":g.random_probe_f1.mean(),
                "mean_diff":(g.bacd_f1-g.random_probe_f1).mean(),
                "W":W1,
                "p":p1,
            },
            {
                "budget":budget,
                "comparison":"BACD vs exhaustive Shapley",
                "mean_a":g.bacd_f1.mean(),
                "mean_b":g.shapley_f1.mean(),
                "mean_diff":(g.bacd_f1-g.shapley_f1).mean(),
                "W":W2,
                "p":p2,
            },
        ])

    pd.DataFrame(test_rows).to_csv(
        TABLES/"FINAL_BACD_significance.csv",
        index=False
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7,4))
    plt.plot(
        active_summary.bacd_probes,
        active_summary.bacd_f1,
        marker="o",
        label="BACD"
    )
    plt.plot(
        active_summary.bacd_probes,
        active_summary.random_probe_f1,
        marker="o",
        label="Random probes"
    )
    plt.axhline(
        active_summary.shapley_f1.mean(),
        linestyle="--",
        label="Exhaustive Shapley"
    )
    plt.xlabel("Mean interventions")
    plt.ylabel("Fault-set F1")
    plt.ylim(0,1)
    plt.title("Active diagnosis: accuracy vs intervention cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS/"FINAL_BACD_accuracy_cost.png",dpi=220)
    plt.show()

# ============================================================
# NOTEBOOK CELL 25 / CODE CELL 17
# ============================================================
if RUN_COST_AWARE_MCR:
    COST_PROFILES={
        "equal":{
            "S":1.0,"R":1.0,"E":1.0,"A":1.0,"G":1.0
        },
        "latency":{
            "S":0.5,"R":2.0,"E":4.0,"A":1.0,"G":8.0
        },
        "cloud_cost":{
            "S":0.2,"R":1.5,"E":5.0,"A":0.8,"G":10.0
        },
    }

    TOLERANCES=[0.01,0.02,0.05]

    cost_rows=[]

    # Use held-out active test cases only.
    cost_cases=test_sig.copy()

    for _,row in tqdm(
        cost_cases.iterrows(),
        total=len(cost_cases),
        desc="Cost-aware MCR"
    ):

        base=float(row["loss__none"])

        losses={
            S:float(row[f"loss__{name}"])
            for S,name in zip(MH_INTERVENTIONS,MH_NAMES)
        }

        for tol_ratio in TOLERANCES:

            tol=max(1e-8,tol_ratio*base)

            valid=[
                S
                for S,l in losses.items()
                if l<=tol
            ]

            if not valid:
                continue

            cardinality_choice=min(
                valid,
                key=lambda S:(
                    len(S),
                    tuple(sorted(S))
                )
            )

            for profile,costs in COST_PROFILES.items():

                cost_choice=min(
                    valid,
                    key=lambda S:(
                        sum(costs[p] for p in S),
                        len(S),
                        tuple(sorted(S))
                    )
                )

                card_cost=sum(
                    costs[p]
                    for p in cardinality_choice
                )

                chosen_cost=sum(
                    costs[p]
                    for p in cost_choice
                )

                cost_rows.append({
                    "dataset":row.dataset,
                    "id":row.id,
                    "truth":row.truth,
                    "seed":row.seed,
                    "severity":row.severity,
                    "tolerance":tol_ratio,
                    "profile":profile,
                    "n_valid_repairs":len(valid),
                    "ambiguous":int(len(valid)>1),
                    "cardinality_choice":"+".join(sorted(cardinality_choice)) or "none",
                    "cost_choice":"+".join(sorted(cost_choice)) or "none",
                    "decision_changed":int(cost_choice!=cardinality_choice),
                    "cardinality_cost":card_cost,
                    "cost_aware_cost":chosen_cost,
                    "absolute_saving":card_cost-chosen_cost,
                    "relative_saving":(
                        (card_cost-chosen_cost)/card_cost
                        if card_cost>0
                        else 0.0
                    ),
                    "cost_choice_residual":losses[cost_choice],
                })

    costdf=pd.DataFrame(cost_rows)
    costdf.to_parquet(COST_ROOT/"cost_mcr_cases.parquet",index=False)

    cost_summary=costdf.groupby(
        ["profile","tolerance"],as_index=False
    ).agg(
        n=("id","size"),
        ambiguous_rate=("ambiguous","mean"),
        changed_rate=("decision_changed","mean"),
        mean_absolute_saving=("absolute_saving","mean"),
        mean_relative_saving=("relative_saving","mean"),
        residual=("cost_choice_residual","mean"),
    )

    cost_summary.to_csv(
        TABLES/"FINAL_cost_aware_MCR.csv",
        index=False
    )

    display(cost_summary)

# ============================================================
# NOTEBOOK CELL 27 / CODE CELL 18
# ============================================================
if RUN_CERTIFICATION_SWEEP:
    # Prefer feature tables already generated by the MASTER notebook.
    previous = DRIVE/"MASTER_Q1"/"C2_MULTIHOP_RAGTRUTH"

    train_path=previous/"ragtruth_train_features.parquet"
    cal_path=previous/"ragtruth_cal_features.parquet"
    test_path=previous/"ragtruth_test_features.parquet"

    if not (
        train_path.exists()
        and cal_path.exists()
        and test_path.exists()
    ):
        raise RuntimeError(
            "RAGTruth train/cal/test feature tables were not found at:\n"
            f"{previous}\n\n"
            "Run the corrected MASTER V2 RAGTruth feature-extraction section first. "
            "This closure notebook intentionally reuses those cached features instead "
            "of recomputing NLI scores."
        )

    traindf=pd.read_parquet(train_path)
    caldf=pd.read_parquet(cal_path)
    testdf=pd.read_parquet(test_path)

    FULL_FEATURES=[
        "sim_min","sim_mean",
        "entail_min","entail_mean",
        "contr_max","contr_mean",
        "lex_min","lex_mean",
        "n_sent","out_chars","ctx_chars"
    ]

    model=make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )

    model.fit(
        traindf[FULL_FEATURES],
        traindf.y
    )

    calprob=model.predict_proba(
        caldf[FULL_FEATURES]
    )[:,1]

    testprob=model.predict_proba(
        testdf[FULL_FEATURES]
    )[:,1]

    print(
        "Held-out test AUROC:",
        roc_auc_score(testdf.y,testprob)
    )
    print(
        "Held-out test AUPRC:",
        average_precision_score(testdf.y,testprob)
    )

# ============================================================
# NOTEBOOK CELL 28 / CODE CELL 19
# ============================================================
if RUN_CERTIFICATION_SWEEP:
    def binom_upper(errors,n,delta):
        if n<=0:return 1.0
        if errors>=n:return 1.0
        return float(
            stats.beta.ppf(
                1-delta,
                errors+1,
                n-errors
            )
        )

    def calibrate_certificate(
        prob,
        y,
        alpha,
        delta,
        min_n=30
    ):
        candidates=np.unique(
            np.quantile(
                prob,
                np.linspace(.01,.95,120)
            )
        )

        delta_each=delta/max(1,len(candidates))

        feasible=[]

        for tau in candidates:

            mask=prob<=tau
            n=int(mask.sum())

            if n<min_n:
                continue

            errors=int(
                np.asarray(y)[mask].sum()
            )

            upper=binom_upper(
                errors,
                n,
                delta_each
            )

            if upper<=alpha:
                feasible.append(
                    (
                        tau,
                        n/len(prob),
                        upper,
                        n
                    )
                )

        if not feasible:
            return None

        return sorted(
            feasible,
            key=lambda z:(
                -z[1],
                z[2],
                z[0]
            )
        )[0]

    ALPHAS=[0.05,0.10,0.15,0.20]
    DELTA=0.05

    cert_rows=[]

    # --------------------------------------------------------
    # Global certificate
    # --------------------------------------------------------

    for alpha in ALPHAS:

        cert=calibrate_certificate(
            calprob,
            caldf.y.values,
            alpha,
            DELTA
        )

        if cert is None:

            cert_rows.append({
                "mode":"global",
                "group":"all",
                "alpha":alpha,
                "tau":np.nan,
                "cal_coverage":0.0,
                "cal_upper_bound":np.nan,
                "test_coverage":0.0,
                "test_risk":np.nan,
                "n_test_cert":0,
            })

        else:

            tau,ccov,cub,cn=cert
            mask=testprob<=tau

            cert_rows.append({
                "mode":"global",
                "group":"all",
                "alpha":alpha,
                "tau":tau,
                "cal_coverage":ccov,
                "cal_upper_bound":cub,
                "test_coverage":float(mask.mean()),
                "test_risk":(
                    float(testdf.y.values[mask].mean())
                    if mask.any()
                    else np.nan
                ),
                "n_test_cert":int(mask.sum()),
            })


    # --------------------------------------------------------
    # Pre-specified task-conditional certificates
    # --------------------------------------------------------

    if "task_type" in caldf.columns:

        groups=sorted(
            set(caldf.task_type.astype(str))
            &
            set(testdf.task_type.astype(str))
        )

        for group in groups:

            cmask=(
                caldf.task_type.astype(str)
                ==
                group
            )

            tmask=(
                testdf.task_type.astype(str)
                ==
                group
            )

            if cmask.sum()<40:
                continue

            for alpha in ALPHAS:

                cert=calibrate_certificate(
                    calprob[cmask.values],
                    caldf.loc[cmask,"y"].values,
                    alpha,
                    DELTA
                )

                if cert is None:

                    cert_rows.append({
                        "mode":"task_conditional",
                        "group":group,
                        "alpha":alpha,
                        "tau":np.nan,
                        "cal_coverage":0.0,
                        "cal_upper_bound":np.nan,
                        "test_coverage":0.0,
                        "test_risk":np.nan,
                        "n_test_cert":0,
                    })

                else:

                    tau,ccov,cub,cn=cert

                    local_prob=testprob[
                        tmask.values
                    ]

                    local_y=testdf.loc[
                        tmask,
                        "y"
                    ].values

                    mask=local_prob<=tau

                    cert_rows.append({
                        "mode":"task_conditional",
                        "group":group,
                        "alpha":alpha,
                        "tau":tau,
                        "cal_coverage":ccov,
                        "cal_upper_bound":cub,
                        "test_coverage":float(mask.mean()),
                        "test_risk":(
                            float(local_y[mask].mean())
                            if mask.any()
                            else np.nan
                        ),
                        "n_test_cert":int(mask.sum()),
                    })

    cert=pd.DataFrame(cert_rows)

    cert.to_csv(
        TABLES/"FINAL_certification_sensitivity.csv",
        index=False
    )

    display(cert)

# ============================================================
# NOTEBOOK CELL 30 / CODE CELL 20
# ============================================================
# ============================================================
# E1. Build final evidence summary
# ============================================================

summary_lines=[
    "# FaulTrace-RAG Final Experimental Evidence",
    "",
    f"Repository commit: `{COMMIT}`",
    f"Generated: {datetime.now(timezone.utc).isoformat()}",
    "",
]

if RUN_CROSS_DOMAIN and "c1i" in globals():
    summary_lines += [
        "## Cross-domain",
        f"- Identifiable cases: {len(c1i):,}",
        f"- Mean Shapley F1: {c1i.shapley_f1.mean():.4f}",
        f"- Mean singleton-delta F1: {c1i.delta_f1.mean():.4f}",
        f"- Mean random F1: {c1i.random_f1.mean():.4f}",
        f"- Mean MCR exact recovery: {c1i.mcr_exact.mean():.4f}",
        "",
    ]

if RUN_ACTIVE_DIAGNOSIS and "active_summary" in globals():
    best=active_summary.sort_values(
        ["bacd_f1","bacd_probes"],
        ascending=[False,True]
    ).iloc[0]

    summary_lines += [
        "## Bayesian Active Diagnosis",
        f"- Best tested budget: {int(best.budget)}",
        f"- BACD F1: {best.bacd_f1:.4f}",
        f"- Mean probes used: {best.bacd_probes:.2f}",
        f"- Exhaustive Shapley F1: {best.shapley_f1:.4f}",
        f"- Exhaustive worlds: {int(best.exhaustive_probes)}",
        "",
    ]

if RUN_COST_AWARE_MCR and "cost_summary" in globals():
    summary_lines += [
        "## Cost-aware MCR",
        "- See `FINAL_cost_aware_MCR.csv` for ambiguity and savings under all pre-specified cost profiles.",
        "",
    ]

if RUN_CERTIFICATION_SWEEP and "cert" in globals():
    positive=cert[
        cert.test_coverage>0
    ]

    summary_lines += [
        "## Certification",
        f"- Tested certification settings: {len(cert)}",
        f"- Non-zero held-out coverage settings: {len(positive)}",
        "- Zero-coverage settings are retained and must not be hidden.",
        "",
    ]

summary_lines += [
    "## Manuscript rule",
    "Only PAPER_MODE results should be used in the final headline tables.",
    "Do not select a model, budget, alpha, tolerance, or cost profile using the held-out test set.",
]

final_md="\n".join(summary_lines)

(ROOT/"FINAL_EVIDENCE.md").write_text(final_md)

print(final_md)

# ============================================================
# NOTEBOOK CELL 31 / CODE CELL 21
# ============================================================
# ============================================================
# E2. GitHub-ready export
# ============================================================

if EXPORT.exists():
    shutil.rmtree(EXPORT)

EXPORT.mkdir(parents=True)

for filename in [
    "FINAL_EVIDENCE.md",
    "environment.txt",
    "checkpoint.json",
    "C1_FINAL_all.csv",
    "C1_FINAL_all.parquet",
]:
    src=ROOT/filename
    if src.exists():
        shutil.copy2(src,EXPORT/src.name)

for folder in [TABLES,PLOTS,FAILS]:
    if folder.exists():
        shutil.copytree(
            folder,
            EXPORT/folder.name
        )

for src in [
    ACTIVE_ROOT/"bacd_results.csv",
    ACTIVE_ROOT/"bacd_results.parquet",
    ACTIVE_ROOT/"all_signatures.parquet",
    COST_ROOT/"cost_mcr_cases.parquet",
]:
    if src.exists():
        shutil.copy2(src,EXPORT/src.name)

(EXPORT/"README.md").write_text(
    f"""# FaulTrace-RAG Final Closure Experiments

Source commit: `{COMMIT}`

This folder contains:
- final cross-domain PAPER_MODE benchmarks,
- Bayesian Active Counterfactual Diagnosis,
- cost-aware MCR stress analysis,
- certification sensitivity,
- paper tables, figures, and evidence summary.

All negative outcomes are intentionally preserved.
"""
)

archive=ROOT/"FaulTrace_FINAL_CLOSURE_GitHub.zip"

if archive.exists():
    archive.unlink()

shutil.make_archive(
    str(archive.with_suffix("")),
    "zip",
    EXPORT
)

print("GitHub package:",archive)
