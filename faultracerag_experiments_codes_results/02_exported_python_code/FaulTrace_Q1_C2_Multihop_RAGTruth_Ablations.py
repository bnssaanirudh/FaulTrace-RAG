"""Exported code cells from FaulTrace_Q1_C2_Multihop_RAGTruth_Ablations.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
%%capture
!pip -q install -U "datasets>=3.0" "sentence-transformers>=3.0" "transformers>=4.45" "rank-bm25>=0.2.2" "scikit-learn>=1.4" "pandas>=2.0" "pyarrow>=15" "scipy>=1.11" "tqdm>=4.66" "matplotlib>=3.8"

# ============================================================
# NOTEBOOK CELL 3 / CODE CELL 2
# ============================================================
from __future__ import annotations
import os,sys,json,time,platform,subprocess,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FAST_MODE=True
PAPER_MODE=False
FORCE_RERUN=False

N_MULTIHOP=80 if FAST_MODE else 300
SEEDS=[42] if FAST_MODE else [13,42,87,2026,31415]
SEVERITIES=[0.40] if FAST_MODE else [0.25,0.40,0.55]
TOP_SENTENCES=16
PLAYERS=("S","R","E","A","G")
FAULTS=[
    ("S",),("R",),("E",),("A",),("G",),
    ("S","R"),("R","E"),("E","A"),("A","G"),
    ("S","R","E"),("R","E","A"),("E","A","G"),("R","E","A","G")
]
STAGE_COST={"S":1.0,"R":2.0,"E":4.0,"A":1.0,"G":8.0}
ACTIVE_BUDGET=9

RAG_TRAIN=400 if FAST_MODE else 3000
RAG_CAL=240 if FAST_MODE else 1800
RAG_TEST=400 if FAST_MODE else 4000
NLI_MODEL="cross-encoder/nli-MiniLM2-L6-H768" if FAST_MODE else "cross-encoder/nli-deberta-v3-base"
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ALPHA=0.10
DELTA=0.05

# ============================================================
# NOTEBOOK CELL 5 / CODE CELL 3
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive",force_remount=False)
    DRIVE=Path("/content/drive/MyDrive/FaulTrace_RAG_Experiments")
except Exception:
    DRIVE=Path("/content/FaulTrace_RAG_Experiments")
ROOT=DRIVE/"C2_FINAL_MULTIHOP_RAGTRUTH"
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
# NOTEBOOK CELL 9 / CODE CELL 5
# ============================================================
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
# NOTEBOOK CELL 11 / CODE CELL 6
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
# NOTEBOOK CELL 13 / CODE CELL 7
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
# NOTEBOOK CELL 15 / CODE CELL 8
# ============================================================
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
# NOTEBOOK CELL 17 / CODE CELL 9
# ============================================================
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
# NOTEBOOK CELL 19 / CODE CELL 10
# ============================================================
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

caldf=score_dataset(cal,"cal");testdf=score_dataset(test,"test")
caldf.to_parquet(ROOT/"ragtruth_cal_features.parquet",index=False);testdf.to_parquet(ROOT/"ragtruth_test_features.parquet",index=False)

# ============================================================
# NOTEBOOK CELL 21 / CODE CELL 11
# ============================================================
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
# NOTEBOOK CELL 23 / CODE CELL 12
# ============================================================
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
# NOTEBOOK CELL 25 / CODE CELL 13
# ============================================================
import matplotlib.pyplot as plt

# Multi-hop failure taxonomy.
fails=mi[
    ((mi.shapley_f1<1)&(mi.active_f1==1)) |
    ((mi.shapley_f1==1)&(mi.mcr_exact==0)) |
    (mi.active_residual>1e-8) |
    (mi.higher_interaction>mi.higher_interaction.quantile(.95))
].sort_values(["baseline_loss","higher_interaction"],ascending=False).head(150)
fails.to_csv(FAILS/"C2_multihop_failures.csv",index=False)

# RAGTruth hardest test errors for full model.
testout=testdf.copy();testout["p_hallucination"]=testprob
testout["error"]=np.abs(testout.y-testout.p_hallucination)
testout.sort_values("error",ascending=False).head(150).to_csv(FAILS/"C2_ragtruth_hard_cases.csv",index=False)

p=main.groupby("dataset")[["shapley_f1","delta_f1","active_f1"]].mean()
ax=p.plot(kind="bar",figsize=(8,4));ax.set_ylim(0,1);ax.set_ylabel("Fault-set F1");ax.set_title("Multi-hop diagnosis ablation")
plt.tight_layout();plt.savefig(PLOTS/"C2_multihop_ablation.png",dpi=220);plt.show()

ax=ab.set_index("model")[["auroc","auprc","f1"]].plot(kind="bar",figsize=(8,4));ax.set_ylim(0,1);ax.set_title("RAGTruth grounding feature ablation")
plt.tight_layout();plt.savefig(PLOTS/"C2_ragtruth_ablation.png",dpi=220);plt.show()

# ============================================================
# NOTEBOOK CELL 27 / CODE CELL 14
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
