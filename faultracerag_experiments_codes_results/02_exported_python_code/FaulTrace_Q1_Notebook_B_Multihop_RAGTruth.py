"""Exported code cells from FaulTrace_Q1_Notebook_B_Multihop_RAGTruth.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
%%capture
!pip -q install -U "datasets>=3.0" "sentence-transformers>=3.0" "transformers>=4.45" "scikit-learn>=1.4" "pandas>=2.0" "pyarrow>=15" "scipy>=1.11" "tqdm>=4.66" "rank-bm25>=0.2.2"

# ============================================================
# NOTEBOOK CELL 3 / CODE CELL 2
# ============================================================
from __future__ import annotations
import os, sys, json, time, random, hashlib, subprocess, platform, re, math, itertools
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

FAST_MODE = True
PAPER_MODE = False
FORCE_RERUN = False
RUN_OPTIONAL_FLAN = False  # secondary sanity only; not required for core claims

N_MULTIHOP = 80 if FAST_MODE else 500
N_RAG_CAL = 160 if FAST_MODE else 800
N_RAG_TEST = 220 if FAST_MODE else 2700
MULTIHOP_SEEDS = [42] if FAST_MODE else [13,42,87,2026,31415]
FAULT_SEVERITIES = [0.40] if FAST_MODE else [0.25,0.40,0.55]
TOP_SENTENCES = 16
PLAYERS = ('S','R','E','A','G')
FAULT_SCENARIOS = [
    ('S',),('R',),('E',),('A',),('G',),
    ('S','R'),('R','E'),('E','A'),('A','G'),
    ('S','R','E'),('R','E','A'),('E','A','G'),('R','E','A','G')
]
NLI_MODEL = 'cross-encoder/nli-MiniLM2-L6-H768'
ALPHA = 0.10       # target certified error rate
DELTA = 0.05       # confidence level for risk upper bound
print({'N_MULTIHOP':N_MULTIHOP,'N_RAG_TEST':N_RAG_TEST,'players':PLAYERS})

# ============================================================
# NOTEBOOK CELL 5 / CODE CELL 3
# ============================================================
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive/FaulTrace_RAG_Experiments')
except Exception:
    DRIVE_ROOT = Path('/content/FaulTrace_RAG_Experiments')
RUN_ROOT = DRIVE_ROOT/'B_RAG_MULTIHOP_CAUSAL'
CHUNK_DIR=RUN_ROOT/'chunks'; RAG_SCORE_DIR=RUN_ROOT/'ragtruth_scores'; TABLE_DIR=RUN_ROOT/'paper_tables'; PLOT_DIR=RUN_ROOT/'plots'; CACHE_DIR=RUN_ROOT/'cache'
for p in [RUN_ROOT,CHUNK_DIR,RAG_SCORE_DIR,TABLE_DIR,PLOT_DIR,CACHE_DIR]: p.mkdir(parents=True,exist_ok=True)

REPO_DIR=Path('/content/FaulTrace-RAG'); REPO_URL='https://github.com/bnssaanirudh/FaulTrace-RAG.git'
if REPO_DIR.exists():
    subprocess.run(['git','-C',str(REPO_DIR),'fetch','origin'],check=False); subprocess.run(['git','-C',str(REPO_DIR),'pull','--ff-only'],check=False)
else:
    subprocess.run(['git','clone','--depth','1',REPO_URL,str(REPO_DIR)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(REPO_DIR)],check=True)
COMMIT=subprocess.check_output(['git','-C',str(REPO_DIR),'rev-parse','HEAD'],text=True).strip()

CHECKPOINT=RUN_ROOT/'checkpoint.json'
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str)); os.replace(tmp,path)
def cp_read():
    if CHECKPOINT.exists():
        try:return json.loads(CHECKPOINT.read_text())
        except:pass
    return {'completed':{},'commit':COMMIT}
def done(k): return (not FORCE_RERUN) and k in cp_read().get('completed',{})
def mark(k,meta):
    cp=cp_read(); cp['completed'][k]=meta; cp['updated_utc']=datetime.now(timezone.utc).isoformat(); atomic_json(CHECKPOINT,cp)
(RUN_ROOT/'environment.txt').write_text(f'commit={COMMIT}\npython={sys.version}\nplatform={platform.platform()}\n')
print('Drive:',RUN_ROOT,'\nCommit:',COMMIT)

# ============================================================
# NOTEBOOK CELL 7 / CODE CELL 4
# ============================================================
import itertools, math
from collections import defaultdict

def all_subsets(players):
    players = tuple(players)
    for r in range(len(players) + 1):
        for comb in itertools.combinations(players, r):
            yield frozenset(comb)

def exact_shapley(values, players):
    """Exact Shapley values for a value game v(S)."""
    players = tuple(players)
    n = len(players)
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [x for x in players if x != p]
        for S in all_subsets(others):
            w = math.factorial(len(S)) * math.factorial(n-len(S)-1) / math.factorial(n)
            phi[p] += w * (values[S | {p}] - values[S])
    return phi

def harsanyi_dividends(values, players):
    """Möbius/Harsanyi dividends. Pair/triple terms are genuine interactions."""
    div = {}
    for S in all_subsets(players):
        total = 0.0
        for T in all_subsets(S):
            total += ((-1) ** (len(S) - len(T))) * values[T]
        div[S] = total
    return div

def minimal_repair_sets(losses, players, tolerance=1e-9):
    """All smallest intervention sets whose loss is at most tolerance."""
    valid = [S for S, loss in losses.items() if loss <= tolerance]
    if not valid:
        return []
    m = min(map(len, valid))
    return sorted([S for S in valid if len(S) == m], key=lambda s: tuple(sorted(s)))

def mrr_at_k(ranking, qrels, k=10):
    for i, docid in enumerate(ranking[:k], start=1):
        if float(qrels.get(docid, 0)) > 0:
            return 1.0 / i
    return 0.0

def recall_at_k(ranking, qrels, k=10):
    rel = {d for d, g in qrels.items() if float(g) > 0}
    if not rel:
        return float('nan')
    return len(rel.intersection(ranking[:k])) / len(rel)

def ndcg_at_k(ranking, qrels, k=10):
    gains = [float(qrels.get(d, 0)) for d in ranking[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted([float(v) for v in qrels.values() if float(v) > 0], reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def stable_seed(*parts):
    import hashlib
    h = hashlib.sha256('||'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0

def set_f1(pred, truth):
    pred, truth = set(pred), set(truth)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    p = len(pred & truth) / len(pred)
    r = len(pred & truth) / len(truth)
    return 2*p*r/(p+r) if (p+r) else 0.0

# ============================================================
# NOTEBOOK CELL 8 / CODE CELL 5
# ============================================================
# 5-player self-test: efficiency must hold; residual is NOT an interaction metric.
rng=np.random.default_rng(42)
_test_v={S:float(rng.random()) for S in all_subsets(PLAYERS)}; _test_v[frozenset()]=0.0
_phi=exact_shapley(_test_v,PLAYERS)
assert abs(sum(_phi.values())-(_test_v[frozenset(PLAYERS)]-_test_v[frozenset()]))<1e-10
print('5-stage Shapley efficiency self-test passed.')

# ============================================================
# NOTEBOOK CELL 10 / CODE CELL 6
# ============================================================
from datasets import load_dataset

def normalize_hotpot(ex):
    titles=list(ex['context']['title']); sents=list(ex['context']['sentences'])
    docs={str(t):list(ss) for t,ss in zip(titles,sents)}
    supports={(str(t),int(i)) for t,i in zip(ex['supporting_facts']['title'],ex['supporting_facts']['sent_id'])}
    return {'id':str(ex['id']),'question':str(ex['question']),'answers':[str(ex['answer'])],'docs':docs,'supports':supports}

def normalize_2wiki(ex):
    m=ex['metadata']; ctx=m['context']; sf=m['supporting_facts']
    docs={str(t):list(ss) for t,ss in zip(ctx['title'],ctx['sentences'])}
    supports={(str(t),int(i)) for t,i in zip(sf['title'],sf['sent_id'])}
    answers=[str(x) for x in ex.get('golden_answers',[])]; answers=answers or ['']
    return {'id':str(ex['id']),'question':str(ex['question']),'answers':answers,'docs':docs,'supports':supports}

def deterministic_take(ds,n,seed=20260910):
    n=min(n,len(ds)); rg=np.random.default_rng(seed); idx=sorted(rg.choice(len(ds),size=n,replace=False).tolist()); return ds.select(idx)

def load_multihop():
    hp=load_dataset('hotpotqa/hotpot_qa','distractor',split='validation')
    hp=deterministic_take(hp,N_MULTIHOP)
    hot=[normalize_hotpot(x) for x in hp]
    # Small parquet mirror with supporting-fact metadata avoids legacy remote-code loaders.
    tw=load_dataset('cmriat/2wikimultihopqa',split='validation')
    tw=deterministic_take(tw,N_MULTIHOP)
    two=[normalize_2wiki(x) for x in tw]
    return {'hotpotqa':hot,'2wikimultihopqa':two}

MULTIHOP=load_multihop()
for k,v in MULTIHOP.items(): print(k,len(v), 'example supports=',len(v[0]['supports']))

# ============================================================
# NOTEBOOK CELL 12 / CODE CELL 7
# ============================================================
from rank_bm25 import BM25Okapi

def tokens(x): return re.findall(r'[A-Za-z0-9]+',str(x).lower())

def flatten_docs(docs):
    rows=[]
    for title,sents in docs.items():
        for i,s in enumerate(sents): rows.append({'title':title,'sent_id':i,'text':str(s),'key':(title,i)})
    return rows

def clean_scope(ex): return set(ex['docs'].keys())

def faulty_scope(ex,severity,rng):
    scope=set(ex['docs'].keys()); support_titles={t for t,_ in ex['supports']}
    candidates=sorted(support_titles & scope) or sorted(scope)
    n=max(1,int(round(max(1,len(candidates))*severity)))
    for t in candidates[:n]: scope.discard(t)
    return scope

def retrieve(ex,scope,k=None):
    rows=[r for r in flatten_docs(ex['docs']) if r['title'] in scope]
    if not rows:return []
    bm=BM25Okapi([tokens(r['text']) for r in rows]); scores=np.asarray(bm.get_scores(tokens(ex['question'])))
    idx=np.argsort(scores)[::-1]
    ranked=[rows[i] for i in idx]
    return ranked if k is None else ranked[:min(k,len(ranked))]

def faulty_retrieve(full_rank,ex,severity,rng,k=TOP_SENTENCES):
    # Remove/demote support-bearing sentences from the observable top-k and replace
    # them with non-support tail sentences. Merely reordering inside top-k would not
    # change evidence coverage, so it would not constitute an observable R fault.
    top=list(full_rank[:k]); tail=list(full_rank[k:]); sup=ex['supports']
    targets=[i for i,r in enumerate(top) if r['key'] in sup]
    n=max(1,int(round(max(1,len(targets))*severity)))
    replacements=[r for r in tail if r['key'] not in sup]
    for i,repl in zip(targets[:n],replacements): top[i]=repl
    if not targets and len(top)>1:
        n2=max(1,int(round(len(top)*severity))); top[:n2]=list(reversed(top[:n2]))
    return top

def clean_extract(rows,ex):
    return [{'key':r['key'],'text':r['text'],'support':int(r['key'] in ex['supports'])} for r in rows]

def faulty_extract(evidence,severity,rng):
    out=[]
    for x in evidence:
        y=dict(x)
        if y['support'] and rng.random()<severity: y['support']=0
        elif not y['support'] and rng.random()<severity*0.10: y['support']=1
        out.append(y)
    return out

def clean_aggregate(evidence,ex):
    required=max(1,len(ex['supports']))
    found=len({x['key'] for x in evidence if x['support'] and x['key'] in ex['supports']})
    coverage=min(1.0,found/required)
    return {'coverage':coverage,'ready':coverage>=1.0}

def faulty_aggregate(evidence,ex,severity):
    clean=clean_aggregate(evidence,ex)
    # Plausible denominator bug: required evidence is effectively over-counted,
    # depressing coverage even when retrieval/extraction were correct.
    bogus=max(0.0,clean['coverage']*(1.0-severity))
    return {'coverage':bogus,'ready':bogus>=0.999}

def clean_generate(agg,ex):
    return ex['answers'][0] if agg['ready'] else '<INSUFFICIENT_EVIDENCE>'

def faulty_generate(agg,ex,severity,rng):
    # Deterministic observable G fault; severity is already varied in upstream experiments.
    return '<HALLUCINATED_ANSWER>' if agg['ready'] else ex['answers'][0]

def execute5(ex,fault_set,repaired_set,severity,seed):
    fs,rs=set(fault_set),set(repaired_set)
    rS=np.random.default_rng(stable_seed(seed,ex['id'],'S',severity))
    scope=clean_scope(ex) if ('S' not in fs or 'S' in rs) else faulty_scope(ex,severity,rS)
    full_rank=retrieve(ex,scope,k=None)
    clean_r=full_rank[:TOP_SENTENCES]
    rR=np.random.default_rng(stable_seed(seed,ex['id'],'R',severity))
    rows=clean_r if ('R' not in fs or 'R' in rs) else faulty_retrieve(full_rank,ex,severity,rR,TOP_SENTENCES)
    ev=clean_extract(rows,ex)
    if 'E' in fs and 'E' not in rs:
        ev=faulty_extract(ev,severity,np.random.default_rng(stable_seed(seed,ex['id'],'E',severity)))
    agg=clean_aggregate(ev,ex) if ('A' not in fs or 'A' in rs) else faulty_aggregate(ev,ex,severity)
    ans=clean_generate(agg,ex) if ('G' not in fs or 'G' in rs) else faulty_generate(agg,ex,severity,np.random.default_rng(stable_seed(seed,ex['id'],'G',severity)))
    return {'coverage':float(agg['coverage']),'answer':ans}

def output_loss(y,target):
    return 0.55*abs(y['coverage']-target['coverage']) + 0.45*float(y['answer']!=target['answer'])

def diagnose5(ex,fault_set,severity,seed):
    target=execute5(ex,(),PLAYERS,severity,seed)
    losses={}; outputs={}
    for S in all_subsets(PLAYERS):
        y=execute5(ex,fault_set,S,severity,seed); outputs[S]=y; losses[S]=output_loss(y,target)
    base=losses[frozenset()]; values={S:base-l for S,l in losses.items()}
    phi=exact_shapley(values,PLAYERS); div=harsanyi_dividends(values,PLAYERS)
    tol=max(1e-8,base*0.02); mcr=minimal_repair_sets(losses,PLAYERS,tol)
    return target,base,losses,values,phi,div,mcr

# Semantics test: each single injected stage should be repairable by its own do-intervention on a controlled example.
_toy={'id':'toy','question':'alpha','answers':['yes'],'docs':{'A':['alpha answer'], 'B':['noise']},'supports':{('A',0)}}
for stage in PLAYERS:
    _,b,losses,_,_,_,mcr=diagnose5(_toy,(stage,),0.9,42)
    assert losses[frozenset({stage})] <= b + 1e-12
print('Five-stage executor smoke test passed.')

# ============================================================
# NOTEBOOK CELL 14 / CODE CELL 8
# ============================================================
def rank_of_truth(phi,truth):
    order=sorted(phi,key=lambda p:(-phi[p],p)); return min(order.index(t)+1 for t in truth)

def run_multihop_block(dataset,examples,seed,severity):
    rows=[]
    for ex in tqdm(examples,desc=f'{dataset}/s{seed}/v{severity}',leave=False):
        for truth in FAULT_SCENARIOS:
            target,base,losses,values,phi,div,mcr=diagnose5(ex,truth,severity,seed)
            k=len(truth); order=sorted(PLAYERS,key=lambda p:(-phi[p],p)); pred=set(order[:k])
            direct={p:values[frozenset({p})] for p in PLAYERS}; dord=sorted(PLAYERS,key=lambda p:(-direct[p],p)); dpred=set(dord[:k])
            rrng=np.random.default_rng(stable_seed(seed,ex['id'],'random-baseline',truth,severity)); rpred=set(rrng.choice(list(PLAYERS),size=k,replace=False).tolist())
            pmcr=set(next(iter(mcr),frozenset()))
            rows.append({
                'dataset':dataset,'id':ex['id'],'seed':seed,'severity':severity,'truth':'+'.join(truth),'k':k,
                'identifiable':base>1e-10,'baseline_loss':base,
                **{f'phi_{p}':phi[p] for p in PLAYERS},
                'shapley_top1':order[0],'shapley_exact_k':int(pred==set(truth)),'shapley_f1':set_f1(pred,truth),'shapley_mrr':1/rank_of_truth(phi,truth),
                'delta_exact_k':int(dpred==set(truth)),'delta_f1':set_f1(dpred,truth),
                'random_exact_k':int(rpred==set(truth)),'random_f1':set_f1(rpred,truth),
                'mcr_pred':'+'.join(sorted(pmcr)),'mcr_exact':int(pmcr==set(truth)),'mcr_residual':losses.get(frozenset(pmcr),np.nan),
                'pair_interaction_abs':sum(abs(v) for S,v in div.items() if len(S)==2),
                'higher_interaction_abs':sum(abs(v) for S,v in div.items() if len(S)>=3),
                'worlds':2**len(PLAYERS),
            })
    return pd.DataFrame(rows)

for dataset,examples in MULTIHOP.items():
    for seed in MULTIHOP_SEEDS:
        for sev in FAULT_SEVERITIES:
            key=f'mh__{dataset}__s{seed}__v{str(sev).replace(".","p")}'
            path=CHUNK_DIR/f'{key}.parquet'
            if done(key) and path.exists(): print('skip',key); continue
            t=time.time(); df=run_multihop_block(dataset,examples,seed,sev)
            tmp=path.with_suffix('.tmp.parquet'); df.to_parquet(tmp,index=False); os.replace(tmp,path)
            mark(key,{'rows':len(df),'seconds':round(time.time()-t,2),'commit':COMMIT}); print('saved',key)

# ============================================================
# NOTEBOOK CELL 16 / CODE CELL 9
# ============================================================
from sentence_transformers import CrossEncoder
from scipy.special import softmax

rag_train=load_dataset('wandb/RAGTruth-processed',split='train')
rag_test=load_dataset('wandb/RAGTruth-processed',split='test')
rag_train=deterministic_take(rag_train,N_RAG_CAL,seed=20260910)
rag_test=deterministic_take(rag_test,N_RAG_TEST,seed=20260911)

nli=CrossEncoder(NLI_MODEL, max_length=512)
LABELS=['contradiction','entailment','neutral']

def parse_hallucinated(ex):
    # RAGTruth-processed schema: {evident_conflict:int, baseless_info:int}.
    x=ex.get('hallucination_labels_processed', ex.get('hallucination_labels'))
    if isinstance(x,dict):
        return int(int(x.get('evident_conflict',0) or 0)>0 or int(x.get('baseless_info',0) or 0)>0)
    if isinstance(x,str):
        st=x.strip()
        try:
            parsed=json.loads(st)
            if isinstance(parsed,list): return int(len(parsed)>0)
            if isinstance(parsed,dict): return int(any(bool(v) for v in parsed.values()))
        except Exception: pass
        return int(st.lower() not in ('','[]','{}','none','null','0','false'))
    if isinstance(x,(list,tuple,set)): return int(len(x)>0)
    return int(bool(x))

def sentence_split(text,max_sents=4):
    parts=[p.strip() for p in re.split(r'(?<=[.!?])\s+',str(text)) if p.strip()]
    return parts[:max_sents] or [str(text)[:500]]

def context_chunks(text,max_chunks=4,chars=1400):
    t=str(text); chunks=[t[i:i+chars] for i in range(0,len(t),chars)]
    return chunks[:max_chunks] or ['']

def grounding_score(ex):
    sents=sentence_split(ex['output'],4 if FAST_MODE else 6)
    chunks=context_chunks(ex['context'],3 if FAST_MODE else 5)
    pairs=[(c,s) for s in sents for c in chunks]
    logits=np.asarray(nli.predict(pairs,batch_size=32,show_progress_bar=False))
    probs=softmax(logits,axis=1)
    entail_idx=LABELS.index('entailment')
    # Each generated sentence needs at least one supporting context chunk; weakest sentence governs.
    vals=[]; j=0
    for _s in sents:
        vals.append(float(np.max(probs[j:j+len(chunks),entail_idx]))); j+=len(chunks)
    return float(min(vals))

def score_split(ds,split_name):
    rows=[]
    existing={}
    for f in RAG_SCORE_DIR.glob(f'{split_name}_*.json'):
        try:
            x=json.loads(f.read_text()); existing[str(x.get('uid',x['id']))]=x
        except:pass
    for ex in tqdm(ds,desc=f'RAGTruth {split_name}'):
        source_id=str(ex['id'])
        uid=hashlib.sha256((source_id+'||'+str(ex.get('model',''))+'||'+str(ex.get('output',''))).encode()).hexdigest()[:20]
        if uid in existing and not FORCE_RERUN:
            rows.append(existing[uid]); continue
        row={'uid':uid,'id':source_id,'task_type':str(ex.get('task_type','')),'hallucinated':parse_hallucinated(ex),'support_score':grounding_score(ex)}
        atomic_json(RAG_SCORE_DIR/f'{split_name}_{uid}.json',row); rows.append(row)
    return pd.DataFrame(rows)

cal=score_split(rag_train,'cal'); test=score_split(rag_test,'test')
cal.to_parquet(RUN_ROOT/'ragtruth_cal.parquet',index=False); test.to_parquet(RUN_ROOT/'ragtruth_test.parquet',index=False)
print(cal.shape,test.shape)

# ============================================================
# NOTEBOOK CELL 18 / CODE CELL 10
# ============================================================
def best_f1_threshold(df):
    y=df.hallucinated.values
    # lower support => more likely hallucination
    candidates=np.quantile(df.support_score,np.linspace(0.02,0.98,97))
    vals=[]
    for t in candidates:
        pred=(df.support_score<t).astype(int); vals.append((f1_score(y,pred),float(t)))
    return max(vals)[1]

thr=best_f1_threshold(cal)
y=test.hallucinated.values; hall_score=1-test.support_score.values; pred=(test.support_score.values<thr).astype(int)
rag_metrics={
    'threshold_from_calibration':thr,
    'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
    'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
    'auroc':roc_auc_score(y,hall_score),'auprc':average_precision_score(y,hall_score),
    'n_test':len(test),
}
display(pd.DataFrame([rag_metrics]))
pd.DataFrame([rag_metrics]).to_csv(TABLE_DIR/'table_B_ragtruth_detection.csv',index=False)

# ============================================================
# NOTEBOOK CELL 20 / CODE CELL 11
# ============================================================
def binom_upper(errors,n,delta=0.05):
    if n<=0:return 1.0
    if errors>=n:return 1.0
    return float(stats.beta.ppf(1-delta, errors+1, n-errors))

def calibrate_cert_threshold(cal_df, alpha=ALPHA, delta=DELTA, min_n=25):
    # CERTIFY as faithful only when support_score >= tau.
    candidates=np.unique(np.quantile(cal_df.support_score,np.linspace(0.05,0.99,120)))
    feasible=[]
    # Bonferroni correction makes threshold search itself part of the confidence accounting.
    delta_each=delta/max(1,len(candidates))
    for tau in candidates:
        sub=cal_df[cal_df.support_score>=tau]
        if len(sub)<min_n: continue
        errors=int(sub.hallucinated.sum())
        ub=binom_upper(errors,len(sub),delta_each)
        if ub<=alpha: feasible.append((tau,len(sub)/len(cal_df),ub,len(sub)))
    if not feasible:
        return None
    # Maximize coverage, then choose the lower-risk option on ties.
    return sorted(feasible,key=lambda x:(-x[1],x[2],x[0]))[0]

cert=calibrate_cert_threshold(cal)
if cert is None:
    print('No threshold met the requested risk bound; this is a valid negative result. Try more calibration data, not test tuning.')
    cert_table=pd.DataFrame([{'tau':np.nan,'coverage':0.0,'test_risk':np.nan,'cal_upper_bound':np.nan}])
else:
    tau,cal_cov,cal_ub,ncal=cert
    cert_test=test[test.support_score>=tau]
    test_risk=float(cert_test.hallucinated.mean()) if len(cert_test) else np.nan
    cert_table=pd.DataFrame([{'tau':tau,'cal_coverage':cal_cov,'cal_upper_bound':cal_ub,'cal_n_cert':ncal,
                              'test_coverage':len(cert_test)/len(test),'test_risk':test_risk,'test_n_cert':len(cert_test),
                              'alpha_target':ALPHA,'delta_familywise':DELTA}])
cert_table.to_csv(TABLE_DIR/'table_B_risk_certificate.csv',index=False); display(cert_table)

# ============================================================
# NOTEBOOK CELL 22 / CODE CELL 12
# ============================================================
mh_files=sorted(CHUNK_DIR.glob('mh__*.parquet')); assert mh_files
mh=pd.concat([pd.read_parquet(f) for f in mh_files],ignore_index=True); mh.to_parquet(RUN_ROOT/'multihop_all.parquet',index=False)
mi=mh[mh.identifiable].copy()
summary=mi.groupby(['dataset','severity'],as_index=False).agg(
    n=('id','size'),shapley_exact=('shapley_exact_k','mean'),shapley_f1=('shapley_f1','mean'),
    delta_exact=('delta_exact_k','mean'),delta_f1=('delta_f1','mean'),random_exact=('random_exact_k','mean'),random_f1=('random_f1','mean'),mcr_exact=('mcr_exact','mean'),
    residual=('mcr_residual','mean'),pair_interaction=('pair_interaction_abs','mean'),higher_interaction=('higher_interaction_abs','mean'))
summary.to_csv(TABLE_DIR/'table_B_multihop_main.csv',index=False); display(summary)

# Paired non-parametric test across identical injected cases.
W,p=stats.wilcoxon(mi.shapley_f1,mi.delta_f1,zero_method='zsplit')
st=pd.DataFrame([{'comparison':'Shapley vs single-fix set-F1','wilcoxon_W':W,'p_value':p,'mean_shapley':mi.shapley_f1.mean(),'mean_delta':mi.delta_f1.mean()}])
st.to_csv(TABLE_DIR/'table_B_statistics.csv',index=False); display(st)

# ============================================================
# NOTEBOOK CELL 24 / CODE CELL 13
# ============================================================
import matplotlib.pyplot as plt
plot=mi.groupby('dataset')[['shapley_exact_k','delta_exact_k','mcr_exact']].mean()
ax=plot.plot(kind='bar',figsize=(8,4)); ax.set_ylim(0,1); ax.set_ylabel('Exact fault-set recovery'); ax.set_title('Multi-hop causal fault localization'); plt.tight_layout(); plt.savefig(PLOT_DIR/'multihop_localization.png',dpi=220); plt.show()

# Risk-coverage curve from held-out RAGTruth test for visualization only (threshold is NOT tuned on test).
ths=np.quantile(test.support_score,np.linspace(0.05,0.99,80)); rc=[]
for t in ths:
    s=test[test.support_score>=t]
    if len(s):rc.append((len(s)/len(test),float(s.hallucinated.mean())))
if rc:
    x,y=zip(*rc); plt.figure(figsize=(6,4)); plt.plot(x,y); plt.xlabel('Coverage'); plt.ylabel('Observed hallucination risk'); plt.title('RAGTruth risk–coverage'); plt.tight_layout(); plt.savefig(PLOT_DIR/'ragtruth_risk_coverage.png',dpi=220); plt.show()

archive=RUN_ROOT/'FaulTrace_Notebook_B_results.zip'
!rm -f "{archive}"
!cd "{RUN_ROOT}" && zip -qr "{archive}" chunks ragtruth_scores paper_tables plots checkpoint.json multihop_all.parquet ragtruth_cal.parquet ragtruth_test.parquet environment.txt
print('Final archive:',archive)
