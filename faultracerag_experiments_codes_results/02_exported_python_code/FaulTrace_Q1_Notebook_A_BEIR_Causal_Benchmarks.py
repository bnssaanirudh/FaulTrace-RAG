"""Exported code cells from FaulTrace_Q1_Notebook_A_BEIR_Causal_Benchmarks.ipynb.
Notebook markdown/output cells are intentionally omitted.
"""

# ============================================================
# NOTEBOOK CELL 1 / CODE CELL 1
# ============================================================
%%capture
!pip -q install -U "beir>=2.0.0" "sentence-transformers>=3.0" "rank-bm25>=0.2.2" "scikit-learn>=1.4" "pandas>=2.0" "pyarrow>=15" "scipy>=1.11" "tqdm>=4.66"

# ============================================================
# NOTEBOOK CELL 3 / CODE CELL 2
# ============================================================
from __future__ import annotations
import os, sys, json, time, random, hashlib, subprocess, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats

FAST_MODE = True          # smoke/quick validation
PAPER_MODE = False        # set True for the final experiment sweep
INCLUDE_TREC_COVID = False # optional larger corpus; turn on for final paper if time allows
FORCE_RERUN = False

TOP_K = 10
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVERS = ["bm25", "dense", "hybrid"]
FAULT_SEVERITIES = [0.35] if FAST_MODE else [0.20, 0.35, 0.50]
SEEDS = [42] if FAST_MODE else [13, 42, 87, 2026, 31415]
N_QUERIES = 40 if FAST_MODE else 300

BENCHMARKS = ["scifact", "nfcorpus", "arguana"] if FAST_MODE else ["scifact", "nfcorpus", "arguana", "fiqa"]
if INCLUDE_TREC_COVID:
    BENCHMARKS.append("trec-covid")

FAULT_SCENARIOS = [
    ("R",), ("E",), ("A",),
    ("R","E"), ("R","A"), ("E","A"),
    ("R","E","A"),
]
PLAYERS = ("R", "E", "A")
print({"FAST_MODE": FAST_MODE, "PAPER_MODE": PAPER_MODE, "benchmarks": BENCHMARKS, "seeds": SEEDS})

# ============================================================
# NOTEBOOK CELL 5 / CODE CELL 3
# ============================================================
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive/FaulTrace_RAG_Experiments')
except Exception:
    DRIVE_ROOT = Path('/content/FaulTrace_RAG_Experiments')

RUN_ROOT = DRIVE_ROOT / 'A_BEIR_CAUSAL'
CHUNK_DIR = RUN_ROOT / 'chunks'
CACHE_DIR = RUN_ROOT / 'cache'
TABLE_DIR = RUN_ROOT / 'paper_tables'
PLOT_DIR = RUN_ROOT / 'plots'
for p in [RUN_ROOT, CHUNK_DIR, CACHE_DIR, TABLE_DIR, PLOT_DIR]: p.mkdir(parents=True, exist_ok=True)

REPO_DIR = Path('/content/FaulTrace-RAG')
REPO_URL = 'https://github.com/bnssaanirudh/FaulTrace-RAG.git'
if REPO_DIR.exists():
    subprocess.run(['git','-C',str(REPO_DIR),'fetch','origin'], check=False)
    subprocess.run(['git','-C',str(REPO_DIR),'pull','--ff-only'], check=False)
else:
    subprocess.run(['git','clone','--depth','1',REPO_URL,str(REPO_DIR)], check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(REPO_DIR)], check=True)
COMMIT = subprocess.check_output(['git','-C',str(REPO_DIR),'rev-parse','HEAD'], text=True).strip()
print('Repository commit:', COMMIT)

CHECKPOINT = RUN_ROOT / 'checkpoint.json'
def read_checkpoint():
    if CHECKPOINT.exists():
        try: return json.loads(CHECKPOINT.read_text())
        except Exception: pass
    return {'completed': {}, 'commit': COMMIT}

def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)

def mark_done(key, meta):
    cp = read_checkpoint(); cp['completed'][key] = meta; cp['updated_utc'] = datetime.now(timezone.utc).isoformat(); atomic_json(CHECKPOINT, cp)

def is_done(key):
    return (not FORCE_RERUN) and key in read_checkpoint().get('completed', {}) and (CHUNK_DIR / f'{key}.parquet').exists()

(RUN_ROOT/'environment.txt').write_text(
    f'utc={datetime.now(timezone.utc).isoformat()}\ncommit={COMMIT}\npython={sys.version}\nplatform={platform.platform()}\n'
)

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
# Mathematical self-tests. These guard against the previous "interaction = v(N)-sum(phi)" mistake.
players = ("R","E","A")
toy_v = {
    frozenset(): 0.0,
    frozenset({"R"}): 0.3, frozenset({"E"}): 0.2, frozenset({"A"}): 0.1,
    frozenset({"R","E"}): 0.7, frozenset({"R","A"}): 0.45, frozenset({"E","A"}): 0.32,
    frozenset({"R","E","A"}): 0.9,
}
phi = exact_shapley(toy_v, players)
assert abs(sum(phi.values()) - (toy_v[frozenset(players)] - toy_v[frozenset()])) < 1e-10
assert abs(toy_v[frozenset(players)] - sum(phi.values())) < 1e-10, "Shapley residual must be zero by efficiency"
div = harsanyi_dividends(toy_v, players)
assert abs(div[frozenset({"R","E"})] - (0.7-0.3-0.2)) < 1e-10
print('Math self-tests passed. Genuine R×E interaction =', div[frozenset({"R","E"})])

# ============================================================
# NOTEBOOK CELL 10 / CODE CELL 6
# ============================================================
from beir import util
from beir.datasets.data_loader import GenericDataLoader

BEIR_BASE = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets'

def load_beir(name):
    target = CACHE_DIR / 'beir'
    target.mkdir(parents=True, exist_ok=True)
    data_path = target / name
    if not data_path.exists():
        util.download_and_unzip(f'{BEIR_BASE}/{name}.zip', str(target))
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split='test')
    # Keep only queries with qrels and use deterministic sample.
    qids = sorted(set(queries) & set(qrels))
    rng = np.random.default_rng(20260910)
    if len(qids) > N_QUERIES:
        qids = sorted(rng.choice(qids, size=N_QUERIES, replace=False).tolist())
    return corpus, {q: queries[q] for q in qids}, {q: qrels[q] for q in qids}

for name in BENCHMARKS:
    print(name)

# ============================================================
# NOTEBOOK CELL 12 / CODE CELL 7
# ============================================================
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

def tok(text):
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())

def corpus_text(doc):
    return (str(doc.get('title','')) + ' ' + str(doc.get('text',''))).strip()

def top_indices(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]

def rrf_merge(a, b, k=10, constant=60):
    scores = defaultdict(float)
    for rank, d in enumerate(a, 1): scores[d] += 1/(constant+rank)
    for rank, d in enumerate(b, 1): scores[d] += 1/(constant+rank)
    return [d for d,_ in sorted(scores.items(), key=lambda x:(-x[1], x[0]))[:k]]

_dense_model = None
def get_dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(DENSE_MODEL)
    return _dense_model

def build_rankings(dataset_name, corpus, queries):
    doc_ids = list(corpus.keys())
    texts = [corpus_text(corpus[d]) for d in doc_ids]
    # BM25
    print('  building BM25...')
    bm25 = BM25Okapi([tok(t) for t in tqdm(texts, leave=False)])
    bm25_rank = {}
    for qid, q in tqdm(queries.items(), desc=f'{dataset_name}: BM25'):
        scores = np.asarray(bm25.get_scores(tok(q)), dtype=np.float32)
        bm25_rank[qid] = [doc_ids[i] for i in top_indices(scores, TOP_K)]

    # Dense corpus vectors cached by dataset/model/commit-independent corpus content.
    model = get_dense_model()
    model_tag = DENSE_MODEL.split('/')[-1].replace('-','_')
    emb_file = CACHE_DIR / f'{dataset_name}_{model_tag}_corpus.npy'
    ids_file = CACHE_DIR / f'{dataset_name}_{model_tag}_docids.json'
    if emb_file.exists() and ids_file.exists() and json.loads(ids_file.read_text()) == doc_ids:
        cemb = np.load(emb_file, mmap_mode='r')
    else:
        cemb = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True).astype('float32')
        np.save(emb_file, cemb); ids_file.write_text(json.dumps(doc_ids))
    qids = list(queries.keys())
    qemb = model.encode([queries[q] for q in qids], batch_size=128, show_progress_bar=False, normalize_embeddings=True).astype('float32')
    dense_rank = {}
    for qi, qid in enumerate(qids):
        scores = np.asarray(cemb @ qemb[qi], dtype=np.float32)
        dense_rank[qid] = [doc_ids[i] for i in top_indices(scores, TOP_K)]
    hybrid_rank = {qid: rrf_merge(bm25_rank[qid], dense_rank[qid], TOP_K) for qid in qids}
    return {'bm25': bm25_rank, 'dense': dense_rank, 'hybrid': hybrid_rank}, doc_ids

# ============================================================
# NOTEBOOK CELL 14 / CODE CELL 8
# ============================================================
def clean_extract(ranking, qrels):
    return [(d, float(qrels.get(d, 0))) for d in ranking]

def clean_aggregate(evidence, qrels):
    ranking = [d for d,_ in evidence]
    # The clean output is standard NDCG@k. This is the quantity the injected pipeline should preserve.
    return ndcg_at_k(ranking, qrels, TOP_K)

def fault_retrieval(clean_rank, doc_pool, qrels, severity, rng):
    # O(k) expected-time corruption: sample negatives by rejection instead of scanning
    # the full corpus for every intervention world (important for FiQA/TREC-COVID).
    out = list(clean_rank)
    n = max(1, int(round(len(out)*severity)))
    positive = [i for i,d in enumerate(out) if float(qrels.get(d,0)) > 0]
    targets = positive[:n] if positive else list(range(min(n,len(out))))
    forbidden = set(out)
    for i in targets:
        replacement = None
        for _ in range(100):
            cand = doc_pool[int(rng.integers(0, len(doc_pool)))]
            if cand not in forbidden and float(qrels.get(cand,0)) <= 0:
                replacement = cand; break
        if replacement is None:
            return list(reversed(out))
        forbidden.add(replacement); out[i] = replacement
    return out

def fault_extract(evidence, severity, rng):
    out = []
    for d,g in evidence:
        if g > 0 and rng.random() < severity:
            out.append((d, 0.0))
        elif g <= 0 and rng.random() < severity*0.15:
            out.append((d, 1.0))
        else:
            out.append((d,g))
    return out

def fault_aggregate(evidence, qrels, severity):
    # Plausible aggregation bug: average extracted gain rather than discounted normalized gain.
    vals = [max(0.0, g) for _,g in evidence]
    wrong = float(np.mean(vals)) if vals else 0.0
    clean = clean_aggregate(evidence, qrels)
    return (1-severity)*clean + severity*wrong

def execute_world(clean_rank, doc_pool, qrels, fault_set, repaired_set, severity, base_seed, qid):
    """do(repaired stages = clean), while all downstream unrepaired stages actually execute."""
    fault_set, repaired_set = set(fault_set), set(repaired_set)
    rngR = np.random.default_rng(stable_seed(base_seed, qid, 'R', severity))
    ranking = list(clean_rank) if ('R' not in fault_set or 'R' in repaired_set) else fault_retrieval(clean_rank, doc_pool, qrels, severity, rngR)

    evidence = clean_extract(ranking, qrels)
    if 'E' in fault_set and 'E' not in repaired_set:
        rngE = np.random.default_rng(stable_seed(base_seed, qid, 'E', severity))
        evidence = fault_extract(evidence, severity, rngE)

    answer = clean_aggregate(evidence, qrels)
    if 'A' in fault_set and 'A' not in repaired_set:
        answer = fault_aggregate(evidence, qrels, severity)
    return float(answer)

def diagnose_query(clean_rank, doc_pool, qrels, fault_set, severity, seed, qid):
    clean_y = execute_world(clean_rank, doc_pool, qrels, (), PLAYERS, severity, seed, qid)
    losses, outputs = {}, {}
    for S in all_subsets(PLAYERS):
        y = execute_world(clean_rank, doc_pool, qrels, fault_set, S, severity, seed, qid)
        outputs[S] = y
        losses[S] = abs(y-clean_y)
    baseline = losses[frozenset()]
    values = {S: baseline-loss for S,loss in losses.items()}
    phi = exact_shapley(values, PLAYERS)
    dividends = harsanyi_dividends(values, PLAYERS)
    tol = max(1e-8, baseline*0.02)
    mcrs = minimal_repair_sets(losses, PLAYERS, tol)
    return clean_y, baseline, losses, values, phi, dividends, mcrs

# Semantic smoke test: R repair must feed the repaired ranking into still-faulty E/A.
_qrels = {'d1':1,'d2':0,'d3':0,'d4':0}
_rank = ['d1','d2','d3']
_pool = list(_qrels)
_, base, losses, vals, phi, divs, mcrs = diagnose_query(_rank,_pool,_qrels,('R',),0.8,42,'toy')
assert base >= 0
assert losses[frozenset({'R'})] <= 1e-8
assert any(set(x)=={'R'} for x in mcrs)
print('Counterfactual semantic self-test passed; MCR =', mcrs)

# ============================================================
# NOTEBOOK CELL 16 / CODE CELL 9
# ============================================================
def true_fault_rank(phi, truth):
    order = sorted(phi, key=lambda p: (-phi[p], p))
    ranks = [order.index(t)+1 for t in truth]
    return min(ranks) if ranks else len(order)+1

def evaluate_block(dataset_name, retriever, rankings, doc_pool, qrels_all, seed, severity):
    rows = []
    for qid, clean_rank in tqdm(rankings.items(), desc=f'{dataset_name}/{retriever}/s{seed}/sev{severity}', leave=False):
        qrels = qrels_all[qid]
        # Standard clean retrieval metrics.
        ir = {
            'ndcg10': ndcg_at_k(clean_rank,qrels,10),
            'recall10': recall_at_k(clean_rank,qrels,10),
            'mrr10': mrr_at_k(clean_rank,qrels,10),
        }
        for fault_set in FAULT_SCENARIOS:
            clean_y, baseline, losses, values, phi, divs, mcrs = diagnose_query(clean_rank,doc_pool,qrels,fault_set,severity,seed,qid)
            if baseline <= 1e-10:
                # Fault had no observable effect for this query. Keep it, but mark non-identifiable.
                identifiable = False
            else:
                identifiable = True
            k = len(fault_set)
            shapley_order = sorted(PLAYERS, key=lambda p:(-phi[p],p))
            shapley_set = set(shapley_order[:k])
            direct_delta = {p: values[frozenset({p})] for p in PLAYERS}
            delta_order = sorted(PLAYERS, key=lambda p:(-direct_delta[p],p))
            delta_set = set(delta_order[:k])
            rrng = np.random.default_rng(stable_seed(seed,qid,'random-baseline',fault_set,severity))
            random_set = set(rrng.choice(list(PLAYERS),size=k,replace=False).tolist())
            pred_mcr = set(next(iter(mcrs), frozenset()))
            pair_abs = sum(abs(v) for S,v in divs.items() if len(S)==2)
            triple = divs.get(frozenset(PLAYERS),0.0)
            rows.append({
                'dataset':dataset_name,'retriever':retriever,'seed':seed,'severity':severity,'qid':qid,
                'fault_truth':'+'.join(fault_set),'fault_k':k,'identifiable':identifiable,
                'clean_output':clean_y,'baseline_loss':baseline,
                **ir,
                **{f'phi_{p}':phi[p] for p in PLAYERS},
                'shapley_top1':shapley_order[0],
                'shapley_exact_k':int(shapley_set==set(fault_set)),
                'shapley_set_f1':set_f1(shapley_set,fault_set),
                'shapley_jaccard':jaccard(shapley_set,fault_set),
                'shapley_fault_mrr':1/true_fault_rank(phi,fault_set),
                'delta_exact_k':int(delta_set==set(fault_set)),
                'delta_set_f1':set_f1(delta_set,fault_set),
                'random_exact_k':int(random_set==set(fault_set)),
                'random_set_f1':set_f1(random_set,fault_set),
                'mcr_pred':'+'.join(sorted(pred_mcr)),
                'mcr_exact':int(pred_mcr==set(fault_set)),
                'mcr_size':len(pred_mcr),
                'mcr_residual_loss':min([losses.get(frozenset(pred_mcr),np.nan)]),
                'pair_interaction_abs_sum':pair_abs,'triple_interaction':triple,
                'interventions_exact':2**len(PLAYERS),
            })
    return pd.DataFrame(rows)

for dataset_name in BENCHMARKS:
    print('\n===',dataset_name,'===')
    corpus, queries, qrels_all = load_beir(dataset_name)
    all_rankings, doc_pool = build_rankings(dataset_name, corpus, queries)
    for retriever in RETRIEVERS:
        for seed in SEEDS:
            for severity in FAULT_SEVERITIES:
                key = f'{dataset_name}__{retriever}__s{seed}__v{str(severity).replace(".","p")}'
                if is_done(key):
                    print('skip completed:', key); continue
                t0=time.time()
                df = evaluate_block(dataset_name,retriever,all_rankings[retriever],doc_pool,qrels_all,seed,severity)
                out = CHUNK_DIR / f'{key}.parquet'
                tmp = out.with_suffix('.tmp.parquet'); df.to_parquet(tmp,index=False); os.replace(tmp,out)
                mark_done(key, {'rows':len(df),'seconds':round(time.time()-t0,2),'commit':COMMIT})
                print('saved',key,len(df),'rows')

# ============================================================
# NOTEBOOK CELL 18 / CODE CELL 10
# ============================================================
files = sorted(CHUNK_DIR.glob('*.parquet'))
assert files, 'No completed chunks found.'
res = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
res.to_parquet(RUN_ROOT/'all_results.parquet', index=False)
res.to_csv(RUN_ROOT/'all_results.csv', index=False)

ident = res[res.identifiable].copy()
summary = ident.groupby(['dataset','retriever','severity'], as_index=False).agg(
    n=('qid','size'), ndcg10=('ndcg10','mean'), recall10=('recall10','mean'), mrr10=('mrr10','mean'),
    shapley_exact=('shapley_exact_k','mean'), shapley_f1=('shapley_set_f1','mean'),
    delta_exact=('delta_exact_k','mean'), delta_f1=('delta_set_f1','mean'),
    random_exact=('random_exact_k','mean'), random_f1=('random_set_f1','mean'),
    mcr_exact=('mcr_exact','mean'), residual=('mcr_residual_loss','mean'),
    interaction=('pair_interaction_abs_sum','mean'))
summary.to_csv(TABLE_DIR/'table_A_main.csv',index=False)
display(summary)

# Dataset-stratified bootstrap CI for the primary comparison: Shapley exact-set vs single-fix delta.
rng=np.random.default_rng(20260910)
def bootstrap_diff(df, a='shapley_exact_k', b='delta_exact_k', B=2000):
    diffs=[]
    groups=[g for _,g in df.groupby('dataset')]
    for _ in range(B):
        vals=[]
        for g in groups:
            idx=rng.integers(0,len(g),len(g))
            s=g.iloc[idx]
            vals.append((s[a]-s[b]).mean())
        diffs.append(np.mean(vals))
    return float(np.mean(diffs)), float(np.quantile(diffs,.025)), float(np.quantile(diffs,.975))

diff,lo,hi=bootstrap_diff(ident)
stat,p=stats.wilcoxon(ident.shapley_set_f1,ident.delta_set_f1,zero_method='zsplit')
stat_table=pd.DataFrame([{'metric':'Shapley - single-fix exact-set','mean_diff':diff,'ci95_low':lo,'ci95_high':hi,'wilcoxon_W':stat,'p_value':p}])
stat_table.to_csv(TABLE_DIR/'table_A_statistics.csv',index=False)
display(stat_table)

# ============================================================
# NOTEBOOK CELL 20 / CODE CELL 11
# ============================================================
import matplotlib.pyplot as plt
plot_df = ident.groupby('dataset')[['shapley_exact_k','delta_exact_k','mcr_exact']].mean()
ax=plot_df.plot(kind='bar', figsize=(9,4))
ax.set_ylabel('Exact fault-set recovery')
ax.set_ylim(0,1)
ax.set_title('FaulTrace fault localization across BEIR domains')
plt.tight_layout(); plt.savefig(PLOT_DIR/'fault_localization_by_dataset.png',dpi=220); plt.show()

# Zip only results/tables/plots/checkpoint, not huge model caches.
archive_base = RUN_ROOT / 'FaulTrace_Notebook_A_results'
!rm -f "{archive_base}.zip"
!cd "{RUN_ROOT}" && zip -qr "{archive_base}.zip" chunks paper_tables plots checkpoint.json all_results.csv all_results.parquet environment.txt
print('Final archive:', archive_base.with_suffix('.zip'))
