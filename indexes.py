import time
import numpy as np, faiss

def _sample(xb, n, seed=0):
    n = min(n, len(xb))
    idx = np.random.default_rng(seed).choice(len(xb), n, replace=False)
    return np.ascontiguousarray(xb[idx])

def build(kind, xb, **p):
    """kind: 'hnsw' (M, efc) | 'ivfflat' (nlist) | 'ivfpq' (nlist, m). Returns (index, build_seconds)."""
    d = xb.shape[1]
    t0 = time.perf_counter()
    if kind == "hnsw":
        index = faiss.index_factory(d, f"HNSW{p['M']},Flat")
        index.hnsw.efConstruction = p.get("efc", 200)
        index.add(xb)
    elif kind == "ivfflat":
        index = faiss.index_factory(d, f"IVF{p['nlist']},Flat")
        index.train(_sample(xb, max(50 * p["nlist"], 100_000)))
        index.add(xb)
    elif kind == "ivfpq":
        index = faiss.index_factory(d, f"IVF{p['nlist']},PQ{p['m']}x8")
        index.train(_sample(xb, max(50 * p["nlist"], 100_000)))
        index.add(xb)
    else:
        raise ValueError(kind)
    return index, time.perf_counter() - t0

def set_search(index, kind, value):
    if kind == "hnsw":
        index.hnsw.efSearch = int(value)      # must be >= 10 for k=10
    else:
        index.nprobe = int(value)             # clamped to nlist by FAISS

def size_bytes(index):
    return len(faiss.serialize_index(index))