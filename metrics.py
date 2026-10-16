import numpy as np

def _d2(db, queries, ids):
    """Squared L2 distances in float64 from each query to each of its ids. ids: (nq, k), all >= 0."""
    diff = db[ids].astype(np.float64) - queries[:, None, :].astype(np.float64)
    return np.einsum("qkd,qkd->qk", diff, diff)

def kth_dist2(queries, db, exact_ids, k=10):
    """Squared distance of the k-th true neighbour (max over the top-k, robust to float ordering noise)."""
    out = np.empty(len(queries))
    for s in range(0, len(queries), 1000):
        sl = slice(s, s + 1000)
        out[sl] = _d2(db, queries[sl], exact_ids[sl, :k]).max(axis=1)
    return out

def recall_at_k(ids, queries, db, kth_d2, k=10, rtol=1e-5, per_query=False):
    """Tie-aware recall@k. A returned id is a hit if its true distance <= the k-th true distance.
    Duplicates count once; -1 (no result) counts as a miss."""
    ids = np.asarray(ids)[:, :k]
    nq = len(ids)
    hits = np.zeros(nq)
    for s in range(0, nq, 1000):
        sl = slice(s, s + 1000)
        I = ids[sl]
        valid = I >= 0
        d2 = _d2(db, queries[sl], np.where(valid, I, 0))
        ok = valid & (d2 <= kth_d2[sl, None] * (1 + rtol))
        order = np.argsort(I, axis=1, kind="stable")           # mark repeated ids
        srt = np.take_along_axis(I, order, axis=1)
        dup_sorted = np.zeros_like(ok)
        dup_sorted[:, 1:] = srt[:, 1:] == srt[:, :-1]
        dup = np.empty_like(dup_sorted)
        np.put_along_axis(dup, order, dup_sorted, axis=1)
        hits[sl] = (ok & ~dup).sum(axis=1)
    r = hits / k
    return r if per_query else float(r.mean())