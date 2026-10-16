import faiss
import data, metrics, indexes
from timing import single_thread, qps_one_at_a_time

N, NQ = 100_000, 500
train, test = data.load("sift", n_train=N)
tune, _ = data.split_queries(test)
I = data.exact_neighbors("sift", train, test, k=100)
q = tune[:NQ]
kd = metrics.kth_dist2(q, train, I[:NQ], 10)

configs = [("hnsw", dict(M=32, efc=200), [10, 40, 160]),
           ("ivfflat", dict(nlist=512), [1, 4, 16, 64]),
           ("ivfpq", dict(nlist=512, m=16), [1, 4, 16, 64])]
for kind, params, sweep in configs:
    index, bt = indexes.build(kind, train, **params)
    print(f"{kind} {params}: build {bt:.1f}s, {indexes.size_bytes(index)/N:.0f} bytes/vector")
    for v in sweep:
        indexes.set_search(index, kind, v)
        _, ids = index.search(q, 10)
        r = metrics.recall_at_k(ids, q, train, kd, 10)
        with single_thread():
            qps = qps_one_at_a_time(index, q, 10, repeats=3)
        print(f"   param={v:4d}  recall={r:.3f}  qps={qps:,.0f}")