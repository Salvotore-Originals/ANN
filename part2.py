import argparse, gc
import pandas as pd, faiss
import data, metrics, indexes
from timing import single_thread, qps_one_at_a_time
from results_log import log_row, RESULTS

K = 10
EF = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
NP = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

def get_configs(name, quick):
    pq_m = [16, 32, 64] if name == "sift" else [10, 20, 50]   # m must divide d (GloVe d=100)
    if quick:
        pq_m = pq_m[:2]
        return ([("hnsw", dict(M=m, efc=200)) for m in (16, 32)] +
                [("ivfflat", dict(nlist=n)) for n in (256, 1024)] +
                [("ivfpq", dict(nlist=1024, m=m)) for m in pq_m])
    return ([("hnsw", dict(M=m, efc=200)) for m in (16, 32, 48)] +
            [("ivfflat", dict(nlist=n)) for n in (1024, 4096, 16384)] +
            [("ivfpq", dict(nlist=4096, m=m)) for m in pq_m])

def fmt(p):
    return ",".join(f"{k}={v}" for k, v in p.items())

def key(ds, kind, n, bp, sp, split):
    return (ds, kind, str(n), bp, str(sp), split)

def load_done():
    done, builds = {}, set()
    if RESULTS.exists():
        df = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
        for _, r in df[df["part"] == "2"].iterrows():
            if r["mode"] == "sweep":
                done[key(r["dataset"], r["index"], r["n_train"], r["build_params"],
                         r["search_param"], r["split"])] = float(r["recall"])
            elif r["mode"] == "build":
                builds.add((r["dataset"], r["index"], r["n_train"], r["build_params"]))
    return done, builds

def measure(index, kind, v, q, kd, db, repeats):
    indexes.set_search(index, kind, v)
    _, ids = index.search(q, K)
    rec = metrics.recall_at_k(ids, q, db, kd, K)
    with single_thread():
        qps = qps_one_at_a_time(index, q, K, repeats=repeats)
    return rec, qps

def run_dataset(name, kinds, n_train, nq, quick, done, builds):
    train, test = data.load(name, n_train)
    n = len(train)
    I = data.exact_neighbors(name, train, test, k=100)
    tune, rep = data.split_queries(test)
    N = data.N_TUNE
    splits = {"tune": (tune[:nq], metrics.kth_dist2(tune[:nq], train, I[:nq], K)),
              "report": (rep[:nq], metrics.kth_dist2(rep[:nq], train, I[N:N + nq], K))}
    for kind, params in get_configs(name, quick):
        if kind not in kinds:
            continue
        bp = fmt(params)
        print(f"\n=== {name} {kind} {bp} (n={n}) ===", flush=True)
        index, bt = indexes.build(kind, train, **params)
        if (name, kind, str(n), bp) not in builds:
            sz = indexes.size_bytes(index)
            log_row(part=2, dataset=name, index=kind, n_train=n, build_params=bp, mode="build",
                    threads=faiss.omp_get_max_threads(), build_time_s=round(bt, 2),
                    index_bytes=sz, bytes_per_vec=round(sz / n, 1), seed=0)
            builds.add((name, kind, str(n), bp))
        print(f"built in {bt:.1f}s", flush=True)
        grid = EF if kind == "hnsw" else [v for v in NP if v <= params["nlist"]]
        hist = []
        for v in grid:
            recs = {}
            for split, (q, kd) in splits.items():
                k_ = key(name, kind, n, bp, v, split)
                if k_ in done:
                    recs[split] = done[k_]
                    continue
                rp = 5 if split == "report" else 1
                rec, qps = measure(index, kind, v, q, kd, train, rp)
                log_row(part=2, dataset=name, index=kind, n_train=n, build_params=bp,
                        search_param=v, split=split, n_queries=len(q), threads=1, mode="sweep",
                        recall=round(rec, 5), qps=round(qps, 2), seed=0, notes=f"repeats={rp}")
                done[k_] = rec
                recs[split] = rec
                print(f"  {split:6s} param={v:5d} recall={rec:.4f} qps={qps:,.0f}", flush=True)
            hist.append(recs["tune"])
            if recs["tune"] >= 0.995:
                break
            if len(hist) >= 4 and hist[-1] - hist[-3] < 0.004:
                print("  recall has plateaued, stopping sweep", flush=True)
                break
        del index
        gc.collect()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dataset", choices=["sift", "glove", "both"], default="both")
    ap.add_argument("--kinds", default="hnsw,ivfflat,ivfpq")
    a = ap.parse_args()
    done, builds = load_done()
    for name in (["sift", "glove"] if a.dataset == "both" else [a.dataset]):
        run_dataset(name, a.kinds.split(","), 100_000 if a.quick else None,
                    1000 if a.quick else 5000, a.quick, done, builds)