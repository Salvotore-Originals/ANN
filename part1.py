import faiss
import data, metrics
from timing import single_thread, qps_one_at_a_time, qps_batch
from results_log import log_row

K = 10
N_SINGLE = 200          # exact single-query search is slow on 1M vectors, so time a subset

def run(name, n_train=None):
    train, test = data.load(name, n_train)
    tune, rep = data.split_queries(test)
    I = data.exact_neighbors(name, train, test, k=100)
    kd_rep = metrics.kth_dist2(rep, train, I[data.N_TUNE:], K)

    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    n = len(train)
    base = dict(part=1, dataset=name, index="flat", n_train=n, split="report", seed=0)

    # sanity: the exact index scores recall 1.0 on the report half
    _, I_rep = index.search(rep, K)
    rec = metrics.recall_at_k(I_rep, rep, train, kd_rep, K)
    print(f"{name}: exact recall@10 on report half = {rec:.4f}")

    with single_thread():
        q1 = qps_one_at_a_time(index, rep[:N_SINGLE], K, repeats=3)
    log_row(**base, n_queries=N_SINGLE, threads=1, mode="single_query", recall=rec, qps=q1)

    with single_thread():
        qb1 = qps_batch(index, rep, K, repeats=3)
    log_row(**base, n_queries=len(rep), threads=1, mode="batch", recall=rec, qps=qb1)

    nthr = faiss.omp_get_max_threads()
    qbn = qps_batch(index, rep, K, repeats=3)
    log_row(**base, n_queries=len(rep), threads=nthr, mode="batch", recall=rec, qps=qbn)

    print(f"{name}: single query 1 thread {q1:,.1f} QPS | batch 1 thread {qb1:,.1f} QPS | "
          f"batch {nthr} threads {qbn:,.1f} QPS | batch-vs-single speedup (1 thread) {qb1/q1:.1f}x")

if __name__ == "__main__":
    run("sift")
    run("glove")