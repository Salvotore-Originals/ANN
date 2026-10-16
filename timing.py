import time, statistics, contextlib
import numpy as np, psutil, faiss

_proc = psutil.Process()

@contextlib.contextmanager
def single_thread(cpu=0):
    """1 FAISS thread, pinned to one logical CPU (reduces P-core/E-core noise on hybrid Intel chips)."""
    n_threads, orig = faiss.omp_get_max_threads(), _proc.cpu_affinity()
    faiss.omp_set_num_threads(1)
    _proc.cpu_affinity([cpu])
    try:
        yield
    finally:
        _proc.cpu_affinity(orig)
        faiss.omp_set_num_threads(n_threads)

def median_time(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def qps_one_at_a_time(index, queries, k=10, repeats=5):
    """Queries/second when sent one by one. Call inside `with single_thread():`."""
    qs = [np.ascontiguousarray(q[None]) for q in queries]   # slicing done outside the timed region
    def run():
        for q in qs:
            index.search(q, k)
    return len(qs) / median_time(run, repeats)

def qps_batch(index, queries, k=10, repeats=3):
    return len(queries) / median_time(lambda: index.search(queries, k), repeats)