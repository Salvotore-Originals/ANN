import csv, time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.csv"
COLUMNS = ["timestamp", "part", "dataset", "index", "n_train", "build_params", "search_param",
           "split", "n_queries", "threads", "mode", "recall", "qps", "build_time_s",
           "index_bytes", "bytes_per_vec", "seed", "notes"]

def log_row(**kw):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    row = {c: "" for c in COLUMNS}
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    for k, v in kw.items():
        assert k in COLUMNS, f"unknown column {k}"
        row[k] = v
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)