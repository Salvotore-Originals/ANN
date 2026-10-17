import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
TARGETS = [0.90, 0.95]
NAMES = {"hnsw": "HNSW", "ivfflat": "IVF-Flat", "ivfpq": "IVF-PQ"}
COLORS = {"hnsw": "tab:blue", "ivfflat": "tab:orange", "ivfpq": "tab:green"}

def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)

def part2(df, ds):
    s = df[(df["part"].astype(str) == "2") & (df.dataset == ds)]
    return s[s.n_train == s.n_train.max()]

def sweep_frame(df, ds):
    s = part2(df, ds)
    s = s[s["mode"] == "sweep"]
    keys = ["index", "build_params", "search_param"]
    t = s[s.split == "tune"].drop_duplicates(keys, keep="last").set_index(keys)[["recall"]]
    t.columns = ["tune_recall"]
    r = s[s.split == "report"].drop_duplicates(keys, keep="last").set_index(keys)[["recall", "qps"]]
    out = t.join(r, how="inner").reset_index()
    out["search_param"] = out["search_param"].astype(int)
    return out

def build_frame(df, ds):
    b = part2(df, ds)
    b = b[b["mode"] == "build"].drop_duplicates(["index", "build_params"], keep="last")
    return b[["index", "build_params", "build_time_s", "index_bytes", "bytes_per_vec"]]

def pareto(d):
    d = d.sort_values("recall", ascending=False)
    keep, best = [], -1.0
    for i, row in d.iterrows():
        if row.qps > best:
            keep.append(i)
            best = row.qps
    return d.loc[keep].sort_values("recall")

def pick(sw, idx, target):
    """Smallest parameter with TUNE recall >= target, per build config; best config by report QPS.
    Returns None if the target is never reached in the sweep."""
    rows = []
    for bp, g in sw[sw["index"] == idx].groupby("build_params"):
        ok = g[g.tune_recall >= target].sort_values("search_param")
        if len(ok):
            rows.append(ok.iloc[0])
    return max(rows, key=lambda r: r.qps) if rows else None

def exact_qps(df, ds):
    p = df[(df["part"].astype(str) == "1") & (df.dataset == ds) & (df["mode"] == "single_query")]
    return float(p.qps.iloc[-1])

def make_tables(df, ds, sw):
    q, b = [], []
    builds = build_frame(df, ds)
    for idx in NAMES:
        g = sw[sw["index"] == idx]
        if g.empty:
            continue
        qrow = {"index": NAMES[idx]}
        for t in TARGETS:
            r = pick(sw, idx, t)
            if r is None:
                qrow[f"QPS @{t:.2f}"] = f"not reached (max recall {g.recall.max():.3f})"
            else:
                qrow[f"QPS @{t:.2f}"] = f"{r.qps:,.0f} ({r.build_params}, param={r.search_param}, recall={r.recall:.3f})"
        q.append(qrow)
        r = pick(sw, idx, 0.95)
        note = ""
        if r is None:                       # report the config with the highest recall instead
            r = g.loc[g.tune_recall.idxmax()]
            note = f"best recall only {r.recall:.3f}"
        bb = builds[(builds["index"] == idx) & (builds.build_params == r.build_params)].iloc[0]
        b.append({"index": NAMES[idx], "config": r.build_params, "build time (s)": bb.build_time_s,
                  "size (MB)": round(bb.index_bytes / 2**20, 1), "bytes/vector": bb.bytes_per_vec,
                  "note": note})
    return pd.DataFrame(q), pd.DataFrame(b)

def make_figure(df, ds, sw):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for idx in NAMES:
        g = sw[sw["index"] == idx]
        if g.empty:
            continue
        f = pareto(g)
        ax.plot(f.recall, f.qps, "-o", ms=3, color=COLORS[idx], label=NAMES[idx])
    ax.plot([1.0], [exact_qps(df, ds)], "k*", ms=11, label="Exact (1 query at a time)")
    for t in TARGETS:
        ax.axvline(t, color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlim(0.2, 1.01)
    ax.set_xlabel("recall@10 (report half)")
    ax.set_ylabel("queries per second (1 thread)")
    ax.set_title(f"{ds.upper()}: best trade-off per index")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"part2_{ds}.png", dpi=200)
    fig.savefig(FIG / f"part2_{ds}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "results.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    TAB.mkdir(parents=True, exist_ok=True)
    for ds in ("sift", "glove"):
        sw = sweep_frame(df, ds)
        if sw.empty:
            continue
        qt, bt = make_tables(df, ds, sw)
        make_figure(df, ds, sw)
        qt.to_csv(TAB / f"part2_qps_{ds}.csv", index=False)
        bt.to_csv(TAB / f"part2_build_{ds}.csv", index=False)
        print(f"\n## {ds}: QPS at recall targets\n{md(qt)}\n\n## {ds}: build cost at recall 0.95\n{md(bt)}")