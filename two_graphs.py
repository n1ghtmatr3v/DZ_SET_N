import argparse
import csv
import os
import math
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

def f(x):
    try:
        return float(x)
    except:
        return math.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--x", choices=["t", "fraction"], default="t")
    args = ap.parse_args()

    stream_path = os.path.join(args.dir, f"stream_{args.stream}.csv")
    stats_path = os.path.join(args.dir, "stats.csv")

    srows = read_csv(stream_path)
    X = [f(r[args.x]) for r in srows]
    true_F0 = [f(r["true_F0"]) for r in srows]
    est_N = [f(r["est_N"]) for r in srows]

    plt.figure()
    plt.plot(X, true_F0, label="true unique")
    plt.plot(X, est_N, label="HLL estimate")
    plt.xlabel(args.x)
    plt.ylabel("unique count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.dir, "graph1.png"), dpi=200)

    rows = read_csv(stats_path)
    X2 = [f(r[args.x]) for r in rows]
    mean_est = [f(r["mean_est"]) for r in rows]
    std_est = [f(r["std_est"]) for r in rows]
    upper = [m + s for m, s in zip(mean_est, std_est)]
    lower = [m - s for m, s in zip(mean_est, std_est)]

    plt.figure()
    plt.plot(X2, mean_est, label="mean estimate")
    plt.fill_between(X2, lower, upper, alpha=0.2, label="± std")
    plt.xlabel(args.x)
    plt.ylabel("unique count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.dir, "graph2.png"), dpi=200)

if __name__ == "__main__":
    main()
