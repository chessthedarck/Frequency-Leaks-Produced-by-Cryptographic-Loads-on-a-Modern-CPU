#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "output_cycles"
OUTPUT_DIR = ROOT / "output_cycles_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
TIME_BIN_MS = 100
PHASE_ORDER = ["baseline", "aes", "rsa", "ed25519", "sha256"]

plt.rcParams.update({"figure.figsize": (12, 6)})


def discover_cycle_csvs() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for path in sorted(INPUT_ROOT.glob("*_plus_baseline/*.csv")):
        algo = path.parent.name.removesuffix("_plus_baseline")
        items.append((algo, path))
    if not items:
        raise SystemExit(f"Aucun CSV consolidé trouvé dans {INPUT_ROOT}")
    return items


def load_cycle_csv(algo: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "cycle",
        "phase",
        "source_csv",
        "time_ms",
        "cpu",
        "scaling_cur_freq_khz",
        "activity",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name}: {missing}")

    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce")
    df["scaling_cur_freq_khz"] = pd.to_numeric(df["scaling_cur_freq_khz"], errors="coerce")
    df = df.dropna(subset=["cycle", "time_ms", "cpu", "scaling_cur_freq_khz"]).copy()

    df["algo"] = algo
    df["freq_mhz"] = df["scaling_cur_freq_khz"] / 1000.0
    df["t_s"] = df["time_ms"] / 1000.0
    df["t_bin_s"] = ((df["time_ms"] // TIME_BIN_MS) * TIME_BIN_MS) / 1000.0
    return df


def summarize_per_run(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["algo", "phase", "cycle", "source_csv"], as_index=False)
        .agg(
            n_rows=("freq_mhz", "size"),
            n_cores=("cpu", "nunique"),
            duration_s=("t_s", "max"),
            freq_mean_mhz=("freq_mhz", "mean"),
            freq_std_mhz=("freq_mhz", "std"),
            freq_min_mhz=("freq_mhz", "min"),
            freq_max_mhz=("freq_mhz", "max"),
            freq_median_mhz=("freq_mhz", "median"),
        )
        .sort_values(["algo", "cycle", "phase", "source_csv"])
    )
    return grouped


def summarize_per_phase(df: pd.DataFrame, per_run: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["algo", "phase"], as_index=False)
        .agg(
            n_rows=("freq_mhz", "size"),
            n_runs=("source_csv", "nunique"),
            n_cycles=("cycle", "nunique"),
            n_cores=("cpu", "nunique"),
            freq_mean_mhz=("freq_mhz", "mean"),
            freq_std_mhz=("freq_mhz", "std"),
            freq_min_mhz=("freq_mhz", "min"),
            freq_max_mhz=("freq_mhz", "max"),
            freq_median_mhz=("freq_mhz", "median"),
        )
    )
    run_stats = (
        per_run.groupby(["algo", "phase"], as_index=False)
        .agg(
            duration_mean_s=("duration_s", "mean"),
            duration_std_s=("duration_s", "std"),
        )
    )
    out = agg.merge(run_stats, on=["algo", "phase"], how="left")
    return out.sort_values(["algo", "phase"])


def build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    ts = (
        df.groupby(["algo", "phase", "t_bin_s"], as_index=False)["freq_mhz"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
        .rename(columns={"mean": "freq_mean_mhz", "min": "freq_min_mhz", "max": "freq_max_mhz", "std": "freq_std_mhz"})
        .sort_values(["algo", "phase", "t_bin_s"])
    )
    return ts


def summarize_differences(phase_summary: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo in sorted(phase_summary["algo"].unique()):
        base = phase_summary[(phase_summary["algo"] == algo) & (phase_summary["phase"] == "baseline")]
        crypto = phase_summary[(phase_summary["algo"] == algo) & (phase_summary["phase"] == algo)]
        if base.empty or crypto.empty:
            continue

        base_mean = float(base["freq_mean_mhz"].iloc[0])
        crypto_mean = float(crypto["freq_mean_mhz"].iloc[0])

        base_ts = timeseries[(timeseries["algo"] == algo) & (timeseries["phase"] == "baseline")][["t_bin_s", "freq_mean_mhz"]]
        crypto_ts = timeseries[(timeseries["algo"] == algo) & (timeseries["phase"] == algo)][["t_bin_s", "freq_mean_mhz"]]
        merged = base_ts.merge(crypto_ts, on="t_bin_s", suffixes=("_baseline", "_crypto"))
        if not merged.empty:
            merged["diff_mhz"] = merged["freq_mean_mhz_crypto"] - merged["freq_mean_mhz_baseline"]
            diff_mean = float(merged["diff_mhz"].mean())
            diff_max = float(merged["diff_mhz"].max())
            diff_min = float(merged["diff_mhz"].min())
        else:
            diff_mean = np.nan
            diff_max = np.nan
            diff_min = np.nan

        rows.append(
            {
                "algo": algo,
                "baseline_mean_mhz": base_mean,
                "crypto_mean_mhz": crypto_mean,
                "delta_mean_mhz": crypto_mean - base_mean,
                "timeseries_diff_mean_mhz": diff_mean,
                "timeseries_diff_max_mhz": diff_max,
                "timeseries_diff_min_mhz": diff_min,
            }
        )
    return pd.DataFrame(rows).sort_values("algo")


def plot_pair_timeseries(timeseries: pd.DataFrame) -> None:
    algos = sorted(timeseries["algo"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, algo in zip(axes, algos):
        subset = timeseries[timeseries["algo"] == algo]
        baseline = subset[subset["phase"] == "baseline"]
        crypto = subset[subset["phase"] == algo]

        ax.plot(baseline["t_bin_s"], baseline["freq_mean_mhz"], label="baseline", color="#4c78a8", linewidth=1.6)
        ax.fill_between(baseline["t_bin_s"], baseline["freq_min_mhz"], baseline["freq_max_mhz"], color="#4c78a8", alpha=0.12)

        ax.plot(crypto["t_bin_s"], crypto["freq_mean_mhz"], label=algo, color="#f58518", linewidth=1.6)
        ax.fill_between(crypto["t_bin_s"], crypto["freq_min_mhz"], crypto["freq_max_mhz"], color="#f58518", alpha=0.12)

        ax.set_title(f"{algo} vs baseline")
        ax.set_xlabel("Temps relatif (s)")
        ax.set_ylabel("Fréquence moyenne (MHz)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(algos):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_pair_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_differences(timeseries: pd.DataFrame) -> None:
    algos = sorted(timeseries["algo"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, algo in zip(axes, algos):
        subset = timeseries[timeseries["algo"] == algo]
        baseline = subset[subset["phase"] == "baseline"]
        crypto = subset[subset["phase"] == algo]
        merged = baseline[["t_bin_s", "freq_mean_mhz"]].merge(
            crypto[["t_bin_s", "freq_mean_mhz"]], on="t_bin_s", suffixes=("_baseline", "_crypto")
        )
        merged["diff_mhz"] = merged["freq_mean_mhz_crypto"] - merged["freq_mean_mhz_baseline"]

        ax.plot(merged["t_bin_s"], merged["diff_mhz"], color="#54a24b", linewidth=1.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
        ax.set_title(f"{algo} - baseline")
        ax.set_xlabel("Temps relatif (s)")
        ax.set_ylabel("Différence moyenne (MHz)")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(algos):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_pair_differences.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame) -> None:
    order = []
    for algo in sorted(df["algo"].unique()):
        order.extend([f"{algo}\nbaseline", f"{algo}\n{algo}"])
    tmp = df.copy()
    tmp["label"] = tmp["algo"] + "\n" + tmp["phase"]

    data = [tmp.loc[tmp["label"] == label, "freq_mhz"].to_numpy() for label in order]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("Fréquence CPU (MHz)")
    ax.set_title("Distribution des fréquences par algorithme et phase")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_boxplot_by_algo_phase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_json_summary(phase_summary: pd.DataFrame, diff_summary: pd.DataFrame) -> None:
    payload = {
        "phase_summary": phase_summary.to_dict(orient="records"),
        "diff_summary": diff_summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "global_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    discovered = discover_cycle_csvs()
    frames = [load_cycle_csv(algo, path) for algo, path in discovered]
    df = pd.concat(frames, ignore_index=True)

    per_run = summarize_per_run(df)
    phase_summary = summarize_per_phase(df, per_run)
    timeseries = build_timeseries(df)
    diff_summary = summarize_differences(phase_summary, timeseries)

    per_run.to_csv(OUTPUT_DIR / "per_run_summary.csv", index=False)
    phase_summary.to_csv(OUTPUT_DIR / "phase_summary.csv", index=False)
    timeseries.to_csv(OUTPUT_DIR / "timeseries_summary.csv", index=False)
    diff_summary.to_csv(OUTPUT_DIR / "diff_vs_baseline_summary.csv", index=False)
    write_json_summary(phase_summary, diff_summary)

    plot_pair_timeseries(timeseries)
    plot_pair_differences(timeseries)
    plot_boxplot(df)

    print(f"[OK] Analyse écrite dans {OUTPUT_DIR}")
    print("[OK] Fichiers:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
