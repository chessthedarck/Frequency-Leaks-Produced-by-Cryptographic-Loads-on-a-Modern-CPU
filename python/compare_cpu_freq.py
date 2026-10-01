#!/usr/bin/env python3
"""Comparaison visuelle des fréquences CPU entre plusieurs captures CSV.

Usage rapide (défaut : test1000 vs workload s0) :
    python3 python/compare_cpu_freq.py

Usage générique :
    python3 python/compare_cpu_freq.py fileA.csv fileB.csv [fileC.csv ...] \
        --labels "run A" "run B" --out graphs/cpu_freq_compare.png --grid-step 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FILES = [
    Path("tests_1000/crypto_workload/cpu_freq_20260218_234028.csv"),
    Path("rasberry_0/cpu_freq_s0.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare des fréquences CPU en fonction du temps (relatif)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="Chemins vers les CSV (2 minimum). Par défaut on compare test1000 vs workload s0.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Légendes à afficher (même ordre que les fichiers).",
    )
    parser.add_argument(
        "--out",
        default="graphs/cpu_freq_compare.png",
        help="Chemin de la figure de sortie (.png).",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.5,
        help="Pas (en secondes) de la grille commune pour la courbe de différence A-B.",
    )
    return parser.parse_args()


def detect_freq_column(df: pd.DataFrame) -> str:
    candidates = [
        "scaling_cur_freq_khz",
        "cpu_freq_khz",
        "cur_freq",
        "freq",
        "frequency",
    ]
    for key in candidates:
        for col in df.columns:
            if key in col:
                return col
    raise ValueError("Colonne fréquence introuvable (ex: scaling_cur_freq_khz)")


def extract_time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "time_ms" in df.columns:
        t_ms = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy()
        valid = np.isfinite(t_ms)
        if not np.any(valid):
            raise ValueError("Toutes les valeurs temporelles sont invalides")
        return t_ms[valid] / 1000.0
    if "timestamp_ns" in df.columns:
        t_ns = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy()
    elif "timestamp_iso" in df.columns:
        t_ns = pd.to_datetime(df["timestamp_iso"], errors="coerce").view("int64")
    else:
        raise ValueError("Aucune colonne de temps trouvée (time_ms, timestamp_ns ou timestamp_iso)")

    valid = np.isfinite(t_ns)
    if not np.any(valid):
        raise ValueError("Toutes les valeurs temporelles sont invalides")

    t_ns = t_ns[valid]
    t0 = t_ns.min()
    return (t_ns - t0) / 1e9  # secondes relatives


def load_series(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    freq_col = detect_freq_column(df)
    time_s = extract_time_seconds(df)

    df = df.loc[np.isfinite(time_s)].copy()
    df["time_s"] = time_s
    df["freq_khz"] = pd.to_numeric(df[freq_col], errors="coerce")

    grouped = (
        df.groupby("time_s")["freq_khz"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_khz", "min": "min_khz", "max": "max_khz"})
        .sort_values("time_s")
    )

    series = {
        "time_s": grouped["time_s"].to_numpy(),
        "mean_mhz": (grouped["mean_khz"].to_numpy()) / 1000.0,
        "min_mhz": (grouped["min_khz"].to_numpy()) / 1000.0,
        "max_mhz": (grouped["max_khz"].to_numpy()) / 1000.0,
    }

    summary = {
        "n_points": int(len(grouped)),
        "t_span_s": float(series["time_s"].max() - series["time_s"].min()),
        "mean_mhz": float(np.nanmean(series["mean_mhz"])),
        "p95_mhz": float(np.nanpercentile(series["mean_mhz"], 95)),
    }

    return {**series, "summary": summary}


def build_common_grid(series_list: List[Dict[str, np.ndarray]], step_s: float) -> np.ndarray:
    t_start = max(s["time_s"].min() for s in series_list)
    t_stop = min(s["time_s"].max() for s in series_list)
    if t_stop <= t_start:
        return np.array([])
    n_steps = int((t_stop - t_start) / step_s) + 1
    return np.linspace(t_start, t_stop, n_steps)


def plot_comparison(series_list: List[Dict[str, np.ndarray]], labels: List[str], out_path: Path, step_s: float):
    colors = plt.cm.tab10(np.linspace(0, 1, len(series_list)))
    fig, (ax_top, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for s, label, color in zip(series_list, labels, colors):
        ax_top.plot(s["time_s"], s["mean_mhz"], label=label, color=color, linewidth=1.5)
        ax_top.fill_between(
            s["time_s"],
            s["min_mhz"],
            s["max_mhz"],
            color=color,
            alpha=0.15,
            linewidth=0,
            label=f"{label} min-max",
        )

    ax_top.set_ylabel("Fréquence moyenne (MHz)")
    ax_top.set_title("Comparaison des fréquences CPU (temps relatif)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    if len(series_list) >= 2:
        grid_t = build_common_grid(series_list[:2], step_s)
        if grid_t.size:
            a = series_list[0]
            b = series_list[1]
            a_interp = np.interp(grid_t, a["time_s"], a["mean_mhz"])
            b_interp = np.interp(grid_t, b["time_s"], b["mean_mhz"])
            diff = a_interp - b_interp
            ax_diff.plot(grid_t, diff, color="black", linewidth=1.2)
            ax_diff.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax_diff.set_ylabel("Différence A-B (MHz)")
            ax_diff.set_xlabel("Temps relatif (s)")
            ax_diff.grid(True, alpha=0.3)
        else:
            ax_diff.text(0.5, 0.5, "Plages temporelles non superposées", ha="center", va="center")
            ax_diff.axis("off")
    else:
        ax_diff.text(0.5, 0.5, "Ajoutez au moins deux fichiers pour la différence", ha="center", va="center")
        ax_diff.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure écrite -> {out_path}")


def main():
    args = parse_args()
    csv_files = [Path(p) for p in (args.csv if args.csv else DEFAULT_FILES)]

    if len(csv_files) < 2:
        raise SystemExit("Merci de fournir au moins deux fichiers CSV.")

    labels = args.labels if args.labels else [p.stem for p in csv_files]
    if len(labels) != len(csv_files):
        raise SystemExit("Le nombre de labels doit correspondre au nombre de CSV.")

    series_list = []
    for path in csv_files:
        if not path.exists():
            raise SystemExit(f"Fichier introuvable: {path}")
        series = load_series(path)
        series_list.append(series)
    print("--- Résumé ---")
    for label, s in zip(labels, series_list):
        info = s["summary"]
        print(
            f"{label}: {info['n_points']} points | portée {info['t_span_s']:.1f}s "
            f"| moyenne {info['mean_mhz']:.1f} MHz | p95 {info['p95_mhz']:.1f} MHz"
        )

    plot_comparison(series_list, labels, Path(args.out), args.grid_step)


if __name__ == "__main__":
    main()
