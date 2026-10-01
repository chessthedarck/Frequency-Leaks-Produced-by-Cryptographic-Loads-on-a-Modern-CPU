#!/usr/bin/env python3
"""
Template Attack — Analyse des Templates (Phase B) — sans poids de Hamming

Charge le CSV de template_collect.py et produit :
  01_mean_by_byte.png       timing moyen pour chaque valeur d'octet (0x00→0xFF)
  02_sample_templates.png   templates gaussiens pour 10 valeurs d'octets échantillonnées
  03_sample_boxplot.png     boxplot pour les mêmes 10 valeurs
  04_fisher.png             critère de Fisher pour les 256 classes
  05_topk_accuracy.png      précision top-K du classifieur centroïde
  template_report.txt       rapport complet

Usage :
    python3 template_analyze.py
    python3 template_analyze.py --indir results/templates
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 10})

N_SAMPLE = 10   # nombre de valeurs d'octets à afficher dans les figures détaillées
SAMPLE_VALS = list(range(0, 256, 256 // N_SAMPLE))[:N_SAMPLE]
SAMPLE_COLORS = plt.cm.tab10(np.linspace(0, 1, N_SAMPLE))


# ── Chargement ────────────────────────────────────────────────────────────────

def load(indir: Path) -> pd.DataFrame:
    csvs = sorted(indir.glob("templates_byte*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise SystemExit(f"Aucun fichier templates_byte*.csv dans {indir}\n"
                         "Lance d'abord : python3 template_collect.py")
    path = csvs[-1]
    print(f"[*] Chargement : {path.name}  ({path.stat().st_size // 1024} Ko)")
    df = pd.read_csv(path)
    df["duration_us"] = df["duration_ns"] / 1_000.0
    return df


# ── Statistiques ──────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("byte_value")["duration_us"].agg(["mean", "std", "count"])
    g.columns = ["mean_us", "std_us", "n"]
    return g.reset_index()


def fisher_criterion(df: pd.DataFrame) -> float:
    groups = [grp["duration_us"].values for _, grp in df.groupby("byte_value")]
    grand_mean = df["duration_us"].mean()
    n_total = len(df)
    k = len(groups)
    between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups) / (k - 1)
    within  = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_total - k)
    return between / within if within > 0 else 0.0


def topk_accuracy(stats: pd.DataFrame, df: pd.DataFrame, ks: list[int]) -> dict[int, float]:
    """Précision top-K : est-ce que la vraie classe est dans les K centroïdes les plus proches ?"""
    centroids = stats.set_index("byte_value")["mean_us"]
    durations = df["duration_us"].values
    labels    = df["byte_value"].values
    cent_vals = centroids.values
    cent_idx  = centroids.index.values

    results = {}
    for k in ks:
        correct = 0
        for dur, true_lbl in zip(durations, labels):
            dists = np.abs(cent_vals - dur)
            top_k = cent_idx[np.argpartition(dists, min(k, len(dists)-1))[:k]]
            if true_lbl in top_k:
                correct += 1
        results[k] = correct / len(labels)
    return results


# ── Figure 1 : timing moyen par valeur d'octet ───────────────────────────────

def plot_mean_by_byte(stats: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(stats["byte_value"], stats["mean_us"],
               s=14, color="#4e79a7", alpha=0.7, zorder=3)
    ax.fill_between(stats["byte_value"],
                    stats["mean_us"] - stats["std_us"],
                    stats["mean_us"] + stats["std_us"],
                    alpha=0.15, color="#4e79a7", label="±1σ")

    ax.set_xlabel("Valeur de l'octet cible (0x00 → 0xFF)")
    ax.set_ylabel("Durée moyenne (µs)")
    ax.set_title("Timing moyen par valeur d'octet — 256 classes directes\n"
                 "Sans regroupement par poids de Hamming")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "01_mean_by_byte.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_mean_by_byte.png")


# ── Figure 2 : templates gaussiens pour N valeurs échantillonnées ─────────────

def plot_sample_templates(stats: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    x_min = stats["mean_us"].min() - 3 * stats["std_us"].max()
    x_max = stats["mean_us"].max() + 3 * stats["std_us"].max()
    x = np.linspace(x_min, x_max, 2000)

    for bv, color in zip(SAMPLE_VALS, SAMPLE_COLORS):
        row = stats[stats["byte_value"] == bv]
        if row.empty:
            continue
        mu, sigma = float(row["mean_us"]), float(row["std_us"])
        pdf = stats_lib.norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, color=color, linewidth=1.8, label=f"0x{bv:02X}")
        ax.fill_between(x, pdf, alpha=0.08, color=color)

    ax.set_xlabel("Durée (µs)")
    ax.set_ylabel("Densité de probabilité")
    ax.set_title(f"Templates gaussiens N(µ, σ²) — {N_SAMPLE} valeurs d'octets\n"
                 "Chevauchement = difficulté à distinguer les 256 classes par timing seul")
    ax.legend(title="byte_value", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "02_sample_templates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_sample_templates.png")


# ── Figure 3 : boxplot pour N valeurs échantillonnées ────────────────────────

def plot_sample_boxplot(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    data   = [df[df["byte_value"] == bv]["duration_us"].values for bv in SAMPLE_VALS]
    labels = [f"0x{bv:02X}" for bv in SAMPLE_VALS]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], SAMPLE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Valeur de l'octet cible")
    ax.set_ylabel("Durée (µs)")
    ax.set_title(f"Distribution des durées — {N_SAMPLE} valeurs d'octets\n"
                 "Chevauchement des boîtes = faible séparabilité par timing seul")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "03_sample_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_sample_boxplot.png")


# ── Figure 4 : critère de Fisher ─────────────────────────────────────────────

def plot_fisher(df: pd.DataFrame, outdir: Path) -> float:
    F = fisher_criterion(df)
    baseline = 1 / df["byte_value"].nunique()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["256 classes\n(byte_value direct)"], [F], color="#4e79a7", width=0.4)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="F = 1 (seuil séparabilité)")
    ax.text(0, F + 0.002 * max(F, 1), f"F = {F:.5f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.set_ylabel("Critère de Fisher F")
    ax.set_title("Séparabilité des 256 classes (byte_value)\n"
                 "F >> 1 = classes bien séparées  |  F << 1 = classes confondues")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "04_fisher.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 04_fisher.png")
    return F


# ── Figure 5 : top-K accuracy ─────────────────────────────────────────────────

def plot_topk(stats: pd.DataFrame, df: pd.DataFrame, outdir: Path) -> dict:
    ks = [1, 3, 5, 10, 20, 50]
    print("[*] Calcul top-K accuracy (peut prendre quelques secondes)...")

    # Sous-échantillon pour la vitesse
    sample = df.sample(min(2000, len(df)), random_state=42)
    topk = topk_accuracy(stats, sample, ks)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, [topk[k] * 100 for k in ks], "o-", color="#4e79a7",
            linewidth=2, markersize=8)
    ax.axhline(100 / 256, color="gray", linestyle="--", linewidth=1,
               label=f"Hasard pur (1/256 = {100/256:.2f} %)")

    for k in ks:
        ax.annotate(f"{topk[k]*100:.1f}%",
                    (k, topk[k] * 100), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("K (nombre de candidats considérés)")
    ax.set_ylabel("Précision top-K (%)")
    ax.set_title("Précision top-K — classifieur centroïde sur 256 classes\n"
                 "Top-1 = 1 seul candidat prédit  |  Top-K = la vraie classe est dans les K meilleurs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(topk.values()) * 120)

    fig.tight_layout()
    fig.savefig(outdir / "05_topk_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 05_topk_accuracy.png")
    return topk


# ── Rapport texte ─────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, stats: pd.DataFrame, F: float, topk: dict, outdir: Path) -> None:
    lines = [
        "=" * 65,
        "RAPPORT — Analyse des Templates ECDSA (256 classes, sans HW)",
        "=" * 65,
        f"\n  Traces totales      : {len(df):,}",
        f"  Classes (byte)      : {df['byte_value'].nunique()}",
        f"  Traces/classe (moy) : {int(stats['n'].mean())}",
        f"  Durée min moyenne   : {stats['mean_us'].min():.1f} µs  (byte={stats.loc[stats['mean_us'].idxmin(), 'byte_value']:#04x})",
        f"  Durée max moyenne   : {stats['mean_us'].max():.1f} µs  (byte={stats.loc[stats['mean_us'].idxmax(), 'byte_value']:#04x})",
        f"  Amplitude totale    : {stats['mean_us'].max() - stats['mean_us'].min():.1f} µs",
        f"  σ moyen intra-classe: {stats['std_us'].mean():.1f} µs",
        "",
        "─" * 65,
        "SÉPARABILITÉ",
        "─" * 65,
        f"  Critère de Fisher F : {F:.6f}",
        f"  Baseline aléatoire  : {100/256:.2f} %",
        "",
        "  Précision top-K (classifieur centroïde) :",
    ]
    for k, acc in topk.items():
        lines.append(f"    top-{k:<3} : {acc*100:5.1f} %")

    lines += [
        "",
        "─" * 65,
        "INTERPRÉTATION",
        "─" * 65,
    ]

    if F > 1.0:
        lines.append("  ✓ F > 1 : les 256 classes sont séparables en moyenne.")
    else:
        lines.append("  ⚠ F < 1 : les 256 classes se chevauchent trop par timing seul.")
        lines.append("    → La variance intra-classe (bruit) > variance inter-classes (signal).")

    top1 = topk.get(1, 0)
    top10 = topk.get(10, 0)
    if top1 < 0.05:
        lines += [
            f"\n  ⚠ Top-1 = {top1*100:.1f} % ≈ hasard ({100/256:.2f} %) :",
            "    → Le timing seul ne suffit PAS pour identifier la valeur d'octet directement.",
            "    → Nécessité de features plus riches : CPU freq dans le temps, spectrogramme, audio.",
        ]
    if top10 > 0.15:
        lines += [
            f"\n  ~ Top-10 = {top10*100:.1f} % > hasard :",
            "    → Le timing permet de réduire l'espace de recherche même sans HW.",
            "    → En combinant plusieurs signatures, la précision peut s'améliorer.",
        ]

    lines += [
        "",
        "─" * 65,
        "PROCHAINE ÉTAPE",
        "─" * 65,
        "  → python3 template_explainability.py --indir results/templates",
        "  → Collecter des features spectrales (CPU freq, audio) pour améliorer la séparabilité",
        "",
    ]

    report_path = outdir / "template_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {report_path}")
    print()
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

import scipy.stats as stats_lib

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse des templates ECDSA — 256 classes directes, sans poids de Hamming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--indir", default="results/templates")
    args = parser.parse_args()

    indir  = Path(args.indir)
    outdir = indir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    df    = load(indir)
    stats = compute_stats(df)

    print(f"[*] {len(df):,} traces  |  {df['byte_value'].nunique()} classes byte")
    print()

    plot_mean_by_byte(stats, outdir)
    plot_sample_templates(stats, outdir)
    plot_sample_boxplot(df, outdir)
    F    = plot_fisher(df, outdir)
    topk = plot_topk(stats, df, outdir)
    write_report(df, stats, F, topk, outdir)


if __name__ == "__main__":
    main()
