#!/usr/bin/env python3
"""
Analyse de la fuite timing ECDSA : naïf vs protégé.

Lit les CSV produits par ecdsa_naive.py et produit :
  - 01_timing_vs_hamming.png   corrélation durée / poids de Hamming de k
  - 02_distributions.png       boxplot et histogramme comparatifs
  - 03_correlation_summary.png résumé statistique (r de Pearson, p-value)
  - correlation_report.txt     rapport texte

Usage :
    python3 analyze_ecdsa_timing.py
    python3 analyze_ecdsa_timing.py --indir results/ecdsa_naive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"figure.figsize": (12, 6)})


# ── Chargement ───────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["duration_us"] = df["duration_ns"] / 1000.0
    return df


# ── Figure 1 : timing vs poids de Hamming ────────────────────────────────────

def plot_timing_vs_hamming(naive: pd.DataFrame, protected: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, df, label, color in [
        (axes[0], naive,     "Naïf (double-and-add)",          "#e15759"),
        (axes[1], protected, "Protégé (double-and-add-always)", "#4e79a7"),
    ]:
        # Moyenne par poids de Hamming
        grouped = df.groupby("hamming_weight_k")["duration_us"].agg(["mean", "std"]).reset_index()

        ax.scatter(df["hamming_weight_k"], df["duration_us"],
                   alpha=0.15, s=6, color=color, label="mesures")
        ax.errorbar(grouped["hamming_weight_k"], grouped["mean"],
                    yerr=grouped["std"], fmt="o-", color="black",
                    linewidth=1.8, markersize=5, label="moyenne ± std")

        r, p = stats.pearsonr(df["hamming_weight_k"], df["duration_us"])
        ax.set_title(f"{label}\nr Pearson = {r:.4f}  (p = {p:.2e})", fontsize=11)
        ax.set_xlabel("Poids de Hamming de k  (nombre de bits à 1)")
        ax.set_ylabel("Durée (µs)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Corrélation timing / poids de Hamming du nonce k", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "01_timing_vs_hamming.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_timing_vs_hamming.png")


# ── Figure 2 : distributions ─────────────────────────────────────────────────

def plot_distributions(naive: pd.DataFrame, protected: pd.DataFrame, outdir: Path) -> None:
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    ax_box.boxplot(
        [naive["duration_us"], protected["duration_us"]],
        labels=["Naïf", "Protégé"],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#e15759", alpha=0.6),
    )
    ax_box.set_ylabel("Durée (µs)")
    ax_box.set_title("Distribution des durées de signature")
    ax_box.grid(True, axis="y", alpha=0.3)

    # Histogramme superposé
    bins = np.linspace(
        min(naive["duration_us"].min(), protected["duration_us"].min()),
        max(naive["duration_us"].max(), protected["duration_us"].max()),
        60,
    )
    ax_hist.hist(naive["duration_us"],     bins=bins, alpha=0.6, color="#e15759", label="Naïf",    density=True)
    ax_hist.hist(protected["duration_us"], bins=bins, alpha=0.6, color="#4e79a7", label="Protégé", density=True)
    ax_hist.set_xlabel("Durée (µs)")
    ax_hist.set_ylabel("Densité")
    ax_hist.set_title("Histogramme des durées")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "02_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_distributions.png")


# ── Figure 3 : résumé corrélations ───────────────────────────────────────────

def plot_correlation_summary(naive: pd.DataFrame, protected: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    results = []
    for df, label in [(naive, "Naïf"), (protected, "Protégé")]:
        r, p = stats.pearsonr(df["hamming_weight_k"], df["duration_us"])
        results.append((label, r, p))

    labels_plot = [r[0] for r in results]
    r_vals      = [r[1] for r in results]
    colors      = ["#e15759" if abs(r) > 0.1 else "#4e79a7" for r in r_vals]

    bars = ax.bar(labels_plot, r_vals, color=colors, width=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline( 0.1, color="gray", linewidth=0.8, linestyle="--", label="seuil |r| = 0.10")
    ax.axhline(-0.1, color="gray", linewidth=0.8, linestyle="--")

    for bar, (_, r, p) in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * np.sign(bar.get_height() + 1e-9),
                f"r={r:.4f}\np={p:.2e}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Corrélation de Pearson (r)")
    ax.set_title("Fuite timing : corrélation durée / poids de Hamming(k)\n"
                 "r ≠ 0 (naïf) → fuite visible   |   r ≈ 0 (protégé) → timing constant")
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "03_correlation_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_correlation_summary.png")


# ── Rapport texte ─────────────────────────────────────────────────────────────

def write_report(naive: pd.DataFrame, protected: pd.DataFrame, outdir: Path) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("RAPPORT — Fuite timing ECDSA naïf vs protégé")
    lines.append("=" * 60)

    for df, label in [(naive, "NAÏF"), (protected, "PROTÉGÉ")]:
        r, p = stats.pearsonr(df["hamming_weight_k"], df["duration_us"])
        stat, pval = stats.mannwhitneyu(
            df.loc[df["hamming_weight_k"] < df["hamming_weight_k"].median(), "duration_us"],
            df.loc[df["hamming_weight_k"] >= df["hamming_weight_k"].median(), "duration_us"],
            alternative="two-sided",
        )
        lines.append(f"\n[ Mode {label} ]")
        lines.append(f"  N signatures       : {len(df)}")
        lines.append(f"  Durée moyenne      : {df['duration_us'].mean():.1f} µs")
        lines.append(f"  Durée std          : {df['duration_us'].std():.1f} µs")
        lines.append(f"  Durée min / max    : {df['duration_us'].min():.0f} / {df['duration_us'].max():.0f} µs")
        lines.append(f"  Pearson r          : {r:.6f}")
        lines.append(f"  p-value Pearson    : {p:.2e}  {'** FUITE **' if p < 0.05 else '(non significatif)'}")
        lines.append(f"  Mann-Whitney U     : {stat:.0f}  (p={pval:.2e})  {'** FUITE **' if pval < 0.05 else '(non significatif)'}")

    lines.append("\n" + "=" * 60)
    lines.append("INTERPRÉTATION")
    lines.append("=" * 60)

    r_naive, p_naive = stats.pearsonr(naive["hamming_weight_k"], naive["duration_us"])
    r_prot,  p_prot  = stats.pearsonr(protected["hamming_weight_k"], protected["duration_us"])

    if p_naive < 0.05 and abs(r_naive) > 0.1:
        lines.append("Naïf    : corrélation SIGNIFICATIVE → fuite du poids de Hamming(k).")
        lines.append("          Un attaquant observant le timing peut estimer combien de bits")
        lines.append("          à 1 se trouvent dans le nonce k, facilitant une lattice attack.")
    else:
        lines.append("Naïf    : corrélation non significative sur ces données.")

    if p_prot >= 0.05 or abs(r_prot) <= 0.1:
        lines.append("Protégé : corrélation NON significative → timing constant, pas de fuite.")
    else:
        lines.append("Protégé : corrélation résiduelle détectée — vérifier l'implémentation.")

    report_path = outdir / "correlation_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {report_path}")
    print()
    print("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse de la fuite timing ECDSA (naïf vs protégé)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--indir", default="results/ecdsa_naive",
                        help="Répertoire contenant naive_timings.csv et protected_timings.csv")
    args = parser.parse_args()

    indir = Path(args.indir)
    naive_path     = indir / "naive_timings.csv"
    protected_path = indir / "protected_timings.csv"

    if not naive_path.exists() or not protected_path.exists():
        raise SystemExit(
            f"Fichiers manquants dans {indir}.\n"
            "Lance d'abord : python3 ecdsa_naive.py"
        )

    naive     = load(naive_path)
    protected = load(protected_path)

    outdir = indir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_timing_vs_hamming(naive, protected, outdir)
    plot_distributions(naive, protected, outdir)
    plot_correlation_summary(naive, protected, outdir)
    write_report(naive, protected, outdir)


if __name__ == "__main__":
    main()
