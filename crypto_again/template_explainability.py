#!/usr/bin/env python3
"""
Template Attack — Explicabilité (Phase D) — sans poids de Hamming

Entraîne un Random Forest sur les données de template_collect.py et explique :
  - Quelles features distinguent les 256 valeurs d'octet (SHAP + importance RF)
  - Comment chaque feature influence les prédictions (SHAP dependence)
  - Pourquoi une prédiction individuelle est juste/fausse (LIME)
  - Impact de n'utiliser QUE duration_us (seule feature observable à l'attaque)

Features utilisées — SANS poids de Hamming :
  duration_us      ← seule observable en attaque réelle  ★
  n_bits_k         ← longueur de k en bits (profiling)
  duration_per_bit ← durée normalisée par bit (profiling)

Sorties :
  01_feature_importance.png  importance globale (Gini RF + SHAP)
  02_shap_beeswarm.png       distribution des valeurs SHAP
  03_shap_dependence.png     effet de duration_us sur les prédictions
  04_lime_sample.png         LIME : 1 bonne + 1 mauvaise prédiction
  05_accuracy_comparison.png accuracy réaliste vs features complètes
  explainability_report.txt  rapport complet

Usage :
    python3 template_explainability.py
    python3 template_explainability.py --indir results/templates
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[!] 'shap' non installé — pip install shap")

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("[!] 'lime' non installé — pip install lime")

plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 10})

# Seule feature observable à l'attaque (Device B)
FEATURES_ATTACK = ["duration_us"]

# Features disponibles en profilage — SANS aucune donnée de poids de Hamming
FEATURES_FULL = [
    "duration_us",        # ← observable à l'attaque  ★
    "n_bits_k",           # longueur de k en bits
    "duration_per_bit",   # durée / n_bits_k
]

FEATURE_LABELS = {
    "duration_us":      "Durée totale (µs)  ★ observable en attaque",
    "n_bits_k":         "Longueur k en bits (profiling)",
    "duration_per_bit": "Durée / bit (profiling)",
}


# ── Chargement ────────────────────────────────────────────────────────────────

def load(indir: Path) -> pd.DataFrame:
    csvs = sorted(indir.glob("templates_byte*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise SystemExit(f"Aucun fichier templates_byte*.csv dans {indir}\n"
                         "Lance d'abord : python3 template_collect.py")
    path = csvs[-1]
    print(f"[*] Chargement : {path.name}  ({path.stat().st_size // 1024} Ko)")
    df = pd.read_csv(path)
    df["duration_us"]    = df["duration_ns"] / 1_000.0
    df["duration_per_bit"] = df["duration_us"] / df["n_bits_k"].replace(0, np.nan)
    df.dropna(inplace=True)
    return df


# ── Entraînement ──────────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series, n_estimators: int = 150) -> tuple:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)
    return model, scores


# ── Figure 1 : importance Gini + SHAP globale ────────────────────────────────

def plot_feature_importance(model, X: pd.DataFrame, outdir: Path):
    feat_names = list(X.columns)
    gini_imp   = model.feature_importances_

    shap_values = None
    shap_imp    = None
    if HAS_SHAP:
        print("  [*] Calcul SHAP (peut prendre quelques secondes)...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_imp    = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    n_feat = len(feat_names)
    labels = [FEATURE_LABELS.get(f, f) for f in feat_names]

    ncols = 2 if HAS_SHAP else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14 if HAS_SHAP else 8, 5))
    if ncols == 1:
        axes = [axes]

    idx = np.argsort(gini_imp)
    colors = ["#e15759" if feat_names[i] == "duration_us" else "#4e79a7" for i in idx]
    axes[0].barh(range(n_feat), gini_imp[idx], color=colors)
    axes[0].set_yticks(range(n_feat))
    axes[0].set_yticklabels([labels[i] for i in idx], fontsize=9)
    axes[0].set_xlabel("Importance Gini (Random Forest)")
    axes[0].set_title("Importance features — Gini\n(rouge = observable à l'attaque)")
    axes[0].grid(True, axis="x", alpha=0.3)

    if HAS_SHAP and shap_imp is not None:
        idx2   = np.argsort(shap_imp)
        colors2 = ["#e15759" if feat_names[i] == "duration_us" else "#4e79a7" for i in idx2]
        axes[1].barh(range(n_feat), shap_imp[idx2], color=colors2)
        axes[1].set_yticks(range(n_feat))
        axes[1].set_yticklabels([labels[i] for i in idx2], fontsize=9)
        axes[1].set_xlabel("Importance SHAP moyenne |φ|")
        axes[1].set_title("Importance features — SHAP\n(rouge = observable à l'attaque)")
        axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Quelles features distinguent les 256 valeurs d'octet ?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "01_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_feature_importance.png")
    return shap_values


# ── Figure 2 : SHAP beeswarm ──────────────────────────────────────────────────

def plot_shap_beeswarm(shap_values, X: pd.DataFrame, outdir: Path) -> None:
    if not HAS_SHAP or shap_values is None:
        return
    mean_shap = np.mean([sv for sv in shap_values], axis=0)
    shap.summary_plot(mean_shap, X,
                      feature_names=[FEATURE_LABELS.get(c, c) for c in X.columns],
                      show=False)
    plt.title("SHAP Beeswarm — Impact sur la prédiction de byte_value\n"
              "Chaque point = une trace  |  couleur = valeur de la feature")
    plt.tight_layout()
    plt.savefig(outdir / "02_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 02_shap_beeswarm.png")


# ── Figure 3 : SHAP dependence pour duration_us ──────────────────────────────

def plot_shap_dependence(shap_values, X: pd.DataFrame, y: pd.Series, outdir: Path) -> None:
    if not HAS_SHAP or shap_values is None:
        return
    feat_names = list(X.columns)
    if "duration_us" not in feat_names:
        return
    dur_idx  = feat_names.index("duration_us")
    shap_dur = np.mean([sv[:, dur_idx] for sv in shap_values], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(X["duration_us"], shap_dur,
                    c=y.values, cmap="plasma", alpha=0.25, s=6)
    plt.colorbar(sc, ax=ax, label="byte_value (0→255)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Durée totale (µs)  ← seule feature observable à l'attaque")
    ax.set_ylabel("Valeur SHAP de duration_us")
    ax.set_title("Dependence Plot — duration_us vs byte_value\n"
                 "Dispersion verticale = bruit masquant la fuite")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "03_shap_dependence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_shap_dependence.png")


# ── Figure 4 : LIME — 1 bonne + 1 mauvaise prédiction ───────────────────────

def plot_lime_sample(model, X: pd.DataFrame, y: pd.Series, outdir: Path) -> None:
    if not HAS_LIME:
        return

    feature_names = [FEATURE_LABELS.get(c, c) for c in X.columns]
    class_names   = [str(c) for c in sorted(y.unique())]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42,
    )

    y_pred = model.predict(X)
    correct_idx   = np.where(y_pred == y.values)[0]
    incorrect_idx = np.where(y_pred != y.values)[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, idx_pool, title_suffix in [
        (axes[0], correct_idx,   "Prédiction CORRECTE"),
        (axes[1], incorrect_idx, "Prédiction INCORRECTE"),
    ]:
        if len(idx_pool) == 0:
            ax.text(0.5, 0.5, "Pas d'exemple", ha="center", va="center")
            ax.set_title(title_suffix)
            continue

        sample_idx = idx_pool[len(idx_pool) // 2]
        true_val   = int(y.values[sample_idx])
        pred_val   = int(y_pred[sample_idx])

        exp = explainer.explain_instance(
            X.values[sample_idx],
            model.predict_proba,
            num_features=len(X.columns),
            labels=[true_val],
        )
        feat_exp = exp.as_list(label=true_val)
        feats  = [f[0] for f in feat_exp]
        vals   = [f[1] for f in feat_exp]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]

        ax.barh(range(len(feats)), vals, color=colors)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution LIME")
        ax.set_title(f"LIME — {title_suffix}\n"
                     f"Vrai byte=0x{true_val:02X}  Prédit=0x{pred_val:02X}")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Explication LIME : pourquoi le classifieur prédit cette valeur d'octet ?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "04_lime_sample.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 04_lime_sample.png")


# ── Figure 5 : accuracy réaliste vs complète (top-1 et top-5) ────────────────

def plot_accuracy_comparison(df: pd.DataFrame, n_estimators: int, outdir: Path) -> dict:
    y = df["byte_value"]
    results = {}

    scenarios = [
        ("Attaque\n[duration_us seul]",        FEATURES_ATTACK, "#e15759"),
        ("Profilage\n[toutes features sans HW]", FEATURES_FULL,   "#4e79a7"),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # top-1 accuracy (cross-val)
    for label, feats, color in scenarios:
        X = df[feats]
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results[label] = {"top1": scores, "color": color}
        print(f"  {label.replace(chr(10),' '):<40} top-1 = {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    # top-5 accuracy (entraînement complet pour predict_proba)
    for label, feats, color in scenarios:
        X = df[feats]
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X, y)
        proba   = model.predict_proba(X)
        top5_acc = top_k_accuracy_score(y, proba, k=5, labels=model.classes_)
        results[label]["top5"] = top5_acc
        print(f"  {label.replace(chr(10),' '):<40} top-5 = {top5_acc*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, k in [(axes[0], "top1", 1), (axes[1], "top5", 5)]:
        labels = list(results.keys())
        colors = [results[l]["color"] for l in labels]

        if metric == "top1":
            means = [results[l]["top1"].mean() for l in labels]
            stds  = [results[l]["top1"].std()  for l in labels]
            ax.bar(labels, means, yerr=stds, color=colors, width=0.5, capsize=8, alpha=0.85)
            for i, (m, label) in enumerate(zip(means, labels)):
                ax.text(i, m + max(stds) + 0.001, f"{m*100:.2f}%",
                        ha="center", fontsize=11, fontweight="bold")
            baseline = 100 / 256
            ax.axhline(baseline / 100, color="gray", linestyle="--",
                       label=f"Hasard (1/256 = {baseline:.2f}%)")
        else:
            vals = [results[l]["top5"] for l in labels]
            ax.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
            for i, (v, label) in enumerate(zip(vals, labels)):
                ax.text(i, v + 0.005, f"{v*100:.2f}%",
                        ha="center", fontsize=11, fontweight="bold")
            baseline = 500 / 256
            ax.axhline(min(1.0, baseline / 100), color="gray", linestyle="--",
                       label=f"Hasard (5/256 = {min(100, baseline):.2f}%)")

        ax.set_ylabel(f"Accuracy top-{k}")
        ax.set_title(f"Précision top-{k} — 256 classes directes\n(sans poids de Hamming)")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Features réalistes (attaque) vs features complètes (profiling) — sans HW",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "05_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 05_accuracy_comparison.png")
    return results


# ── Rapport texte ─────────────────────────────────────────────────────────────

def write_report(model, X_full: pd.DataFrame, y: pd.Series,
                 acc_results: dict, outdir: Path) -> None:
    feat_names = list(X_full.columns)
    gini_imp   = model.feature_importances_
    ranked     = sorted(zip(feat_names, gini_imp), key=lambda x: -x[1])

    lines = [
        "=" * 65,
        "RAPPORT — Explicabilité Template Attack ECDSA (sans HW)",
        "=" * 65,
        f"\n  Features utilisées  : {', '.join(FEATURES_FULL)}",
        f"  Target              : byte_value (256 classes)",
        f"  Baseline hasard     : {100/256:.2f} %",
        "",
        "─" * 65,
        "IMPORTANCE DES FEATURES (Gini)",
        "─" * 65,
    ]
    for rank, (feat, imp) in enumerate(ranked, 1):
        tag = "  ★ OBSERVABLE" if feat == "duration_us" else "  (profiling)"
        lines.append(f"  #{rank}  {FEATURE_LABELS.get(feat, feat):<40} {imp:.4f}{tag}")

    lines += [
        "",
        "─" * 65,
        "ACCURACY PAR SCÉNARIO",
        "─" * 65,
        f"  Hasard top-1 : {100/256:.2f} %",
        f"  Hasard top-5 : {min(100, 500/256):.2f} %",
    ]
    for scenario, data in acc_results.items():
        label = scenario.replace("\n", " ")
        top1  = data["top1"].mean() * 100
        top5  = data.get("top5", 0) * 100
        lines.append(f"  {label:<42} top-1={top1:.2f}%  top-5={top5:.2f}%")

    lines += [
        "",
        "─" * 65,
        "INTERPRÉTATION",
        "─" * 65,
    ]

    acc_attack = list(acc_results.values())[0]["top1"].mean()
    if acc_attack < 0.05:
        lines += [
            f"  ⚠ Avec duration_us seul : {acc_attack*100:.2f} % ≈ hasard ({100/256:.2f} %)",
            "    → Le timing scalaire seul est insuffisant pour les 256 classes directes.",
            "    → Confirmation : il faut des features plus riches pour l'attaque par template.",
            "    → Pistes : séries temporelles de fréquence CPU, spectrogramme audio, MFCC.",
        ]
    else:
        lines += [
            f"  ✓ Avec duration_us seul : {acc_attack*100:.2f} % > hasard ({100/256:.2f} %)",
            "    → La fuite timing est détectable même sans le poids de Hamming.",
        ]

    lines += [
        "",
        "─" * 65,
        "CONCLUSION",
        "─" * 65,
        "  La classification directe sur 256 classes (sans HW) est plus ambitieuse",
        "  et plus réaliste : en vraie attaque, l'attaquant ne connaît pas HW(k).",
        "  La faible accuracy confirme le besoin de traces plus riches que le timing seul.",
        "",
        "  Pour améliorer :",
        "    1. Capturer la fréquence CPU en continu pendant la signature (time series)",
        "    2. Enregistrer le son et extraire MFCC / FFT",
        "    3. Augmenter le nombre de traces par classe (1000 au lieu de 200)",
        "    4. Combiner plusieurs signatures pour une même clé (attaque multi-trace)",
        "",
    ]

    report_path = outdir / "explainability_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {report_path}")
    print()
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicabilité template attack ECDSA — 256 classes, sans poids de Hamming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--indir",        default="results/templates")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--sample",       type=int, default=5000,
                        help="Sous-échantillon pour SHAP/LIME (0 = tout)")
    args = parser.parse_args()

    indir  = Path(args.indir)
    outdir = indir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(indir)
    y  = df["byte_value"]

    print(f"[*] {len(df):,} traces  |  {y.nunique()} classes byte  |  "
          f"features : {FEATURES_FULL}")
    print()

    X_full = df[FEATURES_FULL]

    print("[*] Entraînement Random Forest (features complètes sans HW)...")
    model_full, scores_full = train(X_full, y, args.n_estimators)
    print(f"    accuracy CV : {scores_full.mean()*100:.2f}% ± {scores_full.std()*100:.2f}%")
    print()

    # Sous-échantillon pour SHAP/LIME
    if args.sample > 0 and len(X_full) > args.sample:
        idx    = np.random.RandomState(42).choice(len(X_full), args.sample, replace=False)
        X_shap = X_full.iloc[idx].reset_index(drop=True)
        y_shap = y.iloc[idx].reset_index(drop=True)
    else:
        X_shap, y_shap = X_full, y

    print("[*] Figure 1 : importance features...")
    shap_values = plot_feature_importance(model_full, X_shap, outdir)

    print("[*] Figure 2 : SHAP beeswarm...")
    plot_shap_beeswarm(shap_values, X_shap, outdir)

    print("[*] Figure 3 : SHAP dependence (duration_us)...")
    plot_shap_dependence(shap_values, X_shap, y_shap, outdir)

    print("[*] Figure 4 : LIME (exemples individuels)...")
    plot_lime_sample(model_full, X_shap, y_shap, outdir)

    print("[*] Figure 5 : comparaison accuracy...")
    acc_results = plot_accuracy_comparison(df, args.n_estimators, outdir)

    write_report(model_full, X_full, y, acc_results, outdir)


if __name__ == "__main__":
    main()
