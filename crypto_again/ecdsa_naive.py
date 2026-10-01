#!/usr/bin/env python3
"""
Implémentation naïve vs protégée de la multiplication scalaire ECDSA (P-256).

But : rendre visible la fuite timing du double-and-add non constant-time,
et montrer qu'un double-and-add-always supprime cette fuite.

Sortie CSV : iter, hamming_weight_k, n_bits_k, duration_ns, start_ns, end_ns

Usage :
    python3 ecdsa_naive.py                     # 500 signatures × 2 modes
    python3 ecdsa_naive.py --iters 1000        # plus de données
    python3 ecdsa_naive.py --outdir /tmp/test  # répertoire personnalisé
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import secrets
import time
from pathlib import Path

# ── Paramètres de la courbe P-256 (NIST FIPS 186-4) ────────────────────────
P  = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
A  = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC
B  = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
N  = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
G  = (GX, GY)
INF = None   # Point à l'infini

# ── Arithmétique de corps fini ───────────────────────────────────────────────

def modinv(a: int, m: int) -> int:
    """Inverse modulaire (algorithme d'Euclide étendu)."""
    if a == 0:
        raise ZeroDivisionError
    lm, hm = 1, 0
    low, high = a % m, m
    while low > 1:
        r = high // low
        lm, hm = hm - lm * r, lm
        low, high = high - low * r, low
    return lm % m


def point_add(P1, P2):
    if P1 is INF:
        return P2
    if P2 is INF:
        return P1
    x1, y1 = P1
    x2, y2 = P2
    if x1 == x2:
        return INF if y1 != y2 else point_double(P1)
    lam = ((y2 - y1) * modinv(x2 - x1, P)) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def point_double(Pt):
    if Pt is INF:
        return INF
    x, y = Pt
    if y == 0:
        return INF
    lam = ((3 * x * x + A) * modinv(2 * y, P)) % P
    x3 = (lam * lam - 2 * x) % P
    y3 = (lam * (x - x3) - y) % P
    return (x3, y3)


# ── Multiplication scalaire NAÏVE : double-and-add ───────────────────────────
#
#   Pour chaque bit de k (MSB → LSB) :
#     toujours : R = 2R          (double)
#     si bit=1 : R = R + G       (add — seulement pour les 1 !)
#
#   ⚠ Le nombre d'additions dépend du poids de Hamming de k.
#   ⚠ Plus k a de bits à 1, plus la multiplication est LONGUE.
#   ⚠ → Fuite du secret par le timing.

def scalar_mult_naive(k: int) -> tuple:
    R = INF
    for i in range(k.bit_length() - 1, -1, -1):
        R = point_double(R)
        if (k >> i) & 1:          # ← branche data-dépendante (FUITE ICI)
            R = point_add(R, G)
    return R


# ── Multiplication scalaire PROTÉGÉE : double-and-add-always ─────────────────
#
#   Pour chaque bit de k (MSB → LSB) :
#     toujours : R  = 2R
#     toujours : T  = R + G      (calculé même si bit=0)
#     si bit=1 : R  = T          (on garde T ou on l'ignore)
#
#   Le nombre d'opérations est CONSTANT quel que soit k.
#   → Timing indépendant du secret.

def scalar_mult_protected(k: int) -> tuple:
    R = INF
    for i in range(k.bit_length() - 1, -1, -1):
        R = point_double(R)
        candidate = point_add(R, G)   # toujours calculé
        if (k >> i) & 1:
            R = candidate             # utilisé seulement si bit=1
    return R


# ── Signature ECDSA complète ─────────────────────────────────────────────────

def ecdsa_sign(msg: bytes, privkey: int, k: int, mode: str) -> tuple[int, int]:
    z = int.from_bytes(hashlib.sha256(msg).digest(), "big") % N
    mult = scalar_mult_naive if mode == "naive" else scalar_mult_protected
    R_pt = mult(k)
    if R_pt is INF:
        raise ValueError("k·G = point à l'infini")
    r = R_pt[0] % N
    if r == 0:
        raise ValueError("r == 0")
    s = (modinv(k, N) * (z + r * privkey)) % N
    if s == 0:
        raise ValueError("s == 0")
    return (r, s)


# ── Benchmark ────────────────────────────────────────────────────────────────

def hamming_weight(n: int) -> int:
    return bin(n).count("1")


def run_benchmark(n_iters: int, mode: str, outpath: Path, msg: bytes, privkey: int) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with outpath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "hamming_weight_k", "n_bits_k", "duration_ns", "start_ns", "end_ns"])

        for i in range(1, n_iters + 1):
            k = secrets.randbelow(N - 1) + 1

            start = time.perf_counter_ns()
            ecdsa_sign(msg, privkey, k, mode)
            end = time.perf_counter_ns()

            writer.writerow([i, hamming_weight(k), k.bit_length(), end - start, start, end])

            if i % 100 == 0:
                print(f"  [{mode}] {i}/{n_iters}", flush=True)

    print(f"[OK] {outpath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ECDSA naïf vs protégé — mesure de la fuite timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iters",  type=int, default=500,
                        help="Nombre de signatures par mode")
    parser.add_argument("--outdir", default="results/ecdsa_naive",
                        help="Répertoire de sortie")
    parser.add_argument("--msg",    default="msg.bin",
                        help="Fichier message à signer")
    args = parser.parse_args()

    msg_path = Path(args.msg)
    msg = msg_path.read_bytes() if msg_path.exists() else b"LRE side-channel benchmark"

    privkey = secrets.randbelow(N - 1) + 1
    outdir  = Path(args.outdir)

    print(f"[*] Clé privée : {privkey.bit_length()} bits")
    print(f"[*] {args.iters} signatures × 2 modes → {outdir}")
    print()

    print("[*] Mode NAÏF (double-and-add, fuite timing attendue)...")
    run_benchmark(args.iters, "naive",     outdir / "naive_timings.csv",     msg, privkey)

    print()
    print("[*] Mode PROTÉGÉ (double-and-add-always, timing constant attendu)...")
    run_benchmark(args.iters, "protected", outdir / "protected_timings.csv", msg, privkey)

    print()
    print("[OK] Données prêtes. Lance l'analyse :")
    print(f"     python3 analyze_ecdsa_timing.py --indir {outdir}")


if __name__ == "__main__":
    main()
