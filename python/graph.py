import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.figsize': (12, 6)})

# --- Chemins par défaut---
RAW_PATH  = Path("/home/kamilahezzat/Documents/S7/LRE/gros_pics/Raw data.csv")
FFT_PATH  = Path("/home/kamilahezzat/Documents/S7/LRE/gros_pics/FFT Spectrum.csv")
PEAK_PATH = Path("/home/kamilahezzat/Documents/S7/LRE/gros_pics/Peak History.csv")

# --- Dossier output ---
OUT_DIR = Path("/home/kamilahezzat/Documents/S7/LRE/output")
OUT_DIR.mkdir(exist_ok=True)

def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, encoding="latin-1")

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def estimate_fs(t):
    if len(t) < 2:
        return np.nan, np.nan
    dt = np.median(np.diff(t))
    Fs = 1.0/dt if dt > 0 else np.nan
    return Fs, dt

def save_fig(name: str):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[OK] figure -> {path}")

# --- Charger RAW si t/x pas déjà définis ---
if "t" not in globals() or "x" not in globals():
    raw_df = read_csv_safe(RAW_PATH)
    t_col, x_col = list(raw_df.columns)[:2]
    raw_df = coerce_numeric(raw_df, [t_col, x_col]).dropna(subset=[t_col, x_col])
    t = raw_df[t_col].to_numpy()
    x = raw_df[x_col].to_numpy()
else:
    # construire un df pour export propre
    t_col, x_col = "Time (s)", "Recording (a.u.)"
    raw_df = pd.DataFrame({t_col: t, x_col: x})

# --- Charger FFT/Peaks si pas déjà là ---
if "fft_df" not in globals():
    fft_df = read_csv_safe(FFT_PATH)
if "peaks_df" not in globals():
    peaks_df = read_csv_safe(PEAK_PATH)

# Forcer les colonnes numériques (FFT et Peaks)
f_col, a_col = list(fft_df.columns)[:2]
fft_df = coerce_numeric(fft_df, [f_col, a_col]).dropna(subset=[f_col, a_col])
f = fft_df[f_col].to_numpy()
A = fft_df[a_col].to_numpy()

tp_col, fp_col = list(peaks_df.columns)[:2]
peaks_df = coerce_numeric(peaks_df, [tp_col, fp_col]).dropna(subset=[tp_col, fp_col])
tp = peaks_df[tp_col].to_numpy()
fp = peaks_df[fp_col].to_numpy()

Fs, dt = estimate_fs(t)
print({"n_samples": int(len(t)), "duration_s": float(t[-1]-t[0]) if len(t) else np.nan, "Fs_est_Hz": float(Fs) if not np.isnan(Fs) else None})
# --- Sanity checks RAW ---
x = x.astype(float)
t = t.astype(float)

dc_offset = float(np.mean(x))
x_min, x_max = float(np.min(x)), float(np.max(x))
x_absmax = float(np.max(np.abs(x)))
nan_ratio = float(np.mean(~np.isfinite(x)))

# saturation heuristique (si audio normalisé [-1,1], absmax proche de 1 => possible clipping)
clip_ratio = float(np.mean(np.abs(x) >= 0.999 * x_absmax)) if x_absmax > 0 else 0.0

baseline_summary = {
    "n_samples_raw": int(len(x)),
    "duration_s": float(t[-1]-t[0]) if len(t) else np.nan,
    "Fs_est_Hz": float(Fs) if not np.isnan(Fs) else np.nan,
    "dt_median_s": float(dt) if not np.isnan(dt) else np.nan,
    "dc_offset": dc_offset,
    "x_min": x_min,
    "x_max": x_max,
    "x_absmax": x_absmax,
    "nan_ratio": nan_ratio,
    "clip_ratio_heuristic": clip_ratio
}

pd.DataFrame([baseline_summary]).to_csv(OUT_DIR / "baseline_summary.csv", index=False, encoding="utf-8")
print(f"[OK] baseline_summary.csv -> {OUT_DIR/'baseline_summary.csv'}")

# --- Exports des données propres ---
raw_df.to_csv(OUT_DIR / "raw_clean.csv", index=False, encoding="utf-8")
fft_df.to_csv(OUT_DIR / "fft_clean.csv", index=False, encoding="utf-8")
peaks_df.to_csv(OUT_DIR / "peak_history_clean.csv", index=False, encoding="utf-8")
print("[OK] raw_clean.csv / fft_clean.csv / peak_history_clean.csv écrits")
# Prétraitement
x_detrend = x - np.mean(x)
mx = np.max(np.abs(x_detrend)) if np.max(np.abs(x_detrend)) > 0 else 1.0
x_norm = x_detrend / mx

# Extrait temporel
Nshow = min(len(t), 5000)

plt.figure()
plt.plot(t[:Nshow], x[:Nshow])
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.title("RAW (extrait)")
plt.tight_layout(); save_fig("01_raw_excerpt"); plt.show()

plt.figure()
plt.plot(t[:Nshow], x_norm[:Nshow])
plt.xlabel("Time (s)"); plt.ylabel("Amplitude normalisée")
plt.title("RAW prétraité (centré + normalisé) — extrait")
plt.tight_layout(); save_fig("02_raw_preprocessed_excerpt"); plt.show()

plt.figure()
plt.hist(x_norm, bins=80)
plt.xlabel("Amplitude normalisée"); plt.ylabel("Count")
plt.title("Histogramme amplitude (détection clipping / distribution)")
plt.tight_layout(); save_fig("03_hist_amplitude"); plt.show()
def compute_simple_snr(freq, amp):
    if len(amp) < 10:
        return np.nan, np.nan, np.nan, np.nan
    idx = int(np.nanargmax(amp))
    f_peak = float(freq[idx]); A_peak = float(amp[idx])
    n = len(amp)
    half_win = max(10, int(0.05*n))
    mask = np.ones(n, dtype=bool)
    mask[max(0, idx-half_win):min(n, idx+half_win+1)] = False
    noise = float(np.nanmedian(amp[mask])) if np.any(mask) else np.nan
    snr = 20*np.log10(A_peak/noise) if (noise and noise > 0) else np.nan
    return f_peak, A_peak, noise, snr

fpk, Apk, noise_floor, snr_db = compute_simple_snr(f, A)

plt.figure()
plt.plot(f, A)
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.title("FFT Spectrum (global)")
plt.tight_layout(); save_fig("04_fft_global"); plt.show()

plt.figure()
plt.plot(f, A)
if not np.isnan(fpk):
    plt.axvline(fpk, linestyle="--")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.title(f"FFT Spectrum (pic principal ~ {fpk:.2f} Hz) | SNR~{snr_db:.2f} dB")
plt.tight_layout(); save_fig("05_fft_peak_snr"); plt.show()

pd.DataFrame([{
    "f_peak_Hz": fpk,
    "A_peak": Apk,
    "noise_floor_median": noise_floor,
    "snr_db_simple": snr_db
}]).to_csv(OUT_DIR / "fft_metrics.csv", index=False, encoding="utf-8")
print(f"[OK] fft_metrics.csv -> {OUT_DIR/'fft_metrics.csv'}")
if not np.isnan(Fs) and Fs > 0 and len(x_norm) > 512:
    # Paramètres raisonnables
    NFFT = min(2048, max(256, 2 ** int(np.floor(np.log2(len(x_norm)//16)))))
    noverlap = NFFT // 2

    plt.figure()
    plt.specgram(x_norm, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogramme (NFFT={NFFT}, overlap={noverlap})")
    plt.tight_layout(); save_fig("06_spectrogram"); plt.show()
else:
    print("[INFO] Spectrogramme ignoré (Fs invalide ou signal trop court).")
if np.isnan(Fs) or Fs <= 0:
    raise ValueError("Fs non valide. Pour continuer, impose Fs manuellement (ex: Fs=48000).")

win_s = 0.10  # 100 ms
win = max(32, int(win_s * Fs))
step = win // 2

rms, tt = [], []
for i in range(0, len(x_norm) - win, step):
    seg = x_norm[i:i+win]
    rms.append(np.sqrt(np.mean(seg**2)))
    tt.append(t[i + win//2])

rms = np.array(rms)
tt = np.array(tt)

thr = np.median(rms) + 2*np.std(rms)
active = rms > thr

plt.figure()
plt.plot(tt, rms)
plt.xlabel("Time (s)"); plt.ylabel("RMS")
plt.title("RMS (énergie) dans le temps")
plt.tight_layout(); save_fig("07_rms_time"); plt.show()

plt.figure()
plt.plot(tt, rms)
plt.plot(tt, active.astype(float) * np.max(rms))
plt.xlabel("Time (s)"); plt.ylabel("RMS / activité")
plt.title("Détection segments actifs (seuil sur RMS)")
plt.tight_layout(); save_fig("08_activity_rms_threshold"); plt.show()

# Autocorrélation de RMS
r = rms - np.mean(rms)
acf = np.correlate(r, r, mode="full")
acf = acf[acf.size//2:]
acf = acf / (acf[0] if acf[0] != 0 else 1.0)

lags = np.arange(len(acf)) * (step / Fs)

plt.figure()
plt.plot(lags[:min(600, len(lags))], acf[:min(600, len(acf))])
plt.xlabel("Lag (s)"); plt.ylabel("Autocorrélation (RMS)")
plt.title("Autocorrélation de l’énergie (périodicité potentielle)")
plt.tight_layout(); save_fig("09_acf_rms"); plt.show()

pd.DataFrame([{
    "win_s": win_s,
    "thr_rms": float(thr),
    "active_ratio": float(np.mean(active)) if len(active) else np.nan
}]).to_csv(OUT_DIR / "time_energy_metrics.csv", index=False, encoding="utf-8")
print(f"[OK] time_energy_metrics.csv -> {OUT_DIR/'time_energy_metrics.csv'}")
def spectral_features(segment, fs):
    w = np.hanning(len(segment))
    X = np.fft.rfft(segment * w)
    P = np.abs(X)**2
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)

    Psum = np.sum(P)
    if Psum <= 0:
        return np.nan, np.nan, np.nan

    p = P / Psum
    entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))  # normalisée [0..1]
    centroid = np.sum(freqs * P) / np.sum(P)
    peakf = freqs[int(np.argmax(P))] if len(P) else np.nan
    return float(entropy), float(centroid), float(peakf)

def bandpower(segment, fs, fmin, fmax):
    X = np.fft.rfft(segment * np.hanning(len(segment)))
    P = np.abs(X)**2
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.sum(P[mask])) if np.any(mask) else np.nan

bands = [(0, 200), (200, 1000), (1000, 5000), (5000, 10000)]
rows = []
for i in range(0, len(x_norm) - win, step):
    seg = x_norm[i:i+win]
    ent, cent, peakf = spectral_features(seg, Fs)

    bp = {}
    for (b0, b1) in bands:
        bp[f"bp_{b0}_{b1}Hz"] = bandpower(seg, Fs, b0, b1)

    rows.append({
        "t_mid_s": float(t[i + win//2]),
        "rms": float(np.sqrt(np.mean(seg**2))),
        "entropy": ent,
        "centroid_Hz": cent,
        "peakfreq_window_Hz": peakf,
        **bp
    })

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUT_DIR / "features_timeseries.csv", index=False, encoding="utf-8")
print(f"[OK] features_timeseries.csv -> {OUT_DIR/'features_timeseries.csv'}")

# Plots features
plt.figure()
plt.plot(feat_df["t_mid_s"], feat_df["entropy"])
plt.xlabel("Time (s)"); plt.ylabel("Entropy (norm.)")
plt.title("Entropie spectrale dans le temps")
plt.tight_layout(); save_fig("10_entropy_time"); plt.show()

plt.figure()
plt.plot(feat_df["t_mid_s"], feat_df["centroid_Hz"])
plt.xlabel("Time (s)"); plt.ylabel("Centroid (Hz)")
plt.title("Centroid spectral dans le temps")
plt.tight_layout(); save_fig("11_centroid_time"); plt.show()

# Bandpower plots (un graphique par bande)
for (b0, b1) in bands:
    col = f"bp_{b0}_{b1}Hz"
    plt.figure()
    plt.plot(feat_df["t_mid_s"], feat_df[col])
    plt.xlabel("Time (s)"); plt.ylabel("Bandpower")
    plt.title(f"Énergie spectrale — bande {b0}-{b1} Hz")
    plt.tight_layout(); save_fig(f"12_bandpower_{b0}_{b1}"); plt.show()

# Résumé features global
feat_summary = feat_df.describe(percentiles=[0.1, 0.5, 0.9]).T
feat_summary.to_csv(OUT_DIR / "features_summary.csv", encoding="utf-8")
print(f"[OK] features_summary.csv -> {OUT_DIR/'features_summary.csv'}")
# Stats Peak History
peak_stats = {
    "n_points": int(len(fp)),
    "freq_median_Hz": float(np.nanmedian(fp)) if len(fp) else np.nan,
    "freq_std_Hz": float(np.nanstd(fp)) if len(fp) else np.nan,
    "freq_min_Hz": float(np.nanmin(fp)) if len(fp) else np.nan,
    "freq_max_Hz": float(np.nanmax(fp)) if len(fp) else np.nan,
    "duration_s": float(tp[-1]-tp[0]) if len(tp) else np.nan,
}
pd.DataFrame([peak_stats]).to_csv(OUT_DIR / "peak_metrics.csv", index=False, encoding="utf-8")
print(f"[OK] peak_metrics.csv -> {OUT_DIR/'peak_metrics.csv'}")

plt.figure()
plt.plot(tp, fp, marker=".", linestyle="none")
plt.xlabel("Time (s)"); plt.ylabel("Peak frequency (Hz)")
plt.title("Peak History (points)")
plt.tight_layout(); save_fig("13_peak_history_points"); plt.show()

plt.figure()
plt.hist(fp[~np.isnan(fp)], bins=60)
plt.xlabel("Peak frequency (Hz)"); plt.ylabel("Count")
plt.title("Distribution des fréquences dominantes (Peak History)")
plt.tight_layout(); save_fig("14_peak_histogram"); plt.show()

# Ruptures simples
dfp = np.abs(np.diff(fp))
thr_jump = float(np.nanmedian(dfp) + 3*np.nanstd(dfp)) if len(dfp) else np.nan
jumps = np.where(dfp > thr_jump)[0] if len(dfp) else np.array([], dtype=int)

plt.figure()
plt.plot(tp, fp)
for j in jumps[:80]:
    plt.axvline(tp[j], linestyle="--")
plt.xlabel("Time (s)"); plt.ylabel("Peak frequency (Hz)")
plt.title(f"Peak History + ruptures (thr={thr_jump:.2f})")
plt.tight_layout(); save_fig("15_peak_changepoints"); plt.show()

# Régimes = segments entre ruptures
boundaries = np.concatenate(([0], jumps+1, [len(fp)])) if len(fp) else np.array([0,0])
regimes = []
for k in range(len(boundaries)-1):
    s, e = int(boundaries[k]), int(boundaries[k+1])
    seg_fp = fp[s:e]
    seg_tp = tp[s:e]
    if len(seg_fp) < 2:
        continue
    regimes.append({
        "regime_id": k+1,
        "start_s": float(seg_tp[0]),
        "end_s": float(seg_tp[-1]),
        "duration_s": float(seg_tp[-1]-seg_tp[0]),
        "freq_mean_Hz": float(np.nanmean(seg_fp)),
        "freq_std_Hz": float(np.nanstd(seg_fp)),
        "n_points": int(len(seg_fp)),
    })

reg_df = pd.DataFrame(regimes)
reg_df.to_csv(OUT_DIR / "peak_regimes.csv", index=False, encoding="utf-8")
print(f"[OK] peak_regimes.csv -> {OUT_DIR/'peak_regimes.csv'}")
reg_df.head()
fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(t[:Nshow], x_norm[:Nshow])
ax1.set_title("Signal (prétraité) — extrait")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amp norm.")

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(f, A)
if not np.isnan(fpk):
    ax2.axvline(fpk, linestyle="--")
ax2.set_title("FFT (pic + SNR)")
ax2.set_xlabel("Hz"); ax2.set_ylabel("Amp")

ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(feat_df["t_mid_s"], feat_df["rms"])
ax3.set_title("RMS (énergie)")
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("RMS")

ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(feat_df["t_mid_s"], feat_df["entropy"])
ax4.set_title("Entropie spectrale")
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Entropy")

ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(feat_df["t_mid_s"], feat_df["centroid_Hz"])
ax5.set_title("Centroid spectral")
ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Hz")

ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(tp, fp, marker=".", linestyle="none")
ax6.set_title("Peak History")
ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Hz")

plt.tight_layout()
plt.savefig(OUT_DIR / "dashboard_baseline.png", dpi=200, bbox_inches="tight")
print(f"[OK] dashboard_baseline.png -> {OUT_DIR/'dashboard_baseline.png'}")
plt.show()
