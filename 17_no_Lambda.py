"""
17_no_Lambda.py — Cosa accade al modello f(Q) senza Λ?

f(Q) = Q + α μ² (x-1)/(1+x)²    (no 2Λ)

Risposta breve: il modello esiste ma cade FUORI dalla regione di stabilità.
La chiusura oggi richiede alpha = 2(Ω_m - 1), che per Ω_m = 0.3 dà α = -1.4,
violando sia la stabilità di background (D > 0) che quella tensoriale (f_Q > 0).

Lo dimostro graficamente.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 11})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ==========================================================
# Panel A: chiusura oggi -- valore α richiesto in funzione di Ω_m
# ==========================================================
ax = axes[0, 0]
Om_arr = np.linspace(0.05, 0.99, 100)
# Senza Λ: -α/2 = (1-Ω_m) => α = 2(Ω_m - 1)
alpha_noLam = 2*(Om_arr - 1)
# Con Λ standard: α = 0 dà ΛCDM esatto, α libero per qualsiasi Ω_m
ax.plot(Om_arr, alpha_noLam, 'r-', lw=2.5,
        label='senza $\\Lambda$:  $\\alpha = 2(\\Omega_m - 1)$')
ax.axhline(0, color='blue', lw=2.5, label='con $\\Lambda$:  $\\alpha = 0$ (LCDM)')
ax.fill_between(Om_arr, -1/3, 1/3, color='green', alpha=0.18,
                label='regione stabile $|\\alpha| < 1/3$')
# best-fit dalla nostra analisi
ax.axvline(0.308, color='gray', ls=':', lw=1.5, alpha=0.7)
ax.text(0.31, -1.6, '$\\Omega_m = 0.308$\n(best-fit)', fontsize=8, color='gray')
ax.scatter([0.308], [2*(0.308-1)], color='red', s=80, zorder=5, edgecolor='black')
ax.annotate(f'$\\alpha = {2*(0.308-1):.2f}$\nINSTABILE',
            xy=(0.308, -1.38), xytext=(0.50, -0.7),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'))
ax.set_xlabel('$\\Omega_{m,0}$', fontsize=12)
ax.set_ylabel('$\\alpha$ richiesto dalla chiusura oggi', fontsize=12)
ax.set_title('A. Chiusura $R(x_0) = (1-\\Omega_m)\\,x_0$',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-2.0, 0.6)

# ==========================================================
# Panel B: f_Q(x) per α = -1.4 vs nostro best-fit α = +0.16 + Λ
# ==========================================================
ax = axes[0, 1]
xs = np.linspace(0.01, 30, 2000)

for alpha, Lam, lab, c in [
    (-1.4, 0,        'senza $\\Lambda$, $\\alpha = -1.4$ (chiude $\\Omega_m = 0.3$)', 'red'),
    (+0.159, 0.3875, 'con $\\Lambda$, best-fit $\\alpha = +0.159$',                 'blue'),
    (0,     0.35,    '$\\Lambda$CDM puro ($\\alpha=0$)',                            'black'),
]:
    fQ = 1 + alpha*(3 - xs)/(1 + xs)**3
    style = '-' if c != 'black' else ':'
    ax.plot(xs, fQ, style, color=c, lw=2.0, label=lab)

ax.axhline(0, color='gray', lw=1, ls='-', alpha=0.5)
ax.fill_between(xs, -10, 0, color='red', alpha=0.10, label='regione ghost ($f_Q < 0$)')
ax.axvline(1, color='green', lw=1, ls='--', alpha=0.7, label='oggi ($x = x_0 = 1$)')
ax.set_xlabel('$x = Q/\\mu^2$', fontsize=12)
ax.set_ylabel('$f_Q$', fontsize=12)
ax.set_xscale('log')
ax.set_xlim(0.01, 30)
ax.set_ylim(-3.5, 2.0)
ax.legend(fontsize=8.5, loc='lower right')
ax.set_title('B. Stabilità tensoriale ($f_Q > 0$ richiesto)',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# ==========================================================
# Panel C: D(x) per gli stessi casi
# ==========================================================
ax = axes[1, 0]
for alpha, Lam, lab, c in [
    (-1.4, 0,        'senza $\\Lambda$, $\\alpha = -1.4$', 'red'),
    (+0.159, 0.3875, 'con $\\Lambda$, $\\alpha = +0.159$', 'blue'),
    (0,     0.35,    '$\\Lambda$CDM ($\\alpha=0$)',         'black'),
]:
    D = 1 + 3*alpha*(xs**2 - 6*xs + 1)/(1+xs)**4
    style = '-' if c != 'black' else ':'
    ax.plot(xs, D, style, color=c, lw=2.0, label=lab)

ax.axhline(0, color='gray', lw=1, alpha=0.5)
ax.fill_between(xs, -10, 0, color='red', alpha=0.10,
                label='instabile background ($D < 0$)')
ax.axvline(1, color='green', lw=1, ls='--', alpha=0.7)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$D(x) = f_Q + 2 Q\\, f_{QQ}$', fontsize=12)
ax.set_xscale('log')
ax.set_xlim(0.01, 30)
ax.set_ylim(-4, 3)
ax.legend(fontsize=8.5, loc='lower right')
ax.set_title('C. Stabilità background ($D(x) > 0$ richiesto)',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# ==========================================================
# Panel D: equazione di stato w_DE(z) per i due casi viable e non
# ==========================================================
from scipy.integrate import cumulative_trapezoid

ax = axes[1, 1]

X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 200),
    np.linspace(0.3, 50, 800),
    np.linspace(50, 5000, 200)]))

def w_of_z(alpha, Lam, Om, x0=1.0, zs=np.linspace(0.001, 3, 300)):
    R_s = 2*Lam - alpha*(1 + 6*X_SAMP - 3*X_SAMP**2)/(1 + X_SAMP)**3
    g_s = X_SAMP - R_s
    if not np.all(np.diff(g_s) > 0):
        return zs, np.full_like(zs, np.nan)
    tgt = Om*x0*(1 + zs)**3
    if tgt[0] < g_s.min() or tgt[-1] > g_s.max():
        return zs, np.full_like(zs, np.nan)
    xs = np.interp(tgt, g_s, X_SAMP)
    R = 2*Lam - alpha*(1+6*xs-3*xs**2)/(1+xs)**3
    D = 1 + 3*alpha*(xs**2 - 6*xs + 1)/(1+xs)**4
    fourHdot = -(xs - R)/D
    w = -(xs + fourHdot)/R
    return zs, w

cases = [
    (-1.4, 0,      'senza $\\Lambda$ ($\\alpha=-1.4$, $\\Omega_m=0.3$)', 'red'),
    (+0.159, 0.3875, 'con $\\Lambda$, best-fit', 'blue'),
    (0,    0.35,   '$\\Lambda$CDM', 'black'),
]
for alpha, Lam, lab, c in cases:
    zs, w = w_of_z(alpha, Lam, 0.30)
    style = '-' if c != 'black' else ':'
    ax.plot(zs, w, style, color=c, lw=2.0, label=lab)

ax.axhline(-1, color='gray', lw=1, ls='--', alpha=0.7,
           label='$w = -1$ (LCDM)')
ax.set_xlabel('$z$', fontsize=12)
ax.set_ylabel('$w_{\\rm DE}(z)$', fontsize=12)
ax.set_title('D. Equazione di stato dell\'energia oscura effettiva',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(-2.5, 0.5)

plt.suptitle('Cosa accade al nostro $f(Q)$ se togliamo $\\Lambda$?',
             fontsize=14, fontweight='bold', y=1.005)
plt.tight_layout()

OUT_DIR = '/tmp/TESI_fQ_cosmology/plots'
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(f'{OUT_DIR}/17_no_Lambda.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot salvato in {OUT_DIR}/17_no_Lambda.png")

# ==========================================================
# RIEPILOGO ANALITICO
# ==========================================================
print()
print("="*65)
print("RIEPILOGO ANALITICO -- modello senza Λ")
print("="*65)
print()
print("Vincolo di chiusura oggi (x_0 = 1):")
print("    R(x_0=1) = -α (1+6-3)/(1+1)^3 = -α/2")
print("    deve uguagliare (1-Ω_m) x_0 = (1-Ω_m)")
print("    => α = -2(1-Ω_m) = 2Ω_m - 2")
print()
print("Per Ω_m = 0.30:  α = -1.4")
print("Per Ω_m = 0.50:  α = -1.0")
print("Per Ω_m = 1.00:  α =  0.0   [universo solo materia, no DE]")
print()
print("STABILITÀ:")
print("  - Tensoriale: |α| < 1/3.  α = -1.4 viola (|α|/(1/3) = 4.2x peggio)")
print("  - Background: D(x) > 0 per x ∈ [0, ∞).  α = -1.4 produce D(0) = 1+3α = -3.2 < 0")
print()
print("CONCLUSIONE:")
print("  Senza Λ, il modello descrive cosmologia accelerata SOLO se")
print("  α ~ -1, ben fuori dalla regione di stabilità. f_Q < 0 per x ∈ [0, 0.5]")
print("  significa GHOST nelle perturbazioni tensoriali.")
print("  D(x) < 0 per x → 0 significa BACKGROUND instabile.")
print()
print("  ==> Λ è ESSENZIALE per il funzionamento del modello.")
print("      Non è solo un termine di taratura, ma stabilizzatore dinamico.")
