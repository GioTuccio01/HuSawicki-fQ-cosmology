"""
Equazione di stato dell'energia oscura effettiva per il modello f(Q):

    f(Q) = Q + 2 Lambda + alpha * mu^2 * (Q/mu^2 - 1) / (1 + Q/mu^2)^2

in gauge coincidente, FLRW piatto (Q = 6 H^2).

Convenzione: azione S = (1/16 pi G) int f(Q) sqrt(-g) d^4x,
in cui f = Q riproduce GR e f = Q + 2 Lambda riproduce LambdaCDM.

Unita' lavorative: mu^2 = 1, 16 pi G = 1.
Variabile dimensionless: x = Q/mu^2 = 6 H^2 / mu^2.
"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
# Output directory relativa: ../plots/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "..", "plots")
os.makedirs(_OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Definizioni della teoria
# -----------------------------------------------------------------------------

def R_DE(x, alpha, Lam):
    """R(x) = 16 pi G * rho_DE(x) = 2 Lambda - alpha (1 + 6x - 3x^2)/(1+x)^3
    [Convenzione A: f(Q) = Q + 2Lambda + alpha mu^2 g(x)]"""
    return 2.0*Lam - alpha * (1.0 + 6.0*x - 3.0*x**2) / (1.0 + x)**3

def D_of_x(x, alpha):
    """D(x) = 2 Q f_QQ + f_Q = +1 + 3 alpha (x^2 - 6x + 1)/(1+x)^4
    [Convenzione A]"""
    return +1.0 + 3.0*alpha * (x**2 - 6.0*x + 1.0) / (1.0 + x)**4

def w_DE(x, alpha, Lam):
    """w_DE(x) dalla Raychaudhuri modificata.

    Seconda Friedmann (derivando la prima + conservazione di rho_m):
      (1 - R'(x)) dot(x) = -3H (x - R(x))
    usando l'identita' 1 - R'(x) = -D(x) e dot(x) = 12 H Hdot:
      4 Hdot = (x - R)/D
    w_DE = -(3H^2 + 2 Hdot)/(8 pi G rho_DE) = -(x + 4 Hdot)/R
    """
    R = R_DE(x, alpha, Lam)
    D = D_of_x(x, alpha)
    fourHdot = -(x - R) / D
    return -(x + fourHdot) / R

# -----------------------------------------------------------------------------
# Normalizzazione: oggi x = x0 con Omega_DE,0 = 0.7
# -----------------------------------------------------------------------------
x0  = 1.0    # sceglie mu^2 = 6 H0^2
Om0 = 0.3
OL0 = 1.0 - Om0   # 0.7 = R(x0)/x0

def Lam_from_alpha(alpha, x0=x0, OL0=OL0):
    """Fissa Lambda per avere Omega_DE,0 = 0.7 al tempo presente.
    In conv. A: R(x0) = 2*Lam - alpha*(1+6x-3x^2)/(1+x)^3 = OL0*x0
    => 2*Lam = OL0*x0 + alpha*(...)"""
    Rx0_target = OL0 * x0
    Ralpha_part = alpha * (1.0 + 6.0*x0 - 3.0*x0**2) / (1.0 + x0)**3
    return 0.5 * (Rx0_target + Ralpha_part)  # NB: + invece di -

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
x_vals = np.linspace(1.5, 15, 3000)  # da 0.05 (copre x_- ~0.17) a 15 (oltre x_+ ~5.83)

alphas = [-0.3, -0.1, 0.0, 0.1, 0.3]
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(alphas)))

xp = 3.0 + 2.0*np.sqrt(2)
xm = 3.0 - 2.0*np.sqrt(2)

# --- Figure 1: w_DE(x) ---
fig1, ax1 = plt.subplots(figsize=(7, 5.2))
for alpha, col in zip(alphas, colors):
    Lam = Lam_from_alpha(alpha)
    w   = w_DE(x_vals, alpha, Lam)
    ax1.plot(x_vals, w, lw=2, color=col,
             label=fr'$\alpha={alpha:+.1f}$')

for xc, name in [(xm, r'$x_-=3-2\sqrt{2}$'), (xp, r'$x_+=3+2\sqrt{2}$')]:
    ax1.axvline(xc, ls=':', color='gray', alpha=0.8)
    ax1.text(xc*1.05, -1.28, name, rotation=90, va='bottom', ha='left',
             fontsize=9, color='gray')
ax1.axhline(-1, ls='--', color='k', alpha=0.4)
ax1.axvline(x0, ls='-', color='green', alpha=0.35, label='today')

ax1.set_xlabel(r'$x = Q/\mu^2 = 6H^2/\mu^2$', fontsize=12)
ax1.set_ylabel(r'$w_{\rm DE}$', fontsize=12)
ax1.set_title(r'$w_{\rm DE}(x)$: $f(Q)=Q+2\Lambda+\alpha\mu^2\,(x-1)/(1+x)^2$',
              fontsize=11)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.3, -0.7)
ax1.set_xlim(0.05, 15)

plt.tight_layout()
outpath1 = os.path.join(_OUT_DIR, '02_wDE_phantom_a.png')
plt.savefig(outpath1, dpi=160, bbox_inches='tight')
print(f"Figura salvata: {outpath1}")
plt.close(fig1)

# --- Figure 2: 1 + w_DE zoom sugli attraversamenti ---
fig2, ax2 = plt.subplots(figsize=(7, 5.2))
for alpha, col in zip(alphas, colors):
    Lam = Lam_from_alpha(alpha)
    w   = w_DE(x_vals, alpha, Lam)
    ax2.plot(x_vals, 1.0 + w, lw=2, color=col,
             label=fr'$\alpha={alpha:+.1f}$')

for xc in (xm, xp):
    ax2.axvline(xc, ls=':', color='gray', alpha=0.8)
ax2.axhline(0, ls='--', color='k', alpha=0.4)
ax2.axvline(x0, ls='-', color='green', alpha=0.35)

ax2.set_xlabel(r'$x = Q/\mu^2$', fontsize=12)
ax2.set_ylabel(r'$1+w_{\rm DE}$', fontsize=12)
ax2.set_title(r'Phantom crossing for  $x_\pm = 3\pm 2\sqrt{2}$',
              fontsize=11)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.25, 0.25)
ax2.set_xlim(0.05, 15)

plt.tight_layout()
outpath2 = os.path.join(_OUT_DIR, '02_wDE_phantom_b.png')
plt.savefig(outpath2, dpi=160, bbox_inches='tight')
print(f"Figura salvata: {outpath2}")
plt.close(fig2)

# -----------------------------------------------------------------------------
# Output numerico
# -----------------------------------------------------------------------------
print("\nAttraversamenti fantasma (radici di x^2 - 6x + 1 = 0):")
print(f"  x_- = 3 - 2 sqrt(2) = {xm:.6f}")
print(f"  x_+ = 3 + 2 sqrt(2) = {xp:.6f}")
print(f"\nOggi (x = {x0}, Omega_m = {Om0}, Omega_DE = {OL0}):")
print(f"  {'alpha':>7} | {'Lambda':>8} | {'w_DE(x0)':>10}")
print("  " + "-"*32)
for alpha in alphas:
    Lam = Lam_from_alpha(alpha)
    w0  = w_DE(x0, alpha, Lam)
    print(f"  {alpha:+7.2f} | {Lam:8.4f} | {w0:+10.4f}")
