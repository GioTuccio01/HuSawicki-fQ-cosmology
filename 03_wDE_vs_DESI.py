"""
Stesso f(Q) di prima, ma ora mostro w_DE(z) nella regione DESI-preferita
(alpha < 0, quintessenza oggi, attraversamento di w=-1 verso il passato).

Confronto aggiuntivo: CPL-DESI con (w_0, w_a) = (-0.83, -0.56)
(valori tipici DESI DR1 BAO+CMB+SN best-fit arXiv:2404.03002, Fig.6).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

import os, sys
# Output directory relativa: ../plots/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "..", "plots")
os.makedirs(_OUT_DIR, exist_ok=True)

# -------------------- teoria --------------------
def R_DE(x, a, L):       return 2*L - a*(1 + 6*x - 3*x*x)/(1+x)**3
def D_of_x(x, a):        return +1 + 3*a*(x*x - 6*x + 1)/(1+x)**4
def w_DE(x, a, L):
    R = R_DE(x, a, L); fourHdot = -(x - R)/D_of_x(x, a)
    return -(x + fourHdot)/R

OM0, OL0 = 0.30, 0.70
def x0_of(eta): return 6.0/eta**2
def Lam_of(a, eta):
    x0 = x0_of(eta)
    return 0.5*(OL0*x0 + a*(1+6*x0-3*x0*x0)/(1+x0)**3)

def x_of_z(zs, a, eta):
    x0, L = x0_of(eta), Lam_of(a, eta)
    rhs   = OM0*x0*(1+np.asarray(zs,dtype=float))**3
    g     = lambda x, r: x - R_DE(x, a, L) - r
    out   = np.empty_like(rhs)
    for i, r in enumerate(rhs):
        out[i] = brentq(g, 1e-8, 1e8, args=(r,), xtol=1e-10)
    return out, L

def z_of_x(x, a, eta):
    L = Lam_of(a, eta); x0 = x0_of(eta)
    arg = (x - R_DE(x, a, L))/(OM0*x0)
    return np.cbrt(arg) - 1 if arg > 0 else np.nan

def w_of_z(zs, a, eta):
    xs, L = x_of_z(zs, a, eta)
    return w_DE(xs, a, L)

# -------------------- CPL di riferimento --------------------
def w_CPL(z, w0, wa):
    a = 1.0/(1+z)
    return w0 + wa*(1 - a)

# =================================================================
# FIGURE
# =================================================================
zgrid = np.linspace(0.0, 2.0, 500)

# -------- Figure A: w_DE(z) vs alpha, eta = sqrt(6) fisso ---------
fig1, ax1 = plt.subplots(figsize=(7, 5.2))

eta_fid = np.sqrt(6.0)
alphas  = [-0.20, -0.10, 0.0, +0.10, +0.20]
cmap    = plt.cm.RdBu
colors  = cmap(np.linspace(0.05, 0.95, len(alphas)))

ax1.axhspan(-1.5, -1.0, color='tab:blue',   alpha=0.08, label='phantom  ($w<-1$)')
ax1.axhspan(-1.0, -0.5, color='tab:orange', alpha=0.08, label='quintessence  ($-1<w$)')
ax1.axhline(-1.0, ls='--', color='k', alpha=0.5)

for a, c in zip(alphas, colors):
    w = w_of_z(zgrid, a, eta_fid)
    ax1.plot(zgrid, w, lw=2.2, color=c, label=fr'$\alpha={a:+.2f}$')

wCPL = w_CPL(zgrid, w0=-0.83, wa=-0.56)
ax1.plot(zgrid, wCPL, ls=':', lw=2.5, color='darkgreen',
         label=r'CPL DESI DR1 $(w_0,w_a)=(-0.83,-0.56)$')

ax1.set(xlabel=r'$z$', ylabel=r'$w_{\rm DE}(z)$',
        title=fr'$w_{{\rm DE}}(z)$  at $\eta=\sqrt{{6}}$  (varying $\alpha$)')
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8.5, loc='lower left', ncol=1, framealpha=0.92)
ax1.set_xlim(0, 2); ax1.set_ylim(-1.20, -0.78)

plt.tight_layout()
out1 = os.path.join(_OUT_DIR, '03_wDE_vs_DESI_a.png')
plt.savefig(out1, dpi=160, bbox_inches='tight')
print(f"Figura salvata: {out1}")
plt.close(fig1)

# -------- Figure B: w_DE(z) vs eta, alpha = +0.20 (DESI-preferito in conv A) --------
fig2, ax2 = plt.subplots(figsize=(7, 5.2))

alpha_b = +0.20
etas    = [1.2, np.sqrt(6.0), 4.0, 8.0]
cmap2   = plt.cm.viridis
cols_b  = cmap2(np.linspace(0.1, 0.85, len(etas)))

ax2.axhspan(-1.5, -1.0, color='tab:blue',   alpha=0.08)
ax2.axhspan(-1.0, -0.5, color='tab:orange', alpha=0.08)
ax2.axhline(-1.0, ls='--', color='k', alpha=0.5)

xp, xm = 3 + 2*np.sqrt(2), 3 - 2*np.sqrt(2)
for et, c in zip(etas, cols_b):
    w = w_of_z(zgrid, alpha_b, et)
    ax2.plot(zgrid, w, lw=2.2, color=c,
             label=fr'$\eta={et:.2f}$  ($x_0={x0_of(et):.2f}$)')
    zc = z_of_x(xp, alpha_b, et)
    if np.isfinite(zc) and 0 < zc < zgrid[-1]:
        ax2.axvline(zc, ls=':', color=c, alpha=0.7)
        ax2.text(zc, -0.80, fr'$z_+={zc:.2f}$', rotation=90, va='top',
                 ha='right', fontsize=8, color=c)

ax2.plot(zgrid, wCPL, ls=':', lw=2.5, color='darkgreen',
         label=r'CPL DESI DR1')

ax2.set(xlabel=r'$z$', ylabel=r'$w_{\rm DE}(z)$',
        title=fr'$w_{{\rm DE}}(z)$  at $\alpha={alpha_b:+.2f}$  (varying $\eta=\mu/H_0$)')
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8.5, loc='lower left', framealpha=0.92)
ax2.set_xlim(0, 2); ax2.set_ylim(-1.20, -0.78)

plt.tight_layout()
out2 = os.path.join(_OUT_DIR, '03_wDE_vs_DESI_b.png')
plt.savefig(out2, dpi=160, bbox_inches='tight')
print(f"Figura salvata: {out2}\n")
plt.close(fig2)

# =================================================================
# VALORI NUMERICI
# =================================================================
print("=== w_DE(z=0) e z del phantom crossing (x_+), a eta = sqrt(6) ===")
print(f"{'alpha':>8} | {'w_DE(z=0)':>12} | {'z(x_+)':>10}  {'regime oggi':<14}")
print("-"*58)
for a in alphas:
    w0  = w_of_z(np.array([0.0]), a, eta_fid)[0]
    zp  = z_of_x(xp, a, eta_fid)
    reg = "phantom"  if w0 < -1 else "quintessenza"
    zp_s = f"{zp:+7.3f}" if np.isfinite(zp) else "futuro"
    print(f"{a:+8.3f} | {w0:+12.4f} | {zp_s:>10}  {reg:<14}")

# Confronto con CPL DESI
print("\nPer confronto, CPL DESI DR1 (w0,wa)=(-0.83,-0.56):")
print(f"  w(z=0) = {w_CPL(0,-0.83,-0.56):.3f}  (quintessenza)")
print(f"  w(z=1) = {w_CPL(1,-0.83,-0.56):.3f}  (phantom)")
# z in cui w_CPL = -1
# w0 + wa*(1 - 1/(1+z)) = -1  =>  1 - 1/(1+z) = (-1 - w0)/wa
frac = (-1 - (-0.83))/(-0.56)   # 0.304
z_cross = 1/(1 - frac) - 1
print(f"  z(w=-1) = {z_cross:.3f}")
