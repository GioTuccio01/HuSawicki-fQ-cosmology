"""
f(Q) = Q + 2 Lambda + alpha mu^2 (Q/mu^2 - 1)/(1+Q/mu^2)^2  (convenzione A: Jimenez-Heisenberg-Koivisto)
Variabile dimensionless x = Q/mu^2 = 6 H^2/mu^2,  eta = mu/H_0  ==>  x_0 = 6/eta^2.

Obiettivi:
  (a) x <-> z per vari eta (risolvendo implicitamente la Friedmann)
  (b) Integrazione numerica della Friedmann modificata -> H(z) esplicito
  (c) Confronto con cosmic chronometers (H(z) a ~1-2 sigma) e BAO DESI DR1
      D_M/r_d, D_H/r_d, D_V/r_d a vari z_eff.

ASSUNZIONI:
  * matter (polvere) separatamente conservato: rho_m propto (1+z)^3
  * rho_DE geometrica separatamente conservata (Bianchi in f(Q))
  * Omega_m,0 = 0.30, Omega_DE,0 = 0.70 fissati oggi
  * H_0 e r_d fissati al valore Planck 2018 LCDM (solo per confronto assoluto con dati)
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

# =========================================================================
# 1.  TEORIA  -- unita' mu^2 = 1, 16 pi G = 1
# =========================================================================
def R_DE(x, alpha, Lam):                          # 16 pi G * rho_DE
    return 2.0*Lam - alpha * (1.0 + 6.0*x - 3.0*x*x) / (1.0 + x)**3

def D_of_x(x, alpha):                             # 2Q f_QQ + f_Q
    return +1.0 + 3.0*alpha * (x*x - 6.0*x + 1.0) / (1.0 + x)**4

def w_DE(x, alpha, Lam):
    R = R_DE(x, alpha, Lam)
    fourHdot = -(x - R) / D_of_x(x, alpha)         # 4 Hdot = -(x-R)/D, conv A
    return -(x + fourHdot) / R                    # w_DE = -(3H^2+2Hdot)/(8piG rho_DE)

# ---- parametri oggi ----
OM0, OL0 = 0.30, 0.70
def x0_from_eta(eta):    return 6.0/eta**2
def Lam_of(alpha, eta):
    x0     = x0_from_eta(eta)
    Ralpha = alpha * (1 + 6*x0 - 3*x0*x0) / (1 + x0)**3
    return 0.5*(OL0*x0 + Ralpha)   # conv A: + invece di -

# =========================================================================
# 2.  BACKGROUND SOLVER: dalla continuita' della materia
#     x - R(x) = Omega_m,0 * x_0 * (1+z)^3
# =========================================================================
def x_of_z(z_arr, alpha, eta):
    x0  = x0_from_eta(eta)
    Lam = Lam_of(alpha, eta)
    rhs = OM0 * x0 * (1.0 + np.asarray(z_arr, dtype=float))**3
    def g(x, r): return x - R_DE(x, alpha, Lam) - r
    xs = np.empty_like(rhs)
    for i, r in enumerate(rhs):
        xs[i] = brentq(g, 1e-8, 1e8, args=(r,), xtol=1e-10)
    return xs, Lam

def Ez(z_arr, alpha, eta):
    xs, Lam = x_of_z(z_arr, alpha, eta)
    return np.sqrt(xs / x0_from_eta(eta)), xs, Lam

def z_of_x(x, alpha, eta):
    """Inversa diretta (utile per marcare i phantom crossing a z)."""
    x0 = x0_from_eta(eta);    Lam = Lam_of(alpha, eta)
    arg = (x - R_DE(x, alpha, Lam)) / (OM0 * x0)
    return np.where(arg > 0, np.cbrt(arg) - 1.0, np.nan)

# =========================================================================
# 3.  DISTANZE (universo piatto)
# =========================================================================
C_KMS = 299792.458

def _grids(alpha, eta, zmax=3.2, N=2500):
    z = np.linspace(1e-5, zmax, N)
    E, _, _ = Ez(z, alpha, eta)
    cum_inv_E = np.concatenate(([0.0], cumulative_trapezoid(1.0/E, z)))
    return z, E, cum_inv_E

def DM_over_rd(zs, alpha, eta, H0, rd, zgrid=None):
    if zgrid is None: zgrid = _grids(alpha, eta)
    z, E, I = zgrid
    return (C_KMS / H0 / rd) * np.interp(zs, z, I)

def DH_over_rd(zs, alpha, eta, H0, rd, zgrid=None):
    if zgrid is None: zgrid = _grids(alpha, eta)
    z, E, I = zgrid
    return (C_KMS / H0 / rd) / np.interp(zs, z, E)

def DV_over_rd(zs, alpha, eta, H0, rd, zgrid=None):
    zs = np.asarray(zs, dtype=float)
    DM = DM_over_rd(zs, alpha, eta, H0, rd, zgrid)
    DH = DH_over_rd(zs, alpha, eta, H0, rd, zgrid)
    return (zs * DM*DM * DH)**(1.0/3.0)

# =========================================================================
# 4.  DATI
# =========================================================================
# Cosmic chronometers  [z, H (km/s/Mpc), sigma_H]   -- compilation standard
CC = np.array([
    [0.0700,  69.0, 19.6], [0.0900,  69.0, 12.0], [0.1200,  68.6, 26.2],
    [0.1700,  83.0,  8.0], [0.1791,  75.0,  4.0], [0.1993,  75.0,  5.0],
    [0.2000,  72.9, 29.6], [0.2700,  77.0, 14.0], [0.2800,  88.8, 36.6],
    [0.3519,  83.0, 14.0], [0.3802,  83.0, 13.5], [0.4000,  95.0, 17.0],
    [0.4004,  77.0, 10.2], [0.4247,  87.1, 11.2], [0.4497,  92.8, 12.9],
    [0.4700,  89.0, 49.6], [0.4783,  80.9,  9.0], [0.4800,  97.0, 62.0],
    [0.5930, 104.0, 13.0], [0.6800,  92.0,  8.0], [0.7500,  98.8, 33.6],
    [0.7810, 105.0, 12.0], [0.8750, 125.0, 17.0], [0.8800,  90.0, 40.0],
    [0.9000, 117.0, 23.0], [1.0370, 154.0, 20.0], [1.3000, 168.0, 17.0],
    [1.3630, 160.0, 33.6], [1.4300, 177.0, 18.0], [1.5300, 140.0, 14.0],
    [1.7500, 202.0, 40.0], [1.9650, 186.5, 50.4],
])

# BAO - DESI DR1 (approssimativo, 2024) + Ly-alpha
# [z_eff, measurement, sigma, kind]
BAO = [
    (0.295,  7.93, 0.15, "DV"),     # BGS
    (0.510, 13.62, 0.25, "DM"),     # LRG1
    (0.510, 20.98, 0.61, "DH"),
    (0.706, 16.85, 0.32, "DM"),     # LRG2
    (0.706, 20.08, 0.60, "DH"),
    (0.930, 21.71, 0.28, "DM"),     # LRG3+ELG1
    (0.930, 17.88, 0.35, "DH"),
    (1.317, 27.79, 0.69, "DM"),     # ELG2
    (1.317, 13.82, 0.42, "DH"),
    (1.491, 26.07, 0.67, "DV"),     # QSO
    (2.330, 39.71, 0.94, "DM"),     # Ly-alpha auto+cross
    (2.330,  8.52, 0.17, "DH"),
]

H0_FID = 67.4        # Planck 2018 LCDM
RD_FID = 147.05      # Mpc

# =========================================================================
# 5.  CHI-QUADRI
# =========================================================================
def chi2_CC(alpha, eta, H0):
    E, _, _ = Ez(CC[:,0], alpha, eta)
    return np.sum( ((CC[:,1] - H0*E) / CC[:,2])**2 )

def chi2_BAO(alpha, eta, H0, rd):
    zgrid = _grids(alpha, eta, zmax=3.0)
    c2 = 0.0
    for (z, obs, sig, kind) in BAO:
        if   kind == "DM": mod = DM_over_rd(z, alpha, eta, H0, rd, zgrid)
        elif kind == "DH": mod = DH_over_rd(z, alpha, eta, H0, rd, zgrid)
        else:              mod = DV_over_rd(z, alpha, eta, H0, rd, zgrid)
        c2 += ((obs - mod) / sig)**2
    return c2

# =========================================================================
# 6.  FIGURE
# =========================================================================
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
(ax_wz, ax_Hz), (ax_dm, ax_chi) = axs

# -- (a)  w_DE(z) per vari eta, alpha fisso ---------------------------------
# Per alpha = +0.15, il phantom crossing x_+ cade a z tale che:
#   eta=1.10: z=0.17,  eta=1.50: z=0.70,  eta=sqrt(6)~2.45: z=1.56
# Scelta eta=sqrt(6) canonica (x_0=1, correzione nulla oggi).
alpha_a = 0.15
z_a     = np.linspace(0.0, 2.5, 500)
etas    = [1.5, 2.0, np.sqrt(6.0)]
cols_a  = ['#2e4a8f', '#d1495b', '#2ca02c']
for et, cc in zip(etas, cols_a):
    E, xs, Lam = Ez(z_a, alpha_a, et)
    w = w_DE(xs, alpha_a, Lam)
    ax_wz.plot(z_a, w, lw=2.2, color=cc,
               label=fr'$\eta={et:.2f}$  ($x_0={x0_from_eta(et):.2f}$)')
# phantom crossings in z per ciascun eta (solo se nel passato z>0)
xp, xm = 3.0+2.0*np.sqrt(2), 3.0-2.0*np.sqrt(2)
for et, cc in zip(etas, cols_a):
    for xc in (xp, xm):
        zc = z_of_x(xc, alpha_a, et)
        if np.isfinite(zc) and 0 < zc < z_a[-1]:
            ax_wz.axvline(zc, ls=':', color=cc, alpha=0.7, lw=1.3)
# bande phantom/quintessenza per leggibilita'
ax_wz.axhspan(-1, 0, color='orange', alpha=0.07)
ax_wz.axhspan(-2, -1, color='steelblue', alpha=0.07)
ax_wz.text(2.4, -0.92, 'quintessenza', fontsize=8.5, color='#b35900',
           ha='right', alpha=0.9, fontweight='bold')
ax_wz.text(2.4, -1.22, 'phantom', fontsize=8.5, color='#1f4e79',
           ha='right', alpha=0.9, fontweight='bold')
ax_wz.axhline(-1, ls='--', color='k', alpha=0.55, lw=1, label=r'$w=-1$')
ax_wz.set(xlabel=r'$z$', ylabel=r'$w_{\rm DE}$',
          title=fr'(a)  $w_{{\rm DE}}(z)$ per $\alpha={alpha_a}$, vari $\eta=\mu/H_0$')
ax_wz.set_ylim(-1.25, -0.80)    # zoom stretto per vedere le oscillazioni
ax_wz.set_xlim(0, 2.5)
ax_wz.grid(alpha=0.3); ax_wz.legend(fontsize=8.5, loc='lower right')

# -- (b)  H(z) con cosmic chronometers --------------------------------------
eta_b  = np.sqrt(6.0)
alpha_list = [-0.2, 0.0, 0.2]
cols_b = ['tab:blue', 'k', 'tab:red']
z_b    = np.linspace(1e-3, 2.1, 400)
for al, cc in zip(alpha_list, cols_b):
    E, _, _ = Ez(z_b, al, eta_b)
    ax_Hz.plot(z_b, H0_FID*E, lw=2, color=cc, label=fr'$\alpha={al:+.1f}$')
ax_Hz.errorbar(CC[:,0], CC[:,1], yerr=CC[:,2], fmt='o', ms=4, color='0.35',
               alpha=0.7, label='CC (32 pt)')
ax_Hz.set(xlabel=r'$z$', ylabel=r'$H(z)$  [km s$^{-1}$ Mpc$^{-1}$]',
          title=fr'(b)  $H(z)$ numerico + CC   ($\eta=\sqrt{{6}}$, $H_0={H0_FID}$)')
ax_Hz.grid(alpha=0.3); ax_Hz.legend(fontsize=9)

# -- (c.1)  D_M/r_d con BAO DESI --------------------------------------------
for al, cc in zip(alpha_list, cols_b):
    zg = _grids(al, eta_b, zmax=2.6)
    zc_plot = np.linspace(0.05, 2.45, 300)
    ax_dm.plot(zc_plot, DM_over_rd(zc_plot, al, eta_b, H0_FID, RD_FID, zg),
               lw=2, color=cc, label=fr'$D_M/r_d$, $\alpha={al:+.1f}$')
    ax_dm.plot(zc_plot, DH_over_rd(zc_plot, al, eta_b, H0_FID, RD_FID, zg),
               lw=2, ls='--', color=cc, alpha=0.7,
               label=fr'$D_H/r_d$, $\alpha={al:+.1f}$' if al==0 else None)
for (z, obs, sig, kind) in BAO:
    if kind == "DM":
        ax_dm.errorbar(z, obs, yerr=sig, fmt='o', ms=6, color='crimson',
                       alpha=0.85, zorder=5)
    elif kind == "DH":
        ax_dm.errorbar(z, obs, yerr=sig, fmt='s', ms=6, color='seagreen',
                       alpha=0.85, zorder=5)
    else:
        ax_dm.errorbar(z, obs, yerr=sig, fmt='^', ms=7, color='navy',
                       alpha=0.85, zorder=5)
# legend-hacks per simboli dati
ax_dm.errorbar([], [], yerr=[], fmt='o', color='crimson',  label=r'DESI $D_M/r_d$')
ax_dm.errorbar([], [], yerr=[], fmt='s', color='seagreen', label=r'DESI $D_H/r_d$')
ax_dm.errorbar([], [], yerr=[], fmt='^', color='navy',     label=r'DESI $D_V/r_d$')
ax_dm.set(xlabel=r'$z_{\rm eff}$', ylabel=r'$D_X/r_d$',
          title=r'(c.1)  BAO: $D_M/r_d$ (—), $D_H/r_d$ (- -) vs DESI DR1')
ax_dm.grid(alpha=0.3); ax_dm.legend(fontsize=8, ncol=2)

# -- (c.2)  Delta chi^2 vs alpha --------------------------------------------
alpha_scan = np.linspace(-0.35, 0.35, 15)
cc_arr = np.array([chi2_CC (a, eta_b, H0_FID)         for a in alpha_scan])
bb_arr = np.array([chi2_BAO(a, eta_b, H0_FID, RD_FID) for a in alpha_scan])

ax_chi.plot(alpha_scan, cc_arr - cc_arr.min(), 'o-', lw=1.5, label=r'$\Delta\chi^2_{\rm CC}$')
ax_chi.plot(alpha_scan, bb_arr - bb_arr.min(), 's-', lw=1.5, label=r'$\Delta\chi^2_{\rm BAO}$')
tot = cc_arr + bb_arr
ax_chi.plot(alpha_scan, tot - tot.min(), 'D-', lw=2, color='k', label=r'$\Delta\chi^2_{\rm tot}$')
ax_chi.axvline(0, ls='--', color='gray', alpha=0.6)
ax_chi.axhline(1,    ls=':',  color='gray', alpha=0.5)
ax_chi.axhline(4,    ls=':',  color='gray', alpha=0.5)
ax_chi.text(alpha_scan[0], 1.2, r'$1\sigma$', fontsize=8, color='gray')
ax_chi.text(alpha_scan[0], 4.2, r'$2\sigma$', fontsize=8, color='gray')
ax_chi.set(xlabel=r'$\alpha$', ylabel=r'$\Delta\chi^2$',
           title=fr'(c.2) $\Delta\chi^2(\alpha)$  ($\eta=\sqrt{{6}}$, $H_0,r_d$ fissi)')
ax_chi.grid(alpha=0.3); ax_chi.legend(fontsize=9); ax_chi.set_ylim(-0.3, 14); ax_chi.set_xlim(0,15)

plt.tight_layout()
out_png = os.path.join(_OUT_DIR, '01_background.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Figura: {out_png}\n")

# =========================================================================
# 7.  TABELLE NUMERICHE
# =========================================================================
print("=== Phantom crossings in z (roots di x^2-6x+1=0), varying eta, alpha=0.15 ===")
print(f"{'eta':>8} {'x_0':>7} {'z(x_+)':>10} {'z(x_-)':>10}")
for et in etas:
    zp = z_of_x(xp, alpha_a, et)
    zm = z_of_x(xm, alpha_a, et)
    zp_s = f"{zp:+.3f}" if np.isfinite(zp) else "futuro"
    zm_s = f"{zm:+.3f}" if np.isfinite(zm) else "futuro"
    print(f"{et:8.3f} {x0_from_eta(et):7.3f}  {zp_s:>9}  {zm_s:>9}")

print("\n=== chi^2 (H0, rd fissati a Planck LCDM; eta = sqrt(6)) ===")
print(f"  N_CC = {len(CC)},  N_BAO = {len(BAO)}")
print(f"{'alpha':>7} | {'chi2_CC':>10} | {'chi2_BAO':>10} | {'chi2_tot':>10}")
print("-"*48)
for al in alpha_scan:
    cc = chi2_CC(al, eta_b, H0_FID)
    bb = chi2_BAO(al, eta_b, H0_FID, RD_FID)
    star = "  *" if np.isclose(al, 0.0) else ""
    print(f"{al:+7.3f} | {cc:10.3f} | {bb:10.3f} | {cc+bb:10.3f}{star}")

# alpha migliore sul totale
i_best = np.argmin(tot)
print(f"\nalpha best-fit (CC+BAO) in scan: {alpha_scan[i_best]:+.3f}")
print(f"  Delta chi^2 vs LCDM (alpha=0): {tot[i_best] - tot[alpha_scan.tolist().index(0.0) if 0.0 in alpha_scan.tolist() else len(alpha_scan)//2]:+.3f}")
