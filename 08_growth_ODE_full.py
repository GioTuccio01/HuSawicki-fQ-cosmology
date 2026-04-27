"""
08_growth_ODE_full.py — Validazione dell'approssimazione growth-index con ODE piena.

L'analisi principale usa f(z) ≈ [Om(z) * G_eff/G_N]^0.55 (Linder 2005).
Qui integro l'ODE completa per δ(N) e confronto.

ODE:
  δ''(N) + [2 + E'(N)/E(N)] δ'(N) - (3/2) Ω_m(N) (G_eff/G_N)(N) δ(N) = 0
con N = ln(a).
Condizione iniziale: δ=1, δ'=1 in era di materia (N_init = -6.9 ≈ ln(1e-3)).

Output:
  - tabella di confronto γ=0.55 vs ODE per i 18 punti fσ8
  - rapporto chi2_ODE / chi2_gamma al MAP
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import minimize
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import (load_CC, load_BAO_DESI, load_CMB_Planck,
                          load_PantheonPlus, load_fsigma8)

print("="*65)
print("VALIDAZIONE ODE PIENA vs APPROSSIMAZIONE GROWTH-INDEX")
print("="*65)

CC = load_CC()
BAO_BLOCKS = load_BAO_DESI()
CMB_MEAN, CMB_ICOV = load_CMB_Planck()
SN_Z, SN_MB, C_SN_inv, Ainv_MB, Binv_MB = load_PantheonPlus()
FS8 = load_fsigma8(); FS8_Z, FS8_V, FS8_S = FS8[:,0], FS8[:,1], FS8[:,2]

C_KMS = 299792.458; Z_STAR = 1089.80
ETA_FIX = np.sqrt(6.0); X0_FIX = 1.0

X_SAMP = np.unique(np.concatenate([
    np.linspace(1e-5, 0.3, 100), np.linspace(0.3, 50, 300), np.linspace(50, 15000, 80)]))
Z_LOW = np.linspace(1e-5, 15.0, 300); Z_HIGH = np.linspace(15.0, Z_STAR, 400)

def Lam_of(a, Om0):
    return 0.5*((1.0-Om0)*X0_FIX + a*(1+6*X0_FIX-3*X0_FIX*X0_FIX)/(1+X0_FIX)**3)
def solve_E(z_arr, alpha, Om0):
    Lam = Lam_of(alpha, Om0)
    R_s = 2*Lam - alpha*(1+6*X_SAMP-3*X_SAMP**2)/(1+X_SAMP)**3
    g_s = X_SAMP - R_s
    if not np.all(np.diff(g_s) > 0): return None, None, None
    tgt = Om0*X0_FIX*(1+np.asarray(z_arr,float))**3
    xs = np.interp(tgt, g_s, X_SAMP)
    return np.sqrt(xs/X0_FIX), xs, Lam

def compute_fsigma8_ODE(alpha, Om0, sigma8_0, z_vals):
    """
    Integra l'ODE completa per δ(N). Ritorna fsigma8 a z_vals.
    """
    # Background
    E_lo, xs, Lam = solve_E(Z_LOW, alpha, Om0)
    if E_lo is None:
        return np.full_like(z_vals, np.nan)
    # N = ln(a) = -ln(1+z)
    # Griglia in N dal passato (N_init ≈ -6.9) a oggi (N=0)
    N_init = np.log(1e-3)   # ≈ -6.9
    # Trasformo la griglia z_LOW in N (scartando z molto bassi che danno N>0)
    N_grid_z = -np.log(1 + Z_LOW)   # da -ln(16)≈-2.77 (z=15) a ~0 (z→0)
    # Estendo verso il passato con universo materia-dominato (E² ≈ Om0 e^{-3N})
    N_ext = np.linspace(N_init, N_grid_z.min()-0.01, 100)
    E_ext = np.sqrt(Om0 * np.exp(-3*N_ext))   # matter-dominated
    xs_ext = X0_FIX * E_ext**2   # x = x0 E²
    # G_eff/G_N = 1 in matter domination (x→∞ no correction)
    # Faccio join
    N_full = np.concatenate([N_ext, N_grid_z[::-1]])   # from past to present
    E_full = np.concatenate([E_ext, E_lo[::-1]])
    xs_full = np.concatenate([xs_ext, xs[::-1]])
    # Ordine crescente in N
    idx = np.argsort(N_full)
    N_full = N_full[idx]; E_full = E_full[idx]; xs_full = xs_full[idx]
    # E'/E = d(ln E)/dN
    lnE = np.log(E_full)
    dlnE_dN = np.gradient(lnE, N_full)
    # G_eff/G_N = 1 / (1 - α(3-x)/(1+x)^3)
    fQ_inv = 1.0 / (+1.0 + alpha*(3 - xs_full)/(1+xs_full)**3)
    Ge_over_GN = +fQ_inv   # conv A: G_eff/G_N = +1/f_Q (f_Q -> +1 in GR)
    # Ω_m(N) = Om0 * e^{-3N} / E^2
    Om_N = Om0 * np.exp(-3*N_full) / E_full**2

    # Interpolators
    from scipy.interpolate import interp1d
    dlnE_fn = interp1d(N_full, dlnE_dN, kind='cubic', fill_value='extrapolate')
    Ge_fn   = interp1d(N_full, Ge_over_GN, kind='cubic', fill_value='extrapolate')
    OmN_fn  = interp1d(N_full, Om_N, kind='cubic', fill_value='extrapolate')

    def rhs(N, y):
        d, dp = y
        return [dp,
                -(2 + dlnE_fn(N))*dp + 1.5*OmN_fn(N)*Ge_fn(N)*d]

    # Integro da N_init a N=0
    sol = solve_ivp(rhs, [N_init, 0.0], [1.0, 1.0],
                    t_eval=np.concatenate([[N_init], N_full[N_full>N_init], [0.0]]),
                    method='RK45', rtol=1e-8, atol=1e-10)
    if not sol.success:
        return np.full_like(z_vals, np.nan)
    delta_vals = sol.y[0]; ddelta_vals = sol.y[1]
    N_sol = sol.t
    delta_fn = interp1d(N_sol, delta_vals, kind='cubic', fill_value='extrapolate')
    ddelta_fn = interp1d(N_sol, ddelta_vals, kind='cubic', fill_value='extrapolate')

    # Per ogni z richiesto: calcola f(z) = δ'/δ, δ(z)/δ(0), fσ8
    delta_0 = delta_fn(0.0)
    N_req = -np.log(1 + z_vals)
    d_req = delta_fn(N_req)
    dp_req = ddelta_fn(N_req)
    f_z = dp_req / d_req
    sigma8_z = sigma8_0 * (d_req / delta_0)
    return f_z * sigma8_z


def compute_fsigma8_gamma(alpha, Om0, sigma8_0, z_vals, gamma=0.55):
    """Approssimazione growth-index standard per confronto."""
    E_lo, xs, Lam = solve_E(Z_LOW, alpha, Om0)
    if E_lo is None: return np.full_like(z_vals, np.nan)
    Om_z = Om0*(1+Z_LOW)**3 / E_lo**2
    fQ = +1 + alpha*(3-xs)/(1+xs)**3
    Ge = +1/fQ   # conv A: G_eff = +1/f_Q
    f_z_grid = (Om_z*Ge)**gamma
    cf = np.concatenate(([0.0], cumulative_trapezoid(f_z_grid/(1+Z_LOW), Z_LOW)))
    fs8_full = sigma8_0 * f_z_grid * np.exp(-cf)
    return np.interp(z_vals, Z_LOW, fs8_full)


# Test ai valori MAP trovati in precedenza
alpha_map = -0.154
Om_map = 0.308
s8_map = 0.801
print(f"\nConfronto ai valori MAP (alpha={alpha_map}, Om={Om_map}, s8={s8_map})")
print(f"{'z':>6} {'gamma=0.55':>14} {'ODE piena':>14} {'diff':>10} {'% diff':>10}")

z_test = FS8_Z
fs8_gamma = compute_fsigma8_gamma(alpha_map, Om_map, s8_map, z_test)
fs8_ODE   = compute_fsigma8_ODE(  alpha_map, Om_map, s8_map, z_test)
for i in range(len(z_test)):
    z = z_test[i]; fg = fs8_gamma[i]; fo = fs8_ODE[i]
    print(f"{z:>6.3f} {fg:>14.6f} {fo:>14.6f} {fo-fg:>+10.6f} {100*(fo-fg)/fg:>+9.3f}%")

diff_max = np.max(np.abs(fs8_ODE - fs8_gamma))
diff_rel = np.max(np.abs((fs8_ODE - fs8_gamma)/fs8_gamma))
print(f"\nMax diff assoluta: {diff_max:.5f}")
print(f"Max diff relativa: {diff_rel*100:.3f}%")

# Impact su chi2 fs8
c2_gamma = np.sum(((FS8_V - fs8_gamma)/FS8_S)**2)
c2_ODE   = np.sum(((FS8_V - fs8_ODE)/FS8_S)**2)
print(f"\nchi2(fs8) con gamma=0.55: {c2_gamma:.3f}")
print(f"chi2(fs8) con ODE piena:  {c2_ODE:.3f}")
print(f"Delta chi2: {c2_ODE - c2_gamma:+.4f}")

print("\n--- Validation test ---")
print("Se |Δχ²| < 0.5 l'approssimazione γ=0.55 è giustificata (Linder 2005).")
if abs(c2_ODE - c2_gamma) < 0.5:
    print("==> γ=0.55 è un'approssimazione corretta. OK per analisi principale.")
else:
    print("==> Differenza significativa! L'analisi principale va ripetuta con ODE piena.")
