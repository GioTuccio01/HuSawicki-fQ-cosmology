"""
10_BBN_numerical.py — Controllo numerico della conservazione di BBN.

Il modello f(Q) proposto è UV-soffice: a x→∞, la correzione scompare e si
recupera GR+Λ. Qui lo verifico numericamente calcolando il rapporto
    ΔH²/H²_GR = [H²(fQ) - H²_GR] / H²_GR
a redshift z ~ 10^9 (epoca BBN), per vari valori di α.

Vincolo BBN standard: |ΔH²/H²| < 10% a z_BBN per preservare le abbondanze
primordiali (He-4, D/H).
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("CONTROLLO BBN NUMERICO")
print("="*70)
print()
print("Modello: f(Q) = Q + 2Λ + α μ² (x-1)/(1+x)²   con x = Q/μ² = 6H²/μ²")
print()

# Per BBN siamo in era di radiazione. Uso:
#   H² ≈ H0² * Ω_r * (1+z)^4  con Ω_r ≈ 9.2e-5 (CMB + neutrini)
# e x = 6H²/μ² = (6 H0²/μ²) Ω_r (1+z)^4 = x0 * Ω_r * (1+z)^4 / Ω_m0
# ma a BBN la materia è ancora sub-dominante

OM_R = 9.2e-5       # densità di radiazione + neutrini
OM_M = 0.30         # materia (contribuisce poco a z~10^9)
ETA = np.sqrt(6.0)  # scelta canonica (x_0 = 1, eta = sqrt(6))
X0 = 6/ETA**2       # = 1
H0_eff = 1.0        # unità di H0

# ΔH²/H²_GR = R(x) / (H²_GR / H0² · x0) = R(x) / (E²_GR · x0) * correction
# ma piu' semplice: calcolo direttamente dalla Friedmann
# x - R(x) = Om * x0 * (1+z)^3 in materia-dominata,
# a BBN serve generalizzazione con radiazione

# Formulazione completa:
# H² = H²_GR (1 + δ)
# dove δ dipende da α
# 
# Equazione di Friedmann modificata in f(Q), convenzione A:
#   Q f_Q - f/2 = 8πG ρ_m = (x/x0) [Om(1+z)^3 + Om_r(1+z)^4]
# 
# Sostituendo f = Q + 2Λ + αμ²(x-1)/(1+x)²:
#   f_Q = +1 + α(3-x)/(1+x)³
#   f/2 = x/2 + Λ + αμ²(x-1)/(2(1+x)²)
#   x f_Q = x + αx(3-x)/(1+x)³
#   x f_Q - f/2 = x/2 - Λ + αx(3-x)/(1+x)³ - α(x-1)/(2(1+x)²)
# 
# Riscrivendo come x - R(x) con R(x) = 2Λ - α(1+6x-3x²)/(1+x)³:
#   x - R(x) = (x/x0) · [Om(1+z)^3 + Om_r(1+z)^4]
# 
# A BBN x è enorme. Espandiamo α·(terms) per x→∞:
#   αx(3-x)/(1+x)³ → -α/x  (dominante)
# La correzione f(Q) a H² è O(α/x), quindi per x→∞ (BBN) tende a zero: 
# il vincolo BBN è trivialmente soddisfatto per il nostro modello.

# In GR: x/2 - Λ_GR = (x/x0) · [Om(1+z)^3 + Om_r(1+z)^4]
# La costante Λ è fissata dalla chiusura oggi:
#   R(x0) = (1-Om0)*x0

# Calcolo direttamente x(z) da Newton-Raphson
def x_of_z_in_fQ(alpha, Om_m, Om_r, z, eta=None):
    if eta is None: eta = np.sqrt(6.0)
    """Risolve l'equazione di Friedmann modificata per x(z) in f(Q) + radiazione (conv A)."""
    x0 = 6/eta**2
    # Lambda normalizzata dal vincolo oggi (conv A: + alpha*):
    Lam = 0.5*((1.0-Om_m-Om_r)*x0 + alpha*(1+6*x0-3*x0*x0)/(1+x0)**3)
    # Target: rhs = (1/x0) * [Om_m (1+z)^3 + Om_r (1+z)^4]
    # LHS: x - R(x) = x - 2Λ + α(1+6x-3x²)/(1+x)³  (conv A: R = 2Λ - α...)
    target = Om_m*(1+z)**3 + Om_r*(1+z)**4
    # Newton-Raphson a partire da x_GR
    x_gr = x0 * target / (1-Om_m-Om_r + target) if (1-Om_m-Om_r + target) > 0 else x0*target
    x = max(x_gr, 1e-10)
    for it in range(100):
        Rx = 2*Lam - alpha*(1+6*x-3*x*x)/(1+x)**3   # conv A: -alpha
        f = (x - Rx) - x0*target
        # Derivata: dR/dx = -alpha * d/dx[(1+6x-3x²)/(1+x)³] = -alpha * 3(x²-6x+1)/(1+x)^4
        dRx = -alpha*(3*(x*x-6*x+1))/(1+x)**4
        df = 1 - dRx
        if abs(df) < 1e-15: break
        dx = f/df
        x_new = x - dx
        if x_new <= 0:
            x_new = x * 0.5
        if abs(dx/x) < 1e-12: break
        x = x_new
    return x

def x_of_z_GR(Om_m, Om_r, z, eta=None):
    if eta is None: eta = np.sqrt(6.0)
    x0 = 6/eta**2
    return x0 * (Om_m*(1+z)**3 + Om_r*(1+z)**4 + (1-Om_m-Om_r))

# Scan su alpha e z (conv A: alpha > 0 favorito; tengo entrambi i segni per check)
alphas = [-0.3, -0.135, -0.05, 0.0, 0.05, 0.135, 0.3]
z_vals = [1e4, 1e6, 1e8, 1e9, 3e9, 1e10]

print(f"Vincolo BBN: |ΔH²/H²_GR| < 10% a z ≈ 3e9")
print()
print(f"{'α':>10} " + " ".join(f"{'z=%.0e'%z:>12}" for z in z_vals))
print("-"*90)

for a in alphas:
    row = f"{a:>+10.3f} "
    for z in z_vals:
        x_f = x_of_z_in_fQ(a, OM_M, OM_R, z)
        x_g = x_of_z_GR(OM_M, OM_R, z)
        # H² = x * μ²/6 = x * H0² /η² · 6 / 6 = x H0² / η² / η² · 6...
        # In realtà H²/H0² = x / x0 = x * η² / 6. Quindi ΔH²/H² = Δx/x
        delta = (x_f - x_g) / x_g
        row += f"{delta:>+12.2e} "
    print(row)

print()
print("Conclusioni:")
print("  - A z -> 10^9 (BBN), Delta H^2/H^2 e' dell'ordine di alpha/x")
print("    (asintoticamente la correzione frazionaria a H^2 va come alpha mu^2/x H^2 ~ alpha/x).")
print("    Per alpha ~ 0.135 e z = 3e9, x ~ 6 Om_r (1+z)^4 ~ 5e34, quindi |Delta H^2/H^2| ~ 3e-36.")
print("    Totalmente trascurabile.")
print("  - BBN e' salvo per qualsiasi valore di alpha nel range fisico.")
print("  - La struttura UV-soffice del modello (correzione ~ alpha/x per x->infty) e' la ragione.")

# Calcolo analitico di sanity check
print("\n--- Check analitico a BBN ---")
z_bbn = 3e9
H2_over_H02 = OM_R * (1+z_bbn)**4
x_bbn = 6 * H2_over_H02
print(f"  z_BBN = {z_bbn:.0e}")
print(f"  H^2 / H0^2 ~ Om_r (1+z)^4 ~ {H2_over_H02:.3e}")
print(f"  x_BBN = 6 H^2 / mu^2 ~ {x_bbn:.3e}  (mu = H0 per eta=sqrt(6))")
# [FIX] formula corretta: la correzione frazionaria va come alpha/x, non alpha/x^2
print(f"  |Delta H^2/H^2| ~ alpha / x ~ {0.135/x_bbn:.3e}  (per alpha=0.135)")
print(f"  Il vincolo BBN |Delta H^2/H^2| < 0.1 e' soddisfatto di ~36 ordini di grandezza.")
