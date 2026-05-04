"""
data_loaders.py — Caricamento unificato dei dataset cosmologici.

Tutti gli script MCMC importano da qui. I percorsi sono relativi rispetto
alla directory `scripts/`, cioe' i dati si assumono in `../data/`.

Se i dati si trovano altrove, basta modificare DATA_DIR all'inizio del file
oppure passare il path esplicitamente alle funzioni di caricamento.

Funzioni esportate:
  load_CC()               -> array (N,3): z, H, sigma_H
  load_BAO_DESI()         -> lista di dict con 'z', 'vals', 'kinds', 'icov'
  load_CMB_Planck()       -> (mean, icov)
  load_fsigma8()          -> array (N,3): z, fsigma8, sigma
  load_PantheonPlus()     -> (z, mb, cov_inv, Ainv_MB, Binv_MB)
  load_Pantheon2018_bin() -> (z, mb, cov_inv, Ainv_MB, Binv_MB)
"""
import numpy as np
import os

# Percorso base dei dati (relativo allo script che importa questo modulo)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _path(name):
    return os.path.join(DATA_DIR, name)


# =====================================================================
# Cosmic Chronometers
# =====================================================================
def load_CC(path=None):
    """
    Carica CC Moresco: 32 punti H(z), errori diagonali.
    Ritorna array shape (32, 3) con colonne [z, H, sigma_H].
    """
    path = path or _path("CC_Moresco.dat")
    data = np.loadtxt(path, comments='#')
    assert data.shape == (32, 3), f"Expected (32,3), got {data.shape}"
    return data


# =====================================================================
# DESI DR1 BAO
# =====================================================================
def load_BAO_DESI(path=None):
    """
    Carica BAO DESI DR1: 12 misure in 7 bin.
    Ritorna lista di dict, uno per bin, con chiavi:
      'z'     : redshift effettivo
      'vals'  : array di valori osservati
      'kinds' : lista di tipi ('DV'/'DM'/'DH')
      'sig'   : array di sigma
      'icov'  : inversa covarianza (1x1 o 2x2)
    """
    path = path or _path("BAO_DESI_DR1.dat")
    # Parsing manuale per gestire NaN e struttura a blocchi
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('z_eff'):
                continue
            parts = line.split()
            z = float(parts[0]); kind = parts[1]
            val = float(parts[2]); sig = float(parts[3])
            rho = float('nan') if parts[4] == 'NaN' else float(parts[4])
            rows.append((z, kind, val, sig, rho))

    # Raggruppo per z_eff
    blocks = {}
    for z, kind, val, sig, rho in rows:
        blocks.setdefault(z, []).append((kind, val, sig, rho))

    out = []
    for z, items in sorted(blocks.items()):
        kinds = [it[0] for it in items]
        vals = np.array([it[1] for it in items])
        sigs = np.array([it[2] for it in items])
        rho = items[0][3]  # rho uguale su tutte le righe del bin
        if len(items) == 1:
            icov = np.array([[1.0 / sigs[0]**2]])
        else:
            C = np.array([[sigs[0]**2, rho*sigs[0]*sigs[1]],
                          [rho*sigs[0]*sigs[1], sigs[1]**2]])
            icov = np.linalg.inv(C)
        out.append({'z': z, 'vals': vals, 'kinds': kinds,
                    'sig': sigs, 'icov': icov})
    assert len(out) == 7, f"Expected 7 BAO bins, got {len(out)}"
    return out


# =====================================================================
# CMB Planck 2018 compressed distance priors
# =====================================================================
def load_CMB_Planck(path=None):
    """
    Carica CMB distance priors (R, ell_A, omega_b) da Chen+2019.
    Ritorna (mean, icov):
      mean  : array (3,) = [R, ell_A, omega_b]
      icov  : array (3,3) inversa covarianza
    """
    path = path or _path("CMB_Planck2018_compressed.dat")
    mean = None; sig = None; corr_rows = []
    with open(path) as f:
        mode = None
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('R') and 'omega_b' in s:
                mode = 'mean_header'; continue
            if s.startswith('sigma_R'):
                mode = 'sig_header'; continue
            if s == 'correlation_matrix':
                mode = 'corr'; continue
            parts = s.split()
            if mode == 'mean_header':
                mean = np.array([float(p) for p in parts]); mode = None
            elif mode == 'sig_header':
                sig = np.array([float(p) for p in parts]); mode = None
            elif mode == 'corr':
                corr_rows.append([float(p) for p in parts])

    corr = np.array(corr_rows)
    assert mean.shape == (3,) and sig.shape == (3,) and corr.shape == (3, 3)
    cov = corr * np.outer(sig, sig)
    icov = np.linalg.inv(cov)
    return mean, icov


# =====================================================================
# fsigma8 Gold-18
# =====================================================================
def load_fsigma8(path=None):
    """
    Carica fsigma8(z) Gold-18-like: 18 punti.
    Ritorna array shape (18, 3) con colonne [z, fsigma8, sigma].
    """
    path = path or _path("fsigma8_Gold18.dat")
    data = np.loadtxt(path, comments='#')
    assert data.shape == (18, 3), f"Expected (18,3), got {data.shape}"
    return data


# =====================================================================
# Pantheon+ (1580 SNe dopo cut)
# =====================================================================
def load_PantheonPlus(dat_path=None, cov_path=None, z_min=0.01):
    """
    Carica Pantheon+SH0ES, applica cut z_HD >= z_min & !IS_CALIBRATOR.
    Ritorna (z, mb, cov_inv, Ainv_MB, Binv_MB) dove:
      z      : array shape (N,) di z_HD
      mb     : array shape (N,) di m_b_corr
      cov_inv: array shape (N,N) inversa covarianza STAT+SYS
      Ainv_MB: scalare = 1^T C^-1 1       (per marginalizzare M_B)
      Binv_MB: vettore  = C^-1 1           (per marginalizzare M_B)
    """
    dat_path = dat_path or _path("Pantheon_SH0ES.dat")
    cov_path = cov_path or _path("Pantheon_SH0ES_STAT_SYS.cov")

    D_full = np.genfromtxt(dat_path, skip_header=1,
                           usecols=(2, 8, 13), dtype=float)
    zHD = D_full[:, 0]; mb = D_full[:, 1]; is_cal = D_full[:, 2].astype(int)
    mask = (zHD >= z_min) & (is_cal == 0)
    idx = np.where(mask)[0]
    z_m = zHD[mask]; mb_m = mb[mask]; N = len(z_m)

    with open(cov_path) as f:
        N_cov = int(f.readline())
    C_full = np.loadtxt(cov_path, skiprows=1).reshape(N_cov, N_cov)
    C_m = C_full[np.ix_(idx, idx)]
    del C_full  # libera ~22 MB

    cov_inv = np.linalg.inv(C_m)
    ones = np.ones(N)
    Ainv_MB = ones @ cov_inv @ ones
    Binv_MB = cov_inv @ ones
    return z_m, mb_m, cov_inv, Ainv_MB, Binv_MB


# =====================================================================
# Pantheon 2018 binned (40 pt)
# =====================================================================
def load_Pantheon2018_bin(lc_path=None, sys_path=None):
    """
    Carica Pantheon 2018 binned: 40 punti.
    Ritorna (z, mb, cov_inv, Ainv_MB, Binv_MB) (stessa struttura di Pantheon+).
    """
    lc_path = lc_path or _path("lcparam_DS17f.txt")
    sys_path = sys_path or _path("sys_DS17f.txt")

    L = np.genfromtxt(lc_path, comments='#', usecols=(1, 4, 5))
    z = L[:, 0]; mb = L[:, 1]; dmb = L[:, 2]; N = len(z)
    assert N == 40

    with open(sys_path) as f:
        Ncov = int(f.readline())
    assert Ncov == N
    C_sys = np.loadtxt(sys_path, skiprows=1).reshape(N, N)
    C_stat = np.diag(dmb**2)
    C = C_stat + C_sys
    cov_inv = np.linalg.inv(C)
    ones = np.ones(N)
    return z, mb, cov_inv, ones @ cov_inv @ ones, cov_inv @ ones


def load_Pantheon2018_unbinned(lc_path=None, sys_path=None, z_min=0.01):
    """
    Carica Pantheon 2018 UNBINNED: 1048 SNe individuali.
    Ritorna (z, mb, cov_inv, Ainv_MB, Binv_MB) (stessa struttura di Pantheon+).

    File richiesti (da scaricare da https://github.com/dscolnic/Pantheon):
      lcparam_full_long.txt   (~100 KB,  1048 SNe)
      sys_full_long.txt       (~9 MB,    1048x1048 matrice sistematica)

    Formato lcparam_full_long.txt (19 colonne, separator whitespace):
      #name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar
      cov_m_s cov_m_c cov_s_c set ra dec biascor

    Colonne usate: zcmb (col 1), mb (col 4), dmb (col 5).
    (corrispondenti alle posizioni originali del repo dscolnic)
    """
    lc_path = lc_path or _path("lcparam_full_long.txt")
    sys_path = sys_path or _path("sys_full_long.txt")

    if not os.path.exists(lc_path):
        raise FileNotFoundError(
            f"File {lc_path} non trovato. Scaricalo da:\n"
            f"  https://github.com/dscolnic/Pantheon/raw/master/lcparam_full_long.txt\n"
            f"e salvalo in {os.path.dirname(lc_path)}/")
    if not os.path.exists(sys_path):
        raise FileNotFoundError(
            f"File {sys_path} non trovato. Scaricalo da:\n"
            f"  https://github.com/dscolnic/Pantheon/raw/master/sys_full_long.txt\n"
            f"e salvalo in {os.path.dirname(sys_path)}/")

    # Legge lcparam: comments='#' salta la riga header che inizia con '#name'
    L = np.genfromtxt(lc_path, comments='#', usecols=(1, 4, 5))
    z = L[:, 0]; mb = L[:, 1]; dmb = L[:, 2]
    N_full = len(z)
    # Ci si aspetta 1048; accettiamo anche 1049 se c'e' variazione di versione
    if N_full not in (1048, 1049):
        print(f"[warning] Pantheon 2018 unbinned: letti {N_full} SNe (atteso 1048)")

    # Taglio a z > z_min (come per Pantheon+ escludiamo le SNe molto locali
    # che non vincolano la cosmologia)
    mask = z > z_min
    z = z[mask]; mb = mb[mask]; dmb = dmb[mask]
    idx = np.where(mask)[0]
    N = len(z)

    # Legge covarianza sistematica: prima riga = N_full (numero totale),
    # poi N_full*N_full valori uno per riga.
    with open(sys_path) as f:
        Ncov = int(f.readline())
    assert Ncov == N_full, f"sys_full_long: atteso header {N_full}, letto {Ncov}"
    C_sys_full = np.loadtxt(sys_path, skiprows=1).reshape(N_full, N_full)

    # Applico lo stesso mask sulle righe e colonne della covarianza sistematica
    C_sys = C_sys_full[np.ix_(idx, idx)]

    # Covarianza statistica: diagonale degli errori osservazionali
    C_stat = np.diag(dmb**2)
    C = C_stat + C_sys

    cov_inv = np.linalg.inv(C)
    ones = np.ones(N)
    return z, mb, cov_inv, ones @ cov_inv @ ones, cov_inv @ ones


# =====================================================================
# DATASET EXTRA — in ../data_extra/ (paper-level upgrade)
# =====================================================================
DATA_EXTRA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "data_extra")


def _path_extra(name):
    return os.path.join(DATA_EXTRA_DIR, name)


# ---------------------------------------------------------------------
# DES-Y5 supernovae
# ---------------------------------------------------------------------
def load_DESY5(csv_path=None, covsys_path=None, z_min=0.01, dmu_cut=5.0,
               use_sys=True):
    """
    Carica DES-Y5 Hubble diagram (Vincenzi+24 / Sánchez+24).
    Formato: CSV con header CID,IDSURVEY,zCMB,zHD,zHEL,MU,MUERR_FINAL.
    1829 SNe totali (194 low-z + 1635 DES).

    Applica cut z_HD >= z_min ed esclude punti con errori enormi
    (MUERR_FINAL > dmu_cut).

    Se use_sys=True e covsys_000.txt e' disponibile, carica la covarianza
    sistematica completa 1829x1829 e costruisce C_TOT = C_STAT + C_SYS.
    Se use_sys=False o il file manca, usa solo STAT (errori diagonali).

    Convenzione DES-Y5:
      C_STAT = diag(MUERR_FINAL^2)   (errori statistici)
      C_SYS  = covsys_000.txt         (sistematici con off-diag)
      C_TOT  = C_STAT + C_SYS

    Ritorna (z, mu, cov_inv, Ainv_MB, Binv_MB), stessa struttura degli
    altri SN loader. `mu` sono distance moduli; marginalizzo un offset
    constant comune (equivalente a M_B per apparent magnitudes).
    """
    csv_path = csv_path or _path_extra("DESY5_HD.csv")
    covsys_path = covsys_path or _path_extra("DESY5_covsys.txt")

    # Carica tabella: indice originale serve per slicing della covarianza
    D = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                      usecols=(3, 5, 6), dtype=float)
    zHD = D[:, 0]; mu = D[:, 1]; dmu = D[:, 2]
    N_full = len(zHD)
    assert N_full == 1829

    mask = (zHD >= z_min) & (dmu < dmu_cut)
    idx = np.where(mask)[0]
    z_m = zHD[mask]; mu_m = mu[mask]; dmu_m = dmu[mask]
    N = len(z_m)

    # Statistical covariance (diagonale)
    C_stat = np.diag(dmu_m**2)

    # Systematic covariance, se disponibile
    if use_sys and os.path.exists(covsys_path):
        # File formato: prima riga = 1829, poi 1829^2 numeri
        C_sys_full = np.loadtxt(covsys_path, skiprows=1).reshape(N_full, N_full)
        C_sys = C_sys_full[np.ix_(idx, idx)]
        del C_sys_full  # libera ~25 MB
        C_tot = C_stat + C_sys
        sys_tag = "STAT+SYS"
    else:
        C_tot = C_stat
        sys_tag = "STAT-only"

    cov_inv = np.linalg.inv(C_tot)
    ones = np.ones(N)
    # Stampiamo tag solo alla prima chiamata (log)
    _load_DESY5_last_tag[0] = sys_tag
    return z_m, mu_m, cov_inv, ones @ cov_inv @ ones, cov_inv @ ones


_load_DESY5_last_tag = [""]  # placeholder for log


# ---------------------------------------------------------------------
# Union3 supernovae (binned, Rubin+2023)
# ---------------------------------------------------------------------
def load_Union3(dat_path=None, cov_path=None):
    """
    Carica Union3 binned: 22 bin di redshift.

    Format dat: #name zcmb zhel dz mb dmb ... (CosmoMC-style JLA).
    Colonna mb contiene i distance moduli (NON apparent magnitudes),
    gia' binnati.

    Format cov: prima riga = dimensione N=22, poi N*N numeri (matrice
    22x22 flat). Include stat + sys.

    Ritorna (z, mu, cov_inv, Ainv_MB, Binv_MB), stesso schema degli
    altri SN loader. Il parametro di offset da marginalizzare e' un
    constant overall shift.
    """
    dat_path = dat_path or _path_extra("Union3_data.txt")
    cov_path = cov_path or _path_extra("Union3_cov.txt")

    # Leggo solo colonne z (idx 1) e mu (idx 4)
    D = np.genfromtxt(dat_path, usecols=(1, 4), dtype=float)
    z = D[:, 0]; mu = D[:, 1]; N = len(z)
    assert N == 22

    # Covarianza
    with open(cov_path) as f:
        Ncov = int(f.readline())
    assert Ncov == N
    C = np.loadtxt(cov_path, skiprows=1).reshape(N, N)
    cov_inv = np.linalg.inv(C)
    ones = np.ones(N)
    return z, mu, cov_inv, ones @ cov_inv @ ones, cov_inv @ ones


# ---------------------------------------------------------------------
# DESI DR1 BAO — covarianza ufficiale 12x12
# ---------------------------------------------------------------------
def load_BAO_DESI_official(mean_path=None, cov_path=None):
    """
    Carica DESI DR1 BAO "ALL" combined con covarianza 12x12 ufficiale
    (Cobaya bao_data repo, desi_bao_dr1/).

    Struttura identica a load_BAO_DESI() ma con:
      - valori con piu' decimali (da file ufficiale)
      - covarianza con correlazioni misurate (non stimate)
      - 7 bin, 12 misure totali

    Ritorna lista di dict con 'z', 'vals', 'kinds', 'sig', 'icov',
    compatibile drop-in con load_BAO_DESI().
    """
    mean_path = mean_path or _path_extra("DESI_DR1_mean.txt")
    cov_path = cov_path or _path_extra("DESI_DR1_cov.txt")

    # Parse mean file
    rows = []
    with open(mean_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = s.split()
            z = float(parts[0]); val = float(parts[1])
            kind_raw = parts[2]
            # normalizzo nomi: 'DV_over_rs' -> 'DV'
            kind = kind_raw.replace('_over_rs', '').upper()
            rows.append((z, val, kind))

    # Carica covarianza piena 12x12
    C = np.loadtxt(cov_path)
    assert C.shape == (12, 12)
    # sigma diagonale
    sig_diag = np.sqrt(np.diag(C))
    assert len(rows) == 12

    # Raggruppo per z_eff mantenendo ordine originale
    # (serve per mappare correttamente gli indici nella covarianza)
    z_seen = []
    blocks_by_z = {}
    idx_by_z = {}
    for idx, (z, val, kind) in enumerate(rows):
        if z not in blocks_by_z:
            z_seen.append(z)
            blocks_by_z[z] = []
            idx_by_z[z] = []
        blocks_by_z[z].append((kind, val))
        idx_by_z[z].append(idx)

    out = []
    for z in z_seen:
        items = blocks_by_z[z]
        idxs = idx_by_z[z]
        kinds = [it[0] for it in items]
        vals = np.array([it[1] for it in items])
        # sub-matrice della covarianza per questo bin
        sub_C = C[np.ix_(idxs, idxs)]
        sub_sig = np.sqrt(np.diag(sub_C))
        sub_icov = np.linalg.inv(sub_C)
        out.append({'z': z, 'vals': vals, 'kinds': kinds,
                    'sig': sub_sig, 'icov': sub_icov})

    assert len(out) == 7
    return out


# =====================================================================
# Test standalone (python data_loaders.py)
# =====================================================================
if __name__ == "__main__":
    print("Test caricamento dataset (da", DATA_DIR, "):")
    CC = load_CC(); print(f"  CC:       shape {CC.shape}  z=[{CC[0,0]:.3f}, {CC[-1,0]:.3f}]")
    BAO = load_BAO_DESI(); print(f"  BAO:      {len(BAO)} bin, totale {sum(len(b['kinds']) for b in BAO)} misure")
    CMBmean, CMBicov = load_CMB_Planck(); print(f"  CMB:      mean={CMBmean}")
    FS8 = load_fsigma8(); print(f"  fsigma8:  shape {FS8.shape}")
    try:
        zP, mbP, _, _, _ = load_PantheonPlus()
        print(f"  Pan+:     {len(zP)} SNe, z=[{zP.min():.3f}, {zP.max():.3f}]")
    except FileNotFoundError as e:
        print(f"  Pan+:     file mancante ({e.filename})")
    try:
        zP2, mbP2, _, _, _ = load_Pantheon2018_bin()
        print(f"  Pan18:    {len(zP2)} SNe binned")
    except FileNotFoundError as e:
        print(f"  Pan18:    file mancante ({e.filename})")
    print("\n--- Dataset extra (per paper upgrade) ---")
    try:
        zDE, muDE, _, _, _ = load_DESY5()
        tag = _load_DESY5_last_tag[0] or "unknown"
        print(f"  DESY5:    {len(zDE)} SNe ({tag}), z=[{zDE.min():.3f}, {zDE.max():.3f}]")
    except (FileNotFoundError, IOError) as e:
        print(f"  DESY5:    file mancante")
    try:
        zU3, muU3, _, _, _ = load_Union3()
        print(f"  Union3:   {len(zU3)} bin, z=[{zU3.min():.3f}, {zU3.max():.3f}]")
    except (FileNotFoundError, IOError) as e:
        print(f"  Union3:   file mancante")
    try:
        BAO_full = load_BAO_DESI_official()
        print(f"  DESI-full: {len(BAO_full)} bin, cov 12x12 ufficiale")
    except (FileNotFoundError, IOError) as e:
        print(f"  DESI-full: file mancante")
    print("\nTutto OK.")
from data_loaders_DESI_DR2_patch import load_BAO_DESI_DR2 as load_BAO_DESI
