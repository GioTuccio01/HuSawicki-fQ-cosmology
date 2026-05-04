"""
data_loaders_DESY5_patch.py — VERSIONE CORRETTA (v2)
====================================================
Loader DES-Y5 / DES-Dovekie con covarianza STAT+SYS PIENA.

[FIX CRITICO rispetto a v1]
Il file STAT+SYS.npz fornito da DES contiene la INVERSA della covarianza,
NON la covarianza diretta!  Lo dice il likelihood ufficiale DES
(DES-Dovekie-SN_Likelihood.py riga 67):
    "Covtot_inv is the inverse of stat+sys because it makes no sense to
     invert covsys"

La v1 trattava il contenuto come se fosse C, e poi lo invertiva ancora,
ottenendo numeri sbagliati di un fattore ~1000.

Output (compatibile con la firma originale):
  z, mu, C_inv, A_inv_MB, B_inv_MB

dove:
  - z, mu sono array (N,)
  - C_inv è (N,N) la INVERSA della cov STAT+SYS, letta direttamente
    dal file (NO doppia inversione)
  - A_inv_MB = 1^T C^-1 1   (per marginalizzazione M)
  - B_inv_MB = C^-1 1       (per marginalizzazione M)
"""
import numpy as np
import os


def _read_snana_hd(csv_path):
    """Parser SNANA-style del file Hubble Diagram (righe SN: ...)."""
    rows = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith('SN:'):
                parts = line.split()
                rows.append(parts[1:])
    arr = np.array(rows)
    cols = ['CID', 'IDSURVEY', 'zHD', 'zHEL', 'MU', 'MUERR',
            'MUERR_VPEC', 'MUERR_SYS', 'PROBIA_BEAMS']
    return {c: arr[:, i] for i, c in enumerate(cols)}


def load_DESY5_full(data_dir=None,
                    csv_name="DES-Dovekie_HD.csv",
                    cov_name=None,
                    muerr_cut=None,
                    verbose=True):
    """
    Carica DES-Y5 / DES-Dovekie con la INVERSA della cov STAT+SYS piena.

    Il file STAT+SYS.npz contiene direttamente C^-1 in formato triangolare
    UPPER (np.triu_indices), come da likelihood ufficiale DES.

    Parameters
    ----------
    data_dir : str | None
        Directory dei file. Se None, cerca in posizioni standard.
    csv_name : str
        Nome del file CSV Hubble Diagram.
    cov_name : str | None
        Nome del file .npz con la INVERSA della covarianza.
        Se None, prova in ordine: 'STAT+SYS.npz', 'STAT_SYS.npz',
        'DESY5_STAT_SYS.npz'.
    muerr_cut : float | None
        Cut su MUERR (default None = nessun cut, come fa il likelihood
        ufficiale DES che usa solo zHD>0).
    verbose : bool
        Stampa info.

    Returns
    -------
    z, mu, C_inv, A_inv_MB, B_inv_MB
    """
    # --- trova directory ---
    if data_dir is None:
        for cand in ['./data', '../data', '/mnt/user-data/uploads']:
            cand_csv = os.path.join(cand, csv_name)
            if os.path.exists(cand_csv):
                data_dir = cand
                break
        if data_dir is None:
            raise FileNotFoundError(
                f"Non trovo {csv_name} in ./data, ../data, /mnt/user-data/uploads")

    # --- trova nome file cov ---
    if cov_name is None:
        for cand in ['STAT+SYS.npz', 'STAT_SYS.npz', 'DESY5_STAT_SYS.npz']:
            if os.path.exists(os.path.join(data_dir, cand)):
                cov_name = cand
                break
        if cov_name is None:
            raise FileNotFoundError(
                f"Non trovo nessun file cov (provati: STAT+SYS.npz, STAT_SYS.npz, "
                f"DESY5_STAT_SYS.npz) in {data_dir}")

    csv_path = os.path.join(data_dir, csv_name)
    cov_path = os.path.join(data_dir, cov_name)

    if verbose:
        print(f"  load_DESY5_full v2 (FIX: cov.npz contiene C^-1)")
        print(f"    csv: {csv_path}")
        print(f"    cov: {cov_path}")

    # --- leggi CSV ---
    hd = _read_snana_hd(csv_path)
    z = hd['zHD'].astype(np.float64)
    zhel = hd['zHEL'].astype(np.float64)
    mu = hd['MU'].astype(np.float64)
    muerr = hd['MUERR'].astype(np.float64)

    # Default: come nel likelihood ufficiale DES, mantengo SOLO zHD > 0
    # (cioè elimina solo eventuali sentinel a z=0; mantiene tutte le 1820 SNe)
    mask = z > 0.0
    if muerr_cut is not None:
        mask &= muerr < muerr_cut

    # --- leggi C^-1 dal .npz (formato triangolare upper) ---
    d = np.load(cov_path)
    nsn = int(d[d.files[0]][0])
    if nsn != len(z):
        raise ValueError(f"npz dichiara nsn={nsn}, csv ha {len(z)} SNe")

    inv_cov = np.zeros((nsn, nsn), dtype=np.float64)
    # np.triu_indices riempie SOLO la triangolare superiore in row-major
    inv_cov[np.triu_indices(nsn)] = d[d.files[1]].astype(np.float64)
    # Riflesso lower
    i_lower = np.tril_indices(nsn, -1)
    inv_cov[i_lower] = inv_cov.T[i_lower]

    # --- applica mask ---
    z = z[mask]
    mu = mu[mask]
    inv_cov = inv_cov[np.ix_(mask, mask)]

    N = len(z)
    if verbose:
        print(f"    {N} SNe usate (su {nsn} totali, mask: zHD>0"
              + (f", MUERR<{muerr_cut}" if muerr_cut else "") + ")")
        print(f"    z range: [{z.min():.4f}, {z.max():.4f}]")
        # Sanity check: C^-1 simmetrica e definita positiva
        sym = np.allclose(inv_cov, inv_cov.T, atol=1e-8)
        eig_min = np.linalg.eigvalsh(inv_cov).min()
        print(f"    C^-1 simmetrica? {sym}, min eigenvalue: {eig_min:.4e}")
        if eig_min <= 0:
            print(f"    !!! WARNING: C^-1 non def-positiva, ill-conditioned")

    # A_inv = 1^T C^-1 1, B_inv = C^-1 1 (per marginalizzazione di M)
    ones = np.ones(N)
    Binv_MB = inv_cov @ ones
    Ainv_MB = float(ones @ Binv_MB)

    if verbose:
        print(f"    Ainv_MB = {Ainv_MB:.4f}  (per marginalizzazione M)")

    return z, mu, inv_cov, Ainv_MB, Binv_MB


# Compatibilità retroattiva
def load_DESY5(data_dir=None):
    return load_DESY5_full(data_dir=data_dir)


if __name__ == "__main__":
    z, mu, Cinv, A, B = load_DESY5_full(verbose=True)
    print(f"\nN={len(z)}, mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    # Sanity: chi^2 con M marginalizzato per LCDM(Om=0.3, H0=70)
    c=299792.458; zg=np.linspace(1e-5, 1.5, 5000)
    E=np.sqrt(0.3*(1+zg)**3+0.7); cum=np.concatenate(([0.0], np.cumsum(0.5*(1/E[:-1]+1/E[1:])*np.diff(zg))))
    DM=(c/70)*np.interp(z, zg, cum); mu_th=5*np.log10((1+z)*DM)+25
    r = mu - mu_th
    chi2 = r @ Cinv @ r - (r @ B)**2 / A
    print(f"chi2 LCDM(Om=0.3, H0=70) M-marginalizzato: {chi2:.2f}  (atteso ~ N = {len(z)})")
