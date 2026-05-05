# DES-Y5 systematic covariance file

Il file `DESY5_covsys.txt` (53 MB) non è incluso nel ZIP per motivi di dimensione.
Se hai ricevuto il ZIP senza questo file, l'analisi DES-Y5 funzionerà comunque
in modalità STAT-only (più rapida ma con significatività gonfiata).

## Per ottenere la covarianza sistematica completa

Scarica `covsys_000.txt` dal repository DES-SN5YR:
https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT

E mettilo in `data_extra/` rinominato come `DESY5_covsys.txt`.

Lo script `12_MCMC_DESY5.py` userà automaticamente la covarianza STAT+SYS
se il file è presente, altrimenti fallback a STAT-only con warning.
