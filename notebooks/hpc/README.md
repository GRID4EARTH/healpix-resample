# Révision JSTARS — exécution de E0, E1 et E2 sur le HPC

Protocole : `review/08_protocole_preenregistre.md`.
Code : `notebooks/revision_tools.py` (machinerie),
`notebooks/revision_run.py` (CLI) et `notebooks/revision_selftest.py`.
Sorties : `notebooks/tables/revision/`, uniquement des CSV et du JSON, de
quelques Mo. Aucune table de l'article n'est écrasée.

## À savoir avant de commencer

- **PyTorch est en version CPU.** `pixi.lock` fige `pytorch-2.13.0-cpu_mkl`,
  donc l'environnement `notebooks` tourne sur CPU. Pour E0–E2, ça ne change
  que la durée : les *résultats* sont identiques au GPU (float32, écarts de
  l'ordre de 1e-6, en dessous de la tolérance E0 de 1e-4). Il faut donc
  demander des **nœuds CPU**. Un GPU ne servira qu'à refaire les temps de
  calcul (benchmark L4), pas maintenant.
- **Les nœuds de calcul n'ont souvent pas Internet.** Toutes les étapes
  réseau (pixi, Zenodo, Esri) se font sur un nœud de login. Les jobs tournent
  ensuite hors ligne (`OFFLINE=True` par défaut côté synthétique : une tuile
  Esri manquante provoque une erreur au lieu d'un téléchargement).

## 1. Installation (nœud de login, une fois)

```bash
# pixi dans $HOME (ou un autre préfixe) ; cache sur l'espace de travail
curl -fsSL https://pixi.sh/install.sh | bash
export PIXI_CACHE_DIR=$WORKDIR/.pixi-cache        # à adapter à votre centre

git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
git checkout revision/jstars-r1                   # une fois la branche poussée
pixi install -e notebooks
```

## 2. Données (nœud de login, une fois ; environ 1,2 Go plus 0,3 Go)

```bash
# Archive Zenodo figée de l'article
pixi run -e notebooks jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir /tmp \
    notebooks/load_data_in_zenodo.ipynb
pixi run -e notebooks python notebooks/build_data_manifest.py --check \
    --doi 10.5281/zenodo.22210945

# Textures Esri des 40 extraits synthétiques (Wayback 26334, déterministe)
pixi run -e notebooks python notebooks/revision_run.py fetch-esri
```

## 3. Vérifications rapides (nœud de login ou job court)

```bash
pixi run -e notebooks python notebooks/revision_run.py selftest
pixi run -e notebooks python notebooks/revision_run.py e1e2 real --limit 1
pixi run -e notebooks python notebooks/revision_run.py e1e2 synthetic --limit 1
```

- `selftest` doit afficher **28/28 checks passed**.
- Les deux `--limit 1` traitent une région par classe ; ils servent de test
  de fumée.
- Ensuite, supprimer `notebooks/tables/revision/e1e2_*` (sinon ces 4 régions
  sont reprises telles quelles), ou passer `--force` au run complet.

## 4. Jobs

SLURM (PBS : `revision_pbs.sh`, avec `qsub -v STEP=...` et `-J 0-7`, plus
`NSHARD=8`) :

```bash
sbatch --export=ALL,STEP="e0 real"       notebooks/hpc/revision_slurm.sh
sbatch --export=ALL,STEP="e0 synthetic"  notebooks/hpc/revision_slurm.sh
# Après examen de E0 (voir ci-dessous) :
sbatch --export=ALL,STEP="e1e2 real"      --array=0-3 notebooks/hpc/revision_slurm.sh
sbatch --export=ALL,STEP="e1e2 synthetic" --array=0-7 notebooks/hpc/revision_slurm.sh
# Quand tous les shards sont finis (CPU, quelques minutes) :
pixi run -e notebooks python notebooks/revision_run.py analyze real
pixi run -e notebooks python notebooks/revision_run.py analyze synthetic
```

**Ordres de grandeur sur CPU (estimations, à confirmer par le test de
fumée) :**
- E0 réel : environ 30 min ;
- E0 synthétique : environ 30 min ;
- E1/E2 réel : 1–3 min par région ;
- E1/E2 synthétique : 3–8 min par région.

Avec 8 shards, tout tient dans la limite de 6 h. Chaque région est écrite dès
qu'elle est finie : un job interrompu reprend où il s'était arrêté.

## 5. Lire E0 (point de décision)

`e0 real` et `e0 synthetic` écrivent `tables/revision/e0_<banc>_comparison.csv`
et sortent avec le code 1 si un couple (région, méthode) s'écarte de plus de
1e-4 en relatif des CSV soumis.

- **Aucun écart :** le code actuel reproduit l'article, et E1/E2 sont
  directement comparables aux tables soumises.
- **Écarts :** ne pas lancer E1/E2 avant d'en avoir compris la source. Les
  changements de `knn.py` (gestion des ex-æquo) et de `psf.py` depuis le tag
  `jstars-submission` sont les premiers suspects. Ensuite, soit on fige un
  nouveau commit et on relance toutes les tables, soit on fait tourner les
  expériences de révision au commit de soumission.

## 6. Rapatrier les résultats

```bash
# depuis le poste local
rsync -av hpc:healpix-resample/notebooks/tables/revision/ notebooks/tables/revision/
```

Fichiers principaux :

| Fichier | Contenu |
|---------|---------|
| `analysis_<banc>/comparisons.csv` | Δ appariés, IC, victoires, test de signe (E1 et E2) |
| `analysis_<banc>/selected_configs.csv` | configurations choisies par pli, LOO et oracle |
| `analysis_<banc>/config_curves.csv` | RMSE relatif médian de chaque configuration (figures) |
| `e1e2_<banc>/*.csv` | résultats bruts par région |
| `provenance_*.json` | commit, versions et machine de chaque run |
