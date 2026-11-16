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

### glibc trop ancienne (ex. Datarmor, SLES 12, glibc 2.19)

`pixi.lock` exige glibc ≥ 2.28. Sur un système plus ancien, `pixi install`
refuse l'environnement (`__glibc >=2.28 ... does not match`). Il ne faut pas
forcer avec `CONDA_OVERRIDE_GLIBC` : les binaires planteraient au lancement.
On exécute pixi dans un conteneur qui ne fournit qu'une glibc récente. Le lock
est inchangé.

```bash
curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash   # binaire statique
module load singularity/3.6.4
singularity pull $DATAWORK/containers/noble-scm.sif docker://buildpack-deps:noble-scm
singularity exec --cleanenv -B /home1/datawork,/home1/scratch \
    $DATAWORK/containers/noble-scm.sif env \
    PIXI_CACHE_DIR=$DATAWORK/.pixi-cache \
    ~/.pixi/bin/pixi install -e notebooks
```

Toutes les commandes `pixi run` passent ensuite par le même
`singularity exec`. Le script PBS le fait lui-même quand on lui passe
`HPXREV_SIF` et `PIXI` (voir section 4).

L'image doit contenir **git** et des **certificats CA** :
- `buildpack-deps:noble-scm` (Ubuntu 24.04, glibc 2.39) contient les deux ;
- avec `ubuntu:24.04` nu, pixi plante (« No CA certificates were loaded
  from the system »), puis la construction du paquet échoue
  (setuptools-scm ne trouve pas de version sans git).

## 2. Données (nœud de login, une fois ; environ 230 Mo plus les textures Esri)

```bash
# Archive Zenodo figée de l'article
pixi run -e notebooks jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir /tmp \
    notebooks/load_data_in_zenodo.ipynb
# --output ailleurs : sinon le contrôle réécrit data_manifest.csv (suivi par git)
pixi run -e notebooks python notebooks/build_data_manifest.py --check \
    --doi 10.5281/zenodo.22210945 --output /tmp/data_manifest_check.csv

# Textures Esri des 40 extraits synthétiques (Wayback 26334, déterministe)
pixi run -e notebooks python notebooks/revision_run.py fetch-esri
```

## 3. Vérifications rapides (nœud de login ou job court)

```bash
pixi run -e notebooks python notebooks/revision_run.py selftest
pixi run -e notebooks python notebooks/revision_run.py e1e2 real --limit 1
pixi run -e notebooks python notebooks/revision_run.py e1e2 synthetic --limit 1
```

- `selftest` doit afficher **44/44 checks passed**.
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

Datarmor (PBS, file `omp`, environnement dans le conteneur), depuis la racine
du dépôt :

```bash
V=PIXI=$HOME/.pixi/bin/pixi,PIXI_CACHE_DIR=$DATAWORK/.pixi-cache
V=$V,HPXREV_SIF=$DATAWORK/containers/noble-scm.sif,SINGULARITY_MODULE=singularity/3.6.4
qsub -q omp -v "$V,STEP=e1e2 real,NSHARD=4"      -J 0-3 notebooks/hpc/revision_pbs.sh
qsub -q omp -v "$V,STEP=e1e2 synthetic,NSHARD=8" -J 0-7 notebooks/hpc/revision_pbs.sh
```

E4 (largeurs 0,6–0,8) et E3 (bras S1 et S3, banc synthétique) suivent le
même schéma ; leurs analyses lisent aussi les résultats d'E1 :

```bash
qsub -q omp -v "$V,STEP=e4 real,NSHARD=8"      -J 0-7 notebooks/hpc/revision_pbs.sh
qsub -q omp -v "$V,STEP=e4 synthetic,NSHARD=8" -J 0-7 notebooks/hpc/revision_pbs.sh
qsub -q omp -v "$V,STEP=e3 S1,NSHARD=8"        -J 0-7 notebooks/hpc/revision_pbs.sh
qsub -q omp -v "$V,STEP=e3 S3,NSHARD=8"        -J 0-7 notebooks/hpc/revision_pbs.sh
# puis, dans le conteneur :
pixi run --as-is -e notebooks python notebooks/revision_run.py e4-analyze
pixi run --as-is -e notebooks python notebooks/revision_run.py e3-analyze
```

E5 (SEN2VENµS, environ 35 Go). Le téléchargement se fait sur le nœud de
login, qui a accès à Internet ; le criblage tourne par site. Les fichiers
VENµS sont sous licence CC BY-NC 4.0 et ne doivent pas être redistribués.

```bash
pixi run --as-is -e notebooks python notebooks/revision_run.py e5-fetch      # login, ~1 h
qsub -q omp -v "$V,STEP=e5-manifest --shard 0/13" notebooks/hpc/revision_pbs.sh  # ... jusqu'à 12/13
pixi run --as-is -e notebooks python notebooks/revision_run.py e5-manifest   # tirage, sans --shard
qsub -q omp -v "$V,STEP=e5,NSHARD=32"            -J 0-31 notebooks/hpc/revision_pbs.sh
qsub -q omp -v "$V,STEP=e5 --part ext,NSHARD=32" -J 0-31 notebooks/hpc/revision_pbs.sh  # amendement 3
pixi run --as-is -e notebooks python notebooks/revision_run.py e5-analyze
pixi run --as-is -e notebooks python notebooks/revision_run.py e5-spectral   # E5-S, exploratoire, ~2 min
```

**Ordres de grandeur sur CPU (estimations, à confirmer par le test de
fumée) :**
- E0 réel : environ 30 min ;
- E0 synthétique : environ 30 min ;
- E1/E2 réel : 1–3 min par région ;
- E1/E2 synthétique : 3–8 min par région ;
- E5 : environ 3,5 min par extrait (grille principale) et 1,5 min
  (extension), soit environ 40 min et 16 min pour 307 extraits sur 32 shards.

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
