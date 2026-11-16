# Protocole pré-enregistré — E0, E1, E2 (révision JSTARS)

Ce fichier fixe les grilles, les plis, les critères et les sorties **avant**
tout calcul. Il doit être committé avant le premier run de production ;
ensuite, toute modification doit être datée et justifiée en bas du fichier.
Le code correspondant est `notebooks/revision_tools.py` (constantes en tête
de module) et `notebooks/revision_run.py`.

## Bancs d'essai

| Banc | Unité statistique | Référence | Observation | Grille cible |
|------|-------------------|-----------|-------------|--------------|
| `real` | 40 régions (1 extrait chacune, sélection et filtres de l'article : `gt.load_region_sites` puis `gt.screen_region_patches`) | niveau 19 construit à partir du B04 natif, avant dégradation (`gt.build_reference`, mode `psf_aware`) | B04 flouté par une gaussienne de 25 m puis décimé à 20 m (`gt.build_coarse_grid`) | L19, mêmes cellules et même marge intérieure (200 m) que l'article (`gt.align_and_mask`) |
| `synthetic` | 40 régions, 1 extrait chacune : exactement les `patch_id` de `tables/multi_patch_two_stage_40_metrics.csv` | texture Esri latente, moyennes de cellules L20 (`build_synthetic_patch`) | gaussienne 12,5 m ⊗ ouverture 10 m (`blurred_detector`) | L20, cellules alignées par identifiant (`align_estimate`) |

Les données et le code de génération sont ceux de l'article, exécutés tels
quels. Pour le banc synthétique, les cellules du notebook sont chargées par
marqueur ; aucune copie du code n'est faite.

## Facteurs de largeur supposée

- `real` : f ∈ {0,9 ; 1,0 ; √1,25 ≈ 1,118 ; 1,2} × 25 m. √1,25 correspond à
  la réponse combinée nominale de 27,95 m.
- `synthetic` : f ∈ {0,9 ; 1,0 ; 1,138 ; 1,2} × 12,5 m. 1,138 correspond à la
  gaussienne équivalente au second moment (14,23 m) de gaussienne ⊗ boîte.
- **Mode A :** f = 1 uniquement (largeur nominale imposée à toutes les
  méthodes).
- **Mode B :** f fait partie de la grille de réglage, pour *toutes* les
  méthodes qui modélisent la PSF.

## Méthodes et grilles (E1)

| Famille | Grille |
|---------|--------|
| `psf_aware` | λ ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1} × itérations ∈ {5, 14, 30, 100} ; tol 1e-8 |
| `rl2` (deux étapes : RL puis lecture bicubique en moyenne de cellule) | itérations ∈ {5, 10, 20, 50, 100, 200, 500} × normalisation ∈ {`clip`, `minmax`} ; `clip` = centiles 0,5–99,5 puis écrêtage [0, 1] (réel, article), `minmax` = min–max sans écrêtage (synthétique, article) |
| `tik_id` | Tikhonov de Fourier H*/(\|H\|² + λ), λ ∈ logspace(−5, 0, 11) |
| `tik_lap` | Tikhonov–Laplacien H*/(\|H\|² + λ\|L\|²), même grille |
| `wiener_hunt` | `skimage.restoration.unsupervised_wiener` (Laplacien, hyperparamètres estimés par Gibbs, graine fixe), aucun réglage |
| `geometric` | nearest, bilinear, bicubic (lecture en moyenne de cellule de l'article) |

## Schémas d'inversion (E2)

Ils utilisent les mêmes A et B, à f = 1, sur les mêmes cellules ; les
cellules non calculables valent NaN comme dans le package.

| Famille | Définition | Grille |
|---------|------------|--------|
| `e2_xref` | x_ref = B y | — |
| `e2_wcg` | GC pondéré (Dx) sur (BA + λI)δ = B(y − A x_ref) : réimplémentation, contrôle d'égalité avec le package | λ × itérations (grille E1) |
| `e2_cg_euclid` | même système, produit scalaire euclidien | idem |
| `e2_jacobi` | GC pondéré préconditionné par 1/diag(BA + λ) | idem |
| `e2_sirt` | x ← x + ω B(y − A x) depuis x_ref | ω ∈ {0,5 ; 1 ; 1,5} × itérations ∈ {5, 14, 30, 100} |
| `e2_tik_x0` | (AᵀA + λI)x = Aᵀy, GC euclidien depuis 0 | λ × itérations |
| `e2_tik_dref` | (AᵀA + λI)δ = Aᵀ(y − A x_ref) | λ × itérations |
| `e2_mlem` | MLEM sur HEALPix : x ← x ⊙ Aᵀ(y ⊘ Ax) ⊘ Aᵀ1, données décalées pour être positives (le décalage passe par A, car les lignes de A somment à 1) | itérations ∈ {5, 10, 20, 50, 100, 200} |

## Sélection des hyperparamètres

- **Score de sélection :** RMSE relatif = RMSE_méthode / RMSE_bicubique de la
  même région. Il donne le même poids à chaque classe.
- **Validation croisée stratifiée à 5 plis.** Dans chaque classe, les 10
  régions sont mélangées avec la graine 20261116, puis le pli vaut
  rang mod 5 (2 régions par classe et par pli).
  - Pour chaque famille, on choisit la configuration qui minimise la moyenne
    du score sur les 32 régions d'entraînement.
  - On l'applique aux 8 régions du pli de test.
  - Chaque région est donc notée avec une configuration choisie sans elle.
- **Variante par classe :** leave-one-region-out au sein de la classe
  (9 régions d'entraînement, 1 de test).
- **Oracle :** la meilleure configuration par région, choisie avec la
  référence. C'est une borne optimiste pour chaque famille.
- **Configuration « article » :**
  - `psf_aware` réel : f = 1, λ = 1e-2, 14 itérations ;
  - `psf_aware` synthétique : f = 1, λ = 1e-3, 14 itérations ;
  - RL : 100 itérations (`clip` en réel, `minmax` en synthétique) ;
  - Tikhonov : λ = 1e-2.

## Statistiques et sorties

- Δ apparié = RMSE_concurrent − RMSE_PSF-aware, par région.
- Moyenne et IC bootstrap percentile à 95 %, 5000 réplicats sur les régions
  (graine 20260818, comme dans l'article). Calculés sur l'ensemble et par
  classe.
- Nombre de victoires et test de signe binomial bilatéral exact.
- **Comparaisons rapportées :**
  - PSF-aware (article) contre chaque concurrent (article, CV, oracle) ;
  - PSF-aware (CV) contre chaque concurrent (CV, oracle).
- **Critère E0 (régression) :** écart relatif de RMSE ≤ 1e-4 par (région,
  méthode) par rapport aux CSV soumis. Au-delà, l'écart est documenté et
  toutes les tables sont relancées au commit gelé.
- Toutes les sorties vont dans `notebooks/tables/revision/` ; aucun CSV de
  l'article n'est écrasé.

## Journal des modifications après gel

(vide)
