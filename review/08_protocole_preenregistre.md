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

(Grilles étendues par l'amendement 1, en bas du fichier.)

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

## E4 — sensibilité à la largeur supposée et à λ (ajouté le 2026-11-16, avant tout calcul E4)

- **Données.** Toutes les familles qui modélisent la PSF (`psf_aware`, `rl2`,
  `tik_id`, `tik_lap`, `wiener_hunt`), à chaque largeur supposée f :
  - f ≥ 0,9 : run 2 d'E1 ;
  - f ∈ {0,6 ; 0,7 ; 0,8} : complément E4, mêmes grilles, sans E2.
- **Concurrent à f fixé.** Chaque famille est réglée par la validation croisée
  à 5 plis d'E1, restreinte à ce f. Pour chaque région, le concurrent retenu
  est le meilleur de ces concurrents réglés, choisi avec la référence. Ce
  choix favorise les concurrents.
- **Carte f × λ.** La méthode PSF-aware est prise à chaque (λ, itérations ∈
  {14, 30}), sans réglage, au même f que le concurrent. On rapporte la marge
  relative médiane (concurrent − PSF-aware) / concurrent et le nombre de
  régions gagnées, sur l'ensemble et par classe.
- **Courbes.** RMSE relatif au bicubique (médiane et IQR) de chaque famille
  réglée à chaque f.
- **Zone d'usage sûr.** Pour chaque (λ, itérations), c'est l'intervalle
  contigu de f autour de 1 où la méthode PSF-aware gagne au moins 90 % des
  régions.
- **Sorties.**
  - `analysis_<banc>/e4_width_lambda_map.csv`, `e4_width_curves.csv` et
    `e4_safe_zone.csv` ;
  - `figures/e4_*.pdf`.

## E3 — écart de réponse dû à l'ouverture du détecteur (ajouté le 2026-11-16, avant tout calcul E3)

**Question.** Le banc synthétique génère ses données avec une gaussienne de
12,5 m suivie d'une moyenne sur l'ouverture carrée de 10 m (`block_mean`
4×4). Le modèle gaussien de A ne représente pas cette ouverture. E3 chiffre
ce que coûte cet écart, pour la méthode PSF-aware comme pour ses concurrents.

**Bras (banc synthétique, les 40 régions d'E1, même référence et mêmes
cellules).**

| Bras | Données | Noyau de A (PSF-aware) | Noyau des déconvolutions | Calcul |
|------|---------|------------------------|--------------------------|--------|
| S0 | gaussienne 12,5 m puis moyenne 4×4 (article) | gaussienne 12,5 m | gaussienne 12,5 m | run 2 d'E1, f = 1 |
| S1 | gaussienne 12,5 m échantillonnée aux centres des pixels de 10 m | gaussienne 12,5 m | gaussienne 12,5 m | nouveau |
| S2 | comme S0 | gaussienne équivalente de 14,23 m (second moment) | gaussienne 14,23 m | run 2 d'E1, f = 1,138 |
| S3 | comme S0 | noyau exact gaussienne ⊗ boîte de 10 m | même noyau, échantillonné au pas de 10 m | nouveau |

- **Échantillonnage de S1.** La convolution gaussienne est évaluée
  directement aux centres des pixels de 10 m, sans interpolation.
  - Ces centres tombent à un demi-pixel de la grille fine de 2,5 m.
  - La gaussienne y est échantillonnée aux décalages demi-entiers, avec la
    même troncature (4 σ) et le même bord (`reflect`) que le `gaussian_filter`
    de l'article.
  - Contrôles (selftest) :
    - une constante est conservée ;
    - sur une gaussienne analytique échantillonnée, le résultat suit la
      convolution analytique à 1e-3 du pic près ;
    - le résultat suit l'interpolation spline cubique du `gaussian_filter`,
      sur une texture lisse, à 1e-3 près en RMS relatif.
- **Noyau exact de S3.** Pour σ = 5,31 m et a = 10 m :
  w(u, v) = Π_{t ∈ {u, v}} ½ [erf((t + a/2)/(√2 σ)) − erf((t − a/2)/(√2 σ))].
  - (u, v) est le déplacement du centre de cellule par rapport à
    l'échantillon dans le repère UTM du patch, comme lors de la
    génération : les deux points sont projetés avec le `pyproj` du patch.
  - w est normalisé à un pic de 1, comme la gaussienne, pour que
    `threshold` garde son sens.
  - La recherche de voisins (`sigma_m`) et `Npt` sont ceux de la gaussienne
    de 14,23 m (`recommend_npt`, masse cible 0,99). La masse du noyau exact
    captée par les liens est journalisée.
  - Le noyau discret des déconvolutions vaut w aux décalages entiers de
    10 m, sur un rayon de 3 pixels (≥ 4 σ_éq), normalisé à une somme de 1.
- **Point d'accroche.** Le paramètre `kernel=` de `PSFResampler` (poids brut
  d'un lien en fonction des lon/lat de l'échantillon et de la cellule).
  - Le chemin par défaut garde la même expression gaussienne.
  - On le vérifie par les tests du package et en relançant E0 synthétique
    après la modification, avec le critère d'E0.

**Méthodes et réglage.**
- **Familles calculées dans chaque bras :** `psf_aware` (grille λ ×
  itérations d'E1), `rl2` (`minmax`), `tik_id`, `tik_lap`, `wiener_hunt`.
  Les méthodes géométriques sont recalculées sur les données de S1.
- **Réglage :** chaque famille est réglée par la validation croisée à
  5 plis d'E1 à l'intérieur du bras. Le noyau est fixé dans un bras, comme
  en mode A.
- **Score de sélection :** RMSE / RMSE bicubique du même bras.

**Sorties.**
1. **Récupération par bras et par famille (réglée par CV).** On utilise
   l'équation de récupération de l'article,
   1 − RMSE(méthode, latent) / RMSE(flouté, latent). Son dénominateur ne
   dépend que de la texture et de la gaussienne : il est commun à tous les
   bras d'une région. On rapporte la moyenne et l'IC bootstrap, sur
   l'ensemble et par classe.
2. **À l'intérieur de chaque bras :** PSF-aware (CV) contre chaque
   concurrent (CV). Δ apparié, IC, victoires et test de signe, comme pour
   E1.
3. **D'un bras à l'autre, pour PSF-aware et RL (réglés par CV) :** Δ RMSE
   apparié pour quatre contrastes (positif : le second bras est meilleur) :
   - S0 − S3 : gain à modéliser l'ouverture ;
   - S0 − S2 : gain à la largeur équivalente ;
   - S2 − S3 : forme au-delà du second moment ;
   - S0 − S1 : coût total de l'ouverture, soit la perte d'information plus
     l'écart de modèle.
4. **Banc réel, sans nouveau calcul.** On compare R0 (f = 1, 25 m) et R2
   (f = √1,25, réponse combinée nominale de 27,95 m), PSF-aware contre le
   concurrent le plus fort au même f (tables d'E4).
5. **Fichiers :**
   - `analysis_synthetic/e3_recovery.csv`, `e3_within_arm.csv` et
     `e3_across_arms.csv` ;
   - `analysis_real/e3_real_widths.csv` ;
   - `figures/e3_arms.pdf`.

**Lecture fixée avant calcul.**
- **Si S0 − S3 est petit devant la marge de PSF-aware dans S0 :** l'ouverture
  non modélisée est un effet du second ordre, et le paragraphe « not exactly
  matched » du §IV-E le chiffre.
- **Si S3 est nettement meilleur que S0 :** l'article recommande d'inclure
  l'ouverture dans A quand elle est connue, et les chiffres de S0 sont
  conservateurs.
- **Classement des méthodes dans S1 et S3 :** il dit si l'avantage de
  PSF-aware dépend de l'approximation gaussienne.

## Journal des modifications après gel

### Amendement 1 — 2026-11-16 : extension des grilles ouvertes

**Constat (run 1, commit `b244c3f`, 40 + 40 régions).** La validation croisée
à 5 plis a retenu le bord d'une grille ouverte pour plusieurs familles :

| Famille | Bord retenu au run 1 |
|---------|----------------------|
| `tik_id`, `tik_lap` | λ = 1e-5 (minimum), les deux modes, les deux bancs |
| `rl2` | 500 itérations (maximum) ; mode B réel : f = 1,2 (maximum) |
| `tik_id`, `tik_lap`, `wiener_hunt` (mode B) | f = 1,2 (maximum) |
| `e2_mlem` | 200 itérations (maximum) |
| `e2_sirt` | 100 itérations et ω = 1,5 (maxima) |
| `e2_wcg`, `e2_cg_euclid`, `e2_jacobi`, `e2_tik_x0`, `e2_tik_dref` | 100 itérations (maximum), avec λ = 0 |

Une configuration au bord d'une grille ouverte sous-estime la meilleure
performance de la famille. Cela va contre les concurrents de la méthode
PSF-aware, ce qui biaise la comparaison en sa faveur.

**Règle adoptée.** Si la configuration retenue tombe au bord d'une grille
ouverte, on étend cette grille dans la direction du bord, d'un facteur au
moins 3, et on relance. Les bornes naturelles ne sont pas des bords ouverts :
λ = 0, et ω < 2 pour SIRT. Si la configuration retenue tombe encore au bord
de la grille étendue, on le signale tel quel dans l'article.

**Grilles étendues (run 2).**
- Largeurs : f ∈ {0,9 ; 1,0 ; √1,25 ou 1,138 ; 1,2 ; 1,3 ; 1,4 ; 1,5}, pour
  toutes les méthodes en mode B, PSF-aware comprise.
- `rl2` : itérations ∈ {5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000}.
- `tik_id`, `tik_lap` : λ ∈ logspace(−10, 0, 21).
- `e2_sirt` : ω ∈ {0,5 ; 1 ; 1,5 ; 1,9} × itérations ∈ {5, 14, 30, 100, 300,
  1000}.
- `e2_mlem` : itérations ∈ {5, 10, 20, 50, 100, 200, 500, 1000, 2000}.
- Bras E2 de type GC : itérations ∈ {5, 14, 30, 100, 300, 1000}.

**Grille inchangée pour `psf_aware`** (λ × {5, 14, 30, 100}) :
- mode A : le choix (λ = 0, 30 itérations) est intérieur en itérations ;
- mode B : le choix (λ = 1e-2, 100 itérations) est convergé, car 30 et
  100 itérations donnent le même RMSE à 1e-6 près ;
- à λ = 0, le solveur float32 du package diverge au-delà d'une trentaine
  d'itérations (RMSE 20 à 100 itérations). Prolonger les itérations
  n'apporterait rien.

**Résultat du run 2 (commit `8170b87`).**
- **Largeurs :** les baselines retiennent f = 1,2, désormais un point
  intérieur (la grille va jusqu'à 1,5). Les courbes en largeur sont en U :
  - `tik_id` réel, RMSE relatif médian : 0,847 à f = 1 ; 0,759 à f = 1,2 ;
    1,405 à f = 1,5 ;
  - `psf_aware` : minimum à √1,25 (réel) et à 1,138 (synthétique).
- **Autres bords :** les choix tombent encore au bord des grilles étendues
  (λ = 1e-10 pour Tikhonov ; 5000 itérations RL ; 2000 MLEM ; 1000 pour SIRT
  et les GC). Ce sont des plateaux : entre l'ancien et le nouveau bord, le
  RMSE relatif médian change de 0,04 % au plus. Pour Tikhonov, la limite
  λ → 0 est le filtre inverse exact. Sur la grille grossière, |H|² reste
  ≥ 0,02 dans la bande pour f ≤ 1,2.
- **Comparaisons :** les comparaisons appariées du run 2 diffèrent de celles
  du run 1 de moins de 0,15 point de marge relative. La seule exception est
  SIRT, qui, convergé, rejoint la solution du GC.

**Ce qui ne change pas.** Le run 1 est conservé tel quel dans
`notebooks/tables/revision_run1/`. Le score de sélection, les plis, les
graines et les statistiques sont inchangés. Les comparaisons de l'article
utilisent le run 2, et la réponse aux reviewers mentionne l'extension.
