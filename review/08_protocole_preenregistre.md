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

## E5 — vérité indépendante inter-capteurs : SEN2VENµS (ajouté le 2026-11-16, avant tout calcul E5)

**Question.** Sur de vraies images Sentinel-2 à 10 m, sans dégradation
artificielle, quelle méthode reproduit le mieux une référence acquise le
même jour par un autre capteur, VENµS à 5 m ? Et avec quelle précision
faut-il connaître la largeur de la PSF ?

**Données.** SEN2VENµS v2.0.0 (Michel et al., *Data* 2022, 7, 96 ;
DOI 10.5281/zenodo.14603764).
- Bande B4 uniquement, valeurs en int16 / 10 000.
- Paires du même jour : Sentinel-2 L2A en extraits de 128 × 128 à 10 m,
  VENµS en extraits de 256 × 256 à 5 m. Les deux couvrent la même emprise
  de 1 280 m, en UTM ; les GeoTIFF sont géoréférencés.
- Les fichiers VENµS sont sous licence CC BY-NC 4.0 : on ne redistribue
  que les métriques et le script de téléchargement.

**Sites.** Les 13 sites dont l'angle zénithal de visée de VENµS est
≤ 25° (table 1 de la notice Zenodo) : FR-LQ1, NARYN, FGMANAUS, MAD-AMBO,
ARM, BAMBENW2, ES-IC3XG, ANJI, ATTO, ESGISB-3, ESGISB-1, FR-BIL et
K34-AMAZ. La résolution de VENµS se dégrade hors nadir ; ce critère garde
les références les plus nettes. Il ne dépend que des métadonnées.

**Filtres par extrait.** Ils portent sur les entrées seules, avant toute
méthode. Les zones intérieures s'entendent à plus de 4 σ du bord pour les
lissages.
1. **Invalide :** une valeur B4 < −1 000 ou > 15 000 dans l'une des deux
   images.
2. **Plat :** écart-type relatif de VENµS B4 < 1e-3 (critère du banc réel).
3. **Brillant :** plus de 30 % des pixels VENµS B4 au-dessus de 0,30 de
   réflectance (nuage ou neige ; critère du banc réel).
4. **Incohérence grossière :** corrélation < 0,9 entre Sentinel-2 et VENµS
   ramené à 10 m, les deux lissés par une gaussienne de 40 m de FWHM
   (changement de scène, nuage ou ombre résiduels).
5. **Co-registration :**
   - Le décalage est estimé par corrélation croisée entre Sentinel-2 B4 et
     VENµS B4 ramené à 10 m (amendement 2). Les deux images sont centrées et
     apodisées par une fenêtre de Hann ; le pic est affiné par DFT
     sur-échantillonnée (`skimage.registration.phase_cross_correlation`,
     `normalization=None`, sur-échantillonnage 100). VENµS est ramené à
     10 m par une gaussienne de 10 m de FWHM au pas de 5 m, puis par une
     moyenne 2 × 2.
   - L'extrait est exclu si |décalage| > 1 pixel Sentinel-2.
   - Il est aussi exclu si l'une des quatre estimations sur les quadrants de
     64 × 64 s'écarte de plus de 0,2 pixel de l'estimation sur l'extrait
     entier.

**Échantillonnage.**
- Dans chaque site, 24 extraits sont tirés parmi ceux qui passent les
  filtres (graine 20261116).
- Le tirage se fait à tour de rôle sur les paires (dates), prises dans un
  ordre aléatoire.
- Un site qui a moins de 24 extraits valides les garde tous.

**Référence.**
1. **Harmonisation radiométrique :** ajustement linéaire
   S2₄₀ = a · VEN₄₀ + b sur les images lissées à 40 m (10 m, zone
   intérieure). On l'applique ensuite à VENµS à 5 m. Seules les échelles
   ≥ 40 m entrent dans l'ajustement.
2. **Recalage :** le décalage estimé est appliqué aux positions où on lit
   VENµS. L'image n'est pas rééchantillonnée.
3. **Moyennes de cellules :**
   - Une cellule L20 (6,2 m) vaut la moyenne de VENµS harmonisé, vu comme
     constant par pixel de 5 m, aux centres de ses 64 sous-cellules L23
     (pixel contenant chaque centre).
   - L19 (12,4 m), en robustesse : la même chose sur ses 256 sous-cellules
     L23.
   - Ce code est indépendant des opérateurs PSF. `OverlapConservativeResampler`
     ne convient pas, car il exige une grille source rectiligne en lon/lat.
4. **Cellules évaluées :** celles dont le centre est à ≥ 200 m à l'intérieur
   de l'emprise de l'extrait.
5. **HEALPix :** indexation nested, ellipsoïde WGS84 (comme le banc réel).

**Méthodes.** Elles partent toutes des pixels Sentinel-2 B4 à 10 m.
- **`psf_aware` :**
  - centres des pixels → L20 (et L19) ;
  - gaussienne de largeur f × 12,5 m, l'approximation nominale de la
    réponse Sentinel-2 dans l'article ;
  - f ∈ {0,6 ; 0,7 ; … ; 1,5} ;
  - grille λ × itérations d'E1 ; `Npt` par `recommend_npt` (masse 0,99) ;
    `threshold` 0,1.
- **Géométriques :** plus proche voisin, bilinéaire, bicubique, lus aux
  centres des sous-cellules L22 (L21 pour L19) puis moyennés par cellule.
  C'est l'estimande de l'article.
- **Déconvolutions :** `rl2` (`clip` et `minmax`), `tik_id`, `tik_lap` et
  `wiener_hunt`, avec la gaussienne f × 12,5 m sur la grille de 10 m, les
  grilles d'E1 et la même lecture.

**Score et réglage.**
- **Par extrait :** RMSE contre la référence sur les cellules évaluées, et
  score relatif = RMSE / RMSE bicubique du même extrait.
- **Par site :** moyenne sur ses extraits. L'unité statistique est le site
  (n = 13).
- **Réglage principal : leave-one-site-out (13 plis).** Pour chaque famille,
  on retient la configuration qui minimise la moyenne, sur les sites
  d'entraînement, du score relatif par site. Deux modes :
  - mode A : f = 1 ;
  - mode B : f fait partie de la grille.
- **Réglage transféré (mode A) :** pour chaque famille, la configuration
  qui minimise le score relatif moyen sur les 40 régions synthétiques d'E1,
  à f = 1. C'est la même géométrie (10 m → L20).
- **Oracle :** la meilleure configuration par site.

**Statistiques.**
- **Δ apparié par site** = RMSE moyen du concurrent − RMSE moyen de
  PSF-aware.
- **Inférence :** IC bootstrap sur les sites (5 000 réplicats, graine
  20260818), nombre de sites gagnés et test de signe exact bilatéral.
- **À titre descriptif :** la fraction d'extraits gagnés.
- **Courbes en f :** à chaque f, chaque famille est réglée par
  leave-one-site-out restreint à ce f. On rapporte la médiane et l'IQR
  sur les sites du score relatif, et les victoires de PSF-aware contre le
  concurrent le plus fort au même f.
- **Niveaux :** L20 est le résultat principal ; L19 est rapporté en
  robustesse, avec les modes A et B en leave-one-site-out.

**Lecture fixée avant calcul.**
- **Marges plus faibles qu'en résolution réduite :** c'est attendu, car la
  réponse propre de VENµS floute la référence. Le classement (sites gagnés
  sur 13) est le résultat principal ; le RMSE absolu est secondaire.
- **Optimum de PSF-aware à f < 1 :** il serait cohérent avec cette
  référence floutée. La largeur relative attendue est d'environ
  √(12,5² − r²), où r est la largeur de la réponse de la référence.
- **Données insuffisantes :** si moins de 8 sites gardent au moins 8 extraits
  après les filtres, l'expérience est rapportée comme limitée, sans changer
  les critères.

**Sorties.**
- `tables/revision/e5_manifest.csv` : filtres, décalage et ajustement de
  chaque extrait ;
- `e5/<extrait>.csv` : résultats bruts ;
- `analysis_e5/` : tables ;
- `figures/e5_*.pdf` : figures.

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

### Amendement 2 — 2026-11-16, avant tout calcul E5 : estimateur du décalage

**Constat.** Le selftest d'E5 simule un décalage connu de (0,2 ; 0,3) pixel
entre deux images d'un champ continu. La corrélation de phase prévue au
départ (`skimage`, normalisation de phase, sans fenêtre) le sous-estime d'un
facteur 3 à 10 :
- sans fenêtre, les bords des deux extraits coïncident et tirent le pic vers
  zéro ;
- avec une fenêtre de Hann, la normalisation de phase donne encore le même
  poids aux hautes fréquences sans signal et reste biaisée vers zéro.

**Règle adoptée.** Corrélation croisée des images centrées et apodisées par
une fenêtre de Hann, sans normalisation de phase, affinée par DFT
sur-échantillonnée. Sur trois tirages du test, elle retrouve (0,20 ; 0,30)
pixel, avec un écart entre quadrants inférieur à 0,001 pixel.

**Portée.** Le changement ne touche que l'estimateur. Les seuils
d'exclusion (1 pixel, 0,2 pixel entre quadrants) sont inchangés. Aucun
extrait réel n'avait encore été filtré ni noté.

### Amendement 3 — 2026-11-16, après le run E5, avant tout calcul d'extension : grilles E5

**Constat (run E5, commit `eb73615`, 307 extraits, 13 sites).** Le
leave-one-site-out a retenu le bord d'une grille ouverte, aux niveaux L20
et L19 :

| Famille | Bord retenu |
|---------|-------------|
| `psf_aware` | mode A : λ = 0,1 (maximum) et 5 itérations (minimum) ; mode B : λ = 0,1 (maximum), avec f = 0,7 (intérieur) |
| `rl2` | 5 itérations (minimum), modes A et B ; mode B : f = 0,6 (minimum) |
| `tik_id`, `tik_lap` | mode B : f = 0,6 (minimum) ; `tik_lap` : λ = 1e-10, déjà le bord étendu par l'amendement 1 |
| `wiener_hunt` | mode B, L19 : f = 0,6 dans 6 plis sur 13 |

**Règle appliquée.** Celle de l'amendement 1, sans changement : on étend
chaque grille dans la direction du bord, d'un facteur au moins 3, et on
relance.

**Grilles ajoutées.** Seules les configurations nouvelles sont calculées
(`revision_run.py e5 --part ext`, sorties dans `tables/revision/e5_ext/`).
- **Largeurs des déconvolutions :** f ∈ {0,2 ; 0,3 ; 0,4 ; 0,5} pour `rl2`,
  `tik_id`, `tik_lap` et `wiener_hunt`, avec leurs grilles complètes. Quand
  f → 0, la déconvolution tend vers l'identité et la méthode rejoint la
  lecture bicubique.
- **`psf_aware` :**
  - λ ∈ {0,3 ; 1 ; 3 ; 10}, facteur 100 vers le haut, × itérations
    ∈ {1, 2, 5, 14, 30, 100} ;
  - itérations {1, 2} pour les λ d'origine ;
  - 0 itération, c'est-à-dire x_ref = B y. C'est la borne naturelle commune
    aux deux directions (λ → ∞ et itérations → 0). Elle ne dépend pas de λ
    et est enregistrée une fois, avec λ = 0.
- **`rl2` :** itérations {1, 2} à toutes les largeurs. 1 est la borne
  naturelle, car RL part d'une constante.
- **`psf_aware` garde f ≥ 0,6.**
  - Son choix (0,7) est intérieur.
  - Sous 0,6, l'opérateur ne couvre plus toutes les cellules évaluées : avec
    `threshold` 0,1, le rayon de lien vaut 2,15 σ, soit moins que la
    demi-diagonale de la grille de 10 m (7,07 m) dès f ≤ 0,5. Les scores
    porteraient alors sur moins de cellules que ceux des concurrents.
  - Si, après extension, le choix de f de `psf_aware` tombe à 0,6, on le
    signale tel quel.

**Ce qui ne change pas.** Le run E5 reste tel quel dans
`tables/revision/e5/`. Son analyse est conservée dans `analysis_e5_run1/`.
L'analyse étendue porte sur l'union des deux parties. Le score, les plis, les
graines, les statistiques et les filtres sont inchangés. Si une configuration
retenue tombe encore au bord d'une grille étendue, on le signale tel quel.

## Analyse exploratoire E5-S — 2026-11-16, après l'analyse E5 étendue (non pré-enregistrée)

**Pourquoi.** Après l'amendement 3, aucune méthode ne fait mieux que la
lecture bicubique contre VENµS. Les déconvolutions retiennent f → 0,2, donc
quasiment l'identité. La lecture fixée avant calcul attendait au contraire un
optimum à f < 1, mais pas nul. Deux causes possibles :
- **référence floue :** VENµS B4 n'a pas plus de hautes fréquences que
  Sentinel-2 B4 ;
- **incohérence entre capteurs :** VENµS en a davantage, mais elles ne sont
  pas prévisibles à partir de Sentinel-2. En cause : bruit, repliement de
  Sentinel-2, parallaxe, recalage local, réponses spectrales différentes.

**Statut.** Cette analyse a été décidée après avoir vu les résultats d'E5.
Elle est descriptive : elle ne change aucune sélection, aucune comparaison ni
aucune conclusion d'E5, ne comporte aucun test et sera présentée comme
exploratoire.

**Données.** Les 307 extraits d'E5, bande B4. L'harmonisation (gain a) et le
décalage sont ceux du manifeste E5.

**Calcul, par extrait.**
- Les deux images sont centrées, puis apodisées par la même fenêtre de Hann
  continue, sin²(π u / 1280 m).
- Leur transformée de Fourier est prise autour du coin commun de l'extrait.
- Le décalage E5 est appliqué à VENµS comme une rampe de phase, et son gain
  a comme un facteur.
- Les fréquences communes vont par pas de 1/1280 m⁻¹ jusqu'à la fréquence de
  Nyquist de Sentinel-2 (0,05 m⁻¹, bande carrée).

On somme par anneau radial (largeur 2/1280 m⁻¹) P_S = Σ|S|², P_V = Σ|V|² et
C = Σ S V̄, puis on additionne ces sommes sur les extraits de chaque site.

**Grandeurs par site et par anneau.**
- **Rapport d'amplitude :** R = √(P_V / P_S).
- **Cohérence :** γ² = |C|² / (P_S P_V).
- **Gain linéaire optimal :** |H| = |C| / P_S = γ R. C'est le filtre qui,
  appliqué à Sentinel-2, approche au mieux VENµS au sens des moindres carrés.
  Si |H| ≤ 1 dans toute la bande, la référence ne récompense aucun
  affûtage.
- **Recalage :** les sommes portent sur +k et −k, donc C est réel. Un
  défaut de recalage ne se voit pas comme une phase, mais comme une baisse
  de γ². On calcule donc aussi γ² sans décalage, et γ² avec le décalage E5
  affiné par extrait sur toute la bande (pic de la corrélation croisée,
  suréchantillonnée 8 fois). Si l'affinage relève γ² nettement, le recalage
  global est en cause ; sinon, ce sont les écarts locaux et les autres
  causes.
- **Cohérence de repliement :** la cohérence de Sentinel-2 en k avec VENµS
  aux fréquences partenaires k + p/(10 m), où p ∈ {(±1, 0), (0, ±1), (±1, ±1)}.
  Leur somme est une borne basse de la part de la puissance Sentinel-2 en k
  qui provient de contenu replié.

**Bilan d'erreur.** On le calcule par site, avec et sans filtre boîte de
6,2 m et de 12,4 m. Ce filtre approche les cellules L20 et L19 par des
carrés.
- Énergie VENµS hors de la bande de Sentinel-2, qu'aucune méthode linéaire
  invariante par translation ne peut prédire.
- Erreur dans la bande de l'identité (G = 1).
- Erreur dans la bande du filtre optimal du site (G = H). C'est une borne
  optimiste, puisque H est estimé sur les mêmes données.
- Rapport des RMSE totaux, filtre optimal sur identité : c'est le gain
  maximal qu'un filtre linéaire isotrope pourrait apporter.
- Contrôle de cohérence : le RMSE spectral de l'identité au niveau L20,
  comparé au RMSE bicubique mesuré par E5.

**Lecture (sans seuil de décision).**
- R ≈ 1, avec γ² élevé : la référence est floue.
- R > 1, avec γ² faible aux hautes fréquences : les capteurs sont
  incohérents.
- Les deux causes peuvent s'additionner.

**Sorties.** Elles restent hors de git, comme les autres tables :
- `tables/revision/analysis_e5_spectral/` ;
- `figures/e5_spectral.pdf`.

Seules des sommes spectrales agrégées sortent du module ; la licence VENµS
interdit de redistribuer les images.
