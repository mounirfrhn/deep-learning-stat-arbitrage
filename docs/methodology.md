# Méthodologie mathématique — de A à Z

Ce document décrit formellement **tout ce que fait la pipeline actuelle**
(version *benchmark*, sans deep learning), étape par étape, en collant au code
de `src/`. La dernière section esquisse l'extension CNN+Transformer à venir.

> Convention : les actions sont indexées par $n \in \{1,\dots,N\}$ et les jours
> de bourse par $t \in \{1,\dots,T\}$.

---

## 0. Notation et données

| Symbole | Signification | Code |
|---|---|---|
| $P_{n,t}$ | prix ajusté (dividendes/splits) de l'action $n$ au jour $t$ | `prices` |
| $r_{n,t}$ | rendement simple | `returns` |
| $\mathrm{rf}_t$ | taux sans risque journalier | `factors["RF"]` |
| $f_t \in \mathbb{R}^K$ | vecteur des $K$ facteurs Fama-French | `factors[...]` |
| $R_{n,t}$ | rendement **excédentaire** | `_excess_returns` |
| $\epsilon_{n,t}$ | **résidu** du modèle de facteurs | `residuals` |

---

## 1. Rendements

Rendement simple à partir des prix ajustés :

$$
r_{n,t} = \frac{P_{n,t}}{P_{n,t-1}} - 1
$$

Rendement **excédentaire** (au-dessus du taux sans risque) :

$$
R_{n,t} = r_{n,t} - \mathrm{rf}_t
$$

*(Code : `prices_to_returns` puis `_excess_returns`.)*

---

## 2. Construction des résidus (le cœur)

L'idée : retirer du rendement la part expliquée par le **risque systématique**.
Ce qui reste, le résidu $\epsilon_{n,t}$, est le « portefeuille d'arbitrage »
que l'on tradera. Deux modèles de facteurs sont implémentés.

> ⚠️ **Principe de non-anticipation (no look-ahead).** Le modèle utilisé pour
> le jour $t$ n'est estimé qu'avec l'information jusqu'à $t-1$. On n'utilise
> jamais le futur.

### 2.1. Modèle Fama-French (`fama_french_residuals`)

Sur une fenêtre glissante des $w$ derniers jours ($s \in [t-w,\, t-1]$, avec
$w = $ `ff_loadings_window` $= 60$), on régresse par **MCO** le rendement
excédentaire de chaque action sur les facteurs :

$$
R_{n,s} = \alpha_n + \beta_n^\top f_s + u_{n,s},
\qquad s = t-w,\dots,t-1
$$

En empilant la fenêtre, on résout pour chaque action :

$$
(\hat\alpha_n, \hat\beta_n) = \arg\min_{\alpha,\beta}
\sum_{s=t-w}^{t-1} \big( R_{n,s} - \alpha - \beta^\top f_s \big)^2
$$

Puis on calcule le résidu **hors échantillon** du jour $t$ :

$$
\boxed{\ \epsilon_{n,t} = R_{n,t} - \big(\hat\alpha_n + \hat\beta_n^\top f_t\big)\ }
$$

*(C'est la procédure de Carhart 1997. Dans le code, l'estimation est
vectorisée sur toutes les actions via `np.linalg.lstsq`.)*

### 2.2. Modèle PCA (`pca_residuals`, à la Avellaneda-Lee 2010)

Ici les facteurs ne sont pas donnés : on les **extrait** des données par analyse
en composantes principales.

**(a) Facteurs latents.** Sur une fenêtre de $W = $ `pca_corr_window` $= 252$
jours, on standardise les rendements excédentaires des actions valides :

$$
Z_{n,s} = \frac{R_{n,s} - \mu_n}{\sigma_n},
\qquad
\mu_n = \text{moyenne}_W(R_n),\quad \sigma_n = \text{écart-type}_W(R_n)
$$

On forme la **matrice de corrélation** $C \in \mathbb{R}^{N\times N}$ et sa
décomposition spectrale :

$$
C = Q \Lambda Q^\top, \qquad
\Lambda = \mathrm{diag}(\lambda_1 \le \dots \le \lambda_N)
$$

On garde les $k = $ `pca_num_factors` **plus grandes** valeurs propres ; leurs
vecteurs propres $q_1,\dots,q_k$ définissent les **portefeuilles propres**
(eigen-portfolios). Les poids de chaque facteur reviennent à l'échelle des
rendements :

$$
\omega_j = \frac{q_j}{\sigma}, \qquad j = 1,\dots,k
\quad(\text{division composante par composante})
$$

**(b) Rendements des facteurs.** Le rendement du facteur $j$ est le rendement du
portefeuille propre correspondant :

$$
F_{j,s} = R_{\cdot,s}^\top \, \omega_j
$$

**(c) Résidualisation.** Sur une fenêtre courte ($w = $ `pca_loadings_window`
$= 60$), on régresse chaque action sur ces facteurs latents pour obtenir ses
chargements $\beta_n$, puis :

$$
\boxed{\ \epsilon_{n,t} = R_{n,t} - \big(\hat\alpha_n + \hat\beta_n^\top F_t\big)\ },
\qquad F_t = \big(R_{\cdot,t}^\top \omega_1,\dots,R_{\cdot,t}^\top \omega_k\big)
$$

---

## 3. Prix résiduel cumulé

Le signal ne regarde pas les résidus bruts mais leur **cumul** — le « prix »
du portefeuille résiduel (opérateur d'intégration $\mathrm{Int}$ du papier) :

$$
x_{n,t} = \sum_{s \le t} \epsilon_{n,s}
$$

*(Code : `_cumulative` = `residuals.cumsum()`.)* Intuitivement, si le résidu
mean-reverte, alors $x_{n,t}$ oscille autour d'une tendance et revient vers sa
moyenne locale.

---

## 4. Signal (benchmark mean-reversion)

Sur une fenêtre de $L = $ `lookback` $= 30$ jours, on calcule le **z-score** du
prix résiduel cumulé :

$$
z_{n,t} = \frac{x_{n,t} - \text{moyenne}_L(x_{n})}{\text{écart-type}_L(x_{n})}
$$

Le signal parie sur le **retour à la moyenne** (position opposée à l'écart),
borné pour la robustesse :

$$
\boxed{\ s_{n,t} = -\,\mathrm{clip}(z_{n,t},\, -3,\, 3)\ }
$$

- $z$ très **positif** (résidu monté trop haut) $\Rightarrow s<0$ : on **vend**.
- $z$ très **négatif** (trop bas) $\Rightarrow s>0$ : on **achète**.

*(Code : `mean_reversion_signal`. C'est précisément ce bloc que le
CNN+Transformer remplacera — voir §8.)*

---

## 5. Allocation (politique de portefeuille)

On transforme les signaux bruts en **poids** de portefeuille.

**(a) Neutralité dollar** (marché-neutre) — on retire la moyenne transversale :

$$
\tilde{s}_{n,t} = s_{n,t} - \frac{1}{N_t}\sum_{m=1}^{N_t} s_{m,t}
\quad\Longrightarrow\quad \sum_n \tilde{s}_{n,t} = 0
$$

**(b) Normalisation au levier cible** $\ell = $ `max_gross_leverage` :

$$
\boxed{\ w_{n,t} = \ell \cdot \frac{\tilde{s}_{n,t}}{\sum_m |\tilde{s}_{m,t}|}\ }
\qquad\Longrightarrow\qquad \sum_n |w_{n,t}| = \ell
$$

Le portefeuille est donc **dollar-neutral** ($\sum_n w_{n,t} = 0$ : autant de
long que de short) et de **levier brut** contrôlé.

*(Code : `signals_to_weights`. C'est une règle fixe, pas une optimisation —
contrairement au papier, voir §8.)*

---

## 6. Backtest

On simule la stratégie jour par jour, **sans anticipation** : le poids décidé à
la clôture du jour $t-1$ est détenu sur le jour $t$ et gagne le résidu de $t$.

**Rendement brut de la stratégie :**

$$
\rho_t^{\text{brut}} = \sum_{n=1}^{N} w_{n,t-1}\, \epsilon_{n,t}
$$

**Turnover** (rotation du portefeuille) :

$$
\tau_t = \sum_{n=1}^{N} \big| w_{n,t} - w_{n,t-1} \big|
$$

**Coûts de transaction** (en points de base, $c_{\text{bps}} = 5$) :

$$
\text{coût}_t = \tau_t \cdot \frac{c_{\text{bps}}}{10^4}
$$

**Rendement net :**

$$
\boxed{\ \rho_t = \rho_t^{\text{brut}} - \text{coût}_t\ }
$$

*(Code : `run_backtest`, où `w.shift(1)` encode le décalage $t-1 \to t$.)*

---

## 7. Métriques de performance

Sur la série de rendements nets $\{\rho_t\}$ ($252$ jours de bourse par an) :

$$
\text{Rendement annualisé} = 252 \cdot \overline{\rho}
\qquad
\text{Volatilité annualisée} = \sqrt{252}\cdot \hat{\sigma}_\rho
$$

$$
\boxed{\ \text{Sharpe} = \sqrt{252}\cdot \frac{\overline{\rho}}{\hat{\sigma}_\rho}\ }
$$

**Max drawdown** — pire perte du sommet au creux. Avec l'équité cumulée
$E_t = \prod_{s\le t}(1+\rho_s)$ :

$$
\text{MaxDD} = \min_t \left( \frac{E_t}{\max_{s\le t} E_s} - 1 \right)
$$

*(Code : `metrics.py`.)*

---

## 8. Ce que fait le papier en plus (extension à venir)

Le benchmark ci-dessus utilise un **signal fixe** (§4) et une **allocation
fixe** (§5), indépendants. Le papier les remplace par des fonctions
**apprises et optimisées conjointement**.

**Signal appris** — une fonction $\theta$ (CNN+Transformer) qui prend le vecteur
des $L$ résidus cumulés et sort un signal :

$$
s_{n,t} = \theta\big(x_{n,t-L+1},\dots,x_{n,t}\,;\, \phi\big)
$$

**Allocation apprise** — une fonction $w(\cdot)$ paramétrée, sous contrainte de
levier.

**Objectif** — on optimise les paramètres $\phi$ pour **maximiser directement le
Sharpe** hors échantillon, ce qui revient à la perte :

$$
\boxed{\ \mathcal{L}(\phi) = -\,\sqrt{252}\cdot
\frac{\overline{\rho}(\phi)}{\hat{\sigma}_\rho(\phi)}\ }
$$

C'est la différence centrale : au lieu de règles codées en dur, le réseau
**apprend** le signal et l'allocation qui maximisent le rendement ajusté du
risque. (À implémenter dans `models.py` / `policy.py`.)

---

## Récapitulatif du flux

$$
P_{n,t}
\;\xrightarrow{\text{§1}}\; r_{n,t}, R_{n,t}
\;\xrightarrow{\text{§2}}\; \epsilon_{n,t}
\;\xrightarrow{\text{§3}}\; x_{n,t}
\;\xrightarrow{\text{§4}}\; s_{n,t}
\;\xrightarrow{\text{§5}}\; w_{n,t}
\;\xrightarrow{\text{§6}}\; \rho_t
\;\xrightarrow{\text{§7}}\; \text{Sharpe}
$$

### Références
- Guijarro-Ordóñez, Pelger, Zanotti (2021), *Deep Learning Statistical Arbitrage*, arXiv:2106.04028.
- Avellaneda & Lee (2010), *Statistical Arbitrage in the US Equities Market*.
- Carhart (1997), *On Persistence in Mutual Fund Performance*.
