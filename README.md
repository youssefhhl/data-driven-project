# Prédiction du Risque Cardiovasculaire — Projet DDDM

## Contexte Métier

Ce projet s'inscrit dans le cadre du module **Data-Driven Decision Making**. Il vise à construire un système d'aide à la décision pour identifier les patients à risque élevé de maladie coronarienne, en combinant deux sources de données médicales complémentaires.

**Question décisionnelle** : *Quels patients sont à risque élevé de maladie coronarienne sur un horizon de 10 ans ?*

**Impact attendu** : Réduction des hospitalisations cardiaques non anticipées grâce à une détection précoce et un programme de prévention ciblé.

## Architecture du Projet

```
projet-dddm/
├── notebooks/
│   └── notebook.ipynb        # Analyse complète : 6 phases du pipeline décisionnel
├── dashboard/
│   └── app.py                # Dashboard Streamlit interactif (5 vues par profil)
├── docs/
│   ├── ab_test_plan.md       # Plan A/B test (protocole expérimental)
│   └── data_story.md         # Structure des 15 slides de présentation
├── models/
│   └── best_model.pkl        # Meilleur modèle sauvegardé (généré à l'exécution)
├── requirements.txt          # Dépendances Python versionnées
├── README.md                 # Ce fichier
└── .gitignore
```

## Sources de Données

| Dataset | Source | Lignes | Variables | Particularité |
|---------|--------|--------|-----------|---------------|
| Framingham Heart Study | Kaggle (`aasheesh200/framingham-heart-study-dataset`) | 4 240 | 15 | Suivi longitudinal 10 ans |
| Cardiovascular Disease | Kaggle (`sulianova/cardiovascular-disease-dataset`) | 70 000 | 12 | Classes équilibrées (50/50) |

Les données sont chargées dynamiquement via `kagglehub` — aucun téléchargement manuel nécessaire.

## Prérequis

- Python 3.10+
- Compte Kaggle configuré (clé API dans `~/.kaggle/kaggle.json`)

## Installation

```bash
git clone <url-du-depot>
cd projet-dddm

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancement

### Notebook Jupyter
```bash
cd notebooks
jupyter notebook notebook.ipynb
```

### Dashboard Streamlit
```bash
cd dashboard
streamlit run app.py
```
Accessible sur http://localhost:8501

## Méthodologie (6 Phases)

1. **Définition du problème** — KPI Tree, Business Case, estimation ROI
2. **Collecte & Audit** — Chargement kagglehub, harmonisation, fusion des 2 datasets
3. **EDA & Statistiques** — Distributions, corrélations, tests (Mann-Whitney, Chi²), clustering K-Means, détection outliers (IQR)
4. **Modélisation** — 3 modèles (LR, RF, XGBoost), GridSearchCV, validation croisée 5-fold, SHAP (global + local)
5. **Visualisation** — 5 vues décisionnelles par profil utilisateur (Direction, Opérations, Prévention, Clinique, Segmentation)
6. **Décision & Impact** — 3 recommandations, plan A/B test, estimation financière

## Résultats Clés

| Modèle | AUC-ROC | F1-Score | Recall | Precision |
|--------|---------|----------|--------|-----------|
| Régression Logistique | 0.776 | 0.692 | 0.674 | 0.712 |
| Random Forest | 0.808 | 0.720 | 0.692 | 0.749 |
| **XGBoost** | **0.810** | **0.724** | **0.707** | **0.741** |

## Description des Fichiers

| Fichier | Description |
|---------|-------------|
| `notebooks/notebook.ipynb` | Notebook complet avec les 6 phases, entièrement documenté (Markdown) et exécuté |
| `dashboard/app.py` | Dashboard Streamlit avec 5 vues interactives et filtres par profil |
| `docs/ab_test_plan.md` | Protocole A/B test : hypothèses, taille d'échantillon, durée, métriques, critères d'arrêt |
| `docs/data_story.md` | Structure de la présentation exécutive (15 slides) |
| `models/best_model.pkl` | Modèle XGBoost sérialisé avec scaler et métadonnées |
| `requirements.txt` | Dépendances Python avec versions minimales |

## Auteur

Projet réalisé dans le cadre du module DDDM — Date limite : 07 Juin 2026
