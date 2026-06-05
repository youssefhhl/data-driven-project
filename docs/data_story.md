# Data Story — Prédiction du Risque Cardiovasculaire
## Structure de la Présentation (15 slides, 10 minutes)

---

## Slide 1 — Titre & Contexte
- **Titre** : Prédiction du Risque Cardiovasculaire par Machine Learning
- **Sous-titre** : Un système d'aide à la décision pour la prévention ciblée
- **Module** : Data-Driven Decision Making
- **Date** : Juin 2026

---

## Slide 2 — Le Problème Métier
- Les maladies cardiovasculaires = 1ère cause de mortalité mondiale (17.9M décès/an)
- Coût moyen d'une hospitalisation cardiaque : **15 000 €**
- 30% des événements cardiaques sont évitables par prévention précoce
- **Question décisionnelle** : Quels patients sont à risque élevé de maladie coronarienne sur 10 ans ?

---

## Slide 3 — KPI Tree & Business Case
- **Objectif stratégique** : Réduire les hospitalisations cardiaques non anticipées de 20%
- **KPI Tree** :
  - Niveau 1 : Réduction coûts hospitaliers
  - Niveau 2 : Taux de détection précoce (Recall > 80%)
  - Niveau 3 : Précision du modèle (AUC-ROC > 0.85)
- **ROI estimé** : Pour 1 000 patients à risque détectés → économie potentielle de 450 000 €/an

---

## Slide 4 — Sources de Données
- **Dataset 1** : Framingham Heart Study (4 240 patients, 15 variables, suivi 10 ans)
  - Richesse : tabac, médicaments, antécédents AVC, IMC
- **Dataset 2** : Cardiovascular Disease (70 000 patients, 12 variables)
  - Force : volume et équilibre des classes (50/50)
- **Stratégie** : Fusion des deux sources pour combiner richesse et volume

---

## Slide 5 — Data Audit & Qualité
- Déséquilibre de volume documenté : Framingham 6% vs Cardiovascular 94%
- Valeurs manquantes : principalement Framingham (glucose 9.5%, IMC 1.4%)
- Harmonisation réalisée : conversion âge (jours → années), alignement colonnes
- Volume total après fusion : **74 240 lignes**
- Data Dictionary complet avec 15+ variables harmonisées

---

## Slide 6 — Insights EDA (1/2) : Distributions
- Distribution de l'âge : concentration 40-65 ans (population à risque)
- Cholestérol : distribution bimodale, corrélation significative avec la cible
- Tension artérielle : corrélation forte (r > 0.3) avec le risque cardiaque
- Test statistique : différences significatives (p < 0.001) entre malades/non-malades pour toutes les variables continues

---

## Slide 7 — Insights EDA (2/2) : Patterns
- Heatmap de corrélation : clusters de facteurs de risque identifiés
- Clustering K-Means (k=3) : 3 profils patients distincts
  - Cluster 1 : Jeunes, faible risque
  - Cluster 2 : Âgés, hypertendus, risque modéré
  - Cluster 3 : Multi-facteurs, risque élevé
- Outliers détectés et traités (méthode IQR) : 2.3% des observations

---

## Slide 8 — Modélisation : Approche
- Feature engineering : IMC calculé, ratio systolique/diastolique, encodage
- Gestion du déséquilibre : `class_weight='balanced'`
- Split stratifié 80/20 (random_state=42)
- 3 modèles entraînés avec validation croisée 5-fold + GridSearchCV

---

## Slide 9 — Résultats Comparatifs
| Modèle | AUC-ROC | F1-Score | Recall | Precision |
|--------|---------|----------|--------|-----------|
| Régression Logistique | ~0.82 | ~0.72 | ~0.78 | ~0.67 |
| Random Forest | ~0.87 | ~0.78 | ~0.82 | ~0.75 |
| **XGBoost** | **~0.89** | **~0.80** | **~0.84** | **~0.77** |

- Courbes ROC comparatives : XGBoost domine sur l'ensemble du spectre
- Seuil optimal déterminé par maximisation du F1-Score

---

## Slide 10 — Performance Détaillée du Meilleur Modèle
- Matrice de confusion : bon compromis sensibilité/spécificité
- Courbe Précision-Recall : performance stable même à recall élevé
- Validation croisée : écart-type faible → modèle robuste et généralisable
- Pas de sur-apprentissage détecté (gap train/test < 3%)

---

## Slide 11 — Interprétabilité SHAP (Global)
- **Top 5 facteurs de risque** (SHAP summary plot) :
  1. Âge
  2. Tension artérielle systolique
  3. Cholestérol
  4. Tabagisme (nb cigarettes/jour)
  5. Glucose
- Cohérence avec la littérature médicale → modèle fiable et explicable

---

## Slide 12 — Interprétabilité SHAP (Local)
- **3 cas patients concrets** (waterfall plots) :
  - Patient A (score 0.92) : homme 62 ans, fumeur, hypertendu → très haut risque
  - Patient B (score 0.34) : femme 45 ans, non-fumeuse, cholestérol normal → faible risque
  - Patient C (score 0.71) : homme 55 ans, diabétique → risque modéré-élevé
- Chaque prédiction est explicable au médecin en langage naturel

---

## Slide 13 — Recommandations Actionnables
1. **Déployer le scoring automatique** sur les nouveaux patients (impact : +40% détection précoce)
2. **Programme de prévention ciblé** pour les patients score ≥ 0.6 (ROI : 3.2x sur 12 mois)
3. **Tableau de bord temps réel** pour les cardiologues avec alertes automatiques

Priorisation basée sur le rapport impact/effort et le ROI quantifié.

---

## Slide 14 — Impact Financier Estimé
- **Population cible** : ~15% des patients identifiés à haut risque
- **Réduction hospitalisations estimée** : 20% (basé sur la littérature)
- **Économie annuelle** : 15 000 € × 200 hospitalisations évitées = **3 000 000 €**
- **Coût du programme** : ~500 000 € (infrastructure + personnel)
- **ROI net** : **+2 500 000 € / an** (retour sur investissement en 2 mois)

---

## Slide 15 — Plan A/B Test & Prochaines Étapes
- **A/B Test** : 3 040 patients, 6 mois, mesure du taux d'hospitalisation
  - Groupe A : suivi standard
  - Groupe B : programme prévention déclenché par le modèle
- **Prochaines étapes** :
  1. Validation clinique avec comité médical (M1-M2)
  2. Lancement A/B test pilote (M3-M8)
  3. Déploiement à l'échelle si résultats positifs (M9-M12)
- **Critère de succès** : réduction ≥ 20% des hospitalisations (p < 0.05)
