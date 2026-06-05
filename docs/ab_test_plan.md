# Plan A/B Test — Programme de Prévention Cardiovasculaire

## 1. Contexte et Objectif

Notre modèle prédictif identifie les patients à risque élevé de maladie coronarienne (AUC-ROC > 0.85). L'objectif de ce test A/B est de mesurer l'efficacité d'un **programme de prévention ciblé** déclenché par les prédictions du modèle, comparé au suivi médical standard.

**Question** : L'intervention préventive basée sur le modèle réduit-elle significativement le taux d'hospitalisation cardiaque à 6 mois ?

## 2. Hypothèses

- **H0 (Hypothèse nulle)** : Le programme de prévention ciblé ne réduit pas le taux d'hospitalisation cardiaque par rapport au suivi standard (π_traitement = π_contrôle).
- **H1 (Hypothèse alternative)** : Le programme de prévention ciblé réduit le taux d'hospitalisation cardiaque d'au moins 20% par rapport au suivi standard (π_traitement < π_contrôle).

## 3. Population Cible

### Critères d'inclusion
- Patients identifiés comme « risque élevé » par le modèle (score ≥ 0.6)
- Âge entre 30 et 70 ans
- Pas d'hospitalisation cardiaque dans les 12 derniers mois
- Consentement éclairé signé

### Critères d'exclusion
- Patients déjà sous traitement cardiologique intensif
- Pathologie terminale (espérance de vie < 1 an)
- Incapacité à participer au programme (mobilité, cognition)

## 4. Calcul de la Taille d'Échantillon

| Paramètre | Valeur |
|-----------|--------|
| Puissance statistique (1-β) | 80% |
| Niveau de significativité (α) | 0.05 (bilatéral) |
| Taux d'hospitalisation baseline (contrôle) | 12% |
| Réduction attendue (effet) | 20% relatif → taux traitement = 9.6% |
| Taille par groupe | **1 520 patients** |
| **Taille totale** | **3 040 patients** |

Formule utilisée : test de proportion bilatéral (Z-test), avec correction de continuité.

```
n = (Z_α/2 + Z_β)² × [π1(1-π1) + π2(1-π2)] / (π1 - π2)²
n = (1.96 + 0.84)² × [0.12×0.88 + 0.096×0.904] / (0.12 - 0.096)²
n ≈ 1 520 par groupe
```

**Justification de l'effet attendu (20%)** : La littérature sur les programmes de prévention cardiovasculaire ciblés montre des réductions de 15-30% des événements cardiaques (Yusuf et al., Lancet 2004). Nous retenons 20% comme estimation conservatrice.

## 5. Durée de l'Expérience

- **Phase de recrutement** : 2 mois
- **Phase d'intervention** : 6 mois
- **Phase de suivi post-intervention** : 3 mois
- **Durée totale** : 11 mois

**Justification** : Les événements cardiovasculaires se manifestent sur un horizon de 6+ mois. Une durée plus courte ne capturerait pas suffisamment d'événements pour atteindre la puissance statistique requise.

## 6. Métriques

### Métriques primaires
- **Taux d'hospitalisation cardiaque** à 6 mois (critère principal)
- **Délai avant premier événement cardiaque** (time-to-event)

### Métriques secondaires
- Nombre de consultations d'urgence cardiologiques
- Amélioration des biomarqueurs (cholestérol LDL, tension artérielle)
- Adhérence au programme de prévention (% de séances suivies)
- Score de qualité de vie (questionnaire SF-36)
- Coût total par patient (hospitalisation + prévention)

## 7. Protocole de Randomisation

| | Groupe Contrôle (A) | Groupe Traitement (B) |
|---|---|---|
| **Taille** | 1 520 patients | 1 520 patients |
| **Intervention** | Suivi médical standard | Programme prévention ciblé |
| **Détail** | Consultations annuelles habituelles | Consultation mensuelle + coaching hygiène de vie + alertes automatiques si détérioration des indicateurs |

**Méthode de randomisation** : Randomisation stratifiée par bloc (blocs de 4), stratifiée sur :
- Tranche d'âge (30-50 / 50-70)
- Score de risque du modèle (0.6-0.75 / 0.75-1.0)
- Sexe

**Allocation** : Séquence générée par algorithme (random_state=42), enveloppes scellées.

## 8. Critères d'Arrêt Anticipé

### Arrêt pour efficacité (Futility)
- Si à la revue intermédiaire (3 mois, 50% de la durée), le p-value ajusté (O'Brien-Fleming) est > 0.80, l'étude est arrêtée pour futilité — il est très improbable de démontrer un effet.

### Arrêt pour nocivité (Harm)
- Si le taux d'événements graves dans le groupe traitement dépasse celui du groupe contrôle de plus de 50% (RR > 1.5) à tout moment, arrêt immédiat.
- Comité de surveillance indépendant (DSMB) avec revue trimestrielle.

### Seuils de Lan-DeMets
- Revue intermédiaire à 50% : α_spent = 0.005
- Analyse finale à 100% : α_remaining = 0.048

## 9. Méthode d'Analyse Statistique Post-Test

1. **Analyse primaire** : Test du Chi-deux (ou test exact de Fisher si effectifs faibles) sur les proportions d'hospitalisation entre les deux groupes.
2. **Analyse de survie** : Courbes de Kaplan-Meier + test du log-rank pour le délai avant événement.
3. **Régression logistique ajustée** : Contrôle des covariables (âge, score de risque, comorbidités) pour estimer l'odds ratio ajusté.
4. **Analyse en intention de traiter (ITT)** : Tous les patients randomisés sont analysés dans leur groupe d'affectation initial, indépendamment de l'adhérence.
5. **Analyse per-protocole** : En complément, analyse limitée aux patients ayant suivi ≥ 80% du programme.
6. **Intervalles de confiance à 95%** et calcul du NNT (Number Needed to Treat).

**Correction pour comparaisons multiples** : Benjamini-Hochberg (FDR) pour les métriques secondaires.
