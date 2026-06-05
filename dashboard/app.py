"""
Dashboard Streamlit — Prédiction du Risque Cardiovasculaire
5 vues interactives pour l'aide à la décision clinique
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="Risque Cardiovasculaire", page_icon="❤️", layout="wide")

# === Chargement des données ===
@st.cache_data
def load_data():
    """Charge et harmonise les deux datasets via kagglehub."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df_framingham = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS, "aasheesh200/framingham-heart-study-dataset", "framingham.csv"
    )
    df_cardio = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS, "sulianova/cardiovascular-disease-dataset", "cardio_train.csv",
        pandas_kwargs={"sep": ";"}
    )

    # Harmonisation Cardiovascular
    df_cardio_clean = df_cardio.copy()
    df_cardio_clean['age'] = (df_cardio_clean['age'] / 365).round(0).astype(int)
    df_cardio_clean = df_cardio_clean.rename(columns={
        'ap_hi': 'sysBP', 'ap_lo': 'diaBP', 'gluc': 'glucose',
        'smoke': 'currentSmoker', 'cardio': 'target'
    })
    cols_cardio = ['age', 'gender', 'sysBP', 'diaBP', 'cholesterol', 'glucose', 'currentSmoker', 'target']
    df_c = df_cardio_clean[cols_cardio].copy()
    df_c['source'] = 'cardiovascular'

    # Harmonisation Framingham
    df_fram_clean = df_framingham.rename(columns={
        'male': 'gender', 'totChol': 'cholesterol', 'TenYearCHD': 'target'
    })
    cols_fram = ['age', 'gender', 'sysBP', 'diaBP', 'cholesterol', 'glucose', 'currentSmoker', 'target']
    available = [c for c in cols_fram if c in df_fram_clean.columns]
    df_f = df_fram_clean[available].copy()
    df_f['source'] = 'framingham'

    df_merged = pd.concat([df_f, df_c], ignore_index=True)
    df_merged = df_merged.fillna(df_merged.median(numeric_only=True))
    return df_merged


@st.cache_resource
def load_model():
    """Charge le modèle sauvegardé."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


# Chargement avec spinner
with st.spinner("Chargement des données via KaggleHub..."):
    df = load_data()

model_data = load_model()

# === Sidebar : Filtres ===
st.sidebar.title("🔧 Filtres")
age_range = st.sidebar.slider("Âge", int(df['age'].min()), int(df['age'].max()), (30, 70))
gender_filter = st.sidebar.multiselect("Genre", options=df['gender'].unique().tolist(), default=df['gender'].unique().tolist())
chol_filter = st.sidebar.multiselect("Cholestérol", options=sorted(df['cholesterol'].unique().tolist()), default=sorted(df['cholesterol'].unique().tolist()))

# Appliquer les filtres
mask = (df['age'].between(age_range[0], age_range[1])) & (df['gender'].isin(gender_filter)) & (df['cholesterol'].isin(chol_filter))
df_filtered = df[mask]

st.title("❤️ Dashboard — Risque Cardiovasculaire")
st.markdown(f"**{len(df_filtered):,}** patients affichés (filtres appliqués)")

# === Navigation ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPIs Modèles", "📈 Facteurs de Risque", "🔮 Prédiction Individuelle",
    "🧠 SHAP Importance", "👥 Segmentation Risque"
])

# === VUE 1 : KPIs globaux ===
with tab1:
    st.header("Vue 1 — KPIs des Modèles")
    if model_data and 'metrics' in model_data:
        m = model_data['metrics']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC", f"{m['auc_roc']:.3f}", "✅" if m['auc_roc'] > 0.85 else "⚠️")
        c2.metric("F1-Score", f"{m['f1_score']:.3f}")
        c3.metric("Recall", f"{m['recall']:.3f}", "✅" if m['recall'] > 0.80 else "⚠️")
        c4.metric("Precision", f"{m['precision']:.3f}")

        # Comparatif sous forme de bar chart
        metrics_df = pd.DataFrame({
            'Métrique': ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'],
            'XGBoost (meilleur)': [m['auc_roc'], m['f1_score'], m['recall'], m['precision']]
        })
        fig = px.bar(metrics_df, x='Métrique', y='XGBoost (meilleur)', color='Métrique',
                     title="Performance du Meilleur Modèle (XGBoost)")
        fig.add_hline(y=0.85, line_dash="dash", annotation_text="Objectif AUC-ROC")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Exécutez le notebook pour générer best_model.pkl")

# === VUE 2 : Distribution des facteurs de risque ===
with tab2:
    st.header("Vue 2 — Distribution des Facteurs de Risque")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df_filtered, x='age', color='target', barmode='overlay',
                          title="Distribution de l'Âge par Classe", opacity=0.7,
                          labels={'target': 'Malade', 'age': 'Âge'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df_filtered, x='target', y='sysBP', color='target',
                    title="Tension Systolique par Classe",
                    labels={'target': 'Classe', 'sysBP': 'Tension Systolique (mmHg)'})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(df_filtered, x='cholesterol', color='target', barmode='group',
                          title="Cholestérol par Classe")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(df_filtered.sample(min(5000, len(df_filtered)), random_state=42),
                        x='sysBP', y='diaBP', color='target', opacity=0.5,
                        title="Tension Systolique vs Diastolique")
        st.plotly_chart(fig, use_container_width=True)

# === VUE 3 : Prédiction individuelle ===
with tab3:
    st.header("Vue 3 — Prédiction Individuelle")

    if model_data:
        col1, col2 = st.columns(2)
        with col1:
            p_age = st.number_input("Âge", 20, 90, 55)
            p_gender = st.selectbox("Genre", [0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme")
            p_sysbp = st.number_input("Tension systolique (mmHg)", 80, 250, 130)
            p_diabp = st.number_input("Tension diastolique (mmHg)", 40, 150, 85)

        with col2:
            p_chol = st.number_input("Cholestérol", 1, 600, 200)
            p_glucose = st.number_input("Glucose", 1, 400, 80)
            p_smoker = st.selectbox("Fumeur", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

        if st.button("🔮 Prédire le risque", type="primary"):
            features = model_data['feature_names']
            input_data = pd.DataFrame([[0]*len(features)], columns=features)
            # Remplir les valeurs connues
            for col, val in [('age', p_age), ('gender', p_gender), ('sysBP', p_sysbp),
                            ('diaBP', p_diabp), ('cholesterol', p_chol), ('glucose', p_glucose),
                            ('currentSmoker', p_smoker)]:
                if col in input_data.columns:
                    input_data[col] = val
            # Variables dérivées
            if 'bp_ratio' in input_data.columns:
                input_data['bp_ratio'] = p_sysbp / max(p_diabp, 1)
            if 'pulse_pressure' in input_data.columns:
                input_data['pulse_pressure'] = p_sysbp - p_diabp

            X_scaled = model_data['scaler'].transform(input_data)
            proba = model_data['model'].predict_proba(X_scaled)[0][1]

            # Affichage du résultat
            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                color = "🔴" if proba >= 0.6 else "🟡" if proba >= 0.3 else "🟢"
                niveau = "ÉLEVÉ" if proba >= 0.6 else "MODÉRÉ" if proba >= 0.3 else "FAIBLE"
                st.metric("Score de Risque", f"{proba:.1%}", f"{color} Risque {niveau}")
            with col_r2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba*100,
                    title={'text': "Risque (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "red" if proba >= 0.6 else "orange" if proba >= 0.3 else "green"},
                           'steps': [{'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 60], 'color': "lightyellow"},
                                    {'range': [60, 100], 'color': "lightsalmon"}]}
                ))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Exécutez le notebook pour générer best_model.pkl")

# === VUE 4 : Feature importance SHAP ===
with tab4:
    st.header("Vue 4 — Feature Importance (SHAP)")
    if model_data and 'feature_names' in model_data:
        # Calcul d'importance via le modèle XGBoost
        model = model_data['model']
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': model_data['feature_names'],
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                    title="Importance des Variables (XGBoost)", color='Importance',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Les facteurs les plus importants pour prédire le risque cardiovasculaire sont affichés ci-dessus. "
                "Un SHAP summary plot complet est disponible dans le notebook.")
    else:
        st.warning("Exécutez le notebook pour générer best_model.pkl")

# === VUE 5 : Segmentation par niveau de risque ===
with tab5:
    st.header("Vue 5 — Segmentation des Patients par Niveau de Risque")

    if model_data:
        # Scoring de tous les patients filtrés
        features = model_data['feature_names']
        df_score = df_filtered.copy()
        X_pred = pd.DataFrame(0, index=df_score.index, columns=features)
        for col in features:
            if col in df_score.columns:
                X_pred[col] = df_score[col].values
        if 'bp_ratio' in features and 'sysBP' in df_score.columns and 'diaBP' in df_score.columns:
            X_pred['bp_ratio'] = df_score['sysBP'] / df_score['diaBP'].replace(0, 1)
        if 'pulse_pressure' in features and 'sysBP' in df_score.columns and 'diaBP' in df_score.columns:
            X_pred['pulse_pressure'] = df_score['sysBP'] - df_score['diaBP']

        X_scaled = model_data['scaler'].transform(X_pred)
        scores = model_data['model'].predict_proba(X_scaled)[:, 1]
        df_score['risk_score'] = scores
        df_score['risk_level'] = pd.cut(scores, bins=[0, 0.3, 0.6, 1.0],
                                        labels=['Faible', 'Modéré', 'Élevé'])

        # Distribution des niveaux
        col1, col2 = st.columns(2)
        with col1:
            risk_counts = df_score['risk_level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title="Répartition par Niveau de Risque",
                        color_discrete_map={'Faible': 'green', 'Modéré': 'orange', 'Élevé': 'red'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df_score, x='risk_score', color='risk_level', nbins=50,
                             title="Distribution des Scores de Risque",
                             color_discrete_map={'Faible': 'green', 'Modéré': 'orange', 'Élevé': 'red'})
            st.plotly_chart(fig, use_container_width=True)

        # Profil moyen par segment
        st.subheader("Profil moyen par segment")
        profile = df_score.groupby('risk_level')[['age', 'sysBP', 'diaBP', 'cholesterol']].mean().round(1)
        st.dataframe(profile, use_container_width=True)

        # Métriques par segment
        c1, c2, c3 = st.columns(3)
        for col, level, color in [(c1, 'Faible', '🟢'), (c2, 'Modéré', '🟡'), (c3, 'Élevé', '🔴')]:
            count = (df_score['risk_level'] == level).sum()
            col.metric(f"{color} {level}", f"{count:,}", f"{count/len(df_score)*100:.1f}%")
    else:
        st.warning("Exécutez le notebook pour générer best_model.pkl")

# Footer
st.markdown("---")
st.caption("Dashboard DDDM — Prédiction du Risque Cardiovasculaire | Données : Framingham + Cardiovascular Disease (Kaggle)")
