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

st.set_page_config(page_title="Risque Cardiovasculaire", layout="wide")

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

    df_cardio_clean = df_cardio.copy()
    df_cardio_clean['age'] = (df_cardio_clean['age'] / 365).round(0).astype(int)
    df_cardio_clean = df_cardio_clean.rename(columns={
        'ap_hi': 'sysBP', 'ap_lo': 'diaBP', 'gluc': 'glucose',
        'smoke': 'currentSmoker', 'cardio': 'target'
    })
    cols_cardio = ['age', 'gender', 'sysBP', 'diaBP', 'cholesterol', 'glucose', 'currentSmoker', 'target']
    df_c = df_cardio_clean[cols_cardio].copy()
    df_c['source'] = 'cardiovascular'

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


with st.spinner("Chargement des données via KaggleHub..."):
    df = load_data()

model_data = load_model()

# === Sidebar : Profil Utilisateur ===
st.sidebar.title("Profil Utilisateur")
profil = st.sidebar.selectbox(
    "Je suis :",
    ["Data Analyst", "Direction", "Opérations", "Prévention / Marketing", "Clinicien"],
    help="Sélectionnez votre profil pour afficher les vues adaptées"
)

PROFIL_TABS = {
    "Direction": ["KPIs Modèles", "Segmentation Risque"],
    "Opérations": ["KPIs Modèles", "Facteurs de Risque", "Segmentation Risque"],
    "Prévention / Marketing": ["Facteurs de Risque", "SHAP Importance", "Segmentation Risque"],
    "Clinicien": ["Facteurs de Risque", "Prédiction Individuelle", "SHAP Importance"],
    "Data Analyst": ["KPIs Modèles", "Facteurs de Risque", "Prédiction Individuelle", "SHAP Importance", "Segmentation Risque"],
}

# === Sidebar : Filtres ===
st.sidebar.markdown("---")
st.sidebar.title("Filtres")
age_range = st.sidebar.slider("Âge", int(df['age'].min()), int(df['age'].max()), (30, 70))
gender_filter = st.sidebar.multiselect("Genre", options=df['gender'].unique().tolist(), default=df['gender'].unique().tolist(),
                                        format_func=lambda x: "Homme" if x == 1 else "Femme")
chol_filter = st.sidebar.multiselect("Cholestérol", options=sorted(df['cholesterol'].unique().tolist()), default=sorted(df['cholesterol'].unique().tolist()),
                                      format_func=lambda x: {1: "🟢 Normal", 2: "🟠 Élevé", 3: "🔴 Très élevé"}.get(x, str(x)))

mask = (df['age'].between(age_range[0], age_range[1])) & (df['gender'].isin(gender_filter)) & (df['cholesterol'].isin(chol_filter))
df_filtered = df[mask]

st.title("Dashboard — Risque Cardiovasculaire")
st.markdown(f"**{len(df_filtered):,}** patients affichés | Profil : **{profil}**")

# === Navigation dynamique par profil ===
visible_tabs = PROFIL_TABS[profil]
tabs = st.tabs(visible_tabs)
tab_map = dict(zip(visible_tabs, tabs))

# === VUE 1 : KPIs globaux ===
if "KPIs Modèles" in tab_map:
    with tab_map["KPIs Modèles"]:
        st.header("Vue Direction — KPIs des Modèles")
        if model_data and 'metrics' in model_data:
            m = model_data['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC-ROC", f"{m['auc_roc']:.3f}", "OK" if m['auc_roc'] > 0.85 else "A ameliorer")
            c2.metric("F1-Score", f"{m['f1_score']:.3f}")
            c3.metric("Recall", f"{m['recall']:.3f}", "OK" if m['recall'] > 0.80 else "A ameliorer")
            c4.metric("Precision", f"{m['precision']:.3f}")

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
if "Facteurs de Risque" in tab_map:
    with tab_map["Facteurs de Risque"]:
        st.header("Vue Opérations — Distribution des Facteurs de Risque")

        # Préparer une copie avec target en catégoriel pour de meilleurs graphiques
        df_viz = df_filtered.copy()
        df_viz['Classe'] = df_viz['target'].map({0: 'Sain', 1: 'Malade'})
        df_viz['Cholestérol'] = df_viz['cholesterol'].map({1: 'Normal', 2: 'Élevé', 3: 'Très élevé'})

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df_viz, x='age', color='Classe', barmode='overlay',
                              title="Distribution de l'Âge par Classe", opacity=0.7,
                              labels={'age': 'Âge'},
                              color_discrete_map={'Sain': '#2a9d8f', 'Malade': '#e63946'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df_viz, x='Classe', y='sysBP', color='Classe',
                        title="Tension Systolique par Classe",
                        labels={'sysBP': 'Tension Systolique (mmHg)'},
                        color_discrete_map={'Sain': '#2a9d8f', 'Malade': '#e63946'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df_viz, x='Cholestérol', color='Classe', barmode='group',
                              title="Cholestérol par Classe",
                              category_orders={'Cholestérol': ['Normal', 'Élevé', 'Très élevé']},
                              color_discrete_map={'Sain': '#2a9d8f', 'Malade': '#e63946'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Filtrer les valeurs aberrantes (tension réaliste : 60-250 sys, 40-150 dia)
            df_scatter = df_viz[
                (df_viz['sysBP'].between(60, 250)) &
                (df_viz['diaBP'].between(40, 150)) &
                (df_viz['sysBP'] > df_viz['diaBP'])
            ]
            df_sample = df_scatter.sample(min(5000, len(df_scatter)), random_state=42)
            fig = px.scatter(df_sample, x='sysBP', y='diaBP', color='Classe', opacity=0.5,
                            title="Tension Systolique vs Diastolique",
                            labels={'sysBP': 'Systolique (mmHg)', 'diaBP': 'Diastolique (mmHg)'},
                            color_discrete_map={'Sain': '#2a9d8f', 'Malade': '#e63946'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# === VUE 3 : Prédiction individuelle ===
if "Prédiction Individuelle" in tab_map:
    with tab_map["Prédiction Individuelle"]:
        st.header("Vue Clinicien — Prédiction Individuelle")

        if model_data:
            col1, col2 = st.columns(2)
            with col1:
                p_age = st.number_input("Âge", 20, 90, 55, key="pred_age")
                p_gender = st.selectbox("Genre", [0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme", key="pred_gender")
                p_sysbp = st.number_input("Tension systolique (mmHg)", 80, 250, 130, key="pred_sysbp")
                p_diabp = st.number_input("Tension diastolique (mmHg)", 40, 150, 85, key="pred_diabp")

            with col2:
                p_chol = st.selectbox("Cholestérol", [1, 2, 3],
                                      format_func=lambda x: {1: "Normal", 2: "Élevé", 3: "Très élevé"}[x], key="pred_chol")
                p_glucose = st.selectbox("Glucose", [1, 2, 3],
                                         format_func=lambda x: {1: "Normal", 2: "Élevé", 3: "Très élevé"}[x], key="pred_glucose")
                p_smoker = st.selectbox("Fumeur", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="pred_smoker")

            # Prédiction automatique à chaque changement
            features = model_data['feature_names']
            input_data = pd.DataFrame([[0]*len(features)], columns=features)
            for col, val in [('age', p_age), ('gender', p_gender), ('sysBP', p_sysbp),
                            ('diaBP', p_diabp), ('cholesterol', p_chol), ('glucose', p_glucose),
                            ('currentSmoker', p_smoker)]:
                if col in input_data.columns:
                    input_data[col] = val
            if 'bp_ratio' in input_data.columns:
                input_data['bp_ratio'] = p_sysbp / max(p_diabp, 1)
            if 'pulse_pressure' in input_data.columns:
                input_data['pulse_pressure'] = p_sysbp - p_diabp

            X_scaled = model_data['scaler'].transform(input_data)
            proba = model_data['model'].predict_proba(X_scaled)[0][1]

            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if proba >= 0.6:
                    color_label, emoji = "ÉLEVÉ", "🔴"
                elif proba >= 0.3:
                    color_label, emoji = "MODÉRÉ", "🟠"
                else:
                    color_label, emoji = "FAIBLE", "🟢"
                st.metric("Score de Risque", f"{proba:.1%}", f"{emoji} Risque {color_label}")
            with col_r2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba*100,
                    title={'text': "Risque (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#e63946" if proba >= 0.6 else "#f4a261" if proba >= 0.3 else "#2a9d8f"},
                           'steps': [{'range': [0, 30], 'color': "#d8f3dc"},
                                    {'range': [30, 60], 'color': "#fff3cd"},
                                    {'range': [60, 100], 'color': "#f8d7da"}]}
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚙️ Exécutez le notebook pour générer best_model.pkl")

# === VUE 4 : Feature importance SHAP ===
if "SHAP Importance" in tab_map:
    with tab_map["SHAP Importance"]:
        st.header("Vue Prévention — Importance des Variables")
        if model_data and 'feature_names' in model_data:
            model = model_data['model']
            importances = model.feature_importances_

            # Noms lisibles en français
            feature_labels = {
                'age': 'Âge', 'gender': 'Genre', 'sysBP': 'Tension Systolique',
                'diaBP': 'Tension Diastolique', 'cholesterol': 'Cholestérol',
                'glucose': 'Glucose', 'currentSmoker': 'Tabagisme',
                'bp_ratio': 'Ratio Sys/Dia', 'pulse_pressure': 'Pression Pulsée'
            }

            feat_df = pd.DataFrame({
                'Feature': [feature_labels.get(f, f) for f in model_data['feature_names']],
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                        title="Contribution de chaque variable dans la prédiction (XGBoost Feature Importance)",
                        color='Importance', color_continuous_scale='Reds')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Top 3 insights
            top3 = feat_df.nlargest(3, 'Importance')
            st.markdown("**🔑 Top 3 facteurs de risque :**")
            for i, row in enumerate(top3.itertuples(), 1):
                st.markdown(f"{i}. **{row.Feature}** — contribue à {row.Importance:.1%} des décisions du modèle")

            st.info("💡 Ces importances mesurent combien chaque variable contribue aux splits de l'arbre XGBoost. "
                    "L'analyse SHAP complète (globale + locale par patient) est disponible dans le notebook.")
        else:
            st.warning("⚙️ Exécutez le notebook pour générer best_model.pkl")

# === VUE 5 : Segmentation par niveau de risque ===
if "Segmentation Risque" in tab_map:
    with tab_map["Segmentation Risque"]:
        st.header("Vue Segmentation — Patients par Niveau de Risque")

        if model_data:
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

            st.subheader("Profil moyen par segment")
            profile = df_score.groupby('risk_level')[['age', 'sysBP', 'diaBP', 'cholesterol']].mean().round(1)
            st.dataframe(profile, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            for col, level in [(c1, 'Faible'), (c2, 'Modéré'), (c3, 'Élevé')]:
                count = (df_score['risk_level'] == level).sum()
                col.metric(f"{level}", f"{count:,}", f"{count/len(df_score)*100:.1f}%")
        else:
            st.warning("Exécutez le notebook pour générer best_model.pkl")

# Footer
st.markdown("---")
st.caption("Dashboard DDDM — Prédiction du Risque Cardiovasculaire | Données : Framingham + Cardiovascular Disease (Kaggle)")
