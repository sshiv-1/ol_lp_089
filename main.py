import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(__file__)
IMG_DIR = os.path.join(ROOT, "images")

EDA_DIRS = {
    "Univariate": os.path.join(IMG_DIR, "univariate"),
    "Bivariate": os.path.join(IMG_DIR, "bivariate"),
    "Multivariate": os.path.join(IMG_DIR, "multivariate"),
}

SECTION_IMG_DIRS = {
    "Clustering": os.path.join(IMG_DIR, "clustering"),
}

def show_image_gallery(folder, cols=2, caption_prefix=""):
    if not os.path.exists(folder):
        st.warning(f"Folder not found: {folder}")
        return
    images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    columns = st.columns(cols)
    for idx, img_name in enumerate(images):
        with columns[idx % cols]:
            st.image(os.path.join(folder, img_name), caption=f"{caption_prefix}{img_name}", use_container_width=True)

df = pd.read_csv(os.path.join(ROOT, "Notebooks", "cleaned_dataset.csv"))
classifier = joblib.load(os.path.join(ROOT, "model_pkl", "classifier.pkl"))
regressor = joblib.load(os.path.join(ROOT, "model_pkl", "regressor.pkl"))

st.set_page_config(page_title="OpenLearn 1.0 Capstone Project")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["🏠 Home", "🎯 EDA", "🧮 Classification Task", "📊 Clustering Task"])

if menu == "🏠 Home":
    st.title("🧠 Mental Health in the Tech Workplace")
    st.write("**Cohort:** OpenLearn @ NIT Jalandhar • **Year:** 2025 • **Dataset:** OSMI 2014")
    st.divider()
    st.header("About the Dataset")
    st.write("""
        This dataset dives into **mental health in tech workplaces**, exploring attitudes and prevalence.
        It has **1259 entries** and **25 columns**, with one target column `treatment`.
    """)
    st.info("""
        Roughly half the participants sought treatment for a mental health condition.
        The dataset helps us understand what factors influence this choice.
    """)
    st.subheader("Columns Overview")
    st.markdown("""
    - `Age`, `Gender`, `Country`, `state`, `self_employed`
    - `family_history`, `treatment`, `work_interfere`, `no_employees`
    - `remote_work`, `tech_company`, `benefits`, `care_options`, `wellness_program`
    - `seek_help`, `anonymity`, `leave`, `mental_health_consequence`, `phys_health_consequence`
    - `coworkers`, `supervisor`, `mental_health_interview`, `phys_health_interview`
    - `mental_vs_physical`, `obs_consequence`
    """)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.divider()
    st.header("🎯 Project Goals")
    st.markdown("""
        Explore mental health in tech:
        - Factors influencing treatment-seeking.
        - Predicting treatment likelihood.
        - Identifying employee personas based on mental health experience.
    """)
    st.header("🔑 Key Findings")
    st.info("""
        - **Classification:** Workplace support strongly predicts treatment-seeking.
        - **Clustering:** Employees form distinct personas from 'Quiet Strugglers' to 'Loud-and-Proud Supporters'.
    """)
    st.header("About the Cohort")
    st.markdown("""
        - OpenLearn is a community for AI/ML and finance enthusiasts.
        - Collaboration between NITJ clubs DSC and FinNest.
        - An initiative by batch 2027 for batch 2028.
    """)

elif menu == "🎯 EDA":
    st.title("🔎 Exploratory Data Analysis")
    st.write("Visual exploration of the dataset.")
    tabs = st.tabs(["Univariate", "Bivariate", "Multivariate"])
    with tabs[0]:
        st.subheader("Univariate Analysis")
        show_image_gallery(EDA_DIRS["Univariate"], cols=2, caption_prefix="Univariate • ")
    with tabs[1]:
        st.subheader("Bivariate Analysis")
        show_image_gallery(EDA_DIRS["Bivariate"], cols=2, caption_prefix="Bivariate • ")
    with tabs[2]:
        st.subheader("Multivariate Analysis")
        show_image_gallery(EDA_DIRS["Multivariate"], cols=2, caption_prefix="Multivariate • ")
        st.markdown("""
            #### Key Correlations with 'Treatment':
            - `family_history`: Strong positive correlation with seeking treatment.
            - `work_interfere`: Higher interference increases likelihood of treatment.
            - `benefits` & `care_options`: Awareness and access encourage seeking treatment.
        """)

elif menu == "📊 Clustering Task":
    st.title("🌀 Clustering & Persona Analysis")
    st.write("""
        Segment employees into meaningful groups (personas) using K-Means.
        Visualizations help see clusters in 2D space.
    """)
    show_image_gallery(SECTION_IMG_DIRS["Clustering"], cols=2, caption_prefix="Cluster Viz • ")
    st.markdown("---")
    st.header("Cluster Personas Explained")
    st.write("We identified four personas:")
    with st.expander("👤 Cluster 0 – Keepin’ it to themselves", expanded=True):
        st.write("Prefer to handle mental health quietly. Likely won’t use workplace resources.")
    with st.expander("👤 Cluster 1 – Kinda open, kinda not"):
        st.write("Share some thoughts if the vibe is right but mostly keep it to themselves.")
    with st.expander("👤 Cluster 2 – The quiet strugglers"):
        st.write("Have faced challenges but manage on their own, often perceiving limited workplace support.")
    with st.expander("👤 Cluster 3 – The loud-and-proud supporters"):
        st.write("Actively promote mental health awareness and support colleagues openly.")

elif menu == "🧮 Classification Task":
    st.title("🔎 Classification Task")
    st.divider()
    st.markdown("""
        Predict whether an employee sought treatment using our best model: `XGBoost Classifier`.
    """)
    st.subheader("📌 Model Performance")
    clf_data = {
        "Model": ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "SVC", "XGBoost (Best)"],
        "Accuracy": [0.7428, 0.7301, 0.6920, 0.7365, 0.7429],
        "F1-Score": [0.7460, 0.7301, 0.6689, 0.7365, 0.7460],
        "ROC AUC": [0.7429, 0.7301, 0.6918, 0.7365, 0.7429]
    }
    st.dataframe(pd.DataFrame(clf_data).set_index("Model").style.format("{:.4f}").highlight_max(axis=0, color='#00A36C'))
    st.divider()
    st.header("🔮 Live Treatment Predictor")
    st.write("Fill in details below to predict treatment likelihood.")
    input_data = {}
    display_labels = {
        "Age": "Age",
        "Gender": "Gender",
        "self_employed": "Self-employed?",
        "family_history": "Family history of mental illness?",
        "work_interfere": "Does mental health interfere with work?",
        "no_employees": "Company size",
        "remote_work": "Work remotely 50%+?",
        "tech_company": "Tech company?",
        "benefits": "Employer provides mental health benefits?",
        "care_options": "Aware of care options?",
        "wellness_program": "Employer wellness program?",
        "seek_help": "Employer provides help resources?",
        "anonymity": "Anonymity protected?",
        "leave": "Ease of taking mental health leave?",
        "mental_health_consequence": "Fear negative consequences for mental health?",
        "phys_health_consequence": "Fear negative consequences for physical health?",
        "coworkers": "Discuss with coworkers?",
        "supervisor": "Discuss with supervisor?",
        "mental_health_interview": "Bring up in interview?",
        "phys_health_interview": "Bring up physical issue in interview?",
        "mental_vs_physical": "Employer takes mental health as seriously as physical?",
        "obs_consequence": "Observed negative consequences for coworkers?"
    }
    try:
        cat_features = classifier.named_steps['preprocessor'].transformers_[0][2]
        num_features = classifier.named_steps['preprocessor'].transformers_[1][2]
        features = num_features + cat_features
    except:
        features = [col for col in df.columns if col != 'treatment']
    col1, col2 = st.columns(2)
    midpoint = len(features) // 2
    for col_group, column_ref in zip([features[:midpoint], features[midpoint:]], [col1, col2]):
        with column_ref:
            for feature in col_group:
                label = display_labels.get(feature, feature)
                if feature == "Age":
                    input_data[feature] = st.number_input(label, min_value=18, max_value=100, value=32, key=f"clf_{feature}")
                else:
                    input_data[feature] = st.selectbox(label, df[feature].dropna().unique(), key=f"clf_{feature}")
    if st.button("Predict Treatment Likelihood"):
        if not input_data:
            st.error("Fill in the details above!")
        else:
            input_df = pd.DataFrame([input_data])[features]
            prediction = classifier.predict(input_df)[0]
            confidence = classifier.predict_proba(input_df)[0][np.where(classifier.classes_ == prediction)[0][0]]
            st.subheader("Prediction Result")
            if prediction == "Yes":
                st.success("This person **is likely** to have sought treatment.")
            else:
                st.error("This person **is not likely** to have sought treatment.")
            st.info(f"Confidence: **{confidence:.2%}**")
