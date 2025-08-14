# 🧠 Mental Health in the Tech Workplace

**Cohort:** OpenLearn @ NIT Jalandhar  
**Year:** 2025  
**Dataset:** [OSMI 2014 Mental Health in Tech Survey](https://osmihelp.org/research)

---

## 📌 Project Overview

This project analyzes **mental health in tech workplaces**, focusing on **attitudes, prevalence, and factors** influencing treatment-seeking behavior among employees.  
Using the **OSMI 2014** dataset, we explore patterns through **Exploratory Data Analysis (EDA)**, build **classification models**, and perform **clustering** to identify employee personas.

---

## 📊 Dataset Summary

- **Entries:** 1,259  
- **Columns:** 25 (1 target column: `treatment`)  
- **Target Variable:** Whether the participant sought treatment for a mental health condition.

### Columns
- `Age`, `Gender`, `Country`, `state`, `self_employed`
- `family_history`, `treatment`, `work_interfere`, `no_employees`
- `remote_work`, `tech_company`, `benefits`, `care_options`, `wellness_program`
- `seek_help`, `anonymity`, `leave`, `mental_health_consequence`, `phys_health_consequence`
- `coworkers`, `supervisor`, `mental_health_interview`, `phys_health_interview`
- `mental_vs_physical`, `obs_consequence`

---

## 🔍 Key Insights

- Roughly **half** of participants reported seeking treatment for a mental health condition.
- Workplace culture, benefits, and **perceived support** strongly influence treatment-seeking.
- **Classification:** Supportive workplace policies predict higher likelihood of treatment.
- **Clustering:** Distinct employee personas exist — from *quiet strugglers* to *loud-and-proud supporters*.

---

## 🎯 Project Goals

1. Identify **factors** influencing whether employees seek treatment.
2. Build models to **predict treatment likelihood**.
3. Segment employees into **personas** based on mental health attitudes and experiences.

---

## 🔑 Findings

- **Classification:** Workplace support and mental health benefits are top predictors.
- **Clustering:** Four main personas emerged:
  - *Quiet Strugglers* – prefer to manage issues privately.
  - *Partially Open* – share selectively based on comfort level.
  - *Isolated Fighters* – face challenges but feel unsupported.
  - *Open Advocates* – actively promote mental health awareness.

---

## 👥 About the Cohort

OpenLearn is a learning community for **AI/ML** and **finance enthusiasts**, fostering collaboration across domains.  
This project was a collaboration between NITJ clubs **DSC** and **FinNest**,  
initiated by **Batch 2027** for **Batch 2028** learners.

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** Classification (Logistic Regression, Random Forest, SVC, XGBoost), Clustering (K-Means)

---
# Live Demo

[click here](https://f1analyisis-jn5h455gttaueihhvwtbfy.streamlit.app/?embed_options=light_theme)

---

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
