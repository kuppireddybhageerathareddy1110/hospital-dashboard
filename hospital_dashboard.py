import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pulp import LpMaximize, LpProblem, LpVariable
import warnings

warnings.filterwarnings("ignore")

st.title("Hospital Emergency Analytics Dashboard")

# ----------------------
# Upload Dataset
# ----------------------
uploaded_file = st.file_uploader("Upload Hospital ER CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # ----------------------
    # Data Preprocessing
    # ----------------------
    df['Patient Satisfaction Score'] = df['Patient Satisfaction Score'].fillna(df['Patient Satisfaction Score'].median())
    df['Department Referral'] = df['Department Referral'].fillna('Unknown')

    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Patient Gender'])

    le_race = LabelEncoder()
    df['Race_Encoded'] = le_race.fit_transform(df['Patient Race'])

    le_dept = LabelEncoder()
    df['Department_Encoded'] = le_dept.fit_transform(df['Department Referral'])

    scaler = MinMaxScaler()
    df['Age_Scaled'] = scaler.fit_transform(df[['Patient Age']])
    df['Waittime_Scaled'] = scaler.fit_transform(df[['Patient Waittime']])
    df['Satisfaction_Scaled'] = scaler.fit_transform(df[['Patient Satisfaction Score']])

    # ----------------------
    # Filters
    # ----------------------
    st.sidebar.header("Filter Patients")
    age_min, age_max = st.sidebar.slider("Select Age Range", int(df['Patient Age'].min()), int(df['Patient Age'].max()), (0, 100))
    selected_gender = st.sidebar.multiselect("Select Gender", df['Patient Gender'].unique(), default=df['Patient Gender'].unique())
    selected_department = st.sidebar.multiselect("Select Department", df['Department Referral'].unique(), default=df['Department Referral'].unique())

    df_filtered = df[
        (df['Patient Age'] >= age_min) &
        (df['Patient Age'] <= age_max) &
        (df['Patient Gender'].isin(selected_gender)) &
        (df['Department Referral'].isin(selected_department))
    ]

    st.subheader(f"Filtered Data ({len(df_filtered)} records)")
    st.dataframe(df_filtered.head())

    # ----------------------
    # Clustering Section
    # ----------------------
    st.subheader("Patient Clustering")
    X_cluster = df_filtered[['Age_Scaled', 'Waittime_Scaled', 'Satisfaction_Scaled']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(X_cluster)

    fig_cluster = px.scatter(df_filtered, x='Patient Age', y='Patient Waittime', color='Cluster', title='Patient Clusters')
    st.plotly_chart(fig_cluster)

    # ----------------------
    # Admission Prediction Section
    # ----------------------
    st.subheader("Predict Patient Admission")

    X_ml = df_filtered[['Age_Scaled', 'Waittime_Scaled', 'Gender_Encoded', 'Department_Encoded']]
    y_ml = df_filtered['Patient Admission Flag']

    rf_model = None
    if len(df_filtered) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        st.write("Not enough data in filtered selection for admission prediction.")

    # ----------------------
    # Resource Optimization Section
    # ----------------------
    st.subheader("Optimize Beds & Doctors Allocation")

    total_resources = st.slider("Total Resources Available", 10, 200, 100)

    prob = LpProblem("Hospital_Resource_Optimization", LpMaximize)
    beds = LpVariable("Beds", lowBound=0, cat='Integer')
    doctors = LpVariable("Doctors", lowBound=0, cat='Integer')

    prob += 5*beds + 3*doctors
    prob += beds + doctors <= total_resources
    prob.solve()

    st.write(f"Optimal Beds: {beds.varValue}")
    st.write(f"Optimal Doctors: {doctors.varValue}")

    # ----------------------
    # Satisfaction by Department
    # ----------------------
    st.subheader("Average Satisfaction by Department")
    avg_satisfaction = df_filtered.groupby('Department Referral')['Patient Satisfaction Score'].mean().reset_index()
    fig4 = px.bar(avg_satisfaction, x='Department Referral', y='Patient Satisfaction Score', title='Avg Satisfaction by Department')
    st.plotly_chart(fig4)

    # ----------------------
    # Patient Lookup Section
    # ----------------------
    st.subheader("Patient Lookup")
    patient_id_input = st.text_input("Enter Patient ID to Lookup")

    if patient_id_input:
        patient_row = df[df['Patient Id'] == patient_id_input]
        if not patient_row.empty:
            st.write("### Patient Details")
            st.dataframe(patient_row)

            patient_features = patient_row[['Age_Scaled', 'Waittime_Scaled', 'Satisfaction_Scaled']]
            patient_cluster = kmeans.predict(patient_features)[0]
            st.write(f"Cluster Assignment: {patient_cluster}")

            if rf_model:
                patient_ml_features = patient_row[['Age_Scaled', 'Waittime_Scaled', 'Gender_Encoded', 'Department_Encoded']]
                admission_prob = rf_model.predict_proba(patient_ml_features)[0][1]
                st.write(f"Predicted Admission Probability: {admission_prob:.2f}")

            st.write(f"Patient Wait Time: {patient_row['Patient Waittime'].values[0]}")
            st.write(f"Patient Satisfaction: {patient_row['Patient Satisfaction Score'].values[0]}")
        else:
            st.write("Patient ID not found in dataset.")

    # ----------------------
    # What-If Scenario Simulator
    # ----------------------
    st.subheader("🔮 What-If Scenario Simulator")

    sim_age = st.slider("Age", int(df['Patient Age'].min()), int(df['Patient Age'].max()), 40)
    sim_gender = st.selectbox("Gender", df['Patient Gender'].unique())
    sim_wait = st.slider("Wait Time (minutes)", int(df['Patient Waittime'].min()), int(df['Patient Waittime'].max()), 30)
    sim_dept = st.selectbox("Department", df['Department Referral'].unique())

    # Scale features
    sim_age_scaled = scaler.transform([[sim_age]])[0][0]
    sim_wait_scaled = scaler.transform([[sim_wait]])[0][0]
    sim_satisfaction_scaled = 0.5  # Assume neutral satisfaction for simulation

    # Encode categorical features
    sim_gender_encoded = le_gender.transform([sim_gender])[0]
    sim_dept_encoded = le_dept.transform([sim_dept])[0]

    # Predict cluster
    sim_features_cluster = np.array([[sim_age_scaled, sim_wait_scaled, sim_satisfaction_scaled]])
    sim_cluster = kmeans.predict(sim_features_cluster)[0]

    st.write(f"Predicted Cluster: {sim_cluster}")

    # Predict admission probability
    if rf_model:
        sim_features_ml = np.array([[sim_age_scaled, sim_wait_scaled, sim_gender_encoded, sim_dept_encoded]])
        sim_admission_prob = rf_model.predict_proba(sim_features_ml)[0][1]
        st.write(f"Predicted Admission Probability: {sim_admission_prob:.2f}")
    else:
        st.write("Admission model not available (need more data).")
    # ----------------------
    # Live What-If Charts
    # ----------------------
    st.subheader("📊 Live What-If Charts")

    if rf_model:
        # 1) Admission Probability vs Age
        ages = np.linspace(df['Patient Age'].min(), df['Patient Age'].max(), 30).astype(int)
        probs_age = []
        for a in ages:
            a_scaled = scaler.transform([[a]])[0][0]
            features = np.array([[a_scaled, sim_wait_scaled, sim_gender_encoded, sim_dept_encoded]])
            probs_age.append(rf_model.predict_proba(features)[0][1])
        fig_age = px.line(x=ages, y=probs_age, labels={'x':'Age', 'y':'Admission Probability'}, title="Admission Probability vs Age")
        st.plotly_chart(fig_age)

        # 2) Admission Probability vs Wait Time
        waits = np.linspace(df['Patient Waittime'].min(), df['Patient Waittime'].max(), 30).astype(int)
        probs_wait = []
        for w in waits:
            w_scaled = scaler.transform([[w]])[0][0]
            features = np.array([[sim_age_scaled, w_scaled, sim_gender_encoded, sim_dept_encoded]])
            probs_wait.append(rf_model.predict_proba(features)[0][1])
        fig_wait = px.line(x=waits, y=probs_wait, labels={'x':'Wait Time (min)', 'y':'Admission Probability'}, title="Admission Probability vs Wait Time")
        st.plotly_chart(fig_wait)

        # 3) Heatmap: Age vs Wait Time
        age_grid = np.linspace(df['Patient Age'].min(), df['Patient Age'].max(), 20).astype(int)
        wait_grid = np.linspace(df['Patient Waittime'].min(), df['Patient Waittime'].max(), 20).astype(int)
        Z = []
        for a in age_grid:
            row = []
            for w in wait_grid:
                a_scaled = scaler.transform([[a]])[0][0]
                w_scaled = scaler.transform([[w]])[0][0]
                features = np.array([[a_scaled, w_scaled, sim_gender_encoded, sim_dept_encoded]])
                row.append(rf_model.predict_proba(features)[0][1])
            Z.append(row)
        Z = np.array(Z)

        fig_heatmap = px.imshow(Z,
                                x=wait_grid,
                                y=age_grid,
                                labels=dict(x="Wait Time (min)", y="Age", color="Admission Probability"),
                                title="Admission Probability Heatmap (Age × Wait Time)")
        st.plotly_chart(fig_heatmap)
    else:
        st.write("⚠️ Not enough data for live probability charts.")
