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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings

warnings.filterwarnings("ignore")

st.title("🏥 Hospital Emergency Analytics Dashboard")

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
    # ML: Admission Prediction Section
    # ----------------------
    st.subheader("ML: Predict Patient Admission")

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
    # Optimization Section
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
    # Deep Learning Section
    # ----------------------
    st.subheader("🤖 Deep Learning Admission Prediction")

    if len(df_filtered) >= 50:
        X_dl = df_filtered[['Age_Scaled', 'Waittime_Scaled', 'Gender_Encoded', 'Department_Encoded']]
        y_dl = df_filtered['Patient Admission Flag']
        y_dl_encoded = to_categorical(y_dl)

        X_train, X_test, y_train, y_test = train_test_split(X_dl, y_dl_encoded, test_size=0.2, random_state=42)

        model = Sequential([
            Dense(16, input_dim=X_dl.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(y_dl_encoded.shape[1], activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0, validation_data=(X_test, y_test))

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Deep Learning Test Accuracy: {acc:.2f}")

        fig_dl = px.line(x=range(1, len(history.history['accuracy'])+1),
                         y=history.history['accuracy'],
                         labels={'x':'Epoch', 'y':'Accuracy'},
                         title="Training Accuracy per Epoch")
        st.plotly_chart(fig_dl)
    else:
        st.write("⚠️ Not enough data for deep learning model (need at least 50 records).")

    # ----------------------
    # NLP Section
    # ----------------------
    st.subheader("💬 NLP: Patient Complaints Analysis")

    if 'Patient Complaint Text' in df.columns:
        complaints = df['Patient Complaint Text'].fillna("")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X_text = vectorizer.fit_transform(complaints)
        y_text = (df['Patient Satisfaction Score'] > 3).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_text, y_text, test_size=0.2, random_state=42)
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)

        acc_text = nb_model.score(X_test, y_test)
        st.write(f"NLP Sentiment Classifier Accuracy: {acc_text:.2f}")

        feature_names = np.array(vectorizer.get_feature_names_out())
        top_words = feature_names[np.argsort(nb_model.coef_[0])[-10:]]
        st.write("Top Predictive Complaint Words:", ", ".join(top_words))

        complaint_input = st.text_area("Enter Patient Complaint")
        if complaint_input:
            complaint_vec = vectorizer.transform([complaint_input])
            prediction = nb_model.predict(complaint_vec)[0]
            st.write("Predicted Sentiment:", "😊 Positive" if prediction==1 else "☹️ Negative")
    else:
        st.write("⚠️ No complaint text column found in dataset.")

    # ----------------------
    # Satisfaction by Department
    # ----------------------
    st.subheader("Average Satisfaction by Department")
    avg_satisfaction = df_filtered.groupby('Department Referral')['Patient Satisfaction Score'].mean().reset_index()
    fig4 = px.bar(avg_satisfaction, x='Department Referral', y='Patient Satisfaction Score', title='Avg Satisfaction by Department')
    st.plotly_chart(fig4)

    # ----------------------
    # Patient Lookup
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

    sim_age_scaled = scaler.transform([[sim_age]])[0][0]
    sim_wait_scaled = scaler.transform([[sim_wait]])[0][0]
    sim_satisfaction_scaled = 0.5

    sim_gender_encoded = le_gender.transform([sim_gender])[0]
    sim_dept_encoded = le_dept.transform([sim_dept])[0]

    sim_features_cluster = np.array([[sim_age_scaled, sim_wait_scaled, sim_satisfaction_scaled]])
    sim_cluster = kmeans.predict(sim_features_cluster)[0]

    st.write(f"Predicted Cluster: {sim_cluster}")

    if rf_model:
        sim_features_ml = np.array([[sim_age_scaled, sim_wait_scaled, sim_gender_encoded, sim_dept_encoded]])
        sim_admission_prob = rf_model.predict_proba(sim_features_ml)[0][1]
        st.write(f"Predicted Admission Probability: {sim_admission_prob:.2f}")
    else:
        st.write("Admission model not available (need more data).")
