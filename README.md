
---

# 🏥 Hospital Emergency Analytics Dashboard

This project is a **Streamlit-based interactive dashboard** for analyzing hospital emergency room data. It integrates **data exploration, clustering, machine learning, deep learning, NLP sentiment analysis, optimization, and patient scenario simulation** in a single application.

## 🚀 Features

### 1. Data Upload & Exploration

* Upload CSV files containing hospital ER data.
* View raw data and filtered data based on age, gender, and department.
* Clean and preprocess data with **scaling and encoding**.

### 2. Patient Clustering

* Group patients using **KMeans clustering** based on age, wait time, and satisfaction.
* Interactive scatter plot visualization with Plotly.

### 3. Machine Learning: Admission Prediction

* Predict patient admission using **Random Forest Classifier**.
* Display **classification report**.
* Supports **What-If scenario predictions** for hypothetical patient data.

### 4. Deep Learning Admission Prediction

* Build and train a **simple feed-forward neural network** using TensorFlow/Keras.
* Display training accuracy per epoch using Plotly charts.

### 5. NLP: Patient Complaint Analysis

* Analyze patient complaint text using **TF-IDF** and **Multinomial Naive Bayes**.
* Predict sentiment of new complaints (Positive / Negative).
* Show top predictive words contributing to satisfaction scores.

### 6. Optimization

* Allocate hospital resources (beds and doctors) using **linear programming** (PuLP).
* Interactive slider to adjust total resources and weights.

### 7. Department Analytics

* Visualize **average satisfaction by department** with interactive bar charts.

### 8. Patient Lookup & Scenario Simulator

* Search for patients by **Patient ID** and see their cluster assignment, admission probability, wait time, and satisfaction score.
* Perform **What-If simulations** for hypothetical patients to explore predicted outcomes.

---

## 📁 Project Structure

```
hospital-dashboard/
│
├─ hospital_dashboard.py      # Main Streamlit app
├─ requirements.txt           # Python dependencies
├─ README.md                  # Project documentation
└─ sample_data.csv            # Example dataset (optional)
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/hospital-dashboard.git
cd hospital-dashboard
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run hospital_dashboard.py
```

---

## 📊 Dependencies

* `streamlit` – interactive web app framework
* `pandas` – data manipulation
* `numpy` – numerical computation
* `plotly` – interactive visualization
* `scikit-learn` – machine learning
* `tensorflow` – deep learning
* `pulp` – optimization solver
* `seaborn`, `matplotlib` – additional visualizations

---

## 📝 Usage

1. Upload your **hospital ER CSV dataset** with columns like:

* Patient Id
* Patient Age
* Patient Gender
* Patient Race
* Department Referral
* Patient Waittime
* Patient Satisfaction Score
* Patient Admission Flag
* Patient Complaint Text (optional)

2. Use sidebar filters to explore data by **age, gender, and department**.
3. Navigate tabs to access **clustering, ML predictions, deep learning, NLP, optimization, and what-if simulations**.
4. For patient-specific analysis, enter a **Patient ID** to view their details and predictions.
5. For hypothetical scenarios, adjust sliders and dropdowns in the **What-If Scenario Simulator**.

---

## ⚙️ Notes

* Minimum 20 records required for Random Forest admission prediction.
* Minimum 50 records required for Deep Learning admission prediction.
* SHAP explainability has been removed to avoid shape mismatch issues.

---

## 📌 Author

**Kuppireddy Bhageeratha Reddy**

* GitHub: [https://github.com/kuppireddybhageerathareddy1110/hospital-dashboard.git](https://github.com/kuppireddybhageerathareddy1110/hospital-dashboard.git)
* Streamlit App: [Hospital Dashboard](https://hospital-dashboardgit-bhageeratha.streamlit.app/)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---
