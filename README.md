# 🏙️ UrbanPulse – Smart City Problem Intelligence System

**An end-to-end data analytics, machine learning, and interactive visualization project for analyzing NYC 311 service requests.**

UrbanPulse transforms NYC 311 service request records into meaningful insights through data preprocessing, exploratory data analysis, statistical analysis, machine learning, geographical visualization, and an interactive dashboard.

The project explores complaint patterns, identifies temporal and geographical trends, examines resolution times, and predicts estimated complaint resolution time using machine learning.

---

## 🌐 Live Dashboard

https://urbanpulseak.streamlit.app/

---

## 📌 Project Overview

New York City's 311 service request system receives complaints related to various public services, including noise, sanitation, street conditions, and infrastructure.

UrbanPulse analyzes a collected sample of these service requests to understand:

- Which complaint categories appear most frequently?
- How do complaint patterns vary across months, weekdays, and hours?
- How do resolution times differ across complaint categories?
- Where are complaints geographically concentrated in the collected records?
- Can machine learning estimate the resolution time of a complaint?

The project combines data analysis and predictive modeling into a unified interactive application.

---

## ✨ Key Features

### 1. Data Collection & Preprocessing
- Collection of NYC 311 service request records.
- Data cleaning and handling of missing values.
- Date and time conversion.
- Resolution-time calculation.
- Preparation of structured datasets for analysis.

### 2. Exploratory Data Analysis
- Complaint category distribution.
- Borough-wise complaint distribution.
- Monthly and hourly complaint trends.
- Analysis of complaint status and resolution patterns.

### 3. Statistical Analysis & Feature Engineering
- Descriptive statistical measures.
- Correlation and covariance analysis.
- Resolution-time normalization and binning.
- Extraction of temporal features such as month, hour, and day of the week.

### 4. Machine Learning & Predictive Analytics
Multiple regression models were developed and evaluated:

- Dummy Regressor (Baseline)
- Linear Regression
- Random Forest Regressor

The Random Forest model achieved the best performance among the evaluated models.

| Model | MAE (hours) | RMSE (hours) | R² Score |
|---|---:|---:|---:|
| Baseline | 246.17 | 993.39 | -0.0647 |
| Linear Regression | 242.55 | 778.35 | 0.3463 |
| Random Forest | **220.97** | **742.66** | **0.4049** |

The selected model achieved an MAE of approximately 221 hours on the test set.

### 5. Advanced Data Visualization
- Interactive Plotly charts.
- Complaint distribution visualizations.
- Temporal heatmaps.
- Geographical complaint heatmaps.
- Interactive maps with clustered complaint markers.

### 6. Interactive Streamlit Dashboard
The final dashboard integrates the project's analytical and predictive components.

It includes:

- Interactive filtering by date, complaint category, borough, and status.
- KPI cards and summary statistics.
- Complaint analysis.
- Temporal analysis.
- Geographical analysis.
- Machine learning-based resolution-time prediction.
- Filtered dataset download.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Geographical Visualization | Folium |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Interactive Dashboard | Streamlit |
| Development Environment | VS Code |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
UrbanPulse/
│
├── data/
│   ├── raw/
│   │   └── UrbanPulse_raw.csv
│   │
│   └── processed/
│       ├── UrbanPulse_cleaned.csv
│       └── UrbanPulse_Phase3.csv
│
├── outputs/
│   ├── Phase1/
│   ├── Phase2/
│   ├── Phase3/
│   ├── Phase4/
│   │   ├── UrbanPulse_best_model.pkl
│   │   └── prediction.csv
│   └── Phase5/
│
├── phase1.py
├── phase2.py
├── phase3.py
├── phase4.py
├── phase5.py
├── phase6_dashboard.py
│
├── requirements.txt
└── README.md
```

*The structure above represents the main project organization. Individual output files may vary depending on the implementation.*

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd UrbanPulse
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Dashboard

```bash
streamlit run phase6_dashboard.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

**Note:** The dashboard requires the processed dataset and trained model to be available at their expected project paths.

---

## 📊 Machine Learning Results

The regression models were evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted resolution times.
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures the proportion of target variance explained by the model.

Random Forest performed best among the three evaluated models, achieving:

- MAE: 220.97 hours
- RMSE: 742.66 hours
- R²: 0.4049

Although the model identifies useful patterns, its predictions still have considerable errors and should be treated as estimates rather than guaranteed resolution times.

---

## 🗺️ Dataset

**Source:** NYC Open Data – 311 Service Requests from 2020 to Present

Dataset: [NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9)

The project uses records collected for 2024, including complaint details, timestamps, agency information, boroughs, and geographical coordinates.

**Data Collection Note:** The collection process retrieves up to 3,000 records per month. Therefore, the dataset represents a collected sample and should not be interpreted as a complete or statistically representative record of all NYC 311 complaints.

---

## 🔮 Future Improvements

- Integrate automated and scheduled data collection.
- Expand the dataset to include more historical records.
- Improve predictive performance through additional feature engineering and model experimentation.
- Introduce more detailed geographical and temporal analysis.
- Explore real-time data integration.

---

## 👩‍💻 Author

**Akashita**  
B.Tech – Computer Science and Engineering  
Dr. B. R. Ambedkar National Institute of Technology, Jalandhar

---

## ⭐ Acknowledgements

- NYC Open Data for providing public 311 service request data.
- Python open-source ecosystem for the libraries used in this project.

---

**UrbanPulse – Turning public service requests into actionable insights.**
