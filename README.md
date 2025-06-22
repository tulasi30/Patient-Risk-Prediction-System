# Patient-Risk-Prediction-System

The project provides healthcare providers and administrators with powerful analytics to identify at-risk patients, predict disease progression, and allocate resources effectively. This Streamlit-powered web application offers a comprehensive suite of tools for chronic disease management and risk prediction.

**#Key Features:**

  **Comprehensive Risk Assessment:** Visualizes patient risk categories (Low, Moderate, High) and identifies high-risk populations based on age, geographical location, and medical conditions.
  
 ** Population Demographics Analysis: **Provides in-depth analysis of patient demographics including age distribution, gender, race, and city, with interactive filters for granular data exploration.
  
  **Disease Progression & Forecasting: **Tracks and forecasts chronic disease trends over time using statistical models (e.g., ARIMA) to predict future case prevalence.
  
  **Lifestyle Impact Analysis:** Illustrates the correlation between lifestyle factors, such as smoking and alcohol consumption, and patient risk profiles.
  
  **Machine Learning Predictions:** Integrates machine learning models (e.g., Logistic Regression) to predict patient risk, highlighting significant contributing factors for targeted                     interventions.
  
  **Intuitive User Interface:** Features a modular, web-based interface built with Streamlit and Python, offering dedicated pages for Risk Overview, Trends, Geography, Lifestyle Impact, and a             Data Explorer.
  
  **Dynamic Visualizations:** Employs a variety of interactive charts and graphs (bar charts, line graphs, scatter plots, KPI cards, maps) using Matplotlib, Seaborn, and Plotly, optimized for             both high-level dashboards and detailed analyses.
  
#Technical Stack:

  **Programming Language:** Python
  
  **Web Framework:** Streamlit
  
  **Data Manipulation:** Pandas, NumPy
  
  **Data Visualization:** Matplotlib, Seaborn, Plotly, Altair
  
 ** Machine Learning: **Scikit-learn (Logistic Regression, StandardScaler)
  
  **Time Series Analysis: **Statsmodels (ARIMA)
  
  **Database Interaction (Conceptual):** SQL (for data extraction and validation)
  
  **Business Intelligence Tool (Conceptual):** Power BI (for effective visual creation)
  
#Data Requirements:

The application is designed to operate with structured healthcare datasets, typically provided in CSV format. These include:

  **patients.csv:** Contains core patient demographic information.
  
  **conditions.csv:** Records medical conditions and diagnoses.
  
  **observations.csv:** Stores clinical observations and measurements (e.g., vital signs, lab results).
  
  **encounters.csv:** Details hospital visits and patient encounters.
