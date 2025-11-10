# ⚡ Predicting Electric Vehicle (EV) Range and Building a Generative AI Chatbot

# WEEK-2

### 🤖  Model Creation and Training

This week focuses on the development and training of a Machine Learning (ML) model designed to predict the driving range of Electric Vehicles (EVs) using a combination of vehicle specifications, trip characteristics, and environmental factors. The objective is to construct a robust and accurate model that can subsequently be integrated into a Generative AI chatbot to provide interactive insights into vehicle range performance.
---


### 🧠 Selected Model: Random Forest Regressor

The Random Forest Regressor is an ensemble-based supervised learning algorithm that builds multiple decision trees and combines their results to improve prediction accuracy and reduce overfitting.
---

### ⚙️ Why Random Forest?

- Handles non-linear relationships between EV parameters and driving range.

- Works effectively with mixed data types (numerical and categorical).

- Provides feature importance, helping us understand key factors influencing range.

- Offers high accuracy and robustness against noise and outliers.
---

### 🧩 Model Development Steps

# Data Cleaning & Preprocessing:

- Removed missing and duplicate values.

- Encoded categorical variables using one-hot encoding.

- Split dataset into training (80%) and testing (20%) sets.

# Model Training:

- Used RandomForestRegressor from scikit-learn.

- Tuned parameters such as number of trees (n_estimators) and depth.

# Evaluation Metrics:

- Mean Absolute Error (MAE): Measures average prediction error.

- R² Score: Indicates how well the model explains the data variance.

### 📊 Model Performance
Metric	Value (Approx.)
Mean Absolute Error (MAE)	~10–15 km
R² Score	~0.85–0.90

✅ The model achieved strong prediction accuracy, showing it can effectively estimate EV driving range across different conditions.
---

# WEEK-1

### 🧠 Project Overview
Electric Vehicles (EVs) are transforming the way we travel — but a major issue remains: **range anxiety**, the fear of running out of charge before reaching the destination.  
This project aims to combine **Machine Learning (ML)** and **Generative AI (like GPT)** to solve this problem.

We will develop:
1. An **AI model** that predicts the **driving range** of an EV using trip, vehicle, and environmental data.  
2. A **Generative AI chatbot** that explains predictions, gives battery-saving tips, and interacts with users naturally.  

The final system will make EV driving smarter, more efficient, and more user-friendly.

---
### ❓ Problem Statement
EV users often struggle with:
- Uncertain driving range (affected by weather, speed, or terrain).  
- Limited insights into why their vehicle range fluctuates.  
- Lack of a friendly system to explain performance in simple terms.  

This project addresses these issues by developing:
1. A **machine learning model** to predict EV range more accurately.  
2. A **Generative AI chatbot** that explains results and provides recommendations interactively.

---

### 🔬 Technical Domain: Generative AI
**Generative AI** creates new content — text, images, or code — using transformer-based models such as GPT, DALL·E, or Gemini.  
In this project, Generative AI is used to:
- Build a **GPT-powered EV chatbot** that explains predictions and answers queries.
- Generate **synthetic driving data** when real data is missing.
- Automatically create **reports or summaries** about EV performance.
- Visualize concepts using image generation tools like **DALL·E**.

---

### 🚗 Introduction

#### 1️⃣ EV Range Prediction
Traditional methods for predicting EV range use fixed equations that don’t consider real-world variability.  
Modern **machine learning algorithms** (like Random Forest, XGBoost, and Neural Networks) can model complex interactions for better accuracy.

**Key factors influencing EV range:**
- Battery capacity (kWh)
- Motor efficiency (%)
- Vehicle weight (kg)
- Speed and driving style
- Temperature and weather
- Terrain (flat or hilly)
- Use of air conditioning or heater

**Main challenge:** obtaining accurate, real-world EV driving data.

---

#### 2️⃣ Generative AI in the Automotive Field
Generative AI (ChatGPT, Gemini, LLaMA, or DALL·E) is now being applied in smart mobility systems. It helps to:
- Build **interactive chatbots** for drivers (e.g., Mercedes-Benz ChatGPT assistant).  
- **Generate synthetic data** for training models when limited real-world data exists.  
- **Assist in route planning**, battery management, and vehicle diagnostics.  
- **Explain predictions** and improve user experience through natural conversations.

---

#### 3️⃣ Combining EV and Generative AI
Integrating predictive ML with Generative AI offers a powerful solution:
- ML predicts the **EV range** based on sensor and trip data.  
- GPT explains the results in human-friendly language and provides driving advice.  

**Example:**
> “Based on your trip details and temperature, your EV can travel approximately 220 km.  
> Slowing down by 10 km/h could extend your range by about 12 km.”

This blend of **predictive intelligence** and **conversational AI** enhances reliability, trust, and accessibility.

---

### ⚙️ Requirements

#### **Data Inputs**
| Category | Example Attributes |
|-----------|--------------------|
| Vehicle Specs | Battery capacity (kWh), Motor efficiency (%), Vehicle weight (kg) |
| Trip Data | Distance (km), Average speed (km/h), Route type (city/highway) |
| Environment | Temperature (°C), Weather, Terrain (flat/hilly), Traffic level |
| Load & Passengers | Number of passengers, Cargo weight (kg) |

#### **Outputs**
- **Predicted Range (km)** — estimated distance on a full charge.  
- **Explanation** — feature importance and reasoning behind the prediction.  
- **Chatbot Response** — user-friendly text answers from GPT.

---

### 🧩 Required Software / Libraries / APIs

| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming | Python |
| ML / DL Frameworks | scikit-learn, XGBoost, PyTorch |
| Generative AI | OpenAI API, LangChain, Gemini, DALL·E |
| Visualization | Matplotlib, Plotly |
| Data Handling | Pandas, NumPy |
| UI / Web App | Streamlit |
| Notebook Environment | Jupyter Notebook |

---

### 📊 Datasets (Kaggle Sources)

| Dataset | Description | Source |
|----------|--------------|--------|
| EVs One Electric Vehicle Dataset | EV specs and model features | [Kaggle Link](https://www.kaggle.com/datasets/geoffnel/evs-one-electric-vehicle-dataset) |


---

### Completed Work
✅Built an XGBoost regression model to predict EV driving range using vehicle, trip, and environmental features. 
✅The model was trained, tuned, and evaluated with strong RMSE and MAE scores.                
✅Feature importance highlighted battery capacity and driving conditions as key factors influencing predictions.

---

### Future Work

- Integrate a generative AI-powered chatbot for natural language interaction.
- Develop a Streamlit web app to host the prediction model and chatbot.
- Enhance model accuracy by incorporating real-time data such as traffic, battery health, and driving patterns.
- Explore advanced deep learning models like RNNs and GNNs for improved predictions.

---
