Tourism Package Purchase Prediction 🚀

An end-to-end Machine Learning and MLOps project that predicts whether a customer is likely to purchase a tourism package, helping businesses improve customer targeting and marketing decisions.

🌐 Live Demo

Try the Streamlit App | GitHub Repository | Hugging Face Dataset

⚙️ Project Workflow

Dataset → Data Preparation → Model Training → MLflow Tracking → GitHub Actions → Streamlit Deployment

* Data: Tourism customer data; target variable: ProdTaken.
* Models: Logistic Regression, Random Forest, Gradient Boosting.
* MLOps: Hugging Face dataset registration, MLflow experiment tracking, and automated GitHub Actions workflow.
* Deployment: Streamlit application hosted on Render for interactive customer predictions.

📊 Model Performance

Model	Accuracy	F1-score	ROC-AUC
Logistic Regression	73.12%	51.53%	81.95%
Random Forest	88.50%	67.13%	92.87%
Gradient Boosting	92.62%	77.98%	94.97%

Selected Model: Gradient Boosting, based on the highest F1-score.

📁 Repository Structure

├── data/                  # Tourism dataset
├── src/                   # Registration, preparation & training scripts
├── models/                # Trained model & evaluation metrics
├── deployment/             # Streamlit app & Docker configuration
└── .github/workflows/      # GitHub Actions MLOps pipeline

🛠️ Tech Stack

Python · Pandas · Scikit-learn · MLflow · Hugging Face · Streamlit · Docker · GitHub Actions · Render

⸻

Outcome: An automated model-training workflow and a live web application for tourism customer purchase prediction.
