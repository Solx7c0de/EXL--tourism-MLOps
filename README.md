Tourism Package Purchase Prediction — MLOps Project

Project Overview

This project uses machine learning to predict whether a customer is likely to purchase a tourism package. The objective is to help tourism businesses identify potential customers, improve marketing targeting, and make better use of promotional resources.

The project covers the end-to-end machine learning lifecycle, including data preparation, model experimentation, experiment tracking, automated pipeline execution, and web application deployment.

Live Application

Try the application: https://exl-tourism-mlops.onrender.com

The Streamlit application allows users to upload customer data, obtain purchase predictions, view a prediction summary, and download the results.

GitHub Repository

Repository: https://github.com/Solx7c0de/EXL–tourism-MLOps

Dataset

The project uses a tourism customer dataset containing demographic, travel, and customer interaction features.

* Target variable: ProdTaken
* 1 indicates that the customer purchased the tourism package.
* 0 indicates that the customer did not purchase the package.

The CustomerID and Unnamed: 0 columns are excluded from model training because they are identifiers rather than useful predictive features.

Hugging Face Dataset: https://huggingface.co/datasets/vzz7/tourism-dataset

Dataset registration is automated through the GitHub Actions workflow using the Hugging Face Hub API.

Machine Learning Workflow

The project implements the following workflow:

1. Data Registration: Uploads the dataset to Hugging Face.
2. Data Preparation: Removes identifier columns, separates features and target, and creates stratified training and testing datasets.
3. Preprocessing: Handles missing values, scales numerical features, and encodes categorical features through scikit-learn pipelines.
4. Model Training: Trains and evaluates Logistic Regression, Random Forest, and Gradient Boosting classifiers.
5. Experiment Tracking: Uses MLflow to record model parameters, evaluation metrics, and model artifacts.
6. Model Selection: Selects the model with the highest F1-score among the evaluated candidates.
7. Deployment: Serves customer predictions through a Streamlit web application hosted on Render.
8. Automation: Uses GitHub Actions to execute the data registration, preparation, training, validation, and artifact-generation workflow.

Model Performance

The automated pipeline produced the following evaluation results on its held-out test set:

Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	73.12%	39.46%	74.21%	51.53%	81.95%
Random Forest	88.50%	74.62%	61.01%	67.13%	92.87%
Gradient Boosting	92.62%	91.53%	67.92%	77.98%	94.97%

Gradient Boosting was selected because it achieved the highest F1-score among the three evaluated models.

Its precision of approximately 91.53% indicates that most customers predicted to purchase were actual purchasers in the evaluation set. Its recall of approximately 67.92% also indicates that some actual purchasers were missed, making threshold optimization a possible future improvement.

Note: These figures are from the automated pipeline’s evaluation run. They should not be interpreted as a guarantee of future business performance.

Repository Structure

EXL--tourism-MLOps/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── data/
│   └── tourism.csv
├── src/
│   ├── register_data.py
│   ├── data_prep.py
│   └── train_model.py
├── models/
│   ├── best_model.pkl
│   └── best_metrics.json
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── models/
│       └── best_model.pkl
├── requirements.txt
└── README.md

The structure above describes the intended project layout; generated files and other notebook files may vary depending on the current repository contents.

Key Files

File	Purpose
src/register_data.py	Registers the tourism dataset on Hugging Face
src/data_prep.py	Prepares the dataset for model training and evaluation
src/train_model.py	Trains models, compares metrics, tracks experiments, and saves the selected model
models/best_model.pkl	Stores the selected trained model
models/best_metrics.json	Stores the selected model’s evaluation metrics
deployment/app.py	Implements the Streamlit prediction application
deployment/Dockerfile	Defines the container configuration for deployment
.github/workflows/pipeline.yml	Automates the MLOps workflow using GitHub Actions

MLOps Automation with GitHub Actions

The workflow is configured to run on pushes to the main branch, pull requests targeting main, and manual workflow dispatches.

During execution, it installs dependencies, registers the dataset when appropriate, prepares the data, trains and evaluates models, validates the saved model, checks the Streamlit application syntax, and uploads generated artifacts.

The workflow has been executed successfully, and its generated artifacts include the model comparison results, selected model, evaluation metrics, and MLflow tracking files.

Experiment Tracking

MLflow is used to organise model experiments and record parameters and evaluation metrics. The GitHub Actions run also generates tracking artifacts that can be downloaded for inspection.

The current tracking configuration uses a local SQLite database on the workflow runner. The generated tracking files are retained through workflow artifacts rather than a permanently hosted MLflow server.

Deployment

The Streamlit application is hosted separately on Render:

Live URL: https://exl-tourism-mlops.onrender.com

Users can upload a customer dataset and receive predictions classifying customers as likely or unlikely to purchase the tourism package.

The interface also displays a prediction summary and supports downloading the prediction results.

The current GitHub Actions workflow trains and saves models but does not automatically replace the model used by the live Render application.

Technology Stack

* Language: Python
* Data processing: Pandas, NumPy
* Machine learning: scikit-learn
* Experiment tracking: MLflow
* Dataset registration: Hugging Face Hub
* Web application: Streamlit
* Containerization: Docker
* CI/CD and workflow automation: GitHub Actions
* Application hosting: Render
* Version control: Git and GitHub

Running the Application Locally

Clone the repository:

git clone https://github.com/Solx7c0de/EXL--tourism-MLOps.git
cd EXL--tourism-MLOps

Install the deployment dependencies:

pip install -r deployment/requirements.txt

Run the Streamlit application:

streamlit run deployment/app.py

Ensure that the trained model exists at the model path expected by the application.

Future Improvements

* Tune the classification threshold according to marketing objectives and the relative costs of false positives and false negatives.
* Add cross-validation and more systematic hyperparameter optimization.
* Separate model selection from final test-set evaluation.
* Introduce persistent, remotely hosted MLflow tracking.
* Automate deployment of validated model artifacts to Render.
* Monitor prediction quality and model performance as new customer data becomes available.

Conclusion

This project demonstrates an end-to-end machine learning workflow for tourism package purchase prediction. It combines customer data preparation, model experimentation, experiment tracking, GitHub Actions automation, and a deployed Streamlit application to demonstrate how predictive analytics can support tourism marketing decisions.
