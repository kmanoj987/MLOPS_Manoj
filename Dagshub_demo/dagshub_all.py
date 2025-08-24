import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report
import pickle
import mlflow
import dagshub

# Load and prepare data
data = r'Dagshub_demo\diabetes.csv'
df = pd.read_csv(data)
X = df.drop('Outcome', axis=1)
y = df.Outcome

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

# Handle missing values
from sklearn.impute import SimpleImputer
fill = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill.fit_transform(X_train)
X_test = fill.transform(X_test)

# Initialize DagsHub
dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)

# Define models and their parameters
models = {
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            "solver": "lbfgs",
            "max_iter": 45,
            "multi_class": "auto",
            "random_state": 123
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 123
        }
    },
    'xgboost': {
        'model': xgb.XGBClassifier(),
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 123
        }
    },
    'svm': {
        'model': SVC(),
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': 123
        }
    }
}

# Train and log each model
mlflow.set_experiment("Multi_Classifier_Diabetes_Experiment")

for model_name, model_info in models.items():
    print(f"\nTraining {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Set tags
        mlflow.set_tag("author", "AJ")
        mlflow.set_tag("model_type", model_name)
        
        # Train model
        model = model_info['model']
        model.set_params(**model_info['params'])
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get metrics
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Log parameters
        mlflow.log_params(model_info['params'])
        
        # Log metrics
        metrics = {
            'accuracy': report_dict['accuracy'],
            'recall_class_0': report_dict['0']['recall'],
            'recall_class_1': report_dict['1']['recall'],
            'f1_score_macro': report_dict['macro avg']['f1-score']
        }
        mlflow.log_metrics(metrics)
        
        # Save and log model
        filename = f'{model_name}_model.pkl'
        pickle.dump(model, open(filename, 'wb'))
        mlflow.log_artifact(filename, model_name)
        
        print(f"{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_score_macro']:.4f}")

print("\nAll models have been trained and logged to MLflow")