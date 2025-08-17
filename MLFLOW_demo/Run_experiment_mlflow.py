# ============================================
# Load model from MLflow Experiment and Predict
# ============================================

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 1. Tracking URI must be the same as training
mlflow.set_tracking_uri("file:C:\Users\ankit_aj\Desktop\MLOPS-case_studies\Demo_050725_DVC\SKILLFY_21_JUNE_25\mlruns")


# 2. Replace with your experiment name or ID
experiment_name = "1708_rf_diabetes_prediction_experiment"
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)
experiment_id = exp.experiment_id
print("Using Experiment ID:", experiment_id)

# 3. Get latest run from this experiment
client = MlflowClient()
runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)

if not runs:
    raise Exception("No runs found in experiment:", experiment_name)

run = runs[0]
run_id = run.info.run_id
# can we pass the run ID directly?

print("Latest Run ID:", run_id)

# 4. Construct model path
model_uri = f"runs:/{run_id}/xgboost_classifier"
print("Loading model from:", model_uri)

# 5. Load model
model = mlflow.xgboost.load_model(model_uri)

# 6. Run predictions on X_test (must reuse the same X_test as in training script)
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:10])
print("True Labels:", y_test[:10])