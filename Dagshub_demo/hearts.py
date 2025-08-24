import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import mlflow
import dagshub

data_path = r"data_sets1\heart.csv"

dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)
mlflow.set_experiment("Hearts_Experiment")

df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

#imputing
fil = SimpleImputer(strategy="median")
X_train = fil.fit_transform(X_train)
X_test = fil.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape)

clf1 = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
clf2 = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
clf3 = XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")

voting_clf = VotingClassifier(
    estimators=[("rf", clf1), ("gb", clf2), ("xgb", clf3)],
    voting="soft"
)
with mlflow.start_run(run_name="VotingClassifier_RF_GB_XGB"):

    # Train Voting Classifier
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    # Evaluate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    # Log params of VotingClassifier
    mlflow.log_param("voting", "soft")
    mlflow.log_param("estimators", ["RandomForest", "GradientBoosting", "XGBoost"])

    # Log individual base model params
    mlflow.log_params({f"rf_{k}": v for k, v in clf1.get_params().items()})
    mlflow.log_params({f"gb_{k}": v for k, v in clf2.get_params().items()})
    mlflow.log_params({f"xgb_{k}": v for k, v in clf3.get_params().items()})

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
    mlflow.log_metric("f1_score_macro", report["macro avg"]["f1-score"])
    mlflow.log_metric("recall_class_1", report["1"]["recall"])
    # 'f1_score_macro': report_dict['macro avg']['f1-score']
    # Save model as artifact
    filename = r'Dagshub_demo\voting_classifier.pkl'
    pickle.dump(voting_clf, open(filename, 'wb'))
    #save as articfact
    mlflow.log_artifact("voting_classifier.pkl", artifact_path="models")
