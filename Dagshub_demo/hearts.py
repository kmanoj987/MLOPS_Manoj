import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import mlflow
import dagshub
import matplotlib.pyplot as plt

data_path = r"data_sets1\heart.csv"

# dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)
mlflow.set_experiment("Hearts_Experiment")

df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#imputing
fil = SimpleImputer(strategy="median")
X_train = fil.fit_transform(X_train)
X_test = fil.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape)

clf1 = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
clf2 = GradientBoostingClassifier(n_estimators=15, learning_rate=0.002, random_state=42)
clf3 = XGBClassifier(n_estimators=55, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")

voting_clf = VotingClassifier(
    estimators=[("rf", clf1), ("gb", clf2), ("xgb", clf3)],
    voting="soft"
)
with mlflow.start_run(run_name="VotingClassifier_RF_GB_XGB"):

    # Train Voting Classifier
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    y_probs = voting_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    # Evaluate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(report)
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

        # --- Plot ROC curve with AUC area ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.fill_between(fpr, tpr, alpha=0.3, color="blue", label="AUC Area")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with AUC')
    plt.legend(loc='lower right')
    plt.grid()

    # Save the plot
    roc_plot_path = r"Dagshub_demo\roc_auc_curve.png"
    plt.savefig(roc_plot_path)
    plt.close()

    # Log to MLflow
    mlflow.log_artifact(roc_plot_path)
    mlflow.log_metric("AUC Score", auc_score)

    print(f"AUC Score: {auc_score:.3f}")
    print(f"ROC + AUC Curve saved to: {roc_plot_path}")


    # 'f1_score_macro': report_dict['macro avg']['f1-score']
    # Save model as artifact
    filename = r'Dagshub_demo\voting_classifier.pkl'
    pickle.dump(voting_clf, open(filename, 'wb'))
    #save as articfact
    mlflow.log_artifact(r"Dagshub_demo\voting_classifier.pkl", artifact_path="models")
