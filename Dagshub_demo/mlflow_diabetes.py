import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC as SVClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import dagshub
import mlflow


# dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)
# mlflow.set_experiment("XGBOOST 23_08")

data = r'Dagshub_demo\diabetes.csv'
df = pd.read_csv(data)
df.shape

df.head()
X = df.drop('Outcome',axis=1) # predictor feature coloumns
y = df.Outcome


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.20, random_state = 10,shuffle=True)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))

#impute with mean all 0 readings
fill = SimpleImputer(missing_values = 0 , strategy ="mean")#impute with mean all 0 readings
#fill = Imputer(missing_values = 0 , strategy ="mean", axis=0)

X_train = fill.fit_transform(X_train)
X_train = fill.fit_transform(X_train)
X_test = fill.transform(X_test)


#standardising the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# logistic regression

# log_reg = LogisticRegression(max_iter=1000)
# param_grid = {
#     "penalty": ["l1", "l2", "elasticnet", "none"],  # Regularization types
#     "C": [0.01, 0.1, 1, 10, 100],                  # Inverse regularization strength
#     "solver": ["saga", "liblinear"],                # Solvers that support l1 & elasticnet
#     "l1_ratio": [None, 0.2, 0.5, 0.8]               # Only used when penalty='elasticnet'
# }

# grid_search = GridSearchCV(
#     estimator=log_reg,
#     param_grid=param_grid,
#     scoring="accuracy",
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )
# grid_search.fit(X_train, y_train)

# print("\nBest Parameters:", grid_search.best_params_)
# print(f" Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)


# # for xgboost
# Define XGBoost model
xgb_clf = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "reg_lambda": [1, 2, 5],
    "reg_alpha": [0, 0.1, 1]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

# Get best parameters and accuracy
best_params_xgb = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# # Train the model

# xgb_clf = xgb.XGBClassifier(**params)
# xgb_clf.fit(X_train, y_train)# Predict on the test set
# y_pred = xgb_clf.predict(X_test)

#gradient boosting
# params = {
#     "n_estimators": 750,
#     "learning_rate": 0.015,
#     "max_depth": 6,
#     "subsample": 0.7,
#     "min_samples_split": 6,
#     "min_samples_leaf": 2,
#     'criterion':'squared_error',
#     'ccp_alpha':0.005,
#     'random_state': 42,
# }

# gbclf = GradientBoostingClassifier(**params)
# gbclf.fit(X_train, y_train)

# y_pred = gbclf.predict(X_test)





# neural networks

# mlp = MLPClassifier(
#     hidden_layer_sizes=(128,64, 32, 16),  # 3 hidden layers
#     activation='relu',
#     solver='lbfgs',
#     alpha=0.0001,
#     batch_size=32,
#     learning_rate='adaptive',
#     max_iter=1000,
#     random_state=42,
#     shuffle=True,
#     early_stopping=True,
#     validation_fraction=0.1,
# )

# mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_test)

report = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)


print(report)
# print(report_dict)


#mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# with mlflow.start_run():
#     mlflow.set_tag("author", "satish")  # Replace with your actual name
#     mlflow.log_params(params)
#     mlflow.log_metrics({
#         'accuracy': report_dict['accuracy'],
#         'recall_class_0': report_dict['0']['recall'],
#         'recall_class_1': report_dict['1']['recall'],
#         'f1_score_macro': report_dict['macro avg']['f1-score']
#     })
#     # Save the model to a file
#     filename = 'XGBoost_Model.pkl'
#     pickle.dump(xgb_clf, open(filename, 'wb'))
#     # Log the model file as an artifact
#     mlflow.log_artifact(filename, "Xgboost_model")