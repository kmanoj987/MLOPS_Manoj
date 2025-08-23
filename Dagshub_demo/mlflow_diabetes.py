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


# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 45,
    "multi_class": "auto",
    "random_state": 123,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

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