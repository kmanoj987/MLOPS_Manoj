import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import pickle

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_csv(config["data"]["path"])
X = df.drop(columns=['sales'])
y = df['sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
)

# Train model
model_cfg = config["model"]
model = GradientBoostingRegressor(**model_cfg["params"])
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)