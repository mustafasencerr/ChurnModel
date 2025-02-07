import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import joblib
from sklearn.metrics import f1_score

#%% Reading Datasets
train_data = pd.read_csv("Datasets/train.csv")
validation_data = pd.read_csv("Datasets/validation.csv")
test_data = pd.read_csv("Datasets/test.csv")

X_train = train_data.drop(columns = ["churn"])
y_train = train_data["churn"]

X_validation = validation_data.drop(columns = ["churn"])
y_validation = validation_data["churn"]

X_test = test_data.drop(columns = ["churn"])
y_test = test_data["churn"]

#%% Hyperparameter optimization with Optuna.
def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500, step = 10),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 2, log = True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log = True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log = True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log = True),
    }
    
    # Xgb model
    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state = 42)

    # Fitting model
    model.fit(
        X_train, y_train, 
        eval_set = [(X_validation, y_validation)], 
        verbose = False
    )
    
    # Calculate F1-Score
    y_pred_val = model.predict(X_validation)
    f1 = f1_score(y_validation, y_pred_val)
    
    return f1

# Determine best hyperparameters
study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 10)

# Retrain the model using the best hyperparameters.
best_params = study.best_params
print(best_params)
optimized_xgb = xgb.XGBClassifier(**best_params, eval_metric = "logloss", random_state = 42, use_label_encoder = False)
history = optimized_xgb.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_validation, y_validation)], verbose = True)

# Train - Validation Loss
results = optimized_xgb.evals_result()
train_loss = results["validation_0"]["logloss"]
val_loss = results["validation_1"]["logloss"]

plt.figure(figsize = (8, 5))
plt.plot(train_loss, label = "Train Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("XGBoost")
plt.legend()
plt.show()


#%% Save model
model_path = "Model/xgb_model.pkl"
joblib.dump(optimized_xgb, model_path)
