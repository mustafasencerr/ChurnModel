import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle
import functions as fn

custom_data_path = input("Please enter your custom data path: ")

# Utils
with open("Utils/categorical_features_train_map.pkl", "rb") as f:
    categorical_features_train_map = pickle.load(f)
    
with open("Utils/top_30_features.pkl", "rb") as f:
    top_30_features = pickle.load(f)

with open("Utils/mapping_missing_statistic_train_map.pkl", "rb") as f:
    mapping_missing_statistic_train_map = pickle.load(f)

with open("Utils/iqr_bounds_train.pkl", "rb") as f:
    iqr_bounds_train = pickle.load(f)
    
with open("Utils/train_format_feat.pkl", "rb") as f:
    train_format_feat = pickle.load(f)
    
with open("Utils/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


custom_data = pd.read_csv(custom_data_path)

custom_data = fn.processing_data(df = custom_data, categorical_feat = categorical_features_train_map,
                                     train_missing_stats = mapping_missing_statistic_train_map,
                                     iqr_bounds = iqr_bounds_train, train_format = train_format_feat)

X_custom = custom_data.drop(columns=["churn"])
y_custom = custom_data["churn"]

X_custom_scaled = scaler.transform(X_custom)  #It must be done using the scaler applied to the training data.

X_custom_scaled = pd.DataFrame(X_custom_scaled, columns = X_custom.columns,index = X_custom.index)
X_custom_scaled['churn'] = y_custom
X_custom_scaled = X_custom_scaled[top_30_features]


X_custom = X_custom_scaled.drop(columns=["churn"])
y_custom = X_custom_scaled["churn"]


model_path = "Model/xgb_model.pkl"
xgb_model = joblib.load(model_path)

y_custom_test = xgb_model.predict(X_custom)

accuracy = accuracy_score(y_custom, y_custom_test)
f1 = f1_score(y_custom, y_custom_test)
recall = recall_score(y_custom, y_custom_test)
precision = precision_score(y_custom,y_custom_test)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test Precision: {precision:.4f}")
