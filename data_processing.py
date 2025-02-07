import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import functions as fn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import QuantileTransformer
import pickle
from scipy.stats import pointbiserialr

#%% Obtain Datasets (Train - Validation - Test)

churn_data = pd.read_csv("Datasets/churn_data.csv")
train_validation_data, test_data = train_test_split(churn_data, test_size = 0.15, random_state = 42)
train_data, validation_data = train_test_split(train_validation_data, test_size = 0.15, random_state = 42)

#%% 1 - Noisy Values

train_data.loc[train_data['user_lifetime'] > 4380, 'user_lifetime'] = np.nan

noisy_features_about_day = ["user_no_outgoing_activity_in_days" , "reloads_inactive_days" , "calls_outgoing_inactive_days",
                           "calls_outgoing_to_onnet_inactive_days", "calls_outgoing_to_offnet_inactive_days", "calls_outgoing_to_abroad_inactive_days",
                           "sms_outgoing_inactive_days", "sms_outgoing_to_onnet_inactive_days", "sms_outgoing_to_offnet_inactive_days",
                           "sms_outgoing_to_abroad_inactive_days", "gprs_inactive_days"]

train_data = fn.replace_large_values_with_nan(train_data, noisy_features_about_day)

impossible_negative_features = ["user_spendings", "reloads_sum", "sms_outgoing_spendings", "sms_outgoing_to_offnet_spendings",
                                "sms_incoming_spendings", "gprs_spendings", "last_100_reloads_sum"]

train_data = fn.replace_negative_values_with_nan(train_data, impossible_negative_features)

#%% 2 - Handling Missing Values

missing_count = fn.count_missing_values(train_data)

train_data.drop(["gprs_inactive_days"], axis=1 , inplace = True) # missing ratio = %92.25 !

categorical_features = fn.find_binary_categorical_columns(train_data)
mapping_statistic_grouped = fn.calculate_grouped_statistics(train_data, "churn", categorical_features)
train_data = fn.fill_missing_values(train_data, mapping_statistic_grouped, "churn")

# Missing Values Mapping
categorical_features_train = fn.find_binary_categorical_columns(train_data)
mapping_missing_statistic_train = fn.calculate_statistics_train(train_data, categorical_features_train)

#%% 3 - Handling Outlier Values

# Prepare train dataset for dbscan
dbscan_df = train_data.copy()
scaler = StandardScaler()
dbscan_df_scaled = scaler.fit_transform(dbscan_df)

# DBSCAN 
dbscan = DBSCAN(eps = 2, min_samples = 16)
labels = dbscan.fit_predict(dbscan_df_scaled)

# Outlier values
dbscan_df["cluster"] = labels
outlier_indexes = list(dbscan_df[dbscan_df["cluster"]== -1].index)

train_data = train_data.drop(index = outlier_indexes)
train_data = train_data.reset_index(drop = True)

outlier_count = fn.count_outlier_values(train_data)
iqr_bounds = {feature: fn.get_iqr_bounds(train_data, feature) for feature in train_data.columns}
train_data = fn.clip_outliers(train_data, iqr_bounds, target_column = "churn", categorical_features = categorical_features)

#%% 4 - Unnecessary Features

train_data.drop(["year","month","user_account_id"], axis = 1, inplace = True)
constant_features = [col for col in train_data.columns if train_data[col].nunique() == 1]
train_data.drop(constant_features, axis = 1, inplace = True)

#%% 5 - Feature Extraction

train_data["customer_active_months"] = (train_data["user_lifetime"] / 30).astype(int)

train_data["reloads_per_month"] = np.where(
    train_data["customer_active_months"] == 0, 0,
    train_data["reloads_count"] / train_data["customer_active_months"]
)

train_data["calls_per_month"] = np.where(
    train_data["customer_active_months"] == 0, 0,
    train_data["calls_outgoing_count"] / train_data["customer_active_months"]
)

train_data["sms_per_month"] = np.where(
    train_data["customer_active_months"] == 0, 0,
    train_data["sms_outgoing_count"] / train_data["customer_active_months"]
)

train_data["offnet_call_ratio"] = np.where(
    train_data["calls_outgoing_count"] == 0, 0,
    train_data["calls_outgoing_to_offnet_count"] / train_data["calls_outgoing_count"]
)

train_data["international_call_ratio"] = np.where(
    train_data["calls_outgoing_count"] == 0, 0,
    train_data["calls_outgoing_to_abroad_inactive_days"] / train_data["calls_outgoing_count"]
)

train_data["sms_offnet_ratio"] = np.where(
    train_data["sms_outgoing_count"] == 0, 0,
    train_data["sms_outgoing_to_offnet_count"] / train_data["sms_outgoing_count"]
)

epsilon = 1e-6
train_data["reload_ratio"] = np.where(
    train_data["reloads_count"] == 0, 0,
    train_data["reloads_count"] / (train_data["user_lifetime"] + epsilon)
)

train_data["avg_spending_per_month"] = np.where(
    train_data["customer_active_months"] == 0, 0,
    train_data["user_spendings"] / train_data["customer_active_months"]
)

train_data["avg_call_duration"] = np.where(
    train_data["calls_outgoing_count"] == 0, 0,
    train_data["calls_outgoing_duration"] / train_data["calls_outgoing_count"]
)

train_data["avg_sms_per_reload"] = np.where(
    train_data["reloads_count"] == 0, 0,
    train_data["sms_outgoing_count"] / train_data["reloads_count"]
)

train_data["reload_change"] = np.where(
    train_data["reloads_sum"] == 0, 0,
    train_data["last_100_reloads_sum"] / train_data["reloads_sum"]
)

train_data["call_duration_change"] = np.where(
    train_data["calls_outgoing_duration"] == 0, 0,
    train_data["last_100_calls_outgoing_duration"] / train_data["calls_outgoing_duration"]
)

train_data["sms_count_change"] = np.where(
    train_data["sms_outgoing_count"] == 0, 0,
    train_data["last_100_sms_outgoing_count"] / train_data["sms_outgoing_count"]
)

train_data["avg_call_duration"] = np.where(
    train_data["calls_outgoing_count"] == 0, 
    0,
    train_data["calls_outgoing_duration"] / train_data["calls_outgoing_count"]
)

train_data["daily_call_count"] = np.where(
    train_data["user_lifetime"] == 0, 
    0,
    train_data["calls_outgoing_count"] / train_data["user_lifetime"]
)

train_data["daily_sms_count"] = np.where(
    train_data["user_lifetime"] == 0, 
    0,
    train_data["sms_outgoing_count"] / train_data["user_lifetime"]
)

train_data["offnet_call_ratio"] = np.where(
    train_data["calls_outgoing_count"] == 0, 
    0,
    train_data["calls_outgoing_to_offnet_count"] / train_data["calls_outgoing_count"]
)

services = ['user_has_outgoing_calls', 'user_has_outgoing_sms', 'user_use_gprs']
train_data["multi_service_score"] = train_data[services].sum(axis=1)

#%% 6 - Correlation Analysis

numerical_features = [col for col in train_data.columns if col not in categorical_features_train and col != 'churn']

biserial_corr = {}
for col in numerical_features:
    corr, _ = pointbiserialr(train_data[col], train_data['churn'])
    biserial_corr[col] = corr

biserial_corr = pd.Series(biserial_corr).sort_values(ascending=False)
corr_matrix = train_data[numerical_features].corr(method='spearman')

# Plot the heatmap with annotations
plt.figure(figsize=(40, 32))
sns.heatmap(corr_matrix, 
            annot=True,  # Show correlation values
            fmt=".2f",   # Format to 2 decimal places
            cmap='coolwarm', 
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),  # Hide upper triangle
            linewidths=0.5,
            cbar=True,  # Show color bar
            vmin=-1, vmax=1)  # Set color range to -1 to 1 for better contrast

plt.title("Feature Correlation Matrix (Spearman)", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

high_corr_threshold = 0.7
high_corr_pairs = corr_matrix.abs().stack().reset_index()
high_corr_pairs = high_corr_pairs[high_corr_pairs[0] >= high_corr_threshold]
high_corr_pairs = high_corr_pairs[high_corr_pairs['level_0'] != high_corr_pairs['level_1']]

correlated_features = set()
for _, row in high_corr_pairs.iterrows():
    if row[0] >= 0.85:
        feat1 = row['level_0']
        feat2 = row['level_1']
        
        if abs(biserial_corr.get(feat1, 0)) < abs(biserial_corr.get(feat2, 0)):
            correlated_features.add(feat1)
        else:
            correlated_features.add(feat2)

train_data.drop(columns=correlated_features, inplace=True)

#%% 6 - Data Scaling

X_train = train_data.drop(columns = ["churn"])
y_train = train_data["churn"]
scaler = QuantileTransformer(output_distribution ='uniform')
X_train_scaled = scaler.fit_transform(X_train)

#%% 7 - Up-Sampling with SMOTE

smote = SMOTE(sampling_strategy = 1.0, random_state = 42) # to match %100
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#%% 8 - Feature Importance (Random Forest)

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train_resampled, y_train_resampled)

# Feature Importance Values
importances = rf.feature_importances_

# Features
feature_names = X_train.columns

indices = np.argsort(importances)[::-1]

plt.figure(figsize = (10, 6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[indices], align = "center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation = 90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

top_30_features = [feature_names[i] for i in indices[:40]]
top_30_features.append("churn")

#%% Mapping Train Dataset ###

# Train Dataset Export
X_train_resampled = pd.DataFrame(X_train_resampled, columns = X_train.columns)
X_train_resampled['churn'] = y_train_resampled

X_train_resampled = X_train_resampled[top_30_features]
X_train_resampled.to_csv("Datasets/train.csv", index = False)

# Missing Values
categorical_features_train_map = categorical_features_train
mapping_missing_statistic_train_map = mapping_missing_statistic_train

# Outlier Values
iqr_bounds_train = iqr_bounds

# Train Format
train_format_feat = train_data.columns

#%% Preparing VALIDATION and TEST Datasets ###

validation_data = fn.processing_data(df = validation_data, categorical_feat = categorical_features_train_map,
                                     train_missing_stats = mapping_missing_statistic_train_map,
                                     iqr_bounds = iqr_bounds_train, train_format = train_format_feat)

X_validation = validation_data.drop(columns=["churn"])
y_validation = validation_data["churn"]

X_validation_scaled = scaler.transform(X_validation)  #It must be done using the scaler applied to the training data.

X_validation_scaled = pd.DataFrame(X_validation_scaled, columns = X_validation.columns,index = X_validation.index)
X_validation_scaled['churn'] = y_validation
X_validation_scaled = X_validation_scaled[top_30_features]
X_validation_scaled.to_csv("Datasets/validation.csv",index = False)

test_data = fn.processing_data(df = test_data, categorical_feat = categorical_features_train_map,
                                     train_missing_stats = mapping_missing_statistic_train_map,
                                     iqr_bounds = iqr_bounds_train, train_format = train_format_feat)

X_test = test_data.drop(columns=["churn"])
y_test = test_data["churn"]

X_test_scaled = scaler.transform(X_test)  #It must be done using the scaler applied to the training data.

X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns,index = X_test.index)
X_test_scaled['churn'] = y_test
X_test_scaled = X_test_scaled[top_30_features]
X_test_scaled.to_csv("Datasets/test.csv",index = False)

#%% Custom Prediction
with open("Utils/categorical_features_train_map.pkl", "wb") as f:
    pickle.dump(categorical_features_train_map, f)
    
with open("Utils/top_30_features.pkl", "wb") as f:
    pickle.dump(top_30_features, f)

with open("Utils/mapping_missing_statistic_train_map.pkl", "wb") as f:
    pickle.dump(mapping_missing_statistic_train_map, f)

with open("Utils/iqr_bounds_train.pkl", "wb") as f:
    pickle.dump(iqr_bounds_train, f)
    
with open("Utils/train_format_feat.pkl", "wb") as f:
    pickle.dump(train_format_feat, f)
    
with open("Utils/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    