import pandas as pd
import numpy as np

def replace_large_values_with_nan(df, feature_list, threshold=1270):
    """
    Replaces values greater than the threshold with NaN in the specified features.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    feature_list (list): List of feature names to process.
    threshold (int, float): The threshold above which values will be replaced with NaN (default: 1270).

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    for feature in feature_list:
        if feature in df.columns:
            df.loc[df[feature] > threshold, feature] = np.nan
    return df


def find_columns_with_negative_values(df):
    """
    Checks each column in the DataFrame for negative values and returns a list of column names
    that contain at least one negative value.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    list: A list of column names containing negative values.
    """
    negative_columns = [col for col in df.columns if (df[col] < 0).any()]
    return negative_columns

def replace_negative_values_with_nan(df, feature_list):
    """
    Replaces negative values with NaN in the specified features.

    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    feature_list (list): List of feature names to process.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    for feature in feature_list:
        if feature in df.columns:
            df.loc[df[feature] < 0, feature] = np.nan
    return df

def count_missing_values(df):
    """
    Counts the number of missing values for each column in the dataset and 
    returns a DataFrame representation of missing data statistics.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    pd.DataFrame: A DataFrame showing the count, total number of data points, 
                  and percentage of missing values for each column.
    """
    
    missing_data_count = df.isnull().sum()
    total_rows = len(df)
    missing_ratio = (missing_data_count / total_rows) * 100
    
    result = pd.DataFrame({
        'Missing Data Count': missing_data_count,
        'Total Number of Rows': total_rows,
        'Missing Ratio (%)': round(missing_ratio, 2)
    })
    
    return result


def find_binary_categorical_columns(df):
    """
    Identifies columns that contain only 0 and 1, which may be categorical.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    list: A list of column names that contain only binary values (0 and 1).
    """
    
    """
    binary_categorical_columns = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]
    return binary_categorical_columns
    """
    
    """
    Identifies categorical features in a DataFrame based on unique value count.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    list: A list of column names considered categorical.
    """
    
    categorical_features = [col for col in df.columns if df[col].nunique() <= 5]
    return categorical_features
    


def calculate_grouped_statistics(df, target_column, categorical_features):
    """
    Creates a dictionary where each feature is grouped by the target column.
    For categorical features, it stores the mode value.
    For numerical features, it stores the mean value.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column used for grouping.
    categorical_features (list): List of categorical feature names.
    
    Returns:
    dict: A dictionary where keys are feature names and values are dictionaries
          mapping target column values to either mode (for categorical) or mean (for numerical).
    """
    grouped_stats = {}

    for column in df.columns:
        if column == target_column:
            continue  # Skip the target column itself

        # Group by the target column
        grouped = df.groupby(target_column)[column]

        if column in categorical_features:
            # Calculate mode for categorical features (excluding NaN values)
            grouped_stats[column] = grouped.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
        else:
            # Calculate mean for numerical features (excluding NaN values)
            grouped_stats[column] = grouped.mean().to_dict()

    return grouped_stats

def fill_missing_values(df, grouped_stats, target_column):
    """
    Fills missing values in a DataFrame using precomputed grouped statistics.

    This function replaces NaN values in specified columns with corresponding 
    statistical values from a precomputed dictionary, where the values are grouped 
    based on the target column.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataset containing missing values.
    grouped_stats : dict
        A dictionary where keys are column names and values are dictionaries 
        mapping target column categories to replacement values.
    target_column : str
        The column used for grouping when filling missing values.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with missing values filled according to the grouped statistics.
    """
    df = df.copy()
    
    for column, value_dict in grouped_stats.items():
        if column not in df.columns:
            continue
        
        df[column] = df.apply(
            lambda row: value_dict.get(row[target_column], np.nan) if pd.isna(row[column]) else row[column], axis=1
        )

    return df

def count_outlier_values(df):
    """
    Identifies the number of outliers in all numerical columns using the IQR method 
    and returns a sorted DataFrame from highest to lowest.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.

    Returns:
    pd.DataFrame: A DataFrame listing numerical features and their respective 
                  outlier counts, sorted in descending order.
    """
    outlier_counts = {}

    for column in df.select_dtypes(include=["number"]).columns:  # Select only numerical columns
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        
        outlier_counts[column] = outlier_count
    
    result_df = pd.DataFrame(list(outlier_counts.items()), columns=["Feature", "Outlier Count"])
    result_df = result_df.sort_values(by="Outlier Count", ascending=False)

    return result_df


def get_iqr_bounds(df, feature):
    """
    Computes IQR-based lower and upper bounds for a feature in a DataFrame.

    Parameters:
    df (pd.DataFrame): The training DataFrame.
    feature (str): The feature column name.

    Returns:
    tuple: (lower_bound, upper_bound) for the feature.
    """
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound

def clip_outliers(df, iqr_bounds, target_column, categorical_features):
    """
    Clips outliers in the given numerical features using precomputed IQR bounds.
    The target column and categorical features remain unchanged.

    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    iqr_bounds (dict): Dictionary containing IQR lower and upper bounds for each feature.
    target_column (str): The name of the target column which should not be modified.
    categorical_features (list): List of categorical feature names that should not be modified.

    Returns:
    pd.DataFrame: The modified DataFrame with outliers clipped.
    """
    df = df.copy()
    
    for feature, (lower_bound, upper_bound) in iqr_bounds.items():
        if feature in df.columns and feature not in categorical_features and feature != target_column:
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)  # Clip values to IQR bounds

    return df

def calculate_statistics_train(df, categorical_features):
    """
    Computes mode for categorical features and mean for numerical features in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    categorical_features (list): List of categorical feature names.

    Returns:
    dict: A dictionary where keys are feature names and values are computed statistics
          (mode for categorical, mean for numerical).
    """
    stats = {}

    for column in df.columns:
        if column in categorical_features:
            mode_value = df[column].mode()
            stats[column] = mode_value.iloc[0] if not mode_value.empty else np.nan
        else:
            stats[column] = df[column].mean()

    return stats

def fill_missing_values_with_train_statistic(df, categorical_features, stats_dict):
    """
    Fills missing values in the given DataFrame using precomputed statistics from the train set.
    
    Parameters:
    df (pd.DataFrame): The DataFrame where missing values will be filled.
    categorical_features (list): List of categorical feature names.
    stats_dict (dict): Dictionary containing precomputed statistics from the train set 
                       (mode for categorical, mean for numerical).
    
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    df = df.copy()  
    
    for feature in df.columns:
        if feature in stats_dict:  
            fill_value = stats_dict[feature]  
            
            if df[feature].isnull().sum() > 0:  
                df[feature] = df[feature].fillna(fill_value)

    return df

def clip_outliers_train(df, iqr_bounds, categorical_features, target_column=None):
    """
    Clips outliers in the given numerical features using precomputed IQR bounds.
    The target column (if provided) and categorical features remain unchanged.

    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    iqr_bounds (dict): Dictionary containing IQR lower and upper bounds for each feature.
    categorical_features (list): List of categorical feature names that should not be modified.
    target_column (str, optional): The name of the target column which should not be modified.
                                   If None, no target column will be excluded.

    Returns:
    pd.DataFrame: The modified DataFrame with outliers clipped.
    """
    df = df.copy()
    
    for feature, (lower_bound, upper_bound) in iqr_bounds.items():
        if feature in df.columns and feature not in categorical_features and (target_column is None or feature != target_column):
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

    return df


def processing_data(df, categorical_feat, train_missing_stats, iqr_bounds, train_format):
    """
    Processes a given dataset by handling noisy data, missing values, outliers, 
    removing unnecessary features, and extracting new features.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The dataset to be processed.
    categorical_feat : list
        List of categorical feature names that need special handling for missing values.
    train_missing_stats : dict
        Dictionary containing the statistics for missing value imputation based on the training set.
    iqr_bounds : dict
        Dictionary containing IQR (Interquartile Range) bounds for outlier clipping.
    train_format : list
        List of column names defining the expected format of the final dataset.
    
    Returns:
    -------
    pd.DataFrame
        The processed dataset with cleaned data, feature engineering applied, 
        and aligned to the expected training format.
    """
    
    # Noisy Data
    df.loc[df['user_lifetime'] > 4380, 'user_lifetime'] = np.nan
    
    noisy_features_about_day = ["user_no_outgoing_activity_in_days" , "reloads_inactive_days" , "calls_outgoing_inactive_days",
                               "calls_outgoing_to_onnet_inactive_days", "calls_outgoing_to_offnet_inactive_days", "calls_outgoing_to_abroad_inactive_days",
                               "sms_outgoing_inactive_days", "sms_outgoing_to_onnet_inactive_days", "sms_outgoing_to_offnet_inactive_days",
                               "sms_outgoing_to_abroad_inactive_days", "gprs_inactive_days"]
    
    df = replace_large_values_with_nan(df, noisy_features_about_day) # noisy data about day
    
    impossible_negative_features = ["user_spendings", "reloads_sum", "sms_outgoing_spendings", "sms_outgoing_to_offnet_spendings",
                                    "sms_incoming_spendings", "gprs_spendings", "last_100_reloads_sum"]
    
    df = replace_negative_values_with_nan(df, impossible_negative_features)
    
    # Missing Values
    df.drop(["gprs_inactive_days"],axis=1,inplace=True)
    df = fill_missing_values_with_train_statistic(df, categorical_features = categorical_feat, stats_dict = train_missing_stats)
    
    # Outlier Values
    df = clip_outliers_train(df, iqr_bounds, categorical_features = categorical_feat)
    
    # Unnecessary Features
    df.drop(["year","month","user_account_id"],axis=1,inplace=True)
    
    # Feature Extraction
    df["customer_active_months"] = (df["user_lifetime"] / 30).astype(int)

    df["reloads_per_month"] = np.where(
        df["customer_active_months"] == 0, 0,
        df["reloads_count"] / df["customer_active_months"]
    )
    
    df["calls_per_month"] = np.where(
        df["customer_active_months"] == 0, 0,
        df["calls_outgoing_count"] / df["customer_active_months"]
    )
    
    df["sms_per_month"] = np.where(
        df["customer_active_months"] == 0, 0,
        df["sms_outgoing_count"] / df["customer_active_months"]
    )
    
    df["offnet_call_ratio"] = np.where(
        df["calls_outgoing_count"] == 0, 0,
        df["calls_outgoing_to_offnet_count"] / df["calls_outgoing_count"]
    )
    
    df["international_call_ratio"] = np.where(
        df["calls_outgoing_count"] == 0, 0,
        df["calls_outgoing_to_abroad_inactive_days"] / df["calls_outgoing_count"]
    )
    
    df["sms_offnet_ratio"] = np.where(
        df["sms_outgoing_count"] == 0, 0,
        df["sms_outgoing_to_offnet_count"] / df["sms_outgoing_count"]
    )
    
    epsilon = 1e-6
    df["reload_ratio"] = np.where(
        df["reloads_count"] == 0, 0,
        df["reloads_count"] / (df["user_lifetime"] + epsilon)
    )
    
    df["avg_spending_per_month"] = np.where(
        df["customer_active_months"] == 0, 0,
        df["user_spendings"] / df["customer_active_months"]
    )
    
    df["avg_sms_per_reload"] = np.where(
        df["reloads_count"] == 0, 0,
        df["sms_outgoing_count"] / df["reloads_count"]
    )
    
    df["reload_change"] = np.where(
        df["reloads_sum"] == 0, 0,
        df["last_100_reloads_sum"] / df["reloads_sum"]
    )
    
    df["call_duration_change"] = np.where(
        df["calls_outgoing_duration"] == 0, 0,
        df["last_100_calls_outgoing_duration"] / df["calls_outgoing_duration"]
    )
    
    df["sms_count_change"] = np.where(
        df["sms_outgoing_count"] == 0, 0,
        df["last_100_sms_outgoing_count"] / df["sms_outgoing_count"]
    )

    df["avg_call_duration"] = np.where(
        df["calls_outgoing_count"] == 0, 
        0,
        df["calls_outgoing_duration"] / df["calls_outgoing_count"]
    )

    df["daily_call_count"] = np.where(
        df["user_lifetime"] == 0, 
        0,
        df["calls_outgoing_count"] / df["user_lifetime"]
    )

    df["daily_sms_count"] = np.where(
        df["user_lifetime"] == 0, 
        0,
        df["sms_outgoing_count"] / df["user_lifetime"]
    )

    df["offnet_call_ratio"] = np.where(
        df["calls_outgoing_count"] == 0, 
        0,
        df["calls_outgoing_to_offnet_count"] / df["calls_outgoing_count"]
    )
    
    services = ['user_has_outgoing_calls', 'user_has_outgoing_sms', 'user_use_gprs']
    df["multi_service_score"] = df[services].sum(axis=1)
        
    
    df = df[train_format]
    
    return df
    