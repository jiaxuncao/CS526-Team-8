import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def read_state_data(state_name):
    state_df = df[df['RegionName'] == state_name]
    return state_df

def run_predictions(state_df, all_cols, sample_size=10000, models=['random_forest', 'logistic_regression', 'svm', 'knn', 'ada_boost']):
    
    # Sample data for processing if the dataset is too large
    if len(state_df) > sample_size:
        state_df = state_df.sample(sample_size, random_state=42)
        print(f"Dataset reduced to {sample_size} rows for efficiency.")
    
    concise_state_df = state_df[all_cols]
    X = concise_state_df[all_ivs]
    y = concise_state_df[brfss_target].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the imputer (using mean strategy to fill NaNs)
    imputer = SimpleImputer(strategy='most_frequent')

    # Fit the imputer on the training data and transform both train and test sets
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    pred_res = {}

    if "random_forest" in models:
        # Fit RandomForestClassifier to the data
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_model.predict(X_test)

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)

        # Print the most important features
        # importances = rf_model.feature_importances_
        # feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

        pred_res["random_forest"] = accuracy
        # pred_res["feature_importances"] = feature_importances
    
    if "logistic_regression" in models:
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)
        y_pred_logreg = log_reg.predict(X_test)
        accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

        pred_res["logistic_regression"] = accuracy_logreg
    
    if "svm" in models:
        svm_model = SVC(random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)

        pred_res["svm"] = accuracy_svm
    
    if "knn" in models:
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)

        pred_res["knn"] = accuracy_knn

    if "ada_boost" in models:
        adaboost_model = AdaBoostClassifier(random_state=42)
        adaboost_model.fit(X_train, y_train)
        y_pred_adaboost = adaboost_model.predict(X_test)

        accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)

        pred_res["ada_boost"] = accuracy_adaboost
    
    return pred_res

if __name__=="__main__":
    filename = '../merged_brfss_oxcgrt.csv'
    df = pd.read_csv(filename, encoding='latin1')
    variables_C = [
        'C1M_School closing', 'C2E_Workplace.closing', 'C2M_Workplace closing',
        'C3E_Cancel public events', 'C3M_Cancel public events',
        'C4E_Restrictions on gatherings', 'C4M_Restrictions on gatherings',
        'C5E_Close public transport', 'C5M_Close public transport',
        'C6E_Stay at home requirements', 'C6M_Stay at home requirements',
        'C7E_Restrictions on internal movement', 'C7M_Restrictions on internal movement',
        'C8E_International travel controls', 'C8EV_International travel controls'
    ]
    variables_H = [
        'H1E_Public information campaigns', 'H2E_Testing policy', 'H3E_Contact tracing',
        'H4E_Emergency investment in healthcare', 'H5E_Investment in vaccines',
        'H6E_Facial Coverings', 'H6M_Facial Coverings', 'H7E_Vaccination policy',
        'H8E_Protection of elderly people', 'H8M_Protection of elderly people'
    ]
    variables_E = [
        'E1E_Income support', 'E2E_Debt/contract relief', 
        'E3E_Fiscal measures', 'E4E_International support'
    ]
    brfss_iv = ['MARITAL', 'EDUCA', 'RENTHOM1', 'EMPLOY1', 'CHILDREN', 'INCOME2']
    brfss_target = ['X_MENT14D']

    all_ivs = variables_C + variables_H + variables_E + brfss_iv
    all_ivs = [col.replace(' ', '.').replace('/', '.') for col in all_ivs]
    all_cols = list(set(all_ivs + brfss_target))

    # Create the stringency index
    stringency_cols = variables_C + variables_H + variables_E
    stringency_cols = [col.replace(' ', '.').replace('/', '.') for col in stringency_cols]
    df['stringency_index'] = df[stringency_cols].sum(axis=1)

    # Categorize states by policy stringency
    df['policy_category'] = pd.cut(
        df['stringency_index'],
        bins=[-float('inf'), 30, 60, float('inf')],
        labels=['Low Restriction', 'Moderate Restriction', 'High Restriction']
    )

    # Group by policy category
    group_results = {}
    for category in df['policy_category'].dropna().unique():
        print(f"Running predictions for {category} states...")
        category_df = df[df['policy_category'] == category]
        print(category_df)
        if not category_df.empty:  # Ensure group is not empty
            accuracy = run_predictions(category_df, all_cols, models=['random_forest', 'logistic_regression', 'svm', 'knn', 'ada_boost'])
            group_results[category] = accuracy

    # Save and display results
    group_results_df = pd.DataFrame.from_dict(group_results, orient='index')
    group_results_df.columns = ['random_forest', 'logistic_regression', 'svm', 'knn', 'ada_boost']
    group_results_df.to_csv('policy_group_metrics.csv')

    print("Policy Group Metrics:")
    print(group_results_df)

    # Group by policy category and calculate descriptive statistics for mental health
    mental_health_stats = df.groupby('policy_category')['X_MENT14D'].agg(['mean', 'std', 'min', 'max', 'count'])
    mental_health_stats.to_csv('mental_health_stats_by_policy.csv')
    print("Mental Health Statistics by Policy Category:")
    print(mental_health_stats)

    # Plot distributions of mental health scores by policy category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='policy_category', y='X_MENT14D', data=df)
    plt.title("Distribution of Mental Health Scores by Policy Restriction Level")
    plt.xlabel("Policy Restriction Level")
    plt.ylabel("Mental Health Score (X_MENT14D)")
    plt.savefig("mental_health_by_policy_level.png")

    # Group by policy category and calculate average income levels
    economic_stats = df.groupby('policy_category')['INCOME2'].mean()

    # Plot the results
    economic_stats.plot(kind='bar', figsize=(8, 5), title='Average Income Levels by Policy Restriction Level')
    plt.xlabel('Policy Restriction Level')
    plt.ylabel('Average Income Level')
    plt.xticks(rotation=0)
    plt.savefig("income_by_policy_level.png")





