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
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

def read_state_data(state_name):
    state_df = df[df['RegionName'] == state_name]
    return state_df

def feature_importance_analysis(state_df, model):
    X = state_df[all_ivs]
    y = state_df[brfss_target]
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    return feature_importances

def plot_feature_importances(importances, title):
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title(title)
    plt.savefig(f'{title}.pdf')

# Correlation Analysis
def plot_correlation_matrix(state_df, title):
    state_df = state_df[all_cols]
    print(state_df)
    corr = state_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title(title)
    plt.savefig(f"{title}.pdf")

def analyze_brfss_factors(state_df, title):
    brfss_data = state_df[brfss_iv]

    # Calculate summary statistics
    summary_stats = brfss_data.describe()
    print(f"{title} BRFSS Factors Summary Statistics:\n", summary_stats)

    # Plot distributions
    num_vars = len(brfss_iv)
    fig, axes = plt.subplots(nrows=1, ncols=num_vars, figsize=(15, 6), sharey=False)
    
    for i, var in enumerate(brfss_iv):
        sns.boxplot(y=brfss_data[var], ax=axes[i], palette="Set2")
        axes[i].set_title(var, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Values", fontsize=8)
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle(f"{title} BRFSS Factors Distribution", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{title}_BRFSS_Factors_Distribution.pdf")

def calculate_stringency_index(state_df, state_name):
    
    state_df['stringency_index'] = state_df[policy_factors].sum(axis=1)
    
    # Return the mean stringency index for the state
    state_df.to_csv(f"{state_name}.csv")
    state_df.to_excel(f"{state_name}.xlsx")
    return state_df['stringency_index'].mean()


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
    policy_factors = variables_C + variables_E + variables_H
    brfss_iv = ['MARITAL', 'EDUCA', 'RENTHOM1', 'EMPLOY1', 'CHILDREN', 'INCOME2']
    brfss_target = ['X_MENT14D']

    all_ivs = variables_C + variables_H + variables_E + brfss_iv
    policy_factors = [col.replace(' ', '.').replace('/', '.') for col in policy_factors]
    all_ivs = [col.replace(' ', '.').replace('/', '.') for col in all_ivs]
    all_cols = list(set(all_ivs + brfss_target))

    # Feature Importance for Mississippi
    mississippi_df = read_state_data('Mississippi')
    mississippi_importances = feature_importance_analysis(mississippi_df, RandomForestClassifier(random_state=42))
    plot_feature_importances(mississippi_importances, 'Mississippi Feature Importances')

    # Feature Importance for Michigan
    michigan_df = read_state_data('Michigan')
    michigan_importances = feature_importance_analysis(michigan_df, RandomForestClassifier(random_state=42))
    plot_feature_importances(michigan_importances, 'Michigan Feature Importances')

    plot_correlation_matrix(mississippi_df, 'Mississippi Correlation Matrix')
    plot_correlation_matrix(michigan_df, 'Michigan Correlation Matrix')

    # Check Class Balance
    print("Mississippi class balance:", mississippi_df[brfss_target].value_counts(normalize=True))
    print("Michigan class balance:", michigan_df[brfss_target].value_counts(normalize=True))

    analyze_brfss_factors(mississippi_df, 'Mississippi')
    analyze_brfss_factors(michigan_df, 'Michigan')

    # Calculate and print the stringency index for Mississippi
    mississippi_stringency = calculate_stringency_index(mississippi_df, 'mississippi')
    print(f"Mississippi Stringency Index: {mississippi_stringency}")
    
    # Calculate and print the stringency index for Michigan
    michigan_stringency = calculate_stringency_index(michigan_df, 'michigan')
    print(f"Michigan Stringency Index: {michigan_stringency}")
    
    # Analysis
    if mississippi_stringency < michigan_stringency:
        print("Michigan has a higher overall policy stringency.")
    else:
        print("Mississippi has a higher overall policy stringency.")

