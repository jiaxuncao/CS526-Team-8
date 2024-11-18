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

def read_state_data(state_name):
    state_df = df[df['RegionName'] == state_name]
    return state_df

def run_predictions(state_df, all_cols, models=['random_forest', 'logistic_regression', 'svm', 'knn', 'ada_boost']):
    concise_state_df = state_df[all_cols]
    X = concise_state_df[all_ivs]
    y = concise_state_df[brfss_target]

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

    if os.path.exists('state_predictions.json'):
        with open('state_predictions.json', 'r') as f:
            all_states_res = json.load(f)
    else:
        # iterate through regionNames in df
        all_states_res = {}
        regionNames = df['RegionName'].unique()
        for state in regionNames:
            state_df = read_state_data(state)
            pred_res = run_predictions(state_df, all_cols)
            all_states_res[state] = pred_res
        
        with open('state_predictions.json', 'w') as f:
            json.dump(all_states_res, f)

    # get the best for each
    best_pred = {}
    for state, res in all_states_res.items():
        best_acc = max(res.values())
        best_pred[state] = best_acc
    
    print(best_pred)
    print("Best performing state: ", max(best_pred, key=best_pred.get))
    print()
    print("Worst performing state: ", min(best_pred, key=best_pred.get))
    
    state_predictions_df = pd.DataFrame(all_states_res).T
    state_predictions_df.to_csv('state_predictions.csv')

