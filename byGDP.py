import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define function for predictive modeling
def run_predictions(state_df, all_cols, models=['random_forest']):
    X = state_df[all_cols].dropna()
    y = state_df['X_MENT14D'].loc[X.index]  # Align target with valid IV rows
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    # Load dataset
    filename = 'merged_brfss_oxcgrt.csv'
    df = pd.read_csv(filename, encoding='latin1')

    # Define variables
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

    # Add GDP data to the dataset
    gdp_data = {
        'State': ['California', 'Texas', 'New York', 'Florida', 'Illinois', 'Pennsylvania', 'Ohio', 
                  'Georgia', 'New Jersey', 'North Carolina', 'Vermont', 'Wyoming', 'Montana', 'South Dakota', 'Alaska'],
        'GDP_2020': [3120.39, 1772.13, 1705.12, 1111.70, 875.67, 788.50, 683.46, 
                     627.67, 625.73, 624.90, 33.29, 36.25, 51.91, 55.18, 55.26]
    }
    gdp_df = pd.DataFrame(gdp_data)
    df = df.merge(gdp_df, left_on='RegionName', right_on='State', how='left')

    # Categorize states into "rich" and "poor" based on GDP
    median_gdp = df['GDP_2020'].median()
    df['GDP_Category'] = pd.cut(
        df['GDP_2020'],
        bins=[-float('inf'), median_gdp, float('inf')],
        labels=['Low GDP', 'High GDP']
    )

    # Group by GDP category and calculate descriptive statistics for mental health
    mental_health_stats = df.groupby('GDP_Category')['X_MENT14D'].agg(['mean', 'std', 'min', 'max', 'count'])
    mental_health_stats.to_csv('mental_health_stats_by_gdp.csv')
    print("Mental Health Statistics by GDP Category:")
    print(mental_health_stats)

    # Plot distributions of mental health scores by GDP category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='GDP_Category', y='X_MENT14D', data=df)
    plt.title("Distribution of Mental Health Scores by State GDP Category")
    plt.xlabel("State GDP Category")
    plt.ylabel("Mental Health Score (X_MENT14D)")
    plt.savefig("mental_health_by_gdp_category.png")

    # Run predictions for each GDP category
    group_results = {}
    for category in df['GDP_Category'].dropna().unique():
        print(f"Running predictions for {category} states...")
        category_df = df[df['GDP_Category'] == category]
        if not category_df.empty:  # Ensure group is not empty
            accuracy = run_predictions(category_df, all_cols, models=['random_forest'])
            group_results[category] = accuracy

    # Save and display prediction results
    group_results_df = pd.DataFrame.from_dict(group_results, orient='index', columns=['random_forest'])
    group_results_df.to_csv('prediction_accuracy_by_gdp.csv')
    print("Prediction Accuracy by GDP Category:")
    print(group_results_df)
