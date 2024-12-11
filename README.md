# CS526-Team-8

## Datasets

### Behavioral Risk Factor Surveillance System (BRFSS)

Behavioral Risk Factor Surveillance System (BRFSS) provides self-reported data on mental health, lifestyle factors, and demographics from U.S. residents.

[Codebook for BRFSS](https://www.cdc.gov/brfss/annual_data/2020/pdf/codebook20_llcp-v2-508.pdf)

### COVID-19 Stringency Index

COVID-19 Stringency Index measures the strictness of government responses to the pandemic through nine indicators (e.g., school closures, travel bans).

[Codebook for government response tracker](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md)

## Results

1. State-Level Analysis

I added `prediction.py` to run predictions across different states to get the prediction accuracy based on all policies. Results are saved to `state_predictions.csv` and `state_predictions.json` for different file types.

Meanwhile, I did some rough analysis over all the states and find that Mississippi has the best prediction accuracy: 0.7053422370617696, while Michigan has the worst prediction accuracy: 0.5732586068855084.


2. Group States by Policy Stringency:

A stringency index is computed by summing the values across all selected policy variables. Higher values in these columns indicate stricter policies.

To analyze the impact of varying policy stringency levels, I categorized states into three groups:

- Low Restriction: States with a stringency index ≤ 30.
- Moderate Restriction: States with a stringency index between 30 and 60.
- High Restriction: States with a stringency index > 60.

I used `policy_stringency.py` to run models for each group. The results were saved to `policy_group_metrics.csv.`
