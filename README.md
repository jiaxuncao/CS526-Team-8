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

3. Analysis over Mississippi and Michigan Prediction Accuracy difference:

![Michigan Feature Importances](Michigan%20Feature%20Importances.pdf)
![Mississippi Feature Importances](Mississippi%20Feature%20Importances.pdf)

![Michigan BRFSS Factors Distribution](Michigan_BRFSS_Factors_Distribution.pdf)
![Mississippi BRFSS Factors Distribution](Mississippi_BRFSS_Factors_Distribution.pdf)


### Class Balance
#### Mississippi
- **1**: 69.85%
- **2**: 16.05%
- **3**: 11.63%
- **9**: 2.47%

#### Michigan
- **1**: 57.84%
- **2**: 27.01%
- **3**: 13.70%
- **9**: 1.46%

Class imbalance contributes to the better predicitive ability.

### Policy Stringency Analysis
#### Mississippi
- Mean Values: The mean values for policy measures like school closing and workplace closing are lower (e.g., school closing mean = 2.00).
- Variation: Higher standard deviation in some policies suggests more variability in enforcement.
#### Michigan
- Mean Values: Michigan shows higher mean values (e.g., school closing mean = 2.49), indicating stricter policies generally.
- Consistency: Lower standard deviation in several measures suggests more consistent policy application.

Interpretation: Michigan's stricter and more consistently enforced policies might have had social and economic impacts that could influence mental health negatively.

### BRFSS Factors Analysis
#### Mississippi
- Education: Lower mean education level (mean = 4.74).
- Income: Higher mean income level (mean = 26.14).

##### Michigan
- Education: Higher mean education level (mean = 5.10).
- Income: Lower mean income level (mean = 19.91).

Interpretation:
- Education: Higher education levels in Michigan could correlate with greater awareness of mental health issues or higher stress from professional demands.
- Income: Lower income levels in Michigan might limit access to mental health resources, contributing to lower mental health status.

Overall Interpretation
Policy Impact: Stricter policies in Michigan might have increased stress and anxiety, affecting mental health.
Socioeconomic Factors: Lower income and higher education levels in Michigan could contribute to awareness but also increase stress or limit access to resources.
Service Access: Differences in healthcare access or public health initiatives could also play a role.

### Mental Health Impact
1. Prevalence of Mental Health Issues:
- In Michigan, a survey indicated that 79% of respondents reported concerns about stress, loneliness, anxiety, and/or depression, with 32% of adults endorsing symptoms of anxiety or depressive disorders by mid-2020 [1].
- In contrast, while Mississippi also faced mental health challenges, the state reported a relatively lower prevalence of severe mental health issues, partly due to the effective use of tele-mental health services that increased access to care [1].
2. Telehealth Utilization:
- Mississippi experienced a significant increase in tele-mental health services, which allowed for greater access to mental health care, especially in rural areas. This led to a 190% increase in mental health-related outpatient visits during the pandemic [1].
- Michigan, while also utilizing telehealth, faced more significant challenges in mental health service delivery due to its larger urban populations and higher rates of frontline workers exposed to pandemic stressors [2].

### Socioeconomic Factors
1. Economic Stress:
- Michigan's economy was heavily impacted by the pandemic, leading to increased unemployment and financial stress, which correlated with rising mental health issues [1]. The state saw significant job losses in sectors like hospitality and retail, contributing to anxiety and depression among residents.
- Mississippi, despite its economic challenges, had a lower percentage of individuals in high-stress occupations, which may have contributed to a more stable mental health status during the pandemic [3].
2. Community Support:
- The cultural emphasis on community and family in Mississippi provided a strong support network that helped mitigate the psychological impacts of the pandemic. This social cohesion is crucial for mental health resilience [1].
- In Michigan, the social isolation resulting from strict lockdown measures may have exacerbated mental health issues, as many residents struggled with loneliness and disconnection from their support systems [2].

Conclusion
Overall, while both Mississippi and Michigan faced significant mental health challenges during the COVID-19 pandemic, Mississippi's effective use of telehealth services, lower economic stress in certain sectors, and strong community support contributed to a relatively better mental health status compared to Michigan.

Reference:
1. Preparing for the Behavioral Health Impact of COVID-19 in Michigan - PMC
2. See how Michigan compares to other states in mental health status during pandemic - mlive.com
3. Tracking the Impact of COVID-19 and Lockdown Policies on Public Mental Health Using Social Media: Infoveillance Study - PMC
4. Tele-Mental Health Service: Unveiling the Disparity and Impact on Healthcare Access and Expenditures during the COVID-19
5. Pandemic in Mississippi - PMC
6. Explore Social and Economic Factors in Mississippi | AHR
6. Socioeconomic Status and Well-Being during COVID-19: A Resource Based Examination - PMC