import pandas as pd
import scipy.stats as stats

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

adult_data = pd.read_csv(url, names=columns, na_values=' ?')

adult_data.dropna(inplace=True)

gender_income_crosstab = pd.crosstab(adult_data['sex'], adult_data['income'])
chi2_gender, p_gender, dof_gender, ex_gender = stats.chi2_contingency(gender_income_crosstab)

education_income_crosstab = pd.crosstab(adult_data['education'], adult_data['income'])
chi2_education, p_education, dof_education, ex_education = stats.chi2_contingency(education_income_crosstab)

print(f"Chi-square test for independence of income and gender:")
print(f"Chi2 Statistic: {chi2_gender}, p-value: {p_gender}")

print(f"\nChi-square test for independence of income and education level:")
print(f"Chi2 Statistic: {chi2_education}, p-value: {p_education}")
