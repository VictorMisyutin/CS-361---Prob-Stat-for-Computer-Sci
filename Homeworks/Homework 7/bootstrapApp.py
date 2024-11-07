import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
abalone_data = pd.read_csv(url, names=column_names)

length_data = abalone_data['Length']
population_median = np.median(length_data)
print(f"Population Median: {population_median}")

def bootstrap_confidence_interval(data, n_samples, sample_size, confidence_level=0.90):
    boot_medians = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        boot_medians.append(np.median(sample))
    
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_bound = np.percentile(boot_medians, lower_percentile)
    upper_bound = np.percentile(boot_medians, upper_percentile)
    
    return boot_medians, (lower_bound, upper_bound)

n_samples_a = 2000
sample_size_a = 100
boot_medians_a, ci_a = bootstrap_confidence_interval(length_data, n_samples_a, sample_size_a)
print(f"90% CI for part (a): {ci_a}")

inside_ci_a = np.sum((ci_a[0] <= population_median) & (population_median <= ci_a[1])) / n_samples_a
print(f"Fraction for part (a): {inside_ci_a}")

plt.hist(boot_medians_a, bins=30, edgecolor='k')
plt.title('Bootstrap Sample Medians - Part (a) (100 Records)')
plt.xlabel('Median')
plt.ylabel('Frequency')
plt.show()

n_samples_c = 2000
sample_size_c = 10
boot_medians_c, ci_c = bootstrap_confidence_interval(length_data, n_samples_c, sample_size_c)
print(f"90% CI for part (c): {ci_c}")

inside_ci_c = np.sum((ci_c[0] <= population_median) & (population_median <= ci_c[1])) / n_samples_c
print(f"Fraction for part (c): {inside_ci_c}")

plt.hist(boot_medians_c, bins=30, edgecolor='k')
plt.title('Bootstrap Sample Medians - Part (c) (10 Records)')
plt.xlabel('Median')
plt.ylabel('Frequency')
plt.show()
