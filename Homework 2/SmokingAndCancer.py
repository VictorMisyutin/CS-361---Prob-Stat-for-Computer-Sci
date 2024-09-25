import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Load data
file_path = 'cig.csv'

states = []
cigs = []
blads = []
lungs = []
kids = []
leuks = []

with open(file_path, mode='r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        states.append(row['STATE'])
        cigs.append(float(row['CIG']))
        blads.append(float(row['BLAD']))
        lungs.append(float(row['LUNG']))
        kids.append(float(row['KID']))
        leuks.append(float(row['LEUK']))

# A) plotting
plt.figure(figsize=(10, 6))
for i in range(len(states)):
    plt.scatter(cigs[i], lungs[i], label=states[i])
    plt.text(cigs[i], lungs[i], states[i], fontsize=9)

plt.xlabel('Cigarettes')
plt.ylabel('Lung Cancer Deaths')

plt.show()

states_np = np.array(states)
cigs_np = np.array(cigs)
lungs_np = np.array(lungs)
blads_np = np.array(blads)
kids_np = np.array(kids)
leuks_np = np.array(leuks)

mask = ~(np.isin(states_np, ['DC', 'NE']))
cigs_no_outliers = cigs_np[mask]
lungs_no_outliers = lungs_np[mask]
blads_no_outliers = blads_np[mask]
kids_no_outliers = kids_np[mask]
leuks_no_outliers = leuks_np[mask]

def correlation_coefficient(x, y):
    return np.corrcoef(x, y)[0, 1]

# B) a)
corr_cigs_lungs = correlation_coefficient(cigs_np, lungs_np)
print(f'Correlation coefficient between cigarettes and lung cancer deaths: {corr_cigs_lungs}')
# B) b)
corr_cigs_lungs_no_outliers = correlation_coefficient(cigs_no_outliers, lungs_no_outliers)
print(f'Correlation coefficient between cigarettes and lung cancer deaths (no outliers): {corr_cigs_lungs_no_outliers}')
# C) a)
corr_cigs_blads = correlation_coefficient(cigs_np, blads_np)
print(f'Correlation coefficient between cigarettes and bladder cancer deaths: {corr_cigs_blads}')
# C) b)
corr_cigs_blads_no_outliers = correlation_coefficient(cigs_no_outliers, blads_no_outliers)
print(f'Correlation coefficient between cigarettes and bladder cancer deaths (no outliers): {corr_cigs_blads_no_outliers}')
# D) a)
corr_cigs_kids = correlation_coefficient(cigs_np, kids_np)
print(f'Correlation coefficient between cigarettes and kidney cancer deaths: {corr_cigs_kids}')
# D) b)
corr_cigs_kids_no_outliers = correlation_coefficient(cigs_no_outliers, kids_no_outliers)
print(f'Correlation coefficient between cigarettes and kidney cancer deaths (no outliers): {corr_cigs_kids_no_outliers}')
# E) a)
corr_cigs_leuks = correlation_coefficient(cigs_np, leuks_np)
print(f'Correlation coefficient between cigarettes and leukemia deaths: {corr_cigs_leuks}')
# E) b)
corr_cigs_leuks_no_outliers = correlation_coefficient(cigs_no_outliers, leuks_no_outliers)
print(f'Correlation coefficient between cigarettes and leukemia deaths (no outliers): {corr_cigs_leuks_no_outliers}')

# F) and G)
# if there is a correlation between two sets of data (positive, negative, or otherwise)
# id does not mean that one of the data sets causes the other. It just means
# that there is some similarities or differences between the two. So, even though
# there is a positive correlation between cigarette sales and lung cancer does not
# necessarily mean that  cigarettes cause lung cancer. The same is true with
# cigarettes and leukemia deaths having a negative correlation