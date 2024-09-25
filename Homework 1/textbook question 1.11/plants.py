import csv
import matplotlib.pyplot as plt
import pandas as pd

costs = []
MWatts = []
dates = []

with open('plants.csv', mode='r') as file:
    csvFile = csv.reader(file)
    
    next(csvFile)
    
    for lines in csvFile:
        costs.append(float(lines[0]))
        MWatts.append(float(lines[1]))
        dates.append(float(lines[2]))

df = pd.DataFrame({
    'Cost': costs,
    'MWatts': MWatts,
    'Date': dates
})


# # Box plot for 'Date'
plt.boxplot(dates)
plt.title('Date')

plt.tight_layout()
plt.show()


def find_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Detect outliers in each variable
cost_outliers = find_outliers('Cost')
mwatts_outliers = find_outliers('MWatts')
date_outliers = find_outliers('Date')

print("Cost Outliers:\n", cost_outliers)
print("\nMWatts Outliers:\n", mwatts_outliers)
print("\nDate Outliers:\n", date_outliers)


n = len(costs)
mean_cost = sum(costs) / n

variance = sum((x - mean_cost) ** 2 for x in costs) / n
std_cost = variance ** 0.5

print("\nmean cost: ", mean_cost)
print("standard deviation of cost: ", std_cost)


cost_per_mw = [cost / mwatt for cost, mwatt in zip(costs, MWatts)]

mean_cost_per_mw = sum(cost_per_mw) / n

variance_cost_per_mw = sum((x - mean_cost_per_mw) ** 2 for x in cost_per_mw) / n
std_cost_per_mw = variance_cost_per_mw ** 0.5

print("\nmean cost per MW: ", mean_cost_per_mw)
print("standard deviation of cost per MW: ", std_cost_per_mw)



plt.figure(figsize=(8, 6))
plt.hist(cost_per_mw, bins=10, edgecolor='black', color='skyblue')
plt.title('Histogram of Cost per Megawatt')
plt.xlabel('Cost per Megawatt ($ million per MW)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()