import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
type = []
cals = []
sods = []

with open('foods.csv', mode='r') as file:
    csvFile = csv.reader(file)
    
    next(csvFile)
    for lines in csvFile:
        type.append(lines[0])
        cals.append(int(lines[1]))
        sods.append(int(lines[2]))


data = {
    'Type': type,
    'Calories': cals,
    'Sodium': sods
}

df = pd.DataFrame(data)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

df.groupby('Type')['Calories'].plot(kind='hist', ax=axs[0], bins=10, alpha=0.6, legend=True, title='Calories Distribution by Type')
axs[0].set_xlabel('Calories')

df.groupby('Type')['Sodium'].plot(kind='hist', ax=axs[1], bins=10, alpha=0.6, legend=True, title='Sodium Distribution by Type')
axs[1].set_xlabel('Sodium (mg)')

plt.tight_layout()
plt.show()

