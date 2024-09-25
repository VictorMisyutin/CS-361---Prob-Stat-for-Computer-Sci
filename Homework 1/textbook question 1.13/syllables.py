import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

WDS = []
SEN = []
SYL = []
GROUP = []

# Open and read the CSV file
with open('syllables.csv', mode='r') as file:
    csvFile = csv.reader(file)
    
    next(csvFile)  # Skip the header
    for lines in csvFile:
        WDS.append(int(lines[0]))  # Convert to integer
        SEN.append(int(lines[1]))  # Convert to integer
        SYL.append(int(lines[2]))  # Convert to integer
        GROUP.append(int(lines[4]))  # Convert to integer (assuming the group is in the last column)

# Create a DataFrame with the data
data = {
    'WDS': WDS,
    'SEN': SEN,
    '3SYL': SYL,
    'GROUP': GROUP
}

df = pd.DataFrame(data)

# (a) Boxplot for number of 3 or more syllable words
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
df.boxplot(column='3SYL', by='GROUP')
plt.title('Number of 3-Syllable Words by Group')
plt.suptitle('')  # Hide the default title
plt.xlabel('Education Level Group')
plt.ylabel('Number of 3-Syllable Words')

# (b) Boxplot for number of sentences
plt.subplot(1, 2, 2)
df.boxplot(column='SEN', by='GROUP')
plt.title('Number of Sentences by Group')
plt.suptitle('')
plt.xlabel('Education Level Group')
plt.ylabel('Number of Sentences')

# Display the plots
plt.tight_layout()
plt.show()
