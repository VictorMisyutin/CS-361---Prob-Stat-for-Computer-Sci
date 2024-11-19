import pandas as pd
from collections import Counter

file_path = './agaricus-lepiota.data'
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

data = pd.read_csv(file_path, header=None, names=column_names)

data = data.replace('?', pd.NA).dropna()

edible_data = data[data['class'] == 'e']
poisonous_data = data[data['class'] == 'p']

odor_counts_poisonous = Counter(poisonous_data['odor'])
most_common_poisonous_odor = odor_counts_poisonous.most_common(1)[0][0]

def classify(row):
    if row['odor'] == most_common_poisonous_odor:
        return 'p'
    return 'e'

data['predicted_class'] = data.apply(classify, axis=1)

true_positive = len(data[(data['class'] == 'e') & (data['predicted_class'] == 'e')])
true_negative = len(data[(data['class'] == 'p') & (data['predicted_class'] == 'p')])
false_positive = len(data[(data['class'] == 'e') & (data['predicted_class'] == 'p')])
false_negative = len(data[(data['class'] == 'p') & (data['predicted_class'] == 'e')])

confusion_matrix = {
    "True Positive": true_positive,
    "True Negative": true_negative,
    "False Positive": false_positive,
    "False Negative": false_negative,
}

prob_poisoned_if_edible = false_negative / (false_negative + true_negative)

print(f"Confusion Matrix: {confusion_matrix}")
print(f"Probability of Being Poisoned If Predicted Edible: {prob_poisoned_if_edible:.2f}")
