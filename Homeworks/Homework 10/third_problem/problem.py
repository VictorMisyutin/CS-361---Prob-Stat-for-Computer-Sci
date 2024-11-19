import pandas as pd

file_path = './haberman.data'

column_names = ['Age', 'Year_of_Operation', 'Positive_Auxillary_Nodes', 'Survival_Status']

df = pd.read_csv(file_path, names=column_names)

df['Survival_Status'] = df['Survival_Status'].map({1: 0, 2: 1})

X = df.drop('Survival_Status', axis=1)
y = df['Survival_Status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(max_depth=50, random_state=42)
dt_classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
dt_predictions = dt_classifier.predict(X_test)
dt_cm = confusion_matrix(y_test, dt_predictions)

rf_predictions = rf_classifier.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_predictions)

print(dt_cm, "\n\n", rf_cm)
