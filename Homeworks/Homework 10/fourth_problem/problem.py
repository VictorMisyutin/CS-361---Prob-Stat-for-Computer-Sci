import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

file_path = './EEG Eye State.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

df['eyeDetection'] = df['eyeDetection'].apply(lambda x: int(x.decode('utf-8')))

X = df.drop('eyeDetection', axis=1)
y = df['eyeDetection']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(max_depth=50, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
svc_classifier = SVC(random_state=42)

dt_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)

lr_cv_scores = cross_val_score(lr_classifier, X_train, y_train, cv=5, scoring='accuracy')
svc_cv_scores = cross_val_score(svc_classifier, X_train, y_train, cv=5, scoring='accuracy')

lr_mean_accuracy = np.mean(lr_cv_scores)
svc_mean_accuracy = np.mean(svc_cv_scores)

best_model = None
best_accuracy = 0

if lr_mean_accuracy > svc_mean_accuracy:
    best_model = lr_classifier
    best_model.fit(X_train, y_train)
    best_accuracy = lr_mean_accuracy
else:
    best_model = svc_classifier
    best_model.fit(X_train, y_train)
    best_accuracy = svc_mean_accuracy

best_predictions = best_model.predict(X_test)
best_cm = confusion_matrix(y_test, best_predictions)

print("Best Model Accuracy: {:.2f}%".format(best_accuracy * 100))
print("Best Model Confusion Matrix:\n", best_cm)