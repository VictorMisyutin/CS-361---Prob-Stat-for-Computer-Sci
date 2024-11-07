import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

wine_data_path = 'wine.data'

column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                'Color intensity', 'Hue', 'OD280/OD315', 'Proline']

wine_df = pd.read_csv(wine_data_path, header=None, names=column_names)

X = wine_df.drop('Class', axis=1)
y = wine_df['Class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_eigenvalues = np.sort(eigenvalues)[::-1]

plt.figure(figsize=(10, 6))
plt.plot(sorted_eigenvalues, marker='o')
plt.title('Eigenvalues of the Covariance Matrix')
plt.xlabel('Principal Component Index')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

first_three_eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1][:3]]

plt.figure(figsize=(10, 6))
colors = ['r', 'b', 'm']  

for i in range(3):
    plt.stem(first_three_eigenvectors[:, i], 
             label=f'PC {i + 1}', 
             basefmt=" ", 
             linefmt=colors[i],  
             markerfmt=f'o{colors[i][0]}')

plt.title('Stem Plot of the First 3 Principal Components')
plt.xlabel('Feature Index')
plt.ylabel('Eigenvector Coefficients')
plt.legend()
plt.grid()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for class_label in np.unique(y):
    plt.scatter(X_pca[y == class_label, 0], X_pca[y == class_label, 1], label=f'Class {class_label}')

plt.title('PCA of Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()

num_components = np.sum(sorted_eigenvalues > 1)
num_components
