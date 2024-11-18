import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_path = 'seeds_cleaned.csv'
data = pd.read_csv(file_path)

data = data.apply(pd.to_numeric, errors='coerce')

features = data.drop(columns=['label'])
labels = data['label']

pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)

pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pc_df['label'] = labels

plt.figure(figsize=(10, 6))
scatter = plt.scatter(pc_df['PC1'], pc_df['PC2'], c=pc_df['label'], cmap='viridis', alpha=0.7)
plt.title('Projection of Wheat Kernel Dataset onto First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
cbar.set_label('Wheat Type')
cbar.ax.set_yticklabels(['1', '2', '3'])

plt.grid()
plt.show()

eigenvalues = pca.explained_variance_

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7, color='blue')
plt.title('Eigenvalues of the Covariance Matrix')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.grid()
plt.show()

pc_df.to_csv('seeds_principal_components.csv', index=False)
