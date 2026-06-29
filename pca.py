import numpy as np
x = np.array([[2,1],[3,2],[4,3],[5,4],[6,5]])
print(x)

mean = np.mean(x, axis=0) 
print("Mean:", mean)

x_centered = x - mean
print("Centered Data:\n", x_centered)

cov = np.cov(x_centered.T)
print("Covariance Matrix:\n", cov)
# Diagonal = variance
# Off diagonal = covariance

# np.cov(x_centered.T) transposes the matrix and calculates covariance,
# while np.cov(x_centered, rowvar=False) directly treats columns as variables.
# con1 = np.cov(x_centered, rowvar=False)
# print("Covariance Matrix (rowvar=False):\n", con1)


eigenvalues, eigenvectors = np.linalg.eig(cov)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("Sorted Eigenvalues:\n", eigenvalues)
print("Sorted Eigenvectors:\n", eigenvectors)
Z = x_centered @ eigenvectors

print("Transformed Data (Z):\n", Z)


import matplotlib.pyplot as plt


plt.scatter(x_centered[:,0],
            x_centered[:,1])

plt.axhline(0)
plt.axvline(0)

plt.show()

plt.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], color='r', scale=3)
plt.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], color='g', scale=3)
plt.scatter(x_centered[:, 0], x_centered[:, 1], color='b', alpha=0.5)
plt.axis('equal')
plt.show()



from sklearn.decomposition import PCA

pca = PCA(n_components=1)

X_reduced = pca.fit_transform(x)

print("x_reduced\n", X_reduced)