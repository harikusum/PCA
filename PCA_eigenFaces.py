import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.decomposition import PCA

faces = fetch_olivetti_faces()

X = faces.data
images = faces.images

print("X shape:", X.shape)
print("Images shape:", images.shape)

fig, axes = plt.subplots(2, 5, figsize=(10,4))

for ax, img in zip(axes.ravel(), images[:10]):
    ax.imshow(img, cmap="gray")
    ax.axis("off")

plt.show()

mean_face = np.mean(X, axis=0)

print(mean_face.shape)

pca = PCA(n_components=50)

X_pca = pca.fit_transform(X)

print(X_pca.shape)
eigenfaces = pca.components_

print(eigenfaces.shape)

fig, axes = plt.subplots(2,5, figsize=(10,4))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(
        eigenfaces[i].reshape(64,64),
        cmap="gray"
    )
    ax.set_title(f"PC{i+1}")
    ax.axis("off")

plt.show()

variance = pca.explained_variance_ratio_

print(variance[:10])



cum_var = np.cumsum(variance)

plt.plot(cum_var)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.show()

face = X[0]
compressed = pca.transform(face.reshape(1,-1))
reconstructed = pca.inverse_transform(compressed)
fig, ax = plt.subplots(1,2, figsize=(8,4))

ax[0].imshow(face.reshape(64,64), cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(reconstructed.reshape(64,64), cmap="gray")
ax[1].set_title("Reconstructed")
ax[1].axis("off")

plt.show()

for k in [5, 10, 20, 50, 100]:
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)

    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"k={k}, reconstruction error: {reconstruction_error:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(X[0].reshape(64, 64), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(X_reconstructed[0].reshape(64, 64), cmap="gray")
    axes[1].set_title(f"Reconstructed ({k} components)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
