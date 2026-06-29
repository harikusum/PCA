# PCA Implementation

This repository contains an implementation of Principal Component Analysis (PCA) in Python.

## Features
- Calculates the covariance matrix
- Computes eigenvalues and eigenvectors
- Projects data onto principal components
- Visualizes the results using Matplotlib

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- scikit-learn

## Usage
Run the script using:
```bash
python pca.py
```

## PCA Face Reconstruction Example
This project also includes an eigenfaces example that reconstructs the same face using different numbers of principal components.

### Observation
- **k = 5**: Very blurry
- **k = 10**: Face shape appears
- **k = 20**: Eyes become clearer
- **k = 50**: Looks similar to the original
- **k = 100**: Almost identical

As the number of components increases, the reconstruction becomes sharper and more faithful to the original face.