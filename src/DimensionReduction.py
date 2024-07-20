import numpy as np

def PCA(X):
    A = np.matmul(X,X.T)
    eigenvalue, eigenvector = np.linalg.eig(A)

    idx = eigenvalue.argsort()[::-1]   
    eigenvalue = eigenvalue[idx]
    eigenvector = eigenvector[:,idx]
    
    P = eigenvector.T
    Y = np.matmul(P, X)
    return P, np.real(Y)