import numpy as np
from scipy.sparse import csc_matrix

# Create a large sparse matrix
n = 1000000 # number of rows
m = 3000 # number of columns

# Create a random sparse matrix
non_sparse = (np.random.rand(n, m) < 0.01).astype(int)
sparse = csc_matrix(non_sparse)

# Test multiplication
x = np.random.rand(m)
y = sparse.dot(x)
