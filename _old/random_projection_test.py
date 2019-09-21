import numpy as np
from sklearn.random_projection import SparseRandomProjection as sparse_random
from sklearn.decomposition import PCA as pca
from matplotlib import pyplot as plt
#  ____                           _         ____        _        
# / ___| ___ _ __   ___ _ __ __ _| |_ ___  |  _ \  __ _| |_ __ _ 
#| |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \ | | | |/ _` | __/ _` |
#| |_| |  __/ | | |  __/ | | (_| | ||  __/ | |_| | (_| | || (_| |
# \____|\___|_| |_|\___|_|  \__,_|\__\___| |____/ \__,_|\__\__,_|
#
dim = 10
n_components = 2
dat_points = 1000
total_dat = []
for comp in range(n_components):
      mean_vec = np.random.rand(dim)*10
      cov_mat = np.eye(dim)
      total_dat.append( np.random.multivariate_normal(mean_vec,cov_mat,dat_points))

total_dat = np.asarray(total_dat)
total_dat_long = total_dat[0,:,:]
for comp in range(1,n_components):
    total_dat_long = np.concatenate((total_dat_long,total_dat[comp,:,:]),axis=0)

# ____                 _                 
#|  _ \ __ _ _ __   __| | ___  _ __ ___  
#| |_) / _` | '_ \ / _` |/ _ \| '_ ` _ \ 
#|  _ < (_| | | | | (_| | (_) | | | | | |
#|_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_|
#                                        
# ____            _           _   _             
#|  _ \ _ __ ___ (_) ___  ___| |_(_) ___  _ __  
#| |_) | '__/ _ \| |/ _ \/ __| __| |/ _ \| '_ \ 
#|  __/| | | (_) | |  __/ (__| |_| | (_) | | | |
#|_|   |_|  \___// |\___|\___|\__|_|\___/|_| |_|
#              |__/                            

# Compare with PCA
pca_transformer = pca(n_components=2)
pca_data = pca_transformer.fit_transform(total_dat_long)
plt.subplot(211)
plt.scatter(pca_data[:,0],pca_data[:,1])

# Random projection
transformer = sparse_random (n_components = 2)
X_new = transformer.fit_transform(total_dat_long)
plt.subplot(212)
plt.scatter(X_new[:,0],X_new[:,1])