import numpy as np
import pylab as plt
from baseline_divergence_funcs import *

n = 200
n_cells = np.random.randint(0,4,n)
c_mat = np.zeros((n,n));

for nrn1 in range(n):
    for nrn2 in range(n):
        
        prob = np.random.rand()
        if n_cells[nrn1] == n_cells[nrn2]:
           if prob < 0.6:
               c_mat[nrn1,nrn2] = 1;
        else:
           if prob < 0.1:
              c_mat[nrn1,nrn2] = 1;

corr_r = np.corrcoef(c_mat);
corr_c = np.corrcoef(np.transpose(c_mat));
total_corr = corr_r + corr_c;

thresh_vec = np.linspace(0,1,20)
all_groups = []
all_entropy= []
all_blocks = []
all_global_dists = []
count = 1
for thresh in thresh_vec:
    plt.subplot(4,5,count)
    #test_mat, pred_group, order_vec = blockify(total_corr, thresh)
    test_mat, pred_group, order_vec = blockify(this_dist, thresh,'dis')
    all_blocks.append(test_mat)
    plt.imshow(test_mat,cmap = plt.get_cmap('viridis'))
    this_entropy = entropy_proxy(test_mat)
    #this_global_dist = global_distance(test_mat,pred_group)
    #all_global_dists.append(this_global_dist)
    all_entropy.append(this_entropy)
    all_groups.append(pred_group)
    count += 1

#plt.figure()
#plt.imshow(all_blocks[np.argmax(all_entropy)])
#print(all_groups[np.argmax(all_entropy)])
print(np.argsort(all_entropy,)[-5:])
print(np.sort(all_entropy)[-5:])
#print(np.argsort(all_global_dists))
#print(np.sort(all_global_dists))

#print([len(np.unique(x)) for x in all_groups])