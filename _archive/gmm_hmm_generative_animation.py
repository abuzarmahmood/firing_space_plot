import numpy as np
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pomegranate import *
from tqdm import tqdm,trange

# ____  ____    _              _           _                   
#|___ \|  _ \  | |_ _ __ __ _ (_) ___  ___| |_ ___  _ __ _   _ 
#  __) | | | | | __| '__/ _` || |/ _ \/ __| __/ _ \| '__| | | |
# / __/| |_| | | |_| | | (_| || |  __/ (__| || (_) | |  | |_| |
#|_____|____/   \__|_|  \__,_|/ |\___|\___|\__\___/|_|   \__, |
#                           |__/                         |___/ 

n_components = 3
dims = 2
mean_array = np.random.random((n_components, dims))*20
cov_array = np.array([make_spd_matrix(dims) \
        for x in range(n_components)])
dists = [MultivariateGaussianDistribution(mean,cov) \
                    for mean,cov in zip(mean_array,cov_array)]
trans_mat = numpy.array([[0.99, 0.05, 0.0],
                             [0.0, 0.99, 0.05],
                             [0.0, 0.0, 0.9]])
starts = numpy.array([1.0, 0.0, 0.0])
ends = numpy.array([0.0, 0.0, 0.1])
model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
samples = np.concatenate([model.sample() for x in trange(100)])
plt.scatter(*samples.T);plt.show()
plt.imshow(samples.T,aspect='auto');plt.show()
plt.plot(samples[:,1]);plt.show()

# Generate data
shuffled_dat = np.random.permutation(samples)

max_len = 1000
history_window = 20
inds = [np.arange(i-history_window,i) if i>history_window else np.arange(i+1)\
        for i in range(max_len)]

#dat = shuffled_dat
dat = samples[:max_len]

# First set up the figure, the axis, and the plot element we want to animate
xlims = [np.min(dat[:,0]),np.max(dat[:,0])]
ylims = [np.min(dat[:,1]),np.max(dat[:,1])]

fig,ax = plt.subplots()
ax.set_xlim(xlims)
ax.set_ylim(ylims)
line, = ax.plot([], [], alpha=0.3)
dot, = ax.plot([], [],'o', alpha=0.5)
current_dot, = ax.plot([], [],'o', color='black', markersize = 12)
history_dots, = ax.plot([], [],'o', color='red', alpha=0.5)
#plt.sca(ax)
#plt.axis('off')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    dot.set_data([], [])
    current_dot.set_data([], [])
    history_dots.set_data([], [])
    return dot, line, current_dot, history_dots

# animation function.  This is called sequentially

def animate(i):
    global x,y, history_window
    line.set_data(dat[:i,0],dat[:i,1])
    dot.set_data(dat[:i,0],dat[:i,1])
    current_dot.set_data(dat[i,0],dat[i,1])
    history_dots.set_data(dat[inds[i],0], dat[inds[i],1])
    return dot, line, current_dot, history_dots

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=dat.shape[0], interval=50, blit=True)
#plt.show()

anim.save('hmm_samples.mp4', fps=15, extra_args=['-vcodec', 'libx264'])


fig,ax = plt.subplots(1,2, gridspec_kw = {'width_ratios' : [5,1]})
for this_ax in ax:
    plt.sca(this_ax)
    plt.axis('off')
ax[0].plot(samples[:300,0], alpha = 0.8, linewidth = 2)
ax[1].hist(samples[:,0],bins=70,orientation='horizontal', alpha = 0.8)
ax[1].hist(samples[:,0],bins=70,orientation='horizontal', histtype = 'step', color = 'k') 
plt.show()
