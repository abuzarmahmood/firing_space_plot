import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# ____  ____    _              _           _                   
#|___ \|  _ \  | |_ _ __ __ _ (_) ___  ___| |_ ___  _ __ _   _ 
#  __) | | | | | __| '__/ _` || |/ _ \/ __| __/ _ \| '__| | | |
# / __/| |_| | | |_| | | (_| || |  __/ (__| || (_) | |  | |_| |
#|_____|____/   \__|_|  \__,_|/ |\___|\___|\__\___/|_|   \__, |
#                           |__/                         |___/ 

# Generate data
history_window = 100
t = np.linspace(0, 2*np.pi*4,1000)
inds = [np.arange(i-history_window,i) if i>history_window else np.arange(i+1)\
        for i in range(len(t))]
x = np.sin(t) 
np.random.seed(0)
y = (np.random.random(len(t))-0.5)*0.2

# First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
fig,ax = plt.subplots(3,1)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-1,1)
ax[1].set_xlim(0,max(t))
ax[1].set_ylim(min(x)*1.2,max(x*1.2))
ax[1].set_ylabel('X')
ax[2].set_xlim(0,max(t))
ax[2].set_ylim(min(x)*1.2,max(x*1.2))
ax[2].set_ylabel('Y')
line, = ax[0].plot([], [], lw=2, alpha = 0.5)
dot, = ax[0].plot([], [], 'ro', ms=5)
marker1, = ax[0].plot([], [], 'k', alpha = 0.5, lw=2) 
marker2, = ax[1].plot([], [], 'k', alpha = 0.5, lw=2) 
var1, = ax[1].plot([], [], lw=2, alpha = 0.5)
var2, = ax[2].plot([], [], lw=2, alpha = 0.5)
for this_ax in ax:
    plt.sca(this_ax)
    plt.axis('off')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    dot.set_data([], [])
    marker1.set_data([], [])
    marker2.set_data([], [])
    var1.set_data([], [])
    var2.set_data([], [])
    return dot, line, var1, var2, marker1, marker2

# animation function.  This is called sequentially

def animate(i):
    global x,y, history_window
    line.set_data(x[inds[i]], y[inds[i]])
    marker1.set_data([0,0],[-1,1])
    marker2.set_data([0,max(t)],[0,0])
    var1.set_data(t[:i],x[:i])
    var2.set_data(t[:i],y[:i])
    dot.set_data(x[i],y[i])
    return dot, line, var1, var2, marker1, marker2

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(t), interval=20, blit=True)
plt.show()
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# _____ ____    _____           _           _                   
#|___ /|  _ \  |_   _| __ __ _ (_) ___  ___| |_ ___  _ __ _   _ 
#  |_ \| | | |   | || '__/ _` || |/ _ \/ __| __/ _ \| '__| | | |
# ___) | |_| |   | || | | (_| || |  __/ (__| || (_) | |  | |_| |
#|____/|____/    |_||_|  \__,_|/ |\___|\___|\__\___/|_|   \__, |
#                            |__/                         |___/ 

# Generate data
history_window = 100
t = np.linspace(0, 2*np.pi*4,1000)
inds = [np.arange(i-history_window,i) if i>history_window else np.arange(i+1)\
        for i in range(len(t))]
x = np.sin(t) 
y = np.cos(t)
np.random.seed(0)

def gauss_kern(size):
    x = np.arange(-size,size+1)
    kern = np.exp(-(x**2)/float(size))
    return kern / sum(kern)
def gauss_filt(vector, size):
    kern = gauss_kern(size)
    return np.convolve(vector, kern, mode='same')

z = (np.random.random(len(t))-0.5)*0.2
z = gauss_filt(z,25)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(16,8))
ax0 = fig.add_subplot(1,2,1, projection = '3d')
ax1 = fig.add_subplot(3,2,2)
ax2 = fig.add_subplot(3,2,4)
ax3 = fig.add_subplot(3,2,6)
line, = ax0.plot(x[:1], y[:1], z[:1],lw=2, alpha = 0.5)
dot, = ax0.plot(x[:1], y[:1], z[:1], 'ro', ms=5)
zmarker, = ax0.plot([0], [0], z[:1], 'ko', ms=5)
zline, = ax0.plot([0,0], [0,0], [-1,1], lw=2, alpha=0.5, c='k') 
var1, = ax1.plot([], [], lw=2, alpha = 0.5)
var2, = ax2.plot([], [], lw=2, alpha = 0.5)
var3, = ax3.plot([], [], lw=2, alpha = 0.5)
for this_ax in [ax1,ax2,ax3]:
    plt.sca(this_ax)
    plt.axis('off')


data = np.stack((x,y,z))
ax0.set_xlim3d([-1, 1.0])
ax0.set_ylim3d([-1, 1.0])
ax0.set_zlim3d([-1, 1])
ax0.axes.xaxis.set_ticklabels([])
ax0.axes.yaxis.set_ticklabels([])
ax0.axes.zaxis.set_ticklabels([])
ax1.set_xlim(0,max(t))
ax1.set_ylim(min(x)*1.2,max(x*1.2))
ax2.set_xlim(0,max(t))
ax2.set_ylim(min(x)*1.2,max(x*1.2))
ax3.set_xlim(0,max(t))
ax3.set_ylim(min(x)*1.2,max(x*1.2))

def animate(i):
    global data, history_window, line
    line.set_data(data[:2, inds[i]])
    line.set_3d_properties(data[-1, inds[i]])
    dot.set_data(data[:2, i])
    dot.set_3d_properties(data[-1, i])
    zmarker.set_3d_properties(data[-1, i])
    zline.set_data([0,0],[0,0])
    zline.set_3d_properties([-1,1])
    var1.set_data(t[:i],x[:i])
    var2.set_data(t[:i],y[:i])
    var3.set_data(t[:i],z[:i])
    return line,dot,var1,var2,var3, zmarker, zline 

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, 
                               frames = len(t),interval=20, blit=True)
#plt.show()
anim.save('animation_3d.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# _  _   ____    _____           _           _                   
#| || | |  _ \  |_   _| __ __ _ (_) ___  ___| |_ ___  _ __ _   _ 
#| || |_| | | |   | || '__/ _` || |/ _ \/ __| __/ _ \| '__| | | |
#|__   _| |_| |   | || | | (_| || |  __/ (__| || (_) | |  | |_| |
#   |_| |____/    |_||_|  \__,_|/ |\___|\___|\__\___/|_|   \__, |
#                             |__/                         |___/ 

# Generate data
history_window = 100
t = np.linspace(0, 2*np.pi*3,1000)
new_t = np.linspace(0, 2*np.pi*6,2000)
inds = [np.arange(i-history_window,i) if i>history_window else np.arange(i+1)\
        for i in range(len(new_t))]

def gauss_kern(size):
    x = np.arange(-size,size+1)
    kern = np.exp(-(x**2)/float(size))
    return kern / sum(kern)
def gauss_filt(vector, size):
    kern = gauss_kern(size)
    return np.convolve(vector, kern, mode='same')

np.random.seed(None)
x1 = np.sin(t) 
y1 = np.cos(t)
z1 = (np.random.random(len(t))-0.5)#*0.2
w1 = (np.random.random(len(t))-0.5)#*0.2
#z1 = gauss_filt(z1,25)
#w1 = gauss_filt(w1,25)

z2 = np.sin(t) 
w2 = np.cos(t)
x2 = (np.random.random(len(t))-0.5)#*0.2
#x2 = gauss_filt(x2,25)
y2 = (np.random.random(len(t))-0.5)#*0.2
#y2 = gauss_filt(y2,25)

x = np.concatenate([x1,x2])
y = np.concatenate([y1,y2])
z = np.concatenate([z1,z2])
w = np.concatenate([w1,w2])

data = np.stack([x,y,z,w])
data = np.stack([gauss_filt(var,500) for var in data])

#plt.plot(data.T);plt.show()
#plt.imshow(data,interpolation='nearest',aspect='auto');plt.show()

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(16,8))
line_axs = [fig.add_subplot(4,2,i) for i in [1,3,5,7]]
traj_axs = [fig.add_subplot(2,2,i) for i in [2,4]]
line1, = traj_axs[0].plot(data[0,:1], data[1,:1],lw=2, alpha = 0.5)
line2, = traj_axs[1].plot(data[2,:1], data[3,:1],lw=2, alpha = 0.5, c='orange')
dot1, = traj_axs[0].plot(data[0,:1], data[1,:1],'ro', ms=5)
dot2, = traj_axs[1].plot(data[2,:1], data[3,:1],'ro', ms=5)
var0, = line_axs[0].plot([], [], lw=2, alpha = 0.5)
var1, = line_axs[1].plot([], [], lw=2, alpha = 0.5)
var2, = line_axs[2].plot([], [], lw=2, alpha = 0.5, c='orange')
var3, = line_axs[3].plot([], [], lw=2, alpha = 0.5, c='orange')

for ax in traj_axs:
    ax.axis('equal')
for ax in traj_axs:
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
for ax in line_axs:
    ax.set_xlim(0,2*max(t))
    ax.set_ylim(-1.2,1.2)
for this_ax in [*line_axs,*traj_axs]:
    plt.sca(this_ax)
    plt.axis('off')

def animate(i):
    global data, var_list, history_window
    line1.set_data(data[0,inds[i]], data[1,inds[i]])
    line2.set_data(data[2,inds[i]], data[3,inds[i]])
    dot1.set_data(data[0,i], data[1,i])
    dot2.set_data(data[2,i], data[3,i])
    var0.set_data(new_t[:i],data[0,:i])
    var1.set_data(new_t[:i],data[1,:i])
    var2.set_data(new_t[:i],data[2,:i])
    var3.set_data(new_t[:i],data[3,:i])
    return line1,line2, dot1, dot2, var0, var1, var2, var3, 

anim = animation.FuncAnimation(fig, animate, 
                               frames = 2*len(t),interval=20, blit=True)
anim.save('animation_4d.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
#plt.show()
