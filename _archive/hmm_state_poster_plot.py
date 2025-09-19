import numpy as np
import pylab as plt

# Vanilla sigmoid
def sigmoid(mid, amp, slope, start, end, dt):
    """
    midpoint
    amplitude
    slope
    """
    x = np.linspace(start,end,int(1/dt))
    y = amp / ( 1 + np.exp( slope * -( x - mid)))
    return x,y

# Test plot
test_kwargs = { "mid" : 5, 
                "amp" : 2, 
                "slope" : 10 , 
                "start" : 0 , 
                "end" : 10,
                "dt" : 1/1000}

plt.plot(*sigmoid(**test_kwargs))
plt.show()

# Define top-hat sigmoid function
def tophat_sig(inf1, inf2):
    """
    inflection points 1 and 2
    Hard code other parameters for now
    """
    start = inf1/2
    end = inf2 * 1.5
    param_dict = {  "amp" : 1,
                    "slope" : 1/10,
                    "dt" : 1/1000,
                    "start" : start,
                    "end" : end}
    x,y1 = sigmoid(mid = inf1, **param_dict)
    _, y2 = sigmoid(mid = inf2, **param_dict)

    return x, y1 * (1-y2)

plt.plot(*tophat_sig(5,10))
plt.show()

# Plot state series
transition_points1a = [(0, 250), (250, 800), (800, 1500), (1500,2500)]
transition_points1b = [(0, 150), (200, 650), (700,1400), (1400, 2500)]
transition_points1c = [(0,100), (100,450), (450,2000), (2000,2500)]

def plot_state_sequence(transition_points):
    for points in transition_points:
        plt.fill_between(*tophat_sig(*points), alpha = 0.5)
        plt.plot(*tophat_sig(*points))

fig, ax = plt.subplots(3,1, sharex=True)
plt.sca(ax[0])
plot_state_sequence(transition_points1a)
plt.sca(ax[1])
plot_state_sequence(transition_points1b)
plt.sca(ax[2])
plot_state_sequence(transition_points1c)
plt.xlim(right = 2500)
for this_ax in ax:
    this_ax.axis('off')
#    this_ax.spines['top'].set_visible(False)
#    this_ax.spines['right'].set_visible(False)
fig.savefig('bla_gc_dummy_hmm_states_no_border',dpi=300)
plt.show()
