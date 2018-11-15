import numpy as np


p0 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
p1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
p2 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]

p = np.array([p0, p1, p2])

from bayespy.utils import random
z = random.categorical([1/3, 1/3, 1/3], size=100)
x = random.bernoulli(p[z])

N = 100
D = 10
K = 10

from bayespy.nodes import Categorical, Dirichlet

R = Dirichlet(K*[1e-5], name='R')
Z = Categorical(R, plates=(N,1), name='Z')

from bayespy.nodes import Beta
P = Beta([0.5, 0.5], plates=(D,K), name='P')

from bayespy.nodes import Mixture, Bernoulli
X = Mixture(Z, Bernoulli, P)

from bayespy.inference import VB
Q = VB(Z, R, X, P)
P.initialize_from_random()
X.observe(x)

Q.update(repeat=1000)
import bayespy.plot as bpplt
bpplt.hinton(R)
bpplt.hinton(P)
bpplt.hinton(Z)