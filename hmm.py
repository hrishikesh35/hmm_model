import numpy as np
from hmmlearn import *
# from hmmlearn.hmm import MultinomialHMM
from hmmlearn.hmm import CategoricalHMM
startprob = np.array([0.6, 0.4])
transmat = np.array([[0.7, 0.3],[0.4, 0.6]])
covar = np.array([[0.1, 0.4, 0.5],[0.6, 0.3, 0.1]])
model = CategoricalHMM(n_components=2, startprob_prior=startprob, transmat_prior=transmat)
X = [[0,0,1,0],[0,0,1,0],[1,1,1,0],[0,0,1,0]]
model.fit(X)
print('Transition Probabilities:\n\n',model.transmat_, "\n")
prob = model.decode(np.array([0,1,0,1]).reshape(4,1))
print(np.exp(prob[0]))