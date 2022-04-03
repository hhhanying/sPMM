import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

def test_supervised_Normal(X, nsample, a, rho, Tau, Lambda, T, w = None):
    '''
    Given the estimated model, return the retimated labels and the posterior distribution of labels.
    res["labels"]: the list of estmate labels
    res["loglikelihood"]: the list of the posterior distribution.
    '''
    nlabel = len(T)
    N = len(X)
    d = len(X[0])
    if not w: # if no info, assume uniform
        w = np.zeros(nlabel)


    probs = []
    for j in range(N):
        tem = []

        for y in range(nlabel):
            gs = np.random.dirichlet(a * rho, nsample)
            us = np.dot(gs, T[y].T)
            lambdas = np.dot(us, Lambda)
            taus = np.dot(us, Tau)
            sigmas = np.sqrt(1 / lambdas)
            mus = taus / lambdas

            xs = np.tile(X[j,], (nsample, 1))
            likelihoods = norm.pdf(xs, mus, sigmas)
            loglikelihood = np.log(w[y]) + logsumexp(np.log(likelihoods).sum(axis = 1)) # since we use the same nsample, no need to subtract log(nsample)

            tem.append(loglikelihood)

        probs.append(tem)

    labels = []
    for i in range(N):
        post_prob = probs[i]
        label = post_prob.index(max(post_prob))
        labels.append(label)
    
    return {"labels": labels, "loglikelihood": probs}