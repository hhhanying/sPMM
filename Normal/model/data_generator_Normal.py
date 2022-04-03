from tkinter import N
import numpy as np

def document_generator_Normal(a, rho, T, Lambda, Tau, N, w = None):
    '''
    a, rho: corpus-level parameters
    T: transformation matrix. ntopic * K * dg
    Lambda, Tau: topics. K*d matrix. Lambda are positive.
    N: the number of documents.
    w: the probability of Y.
    
    Lambda = 1/sigma^2
    Tau = mu/sigma^2
    or
    sigma = 1/Lambda
    mu = Tau/Lambda

    x|u ~ normal(lambda_x, tau_x), where lambda_x = sum(u_i * lambda_i) and tau_x = sum(u_i * tau_i)
    
    output: 
    X: N*d, X[i] = document[i]
    Y: Y[i] = label[i]
    G: membership
    U: transformed membership
    '''

    nlabel = len(T) # number of classes
    d = len(Tau[0]) # dim(x)
    
    Y = np.random.choice(list(range(nlabel)), size = N, p = w) # labels

        
    G = np.random.dirichlet(a * rho, N)
    U = np.array([np.dot(T[Y[i]], G[i]) for i in range(N)])

    X = np.zeros((N, d))
    for i in range(N):
        u = U[i]
        for j in range(d):
            lambdax = np.dot(u, Lambda[:, j])
            taux = np.dot(u, Tau[:, j])
            sx = 1 / lambdax
            mux = sx * taux
            X[i][j] = np.random.normal(mux, np.sqrt(sx))

    return X, Y, G, U