import numpy as np

def document_generator(a, rho, T, Lambda, Tau, N):
    '''
    a, rho: corpus-level parameters
    T: transformation matrix. ntopic * K * dg
    Lambda, Tau: topics. K*d matrix. Lambda are positive.
    N: the number of documents.
    
    Lambda = 1/sigma
    Tau = mu/sigma
    or
    sigma = 1/Lambda
    mu = Tau/Lambda
    
    output: 
    X: N*d, X[i] = document[i]
    Y: Y[i] = label[i]
    G: membership
    U: transformed membership
    '''

    nlabel = len(T) # number of classes
    d = len(Tau[0]) # dim(x)
    
    Y = np.random.choice(list(range(nlabel)), N) # labels
    G = np.random.dirichlet(a*rho, N)
    U = np.array([np.dot(T[Y[i]], G[i]) for i in range(N)])

    X = np.zeros((N, d))
    for i in range(N):
        u = U[i]
        for j in range(d):
            lambdax = np.dot(u, Lambda[:, j])
            taux = np.dot(u, Tau[:, j])
            sx = 1/lambdax
            mux = sx*taux
            X[i][j] = np.random.normal(mux, np.sqrt(sx))

    return X, Y, G, U