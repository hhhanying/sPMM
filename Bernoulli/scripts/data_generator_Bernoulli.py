import numpy as np

def document_generator(a, rho, T, P, N):
    '''
    a, rho: corpus-level parameters
    T: transformation matrix. ntopic * K * dg
    P: success probability for each topics. K*d matrix. Between 0 and 1.
    N: the number of documents.
    
    output: 
    X: N*d, X[i] = document[i]
    Y: Y[i] = label[i]
    G: membership
    U: transformed membership
    '''

    nlabel = len(T) # number of classes
    d = len(P[0]) # dim(x)
    
    Y = np.random.choice(list(range(nlabel)),N) # labels
    G = np.random.dirichlet(a*rho,N)
    U = np.array([np.dot(T[Y[i]], G[i]) for i in range(N)])

    logitP = np.log(P/(1-P))
    logitX = np.dot(U,logitP)
    PX = 1/(1+np.exp(-logitX))

    X = np.random.binomial(1,PX,(N,d))
    
    return X, Y, G, U
