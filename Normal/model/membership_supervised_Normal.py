import numpy as np
import pymc3 as pm
import theano

def membership_supervised_Normal(X, Y, a, rho, Mu, Lambda, T, ntrace, nchain, nskip):
    '''
    Given the covariates, responses and the model, estiimate the membership
    '''
    
    N = len(X)
    dg = len(T[0][0])
    nlabel = len(T)
    
    Tau = Mu * Lambda

    model = pm.Model()
    with model:
        G_ = pm.Dirichlet("G", a = a * rho, shape = (N, dg))  
        T_ = theano.shared(np.array(T))
        
        for i in range(N):
            u_ = pm.math.dot(G_[i: (i + 1)], T_[Y[i]].T) 
            Taux_ = pm.math.dot(u_, Tau)
            Lambdax_ = pm.math.dot(u_, Lambda)
            Sx_ = 1 / Lambdax_
            Mux_ = Sx_ * Taux_
            
            X_ = pm.Normal("x" + str(i), mu = Mux_, sigma = pm.math.sqrt(Sx_), observed = X[i])
            
        trace = pm.sample(ntrace, chains = nchain) 

    nsample = ntrace * nchain
    nsave = nsample // nskip
    index_save = [i * nskip - 1 for i in range(1, 1 + nsave)]
    
    return trace["G"][index_save].mean(axis = 0)
