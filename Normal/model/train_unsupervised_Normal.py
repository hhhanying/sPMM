import numpy as np
import pymc3 as pm
import theano

def train_unsupervised_Normal(X, Y, K, b, alpha, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip):
    '''
    Given training set and hyperparameters, estimate the parameters of the model and the estimated SVM.
    X, Y: training set
    Y is only used for training SVM
    K: total topic numbers
    b, alpha: hyperparameters for the prior of a and rho
    (here we use all-one-vector as alpha)
    mu_Mu, sigma2_Mu: the prior of Mu is N(mu_Mu, sigma2_Mu/Lambda)
    alpha_Lambda, beta_Lambda: the prior of Lambda is Gamma(alpha_Lambda, beta_Lambda)
    ntrace, nchain, nskip: parameters of sampling
    output: a directionary 
    a,rho, Mu Lambda: estimate of parameters
    U: estimate of memberships of training set
   
    ''' 
    
    N = len(X)
    d = len(X[0])

    model = pm.Model()
    with model:
        # topics
        Lambda_ = pm.Gamma("Lambda", alpha = alpha_Lambda, beta = beta_Lambda, shape = (K, d))
        Mu_ = pm.Normal("Mu", mu = mu_Mu, sigma = pm.math.sqrt(sigma2_Mu / Lambda_), shape = (K, d))
        S_ = 1 / Lambda_
        Tau_ = Mu_ / S_       
        # corpus-level parameters
        a_ = pm.Exponential("a", b)  
        rho_ = pm.Dirichlet("rho", a = alpha)
        # membership
        U_ = pm.Dirichlet("U", a = a_ * rho_, shape = (N, K))
    
        for i in range(N):
            u_ = U_[i: (i + 1)]
            Taux_ = pm.math.dot(u_, Tau_)
            Lambdax_ = pm.math.dot(u_, Lambda_)
            Sx_ = 1 / Lambdax_
            Mux_ = Sx_ * Taux_
            X_ = pm.Normal("x" + str(i), mu = Mux_, sigma = pm.math.sqrt(Sx_), observed = X[i])
            
        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace * nchain
    nsave = nsample // nskip
    index_save = [i * nskip - 1 for i in range(1, 1 + nsave)]
    estimate = {}      
    
    for para in ["a", 'rho', 'Mu', "Lambda", "U"]:
        estimate[para] = trace[para][index_save].mean(axis = 0)
        
    return estimate  