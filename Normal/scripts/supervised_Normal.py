import numpy as np
import pymc3 as pm
import theano
from sklearn.model_selection import KFold
from collections import Counter

def train_supervised(X, Y, K, b, alpha, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip):
    '''
    Given training set and hyperparameters, estimate the parameters of the model.

    X, Y: training set
    K: total topic numbers
    b, alpha: hyperparameters for the prior of a and rho
    T: transformation matrix
    mu_Mu, sigma2_Mu: the prior of Mu is N(mu_Mu, sigma2_Mu/Lambda)
    alpha_Lambda, beta_Lambda: the prior of Lambda is Gamma(alpha_Lambda, beta_Lambda)
    ntrace, nchain, nskip: parameters of sampling

    output: a directionary.
    a,rho, Mu Lambda: estimate of parameters
    G: estimate of memberships of training set
    '''

    N = len(Y) # sample size
    dg = len(alpha) # dim(untransformed membership)
    d = len(X[0]) # dim(x)
    T_arr = np.array(T) # nlabel * K * dg

    model = pm.Model()
    with model:
        Lambda_ = pm.Gamma("Lambda", alpha = alpha_Lambda, beta = beta_Lambda, shape = (K, d)) # notice np.random.gamma require shape and scale
        Mu_ = pm.Normal("Mu", mu = mu_Mu, sigma = pm.math.sqrt(sigma2_Mu / Lambda_), shape = (K, d)) # notice sigma rather than sigma^2
        S_ = 1 / Lambda_
        Tau_ = Mu_ / S_       

        a_ = pm.Exponential("a", b)  # mean = 1/b
        rho_ = pm.Dirichlet("rho", a = alpha)
        G_ = pm.Dirichlet("G", a = a_ * rho_, shape = (N, dg))        
        T_ = theano.shared(T_arr)
    
        for i in range(N):
            u_ = pm.math.dot(G_[i: (i + 1)], T_[Y[i]].T) # 1 * K
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
    
    for para in ["a", 'rho', 'Mu', "Lambda", "G"]:
        estimate[para] = trace[para][index_save].mean(axis = 0)
    return estimate

def predict_supervised(X, estimate, T, ntrace, nchain, nskip):
    '''
    Given the test set and the estimated parameters of model, predict the labels and estimate the memberships.

    X: test set
    estimate: parameters estimated from training set, a dictionary containing a, rho, Mu and Lambda
    T: transformation matrix
    ntrace, nchain, nskip: parameters of sampling
    
    output: a directionary.
    G: estimate of memberships of test set
    Y: estimate of labels of test set
    '''
    
    N = len(X)
    dg = len(T[0][0])
    nlabel = len(T)
    a, rho, Mu, Lambda = estimate["a"], estimate["rho"], estimate["Mu"], estimate["Lambda"] # extrace estimated parameters
    Tau = Mu * Lambda
    T_arr = np.array(T)
    
    model = pm.Model()
    with model:
        G_ = pm.Dirichlet("G", a = a * rho, shape = (N, dg))
        Y_ = pm.Categorical("Y", p = np.ones(nlabel) / nlabel, shape = N)    
        T_ = theano.shared(T_arr)
        
        for i in range(N):
            t_ = T_[Y_[i]]
            u_ = pm.math.dot(G_[i: (i + 1)], t_.T)
            Taux_ = pm.math.dot(u_, Tau)
            Lambdax_ = pm.math.dot(u_, Lambda)
            Sx_ = 1 / Lambdax_
            Mux_ = Sx_ * Taux_
            
            X_ = pm.Normal("x" + str(i), mu = Mux_, sigma = pm.math.sqrt(Sx_), observed = X[i])
            
        trace = pm.sample(ntrace, chains = nchain) 

    nsample = ntrace * nchain
    nsave = nsample // nskip
    index_save = [i * nskip - 1 for i in range(1, 1 + nsave)]
    prediction = {}

    prediction["G"] = trace["G"][index_save].mean(axis = 0)
    
    pre_Y = []
    for i in range(N):
        y = trace["Y"][index_save, i].tolist() # saved prediction for ith subject
        frequency = Counter(y).most_common() # the labels and their frequencies sorted by frequency
        possible_prediction = [] # used to save the most common labels
        j = 0
        while j < len(frequency) and frequency[j][1] == frequency[0][1]:
            possible_prediction.append(frequency[j][0])
            j += 1
        tem_pre = np.random.choice(possible_prediction, 1)[0] # break the tie randomly
        pre_Y.append(tem_pre)       
    prediction["Y"] = np.array(pre_Y)  
    
    return prediction

def accuracy_supervised_Normal(trainingX, trainingY, testX, testY, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set, test set and needed parameters
    # this function will return the test accuracy
    estimation = train_supervised(trainingX, trainingY, K, b, alpha, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip)
    prediction = predict_supervised(testX, estimation, T, ntrace, nchain, nskip)
    Ntest = len(testY)
    accuracy = sum(testY == prediction["Y"])/Ntest
    return accuracy, estimation   

def CV_supervised_Normal(X, Y, Nfold, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set
    # This function will perform K-fold cross validation and return the list of the accuracies
    # Output:
    # 1. the list of the accuracies
    # 2. the estimation corresponding to the best accuracy

    kf = KFold(n_splits = Nfold)
    res = []
    best_accuracy = 0

    for train_index, test_index in kf.split(X):
        tem_accuracy, tem_estimation = accuracy_supervised_Normal(X[train_index], Y[train_index], X[test_index], Y[test_index], T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip)
        res.append(tem_accuracy)

        # save the best accuracy it achieved and the corresponding model
        if tem_accuracy > best_accuracy:
            best_accuracy = tem_accuracy
            best_estimation = tem_estimation
            
    return res, best_estimation 