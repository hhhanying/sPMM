import numpy as np
import pymc3 as pm
import theano
from sklearn.model_selection import KFold
from collections import Counter

def train_supervised(X, Y, K, b, alpha, T, alpha_p, beta_p, ntrace, nchain, nskip):
    '''
    X, Y: training set
    K: total topic numbers
    b, alpha: hyperparameters for the prior of a and rho
    T: transformation matrix
    alpha_p, beta_p: the hyperparameter for the prior of success probability (Beta distribution)
    ntrace, nchain, nskip: parameters of sampling
    '''
    
    N = len(Y)
    dg = len(alpha)
    d = len(X[0])

    model = pm.Model()
    with model:
        P_ = pm.Beta("P", alpha = alpha_p, beta = beta_p, shape = (K,d))
        a_ = pm.Exponential("a", b)
        rho_ = pm.Dirichlet("rho", a = alpha)
        G_ = pm.Dirichlet("G", a = a_*rho_, shape = (N, dg))
        T_= np.array([T[Y[i]] for i in range(N)])
        logit_P = pm.math.logit(P_)

        for i in range(N):
            u_ = pm.math.dot(G_[i:(i+1)],T_[i].T)
            logit_x = pm.math.dot(u_, logit_P)
            X_ = pm.Bernoulli("x"+str(i), logit_p = logit_x, observed = X[i])

        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]
    estimate = {}
    
    for para in ["a", 'rho', "P", "G"]:
        estimate[para] = trace[para][index_save].mean(axis=0)
    return estimate

def predict_supervised(X, estimate, T, ntrace, nchain, nskip):
    '''
    X: test set
    estimate: parameters estimated from training set, a dictionary containing a, rho and P 
    T: transformation matrix
    ntrace, nchain, nskip: parameters of sampling
    output: estimated label Y and membership G
    '''
    N = len(X)
    dg = len(T[0][0])
    nlabel = len(T)
    a, rho, P = estimate["a"], estimate["rho"], estimate["P"]
    T_arr = np.array(T)
    
    model = pm.Model()
    with model:
        G_ = pm.Dirichlet("G", a=a*rho, shape = (N, dg))
        Y_ = pm.Categorical("Y", p=np.ones(nlabel)/nlabel, shape=N)
        T_ = theano.shared(T_arr)
        logit_P = pm.math.logit(P)
        
        for i in range(N):
            t_ = T_[Y_[i]]
            u_ = pm.math.dot(G_[i:(i+1)],t_.T)
            logit_x = pm.math.dot(u_, logit_P)
            X_ = pm.Bernoulli("x"+str(i), logit_p = logit_x, observed = X[i])

        trace = pm.sample(ntrace, chains = nchain) 

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]    
    prediction = {}

    prediction["G"] = trace["G"][index_save].mean(axis=0)

    pre_Y = []
    for i in range(N):
        y = trace["Y"][index_save,i].tolist()
        frequency = Counter(y).most_common()
        possible_prediction = []
        j = 0
        while j<len(frequency) and frequency[j][1] == frequency[0][1]:
            possible_prediction.append(frequency[j][0])
            j += 1
        tem_pre = np.random.choice(possible_prediction, 1)[0]
        pre_Y.append(tem_pre)       
    prediction["Y"] = np.array(pre_Y)   
    return prediction

def accuracy_supervised_Bernoulli(trainingX, trainingY, testX, testY, T, alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set, test set and needed parameters
    # this function will return the test accuracy
    estimation = train_supervised(trainingX, trainingY, K, b, alpha, T, alpha_p, beta_p, ntrace, nchain, nskip)
    prediction = predict_supervised(testX, estimation, T, ntrace, nchain, nskip)
    Ntest = len(testY)
    accuracy = sum(testY == prediction["Y"])/Ntest
    print(accuracy)
    return accuracy

def CV_supervised_Bernoulli(X, Y, Nfold, T, alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set
    # This function will perform K-fold cross validation and return the list of the accuracies
    kf = KFold(n_splits=Nfold)
    res = []
    for train_index, test_index in kf.split(X):
        tem_accuracy = accuracy_supervised_Bernoulli(X[train_index], Y[train_index], X[test_index], Y[test_index], T, alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip)
        res.append(tem_accuracy)
    return res  
    