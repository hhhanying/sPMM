import numpy as np
import pymc3 as pm
import theano
from sklearn import svm
from sklearn.model_selection import KFold

def train_unsupervised(X, Y, K, b, alpha, alpha_p, beta_p, ntrace, nchain, nskip):
    '''
    Given training set and hyperparameters, estimate the parameters of the model and the estimated SVM.

    X, Y: training set
    Y is only used for training SVM
    K: total topic numbers
    b, alpha: hyperparameters for the prior of a and rho
    (here we use all-one-vector as alpha)
    alpha_p, beta_p: the hyperparameter for the prior of success probability (Beta distribution)
    ntrace, nchain, nskip: parameters of sampling

    output: a directionary and a classifier.
    a,rho, P: estimate of parameters
    U: estimate of memberships of training set
    '''
    
    N = len(X)
    d = len(X[0])
    
    model = pm.Model()
    with model:
        P_ = pm.Beta("P", alpha = alpha_p, beta = beta_p, shape = (K,d))
        a_ = pm.Exponential("a", b) 
        rho_ = pm.Dirichlet("rho", a = alpha)        
        U_ = pm.Dirichlet("U", a=a_*rho_, shape = (N, K)) # difference
        logit_P = pm.math.logit(P_)
     
        for i in range(N):
            u_ = U_[i:(i+1)]  # attention
            logit_x = pm.math.dot(u_, logit_P)
            X_ = pm.Bernoulli("x"+str(i), logit_p = logit_x, observed = X[i])

        trace = pm.sample(ntrace, chains = nchain)
        
    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]
    estimate = {}
    
    for para in ["a", 'rho', "P", "U"]:
        estimate[para] = trace[para][index_save].mean(axis=0)

    classifier = svm.SVC(C=2,kernel='rbf', decision_function_shape='ovo') 
    classifier.fit(estimate["U"], Y)
        
    return estimate, classifier

def predict_unsupervised(X, estimate, classifier, ntrace, nchain, nskip):
    '''
    Given the test set, the estimated parameters of model and the estimated SVM, predict the labels and estimate the memberships.

    X: test set
    estimate: parameters estimated from training set, a dictionary containing a, rho and P 
    classifier: the SVM trained from training set
    ntrace, nchain, nskip: parameters of sampling

    output: a directionary.
    U: estimate of memberships of test set
    Y: estimate of labels of test set
    '''
    
    N = len(X)
    a, rho, P = estimate["a"], estimate["rho"], estimate["P"]
    K = len(P)

    model = pm.Model()
    with model:
        U_ = pm.Dirichlet("U", a=a*rho, shape = (N, K)) # difference
        logit_P = pm.math.logit(P)
           
        for i in range(N):
            u_ = U_[i:(i+1)]  # attention
            logit_x = pm.math.dot(u_, logit_P)
            X_ = pm.Bernoulli("x"+str(i), logit_p = logit_x, observed = X[i])

        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]    
    prediction = {}
    
    for i in ["U"]:
        prediction[i] = trace[i][index_save].mean(axis=0) 
    prediction["Y"] = classifier.predict(prediction["U"])
     
    return prediction

def accuracy_unsupervised_Bernoulli(trainingX, trainingY, testX, testY, alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set, test set and needed parameters
    # this function will return the test accuracy, the estimated parameters of model and the estimated SVM
    estimation, classifier = train_unsupervised(trainingX, trainingY, K, b, alpha, alpha_p, beta_p, ntrace, nchain, nskip)
    prediction = predict_unsupervised(testX, estimation, classifier, ntrace, nchain, nskip)
    Ntest = len(testY)
    accuracy = sum(testY == prediction["Y"])/Ntest
    print(accuracy)
    return accuracy, estimation, classifier

def CV_unsupervised_Bernoulli(X, Y, Nfold, alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set
    # This function will perform K-fold cross validation and return the list of the accuracies
    # Notica the alpha here is not the same as the alpha used before
    # Its dimension become K
    # It's defined by the user. We choose all-1-vector here
    kf = KFold(n_splits=Nfold)
    res = []
    best_accuracy = 0

    for train_index, test_index in kf.split(X):
        tem_accuracy, tem_estimation, tem_classifier = accuracy_unsupervised_Bernoulli(X[train_index], Y[train_index], X[test_index], Y[test_index], alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip)
        res.append(tem_accuracy)
        
        if tem_accuracy > best_accuracy:
            best_accuracy = tem_accuracy
            best_estimation = tem_estimation
            best_classifier = tem_classifier
    return res, best_estimation, best_classifier