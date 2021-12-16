import numpy as np
import pymc3 as pm
import theano
from sklearn import svm
from sklearn.model_selection import KFold

def train_unsupervised(X, Y, K, b, alpha, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip):
    '''
    X, Y: training set
    Y is only used for training SVM
    K: total topic numbers
    b, alpha: hyperparameters for the prior of a and rho
    (here we use all-one-vector as alpha)
    mu_Mu, sigma2_Mu: the prior of Mu is N(mu_Mu, sigma2_Mu/Lambda)
    alpha_Lambda, beta_Lambda: the prior of Lambda is Gamma(alpha_Lambda, beta_Lambda)
    ntrace, nchain, nskip: parameters of sampling
    ''' 
    
    N = len(X)
    d = len(X[0])

    model = pm.Model()
    with model:
        # topics
        Lambda_ = pm.Gamma("Lambda", alpha = alpha_Lambda, beta = beta_Lambda, shape = (K, d))
        Mu_ = pm.Normal("Mu", mu = mu_Mu, sigma = pm.math.sqrt(sigma2_Mu/Lambda_), shape = (K, d))
        S_ = 1/Lambda_
        Tau_ = Mu_/S_       
        # corpus-level parameters
        a_ = pm.Exponential("a", b)  
        rho_ = pm.Dirichlet("rho", a = alpha)
        # membership
        U_ = pm.Dirichlet("U", a=a_*rho_, shape = (N, K))
    
        for i in range(N):
            u_ = U_[i:(i+1)]
            Taux_ = pm.math.dot(u_, Tau_)
            Lambdax_ = pm.math.dot(u_, Lambda_)
            Sx_ = 1/Lambdax_
            Mux_ = Sx_*Taux_
            X_ = pm.Normal("x"+str(i), mu=Mux_, sigma=pm.math.sqrt(Sx_), observed=X[i])
            
        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]
    estimate = {}         
    
    for para in ["a", 'rho', 'Mu', "Lambda", "U"]:
        estimate[para] = trace[para][index_save].mean(axis=0)
        
    classifier = svm.SVC(C=2,kernel='rbf', decision_function_shape='ovo') 
    classifier.fit(estimate["U"], Y)
        
    return estimate, classifier   

def predict_unsupervised(X, estimate, classifier, ntrace, nchain, nskip): 
    '''
    X: test set
    estimate: parameters estimated from training set, a dictionary containing a, rho and P 
    classifier: the SVM trained from training set
    ntrace, nchain, nskip: parameters of sampling
    output: estimated label Y and membership U
    '''
    
    N = len(X)
    a, rho, Mu, Lambda = estimate["a"], estimate["rho"], estimate["Mu"], estimate["Lambda"]
    Tau = Mu*Lambda
    K = len(Tau)

    model = pm.Model()
    with model:
        U_ = pm.Dirichlet("U", a=a*rho, shape = (N, K))
        for i in range(N):
            u_ = U_[i:(i+1)] 
            Taux_ = pm.math.dot(u_, Tau)
            Lambdax_ = pm.math.dot(u_, Lambda)
            Sx_ = 1/Lambdax_
            Mux_ = Sx_*Taux_
            
            X_ = pm.Normal("x"+str(i), mu=Mux_, sigma=pm.math.sqrt(Sx_), observed=X[i])    
            
        trace = pm.sample(ntrace, chains = nchain) 
        
    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]    
    prediction = {}
    
    for i in ["U"]:
        prediction[i] = trace[i][index_save].mean(axis=0) 
    prediction["Y"] = classifier.predict(prediction["U"])
     
    return prediction

def accuracy_unsupervised_Normal(trainingX, trainingY, testX, testY, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set, test set and needed parameters
    # this function will return the test accuracy
    estimation, classifier = train_unsupervised(trainingX, trainingY, K, b, alpha, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip)
    prediction = predict_unsupervised(testX, estimation, classifier, ntrace, nchain, nskip)
    Ntest = len(testY)
    accuracy = sum(testY == prediction["Y"])/Ntest
    print(accuracy)
    return accuracy      

def CV_unsupervised_Normal(X, Y, Nfold, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip):
    # Given the training set
    # This function will perform K-fold cross validation and return the list of the accuracies
    # Notica the alpha here is not the same as the alpha used before
    # Its dimension become K
    # It's defined by the user. We choose all-1-vector here
    kf = KFold(n_splits = Nfold)
    res = []
    for train_index, test_index in kf.split(X):
        tem_accuracy = accuracy_unsupervised_Normal(X[train_index], Y[train_index], X[test_index], Y[test_index], mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip):        
        res.append(tem_accuracy)
    return res     