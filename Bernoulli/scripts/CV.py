import numpy as np
import json 
import pymc3 as pm
import arviz as az
from sklearn import svm
from sklearn.model_selection import KFold
import theano
from collections import Counter
import sys

confi_file = "configurations.csv"

def document_generator(a, rho, T, P, N):
    '''
    a, rho: corpus-level parameters
    T: ntopic * K * dg
    P: topics. K*d matrix
    N: the number of documents.
    
    output: 
    X: N*d, X[i] = document[i]
    Y: Y[i] = label[i]
    G: membership
    U: transformed membership
    '''

    nlabel = len(T) # number of y
    d = len(P[0]) # dim(x)
    
    Y = np.random.choice(list(range(nlabel)),N) # labels
    G = np.random.dirichlet(a*rho,N)
    U = np.array([np.dot(T[Y[i]], G[i]) for i in range(N)])

    logitP = np.log(P/(1-P))
    logitX = np.dot(U,logitP)
    PX = 1/(1+np.exp(-logitX))

    X = np.random.binomial(1,PX,(N,d))
    
    return X, Y, G, U

def train_supervised(X, Y, K, b, alpha, T, alpha_p, beta_p, ntrace, nchain, nskip):
    '''
    data: training set
    K, d, dg
    #T?
    prior: b, alpha
    
    not include: prior for Lambda, prior for Tau
    '''
    N = len(Y)
    dg = len(alpha)
    d = len(X[0])

    model = pm.Model()
    with model:
        P_ = pm.Beta("P", alpha = alpha_p, beta = beta_p, shape = (K,d))
        a_ = pm.Exponential("a", b)
        rho_ = pm.Dirichlet("rho", a=alpha)
        G_ = pm.Dirichlet("G", a=a_*rho_, shape = (N, dg))
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

def predict_supervised(X, estimate, dg, T, ntrace, nchain, nskip):
    '''
    test set
    estimate parameters
    output: label Y and membership G
    #T?
    '''
    N = len(X)
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

def accuracy_supervised(trainingX, trainingY, testX, testY, T, alpha_p, beta_p, K, dg, b, alpha, ntrace, nchain, nskip):
    estimate = train_supervised(trainingX, trainingY, K, b, alpha, T, alpha_p, beta_p, ntrace, nchain, nskip)
    prediction = predict_supervised(testX, estimate, dg, T, ntrace, nchain, nskip)
    Ntest = len(testY)
    accuracy = sum(testY == prediction["Y"])/Ntest
    return accuracy, estimate

def train_unsupervised(X, K, b, alpha_p, beta_p, ntrace, nchain, nskip):
    '''
    X: training set
    K, d, dg
    
    prior: b, alpha
    
    not include: prior for Lambda, prior for Tau
    '''

    N = len(X)
    d = len(X[0])
    
    model = pm.Model()
    with model:
        P_ = pm.Beta("P", alpha = alpha_p, beta = beta_p, shape = (K,d))
        a_ = pm.Exponential("a", b)
        rho_ = pm.Dirichlet("rho", a = np.ones(K))  # difference

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
    return estimate

def predict_unsupervised(X, estimate, K, ntrace, nchain, nskip):
    '''
    test set
    estimate parameters
    output: label Y and membership G
    #T?
    '''
    N = len(X)
    a, rho, P = estimate["a"], estimate["rho"], estimate["P"]

    model = pm.Model()
    with model:
        U_ = pm.Dirichlet("U", a=a*rho, shape = (N, K)) # difference
        logit_P = pm.math.logit(P)
           
        for i in range(N):
            # u_ = pm.math.dot(G_[i:(i+1)],T_[i].T)
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
    return prediction

def accuracy_unsupervised(trainingX, trainingY, testX, testY, alpha_p, beta_p, K, b, ntrace, nchain, nskip):
    estimate_un = train_unsupervised(trainingX, K, b, alpha_p, beta_p, ntrace, nchain, nskip)
    classifier = svm.SVC(C=2,kernel='rbf', decision_function_shape='ovo') 
    classifier.fit(estimate_un["U"], trainingY)
    prediction_un = predict_unsupervised(testX, estimate_un, K, ntrace, nchain, nskip)
    accuracy = classifier.score(prediction_un["U"],testY)
    return accuracy, estimate_un, classifier

def train_cv_supervised(data, Nfold, T, K, dg, b, alpha, ntrace, nchain, nskip):
    X, Y = data["X"], data["Y"]
    best_accuracy = 0
    kf = KFold(n_splits=Nfold)

    for train_index, test_index in kf.split(X):
        tem_accuracy, tem_estimate = accuracy_supervised(X[train_index], Y[train_index], X[test_index], Y[test_index], T, K, dg, b, alpha, ntrace, nchain, nskip)
        if tem_accuracy > best_accuracy:
            best_accuracy, estimate = tem_accuracy, tem_estimate

    return estimate

def train_cv_unsupervised(data, Nfold, K, b, ntrace, nchain, nskip):
    X, Y = data["X"], data["Y"]
    best_accuracy = 0
    kf = KFold(n_splits=Nfold)   
    for train_index, test_index in kf.split(X):
        tem_accuracy, tem_estimate_un, tem_classifier = accuracy_unsupervised(X[train_index], Y[train_index], X[test_index], Y[test_index], K, b, ntrace, nchain, nskip)
        if tem_accuracy > best_accuracy:
            best_accuracy, estimate, classifier = tem_accuracy, tem_estimate_un, tem_classifier
    return estimate, classifier

f = open(confi_file,"r")
x = f.readlines()
f.close()
index = int(sys.argv[1]) + 1 # 1st line is header
para = x[index].strip().split(",")
d, k0, k1, nlabel, alpha_p, beta_p, ntrace, nchain, nskip, Ntrain, Ntest, Nfold = [int(i) for i in para]

# d = 20
# k0=2
# k1=5
# alpha_p, beta_p = 1, 1
# nlabel = 3
# ntrace = 500
# nchain = 2
# nskip = 2
# Ntrain = 100
# Ntest = 100


dg = k0 + k1 
K = ntopic = nlabel*k0+k1

alpha = np.ones(dg)
b = 0.1
# define T
T = []
for i in range(nlabel):
    tem = np.block([
        [np.zeros((k0*i,k0+k1))],
        [np.eye(k0), np.zeros((k0, k1))],
        [np.zeros((k0*(nlabel-i-1),k0+k1))],
        [np.zeros((k1,k0)), np.eye(k1)]
    ])
    T.append(tem)
# draw corpus-level parameters
rho = np.random.dirichlet(alpha, 1)[0]
a = np.random.exponential(1/b,1)[0]
P = np.random.beta(alpha_p, beta_p,(K,d))

print("the value of a:", a)
print("the value of rho:", rho)


# generate parameter
X, Y, G, U = document_generator(a, rho, T, P, Ntrain)
training_set = {'X': X, "Y": Y, "G": G, "U":U}
X, Y, G, U = document_generator(a, rho, T, P, Ntest)
test_set = {'X': X, "Y": Y, "G": G, "U":U}


# supervised part
accuracy_su, estimate_supervised = accuracy_supervised(training_set["X"], training_set["Y"], test_set["X"], test_set["Y"], T, alpha_p, beta_p, K, dg, b, alpha, ntrace, nchain, nskip)

# unsupervised part
accuracy_un, estimate_un, classifier  = accuracy_unsupervised(training_set["X"], training_set["Y"], test_set["X"], test_set["Y"], alpha_p, beta_p, K, b, ntrace, nchain, nskip)

print(nlabel,accuracy_su, accuracy_un)
filename = "result"+str(nlabel)+".txt"
filename = "output{}.txt".format(str(index))
f = open(filename,"w")
f.write(str(accuracy_su)+'\n')
f.write(str(accuracy_un)+'\n')
f.close()


    
