# test correctness
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

def document_generator(a, rho, T, Lambda, Tau, N):
    '''
    a, rho: corpus-level parameters
    T: ntopic * K * dg
    Lambda, Tau: topics. K*d matrix
    N: the number of documents.
    
    output: 
    X: N*d, X[i] = document[i]
    Y: Y[i] = label[i]
    G: membership
    U: transformed membership
    '''

    nlabel = len(T) # number of y
    d = len(Tau[0]) # dim(x)
    
    Y = np.random.choice(list(range(nlabel)),N) # labels
    G = np.random.dirichlet(a*rho,N)
    U = np.array([np.dot(T[Y[i]], G[i]) for i in range(N)])

    X = np.zeros((N, d))
    for i in range(N):
        u = U[i]
        for j in range(d):
            lambdax = (u*Lambda[:,j]).sum()
            taux = (u*Tau[:,j]).sum()
            sx = 1/lambdax
            mux = sx*taux
            X[i][j] = np.random.normal(mux, np.sqrt(sx))

    return X, Y, G, U

def train_supervised(X, Y, K, b, alpha, T, ntrace, nchain, nskip):
    '''
    data: training set
    K, d, dg
    #T?
    prior: b, alpha
    
    not include: prior for Lambda, prior for Tau
    '''
    
    # X, Y, G, U = data["X"], data["Y"], data["G"], data["U"]
    N = len(Y)
    dg = len(alpha)
    d = len(X[0])

    model = pm.Model()
    with model:
        Lambda_ = pm.Gamma("Lambda",alpha=1,beta=2,shape=(K,d))
        Tau_ = pm.Normal("Tau", mu=0, sigma=1, shape = (K,d))
        a_ = pm.Exponential("a", b)
        rho_ = pm.Dirichlet("rho", a=alpha)
        G_ = pm.Dirichlet("G", a=a_*rho_, shape = (N, dg))
        T_= np.array([T[Y[i]] for i in range(N)])

        for i in range(N):
            u_ = pm.math.dot(G_[i:(i+1)],T_[i].T)

            Taux_ = pm.math.dot(u_,Tau_)
            Lambdax_ = pm.math.dot(u_,Lambda_)
            Sx_ = 1/Lambdax_
            Mux_ = Sx_*Taux_

            X_ = pm.Normal("x"+str(i),mu=Mux_,sigma=pm.math.sqrt(Sx_),observed=X[i])

        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]
    estimate = {}
    for para in ["a", 'rho', 'Tau', "Lambda", "G"]:
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
    a, rho, Tau, Lambda = estimate["a"], estimate["rho"], estimate["Tau"], estimate["Lambda"]
    T_arr = np.array(T)
    
    model = pm.Model()
    with model:
        G_ = pm.Dirichlet("G", a=a*rho, shape = (N, dg))
        Y_ = pm.Categorical("Y", p=np.ones(nlabel)/nlabel, shape=N)
        T_ = theano.shared(T_arr)
        for i in range(N):
            t_ = T_[Y_[i]]
            u_ = pm.math.dot(G_[i:(i+1)],t_.T)

            Taux_ = pm.math.dot(u_,Tau)
            Lambdax_ = pm.math.dot(u_,Lambda)
            Sx_ = 1/Lambdax_
            Mux_ = Sx_*Taux_
            X_ = pm.Normal("x"+str(i),mu=Mux_,sigma=pm.math.sqrt(Sx_),observed=X[i])
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


def train_unsupervised(X, K, b, ntrace, nchain, nskip):
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
        Lambda_ = pm.Gamma("Lambda",alpha=1,beta=2,shape=(K,d))
        Tau_ = pm.Normal("Tau", mu=0, sigma=1, shape = (K,d))
        a_ = pm.Exponential("a", b)
        rho_ = pm.Dirichlet("rho", a = np.ones(K))  # difference

        U_ = pm.Dirichlet("U", a=a_*rho_, shape = (N, K)) # difference
           
        for i in range(N):
            # u_ = pm.math.dot(G_[i:(i+1)],T_[i].T)
            u_ = U_[i:(i+1)]  # attention
            Taux_ = pm.math.dot(u_,Tau_)
            Lambdax_ = pm.math.dot(u_,Lambda_)
            Sx_ = 1/Lambdax_
            Mux_ = Sx_*Taux_

            X_ = pm.Normal("x"+str(i),mu=Mux_,sigma=pm.math.sqrt(Sx_),observed=X[i])

        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]  
    estimate = {}
    for para in ["a", 'rho', 'Tau', "Lambda", "U"]:
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
    a, rho, Tau, Lambda = estimate["a"], estimate["rho"], estimate["Tau"], estimate["Lambda"]

    model = pm.Model()
    with model:
        U_ = pm.Dirichlet("U", a=a*rho, shape = (N, K)) # difference
           
        for i in range(N):
            # u_ = pm.math.dot(G_[i:(i+1)],T_[i].T)
            u_ = U_[i:(i+1)]  # attention
            Taux_ = pm.math.dot(u_,Tau)
            Lambdax_ = pm.math.dot(u_,Lambda)
            Sx_ = 1/Lambdax_
            Mux_ = Sx_*Taux_

            X_ = pm.Normal("x"+str(i),mu=Mux_,sigma=pm.math.sqrt(Sx_),observed=X[i])

        trace = pm.sample(ntrace, chains = nchain)

    nsample = ntrace*nchain
    nsave = nsample//nskip
    index_save = [i*nskip-1 for i in range(1,1 + nsave)]    
    prediction = {}  
    for i in ["U"]:
        prediction[i] = trace[i][index_save].mean(axis=0)        
    return prediction

def accuracy_supervised(trainingX, trainingY, testX, testY, T, K, dg, b, alpha, ntrace, nchain, nskip):
    estimate = train_supervised(trainingX, trainingY, K, b, alpha, T, ntrace, nchain, nskip)
    prediction = predict_supervised(testX, estimate, dg, T, ntrace, nchain, nskip)
    Ntest = len(testY)
    accuracy = sum(testY == prediction["Y"])/Ntest
    return accuracy, estimate

def train_cv_supervised(data, Nfold, T, K, dg, b, alpha, ntrace, nchain, nskip):
    X, Y = data["X"], data["Y"]
    best_accuracy = 0
    kf = KFold(n_splits=Nfold)

    for train_index, test_index in kf.split(X):
        tem_accuracy, tem_estimate = accuracy_supervised(X[train_index], Y[train_index], X[test_index], Y[test_index], T, K, dg, b, alpha, ntrace, nchain, nskip)
        if tem_accuracy > best_accuracy:
            best_accuracy, estimate = tem_accuracy, tem_estimate

    return estimate

def accuracy_unsupervised(trainingX, trainingY, testX, testY, K, b, ntrace, nchain, nskip):
    estimate_un = train_unsupervised(trainingX, K, b, ntrace, nchain, nskip)
    classifier = svm.SVC(C=2,kernel='rbf', decision_function_shape='ovo') 
    classifier.fit(estimate_un["U"], trainingY)

    prediction_un = predict_unsupervised(testX, estimate_un, K, ntrace, nchain, nskip)
    accuracy = classifier.score(prediction_un["U"],testY)
    return accuracy, estimate_un, classifier

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
d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest, Nfold, n1, n2 = [int(i) for i in para]
# the variance of n1 shared topics will time n2

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
Lambda = np.random.gamma(1,2,K*d).reshape((K,d))
Tau = np.random.normal(0,10,K*d).reshape((K,d))
S = 1/Lambda
R = np.log(Lambda)
Mu = S*Tau

print("the value of a:", a)
print("the value of rho:", rho)

Lambda[-n1:] *= n2

# generate parameter
X, Y, G, U = document_generator(a, rho, T, Lambda, Tau, Ntrain)
training_set = {'X': X, "Y": Y, "G": G, "U":U}
X, Y, G, U = document_generator(a, rho, T, Lambda, Tau, Ntest)
test_set = {'X': X, "Y": Y, "G": G, "U":U}



# supervised part
estimate_supervised = train_cv_supervised(training_set, Nfold, T, K, dg, b, alpha, ntrace, nchain, nskip)
prediction_test_supervised = predict_supervised(test_set["X"], estimate_supervised, dg, T, ntrace, nchain, nskip)
accuracy_su = sum(test_set["Y"] == prediction_test_supervised["Y"])/Ntest

# unsupervised part
estimate_unsupervised, classifier_unsupervised = train_cv_unsupervised(training_set, Nfold, K, b, ntrace, nchain, nskip)
prediction_test_unsupervised = predict_unsupervised(test_set["X"], estimate_unsupervised, K, ntrace, nchain, nskip)
accuracy_un = classifier_unsupervised.score(prediction_test_unsupervised["U"],test_set["Y"])


print(nlabel,accuracy_su, accuracy_un)
# filename = "result"+str(nlabel)+".txt"
filename = "output{}.txt".format(str(index))
f = open(filename,"w")
f.write(str(accuracy_su)+'\n')
f.write(str(accuracy_un)+'\n')
f.close()
