import numpy as np
import sys
# from data_generator_Normal import document_generator 
# import supervised_Normal
# import unsupervised_Normal


k0 = k1 = int(sys.argv[1])
Ntrain, Ntest = 47, 10
ntrace = 1000
nchain = 2
nskip = 2
nlabel = 3
d = 240
nlabel=4

dg = k0 + k1 
K = ntopic = nlabel*k0+k1

alpha = np.ones(dg)
b = 0.1
'''
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
'''
data_file = "../data/cleaned_data.csv"
f = open(data_file,"r")
x = f.readlines()
f.close()

X = []
Y = []
for i in range(1, len(x)):
    tem = x[i].strip().split(",")
    X.append([float(j) for j in tem[2:-1]])
    Y.append(int(tem[-1]))
X = np.array(X)
Y = np.array(Y)

alpha_Lambda =  np.array([1]*180 + [3]*18 + [3]*42)
beta_Lambda = np.array([2]*180 + [3]*18 + [0.3]*42)
mu_Mu = np.array([1]*180 + [1]*18 + [-0.5]*42)
sigma2_Mu = np.array([0.01]*180 + [2]*18 + [0.000001]*42)

train_index = random.choice(0,57,Ntrain)
test_index = []
for i in list(range(57)): 
    if i not in set(train_index):
        test_index.append(i)


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

    N = len(Y)
    dg = len(alpha)
    d = len(X[0])
    T_arr = np.array(T)

    model = pm.Model()
    with model:
        Lambda_ = pm.Gamma("Lambda", alpha = alpha_Lambda, beta = beta_Lambda, shape = (K, d))
        Mu_ = pm.Normal("Mu", mu = mu_Mu, sigma = pm.math.sqrt(sigma2_Mu/Lambda_), shape = (K, d))
        S_ = 1/Lambda_
        Tau_ = Mu_/S_       

        a_ = pm.Exponential("a", b)  
        rho_ = pm.Dirichlet("rho", a = alpha)
        G_ = pm.Dirichlet("G", a = a_*rho_, shape = (N, dg))        
        T_ = theano.shared(T_arr)
    
        for i in range(N):
            u_ = pm.math.dot(G_[i:(i+1)],T_[Y[i]].T)
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
    
    for para in ["a", 'rho', 'Mu', "Lambda", "G"]:
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

def accuracy_unsupervised(trainingX, trainingY, testX, testY, K, b, ntrace, nchain, nskip):
    estimate_un = train_unsupervised(trainingX, K, b, ntrace, nchain, nskip)
    classifier = svm.SVC(C=2,kernel='rbf', decision_function_shape='ovo') 
    classifier.fit(estimate_un["U"], trainingY)

    prediction_un = predict_unsupervised(testX, estimate_un, K, ntrace, nchain, nskip)
    accuracy = classifier.score(prediction_un["U"],testY)
    return accuracy, estimate_un, classifier
    
# print("fine")
estimate = train_supervised(X[train_index], Y[train_index], K, b, alpha, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip)
print("training_accuracy:", sum(estimate["Y"]==Y[train_index])/Ntest)



