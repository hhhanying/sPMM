import numpy as np
import sys
import pymc3 as pm
import theano
from collections import Counter
from sklearn.model_selection import KFold
import json

'''
Changes:
After training and testing, this file will again use the learned topics to estimate membership under unsupervised model.

Input:
1. data_file: file name of dataset (with header)
2. confi_file: file name of configurations (with header)
3. res_file: The result will be stored in res_file + str(index) + ".txt"
4. $id: we will use the ($id+1)-th configurations

Configuration file need to contain:
k0: number of class-specific topics
k1: number of shared topics
which_index: we perform 5-fold CV here, and this parameter detemines the way to partition the dataset (0-4)

Results:
A dictionary:
res = {"estimation":{}, "prediction":{}, "test":{}}
estimation = {a, rho, G, Mu1, Mu2, Mu3, Lambda1, Lambda2, Lambda3}
prediction = {G, Y}
test = {U}
'''

# get input
data_file = sys.argv[1]
confi_file = sys.argv[2]
res_file = sys.argv[3]
index = int(sys.argv[4]) + 1

# read configurations
with open(confi_file,"r") as f:
    x = f.readlines()
confi = x[index].strip().split(",")
k0, k1, which_index = [int(i) for i in confi]

# hyperparameters
alpha_Lambda1, beta_Lambda1, mu_Mu1, sigma2_Mu1 =   3.2438328, 11.1157220, -1, 0.9138574
alpha_Lambda2, beta_Lambda2, mu_Mu2, sigma2_Mu2 =  3.416767, 11.05094, 0, 0.6660274
alpha_Lambda3, beta_Lambda3, mu_Mu3, sigma2_Mu3 =  0.955197 , 0.4841954, 0, 0.043003

# read data
with open(data_file,"r") as f:
    x = f.readlines()
X = []
Y = []
for i in range(1, len(x)):
    tem = x[i].strip().split(",")
    X.append([float(j) for j in tem[1:-1]])
    Y.append(int(tem[-1]))
X = np.array(X)
Y = np.array(Y)

# Train-Test split
All_data = list(range(23)) + list(range(32,52)) # the index of sepsis and healthy&no antibiotics
if which_index == 5:
    train_index = test_index = All_data
else:
    indexes = [[1, 15, 13, 12,  8, 25, 28, 50, 41, 35, 34, 54],
    [16,  3, 18, 19, 21, 30, 24 ,51, 47, 46, 33, 55],
    [17,  0, 11 , 6 ,29 ,23 ,44 ,39 ,43 ,40, 53],
    [10,  4,  9 , 2 ,26 ,27 ,38 ,36, 32, 42 ,52],
    [5 ,22 ,20, 14 , 7 ,31 ,45, 37, 48, 49, 56]]
    test_index = []
    train_index = []
    for i in All_data: 
        if i not in set(indexes[which_index]):
            train_index.append(i)
        else:
            test_index.append(i)
        
training_X1, training_X2, training_X3 = X[train_index, 0:180], X[train_index, 180:198], X[train_index, 198:240]
training_Y = Y[train_index]
test_X1, test_X2, test_X3 = X[test_index, 0:180], X[test_index, 180:198], X[test_index, 198:240]
test_Y = Y[test_index]

# set parameters
ntrace = 2000
nchain = 2
nskip = 2
nlabel = 2
d = 240
d1, d2, d3 = 180, 18, 42
dg = k0 + k1 
K = ntopic = nlabel*k0+k1
Ntrain, Ntest = len(train_index), len(test_index)

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
T_arr = np.array(T)

# supervised training
model_training = pm.Model()
with model_training:
    a_ = pm.Exponential("a", b)  
    rho_ = pm.Dirichlet("rho", a = alpha)
    G_ = pm.Dirichlet("G", a = a_*rho_, shape = (Ntrain, dg))        
    T_ = theano.shared(T_arr)  

    Lambda1_ = pm.Gamma("Lambda1", alpha = alpha_Lambda1, beta = beta_Lambda1, shape = (K, d1))
    Mu1_ = pm.Normal("Mu1", mu = mu_Mu1, sigma = pm.math.sqrt(sigma2_Mu1/Lambda1_), shape = (K, d1))
    S1_ = 1/Lambda1_
    Tau1_ = Mu1_/S1_       

    Lambda2_ = pm.Gamma("Lambda2", alpha = alpha_Lambda2, beta = beta_Lambda2, shape = (K, d2))
    Mu2_ = pm.Normal("Mu2", mu = mu_Mu2, sigma = pm.math.sqrt(sigma2_Mu2/Lambda2_), shape = (K, d2))
    S2_ = 1/Lambda2_
    Tau2_ = Mu2_/S2_   

    Lambda3_ = pm.Gamma("Lambda3", alpha = alpha_Lambda3, beta = beta_Lambda3, shape = (K, d3))
    Mu3_ = pm.Normal("Mu3", mu = mu_Mu3, sigma = pm.math.sqrt(sigma2_Mu3), shape = (K, d3))
    S3_ = 1/Lambda3_
    Tau3_ = Mu3_/S3_  

    for i in range(Ntrain):
        u_ = pm.math.dot(G_[i:(i+1)],T_[training_Y[i]].T)

        Taux1_ = pm.math.dot(u_, Tau1_)
        Lambdax1_ = pm.math.dot(u_, Lambda1_)
        Sx1_ = 1/Lambdax1_
        Mux1_ = Sx1_*Taux1_
        X1_ = pm.Normal("x_1_"+str(i), mu=Mux1_, sigma=pm.math.sqrt(Sx1_), observed=training_X1[i])

        Taux2_ = pm.math.dot(u_, Tau2_)
        Lambdax2_ = pm.math.dot(u_, Lambda2_)
        Sx2_ = 1/Lambdax2_
        Mux2_ = Sx2_*Taux2_
        X2_ = pm.Normal("x_2_"+str(i), mu=Mux2_, sigma=pm.math.sqrt(Sx2_), observed=training_X2[i])

        Taux3_ = pm.math.dot(u_, Tau3_)
        Lambdax3_ = pm.math.dot(u_, Lambda3_)
        Sx3_ = 1/Lambdax3_
        Mux3_ = Sx3_*Taux3_
        X3_ = pm.Normal("x_3_"+str(i), mu=Mux3_, sigma=pm.math.sqrt(Sx3_), observed=training_X3[i])

    trace = pm.sample(ntrace, chains = nchain)

nsample = ntrace*nchain
nsave = nsample//nskip
index_save = [i*nskip-1 for i in range(1,1 + nsave)]
estimate = {}
    
for para in ["a", 'rho', "G"]+[para+str(i) for para in ["Mu", "Lambda"] for i in range(1,4)]:
    estimate[para] = trace[para][index_save].mean(axis=0)


# prediction
a, rho, Mu1_, Lambda1_, Mu2_, Lambda2_, Mu3_, Lambda3_ = estimate["a"], estimate["rho"], estimate["Mu1"], estimate["Lambda1"], estimate["Mu2"], estimate["Lambda2"], estimate["Mu3"], estimate["Lambda3"]
Tau1_, Tau2_, Tau3_ = Mu1_*Lambda1_, Mu2_*Lambda2_, Mu3_*Lambda3_

model_testing = pm.Model()
with model_testing:
    G_ = pm.Dirichlet("G", a=a*rho, shape = (Ntest, dg))
    Y_ = pm.Categorical("Y", p=np.ones(nlabel)/nlabel, shape=Ntest)
    T_ = theano.shared(T_arr)

    for i in range(Ntest):
        u_ = pm.math.dot(G_[i:(i+1)],T_[Y_[i]].T)

        Taux1_ = pm.math.dot(u_, Tau1_)
        Lambdax1_ = pm.math.dot(u_, Lambda1_)
        Sx1_ = 1/Lambdax1_
        Mux1_ = Sx1_*Taux1_
        X1_ = pm.Normal("x_1_"+str(i), mu=Mux1_, sigma=pm.math.sqrt(Sx1_), observed=test_X1[i])

        Taux2_ = pm.math.dot(u_, Tau2_)
        Lambdax2_ = pm.math.dot(u_, Lambda2_)
        Sx2_ = 1/Lambdax2_
        Mux2_ = Sx2_*Taux2_
        X2_ = pm.Normal("x_2_"+str(i), mu=Mux2_, sigma=pm.math.sqrt(Sx2_), observed=test_X2[i])

        Taux3_ = pm.math.dot(u_, Tau3_)
        Lambdax3_ = pm.math.dot(u_, Lambda3_)
        Sx3_ = 1/Lambdax3_
        Mux3_ = Sx3_*Taux3_
        X3_ = pm.Normal("x_3_"+str(i), mu=Mux3_, sigma=pm.math.sqrt(Sx3_), observed=test_X3[i])

    trace = pm.sample(ntrace, chains = nchain)

prediction = {}

prediction["G"] = trace["G"][index_save].mean(axis=0)

pre_Y = []
for i in range(Ntest):
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

accuracy = sum(test_Y == prediction["Y"])/Ntest
print("accuracy:", accuracy)

# test under unsupervised model
rho = np.array([1/2/nlabel/k0] * nlabel*k0 + [1/2/k1] * k1) # redefine rho

model_testing_unsupervised = pm.Model()
with model_testing_unsupervised:
    U_ = pm.Dirichlet("U", a=a*rho, shape = (Ntest, ntopic))

    for i in range(Ntest):
        u_ = U_[i:(i+1)]

        Taux1_ = pm.math.dot(u_, Tau1_)
        Lambdax1_ = pm.math.dot(u_, Lambda1_)
        Sx1_ = 1/Lambdax1_
        Mux1_ = Sx1_*Taux1_
        X1_ = pm.Normal("x_1_"+str(i), mu=Mux1_, sigma=pm.math.sqrt(Sx1_), observed=test_X1[i])

        Taux2_ = pm.math.dot(u_, Tau2_)
        Lambdax2_ = pm.math.dot(u_, Lambda2_)
        Sx2_ = 1/Lambdax2_
        Mux2_ = Sx2_*Taux2_
        X2_ = pm.Normal("x_2_"+str(i), mu=Mux2_, sigma=pm.math.sqrt(Sx2_), observed=test_X2[i])

        Taux3_ = pm.math.dot(u_, Tau3_)
        Lambdax3_ = pm.math.dot(u_, Lambda3_)
        Sx3_ = 1/Lambdax3_
        Mux3_ = Sx3_*Taux3_
        X3_ = pm.Normal("x_3_"+str(i), mu=Mux3_, sigma=pm.math.sqrt(Sx3_), observed=test_X3[i])

    trace = pm.sample(ntrace, chains = nchain)

test = {}
test["U"] = trace["U"][index_save].mean(axis=0)

# save the result
for i in estimate.keys():
    if type(estimate[i]) is type(np.array([1])):
        estimate[i] = estimate[i].tolist()
for i in prediction.keys():
    prediction[i] = prediction[i].tolist()
for i in test.keys():
    test[i] = test[i].tolist()
res = {"estimation": estimate, "prediction": prediction, "test": test}

filename = res_file + str(index) + ".txt"
f = open(filename,"w")
f.write(json.dumps(res))
f.close()