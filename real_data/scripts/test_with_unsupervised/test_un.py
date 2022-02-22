import numpy as np
import sys
import pymc3 as pm
import theano
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import KFold
import json

'''
We have learned the topics and other parameters in a supervised model.
This file is used to estimated the membership under an unsupervised model with these parameters.
Overall, we want to see whether the topic are class specific.

Require input:
1. data_file: file name of dataset (with header)
2. confi_file: file name of configurations (with header)
3. res_file: The result will be stored in res_file + str(index) + ".txt"
4. index = $id: we will use the ($id+1)-th configurations
5. para_file: where we saved the res file
'''

# get input
data_file = sys.argv[1] # name of data
confi_file = sys.argv[2] # confi name
res_file = sys.argv[3] # where the result is saved
index = int(sys.argv[4]) + 1 # no_confi we use
para_file = sys.argv[5] # where the learned topics are saved

# read configurations
f = open(confi_file,"r")
x = f.readlines()
f.close()
confi = x[index].strip().split(",")
k0, k1, which_index = [int(i) for i in confi]

# read data
f = open(data_file,"r")
x = f.readlines()
f.close()
X = []
Y = []
for i in range(1, len(x)):
    tem = x[i].strip().split(",")
    X.append([float(j) for j in tem[1:-1]])
    Y.append(int(tem[-1]))
X = np.array(X)
Y = np.array(Y)

# Train-Test split
indexes = [[1, 15, 13, 12,  8, 25, 28, 50, 41, 35, 34, 54],
[16,  3, 18, 19, 21, 30, 24 ,51, 47, 46, 33, 55],
[17,  0, 11 , 6 ,29 ,23 ,44 ,39 ,43 ,40, 53],
[10,  4,  9 , 2 ,26 ,27 ,38 ,36, 32, 42 ,52],
[5 ,22 ,20, 14 , 7 ,31 ,45, 37, 48, 49, 56]]
test_index = indexes[which_index]
# train_index = []
# for i in list(range(57)): 
#     if i not in set(test_index):
#         train_index.append(i)
# training_X1, training_X2, training_X3 = X[train_index, 0:180], X[train_index, 180:198], X[train_index, 198:240]
# training_Y = Y[train_index]
test_X1, test_X2, test_X3 = X[test_index, 0:180], X[test_index, 180:198], X[test_index, 198:240]
test_Y = Y[test_index]

# assign values to parameters
ntrace = 2000
nchain = 4
nskip = 2
nlabel = 4
d = 240
d1, d2, d3 = 180, 18, 42
K = ntopic = nlabel*k0+k1
Ntest = len(test_index)

# read the parameters: a mu lambda
# rho need to be changed
with open(para_file, "r") as f:
    x = f.read()
estimate = json.loads(x)["supervised"][str(k0)][str(which_index)]["estimation"]

rho = np.array([1/2/nlabel/k0] * nlabel*k0 + [1/2/k1] * k1) # redefine rho

a = estimate["a"]
Mu1_, Lambda1_, Mu2_, Lambda2_, Mu3_, Lambda3_ = [ np.array(i) for i in [estimate["Mu1"], estimate["Lambda1"], estimate["Mu2"], estimate["Lambda2"], estimate["Mu3"], estimate["Lambda3"]]]
Tau1_, Tau2_, Tau3_ = Mu1_*Lambda1_, Mu2_*Lambda2_, Mu3_*Lambda3_

# estimate the memberhsips 
model_testing = pm.Model()
with model_testing:
    U_ = pm.Dirichlet("U", a=a*rho, shape = (Ntest, ntopic))
    Y_ = pm.Categorical("Y", p=np.ones(nlabel)/nlabel, shape=Ntest)

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


# save the result
prediction = {}

nsample = ntrace*nchain
nsave = nsample//nskip
index_save = [i*nskip-1 for i in range(1,1 + nsave)]

prediction["U"] = trace["U"][index_save].mean(axis=0).tolist()

filename = res_file + str(index) + ".txt"
with open(filename,"w") as f:
    f.write(json.dumps(prediction))