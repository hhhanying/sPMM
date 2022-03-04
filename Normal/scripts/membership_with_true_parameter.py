'''
This file wll try to learn the membership given the true paratemeters of the model
input:
1. index of configuration: 0 to 80
2. name of configuration file
3. name to save result
4. name of data file, sys.argv[4].format(str(set_index))
5. method: supervised, unsupervised
6. dataset: training_set, test_set
'''
import numpy as np
import sys
import json
from supervised_Normal import predict_supervised
from unsupervised_Normal import predict_unsupervised


confi_index = int(sys.argv[1]) + 1 # 1st line is header, 1-81
confi_file = sys.argv[2] # name of configuration file
res_file = sys.argv[3] # result will be saved in res_file.format(str(set_index), method) 
data_file = sys.argv[4] # where the simulated data is saved
method = sys.argv[5]
# train_test = sys.argv[6]

# read configuration
with open(confi_file, "r") as f:
    x = f.readlines()
res = x[confi_index].strip()
para = res.split(",")
# mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda = [float(i) for i in para[: 4]] # not needed
set_index, d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest = [int(i) for i in para]
data_file = data_file.format(str(set_index))

# read model
with open(data_file, "r") as f:
    x = f.read()
dat = json.loads(x)

estimate = dat["model"]
for i in ["rho", "Lambda", "Mu"]:
    estimate[i] = np.array(estimate[i])

# X = np.array(dat[train_test]["X"])
# Y = np.array(dat[train_test]["Y"])

print("check 1")  # will be deleted

if method == "supervised":
    X = np.array(dat["test_set"]["X"])
    Y = np.array(dat["test_set"]["Y"])
    # define T
    T = []
    for i in range(nlabel):
        tem = np.block([
            [np.zeros((k0 * i, k0 + k1))],
            [np.eye(k0), np.zeros((k0, k1))],
            [np.zeros((k0 * (nlabel - i - 1), k0 + k1))],
            [np.zeros((k1, k0)), np.eye(k1)]
        ])
        T.append(tem)
    prediction = predict_supervised(X, estimate, T, ntrace, nchain, nskip)

if method == "unsupervised":    
    X = np.concatenate((np.array(dat["training_set"]["X"]), np.array(dat["test_set"]["X"])), axis = 0)
    Y = np.concatenate((np.array(dat["training_set"]["Y"]), np.array(dat["test_set"]["Y"])), axis = 0)
    K = ntopic = nlabel * k0 + k1
    estimate["rho"] = np.ones(K)
    prediction = predict_unsupervised(X, estimate, None, ntrace, nchain, nskip)
    
print("check 2") # will be deleted

try:  # will be deleted
    print("check 3")  # will be deleted
    for i in prediction.keys():
        np.savetxt("{}_{}.csv".format(i, set_index), prediction[i], delimiter=',', header='', fmt="%.5f")
        print(i)  # will be deleted
    print("check 3 finish")
except:
    print("check 3 fail") 

try:  # this section will will be deleted       
    print("check 4")  # will be deleted
    for i in prediction.keys():
        if type(prediction[i]) is type(np.array([1])):
            prediction[i] = prediction[i].tolist()    
    print("check 4 finish")
except:
    print("check 4 fail") 

with open(res_file.format(str(set_index), method), "w") as f:
    f.write(json.dumps(prediction))
    print(json.dumps(prediction))
print("finished")

