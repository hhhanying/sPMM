'''
This file wll try to learn the membership given the true paratemeters of the model
input:
1. index of configuration: 0 to 80
2. name of configuration file
3. name to save result
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
res_file = sys.argv[3] # result will be saved in res_file.format(str(set_index), method, train_test)
method = sys.argv[5]
train_test = sys.argv[6]
confi_index = 80 # to see the largest memory and disk
# read configuration
with open(confi_file, "r") as f:
    x = f.readlines()
res = x[confi_index].strip()
para = res.split(",")
# mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda = [float(i) for i in para[: 4]] # not needed
set_index, d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest = [int(i) for i in para]
data_file = sys.argv[4].format(str(set_index))

# read model
with open(data_file, "r") as f:
    x = f.read()
dat = json.loads(x)

estimate = dat["model"]
X = np.array(dat[train_test]["X"])
Y = np.array(dat[train_test]["Y"])

if method == "supervised":
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
    K = ntopic = nlabel * k0 + k1
    estimate["rho"] = np.ones(K)
    prediction = predict_unsupervised(X, estimate, None, ntrace, nchain, nskip)
    

for i in prediction.keys():
    if type(prediction[i]) is type(np.array([1])):
        prediction[i] = prediction[i].tolist()    


with open(res_file.format(str(set_index), method, train_test), "w"):
    f.write(json.dumps(prediction))




