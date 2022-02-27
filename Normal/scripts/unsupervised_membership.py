import numpy as np
import sys
import json
from unsupervised_Normal import train_unsupervised,predict_unsupervised


confi_index = int(sys.argv[1]) + 1 # 1st line is header, 1-81
confi_file = sys.argv[2] # name of configuration file
res_file = sys.argv[3] # result will be saved in res_file.format(str(confi_index))
data_file = sys.argv[4] # data is saved in data_file.format(str(set_index)

# extract configurations
with open(confi_file, "r") as f:
    x = f.readlines()
res = x[confi_index].strip()
para = res.split(",")
mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda = [float(i) for i in para[: 4]]
set_index, d, k0, k1, k0_train, k1_train, nlabel, ntrace, nchain, nskip, Ntrain, Ntest = [int(i) for i in para[4:]]
data_file = data_file.format(str(set_index))

# extract data
with open(data_file, "r") as f:
    x = f.read()
dat = json.loads(x)
training_X = np.array(dat["training_set"]["X"])
training_Y = np.array(dat["training_set"]["Y"])
test_X = np.array(dat["test_set"]["X"])
test_Y = np.array(dat["test_set"]["Y"])

k0, k1 = k0_train, k1_train
dg = k0 + k1 
K = ntopic = nlabel * k0 + k1
alpha = np.ones(K) # pay attention to this change
b = 0.1
# define T

estimation, classifier = train_unsupervised(training_X, training_Y, K, b, alpha, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip)
prediction = predict_unsupervised(test_X, estimation, classifier, ntrace, nchain, nskip) # only contain U

def dir_to_list(x):
    # x is a directory, which only contains np array or const
    for i in x.keys():
        if type(x[i]) is type(np.array([1])):
            x[i] = x[i].tolist()
dir_to_list(estimation)
dir_to_list(prediction)            
res = {"estimation": estimation, "prediction": prediction}

with open(res_file.format(str(confi_index)), "w") as f:
    f.write(json.dumps(res))
