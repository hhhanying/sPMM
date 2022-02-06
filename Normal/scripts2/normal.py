import numpy as np
import sys
from data_generator_Normal import document_generator 
from supervised_Normal import CV_supervised_Normal, predict_supervised
from unsupervised_Normal import CV_unsupervised_Normal, predict_unsupervised

'''
The configuration file is supposed to provide:
mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest, Nfold
The first four parameters are float, the remainings are int
The name of configuration file need to be provided

Input: 
line number of configuration: 0 - (queue_no-1)
name of configuration file
name of output file

Output:
file_name: name + index + .txt
index = line number + 1

The file contains:
mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest, Nfold, a
supervised, unsupervised: the accuracy on test set
supervised_1 to supervised_Nfold: list of accuracies in CV for supervised model
unsupervised_1 to unsupervised_Nfold: list of accuracies in CV for unsupervised model

Other settings:
alpha = (1, ..., 1)
b = 0.1
In unsupervised model, I'll also use all-one-vector as hyperparameter.
'''

confi_file = sys.argv[2]
res_file = sys.argv[3]

# read parameters
f = open(confi_file,"r")
x = f.readlines()
f.close()

index = int(sys.argv[1]) + 1 # 1st line is header
res = x[index].strip()
para = res.split(",")
mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda = [float(i) for i in para[:4]]
d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest, Nfold = [int(i) for i in para[4:]]

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
Lambda = np.random.gamma(shape = alpha_Lambda, scale = 1/beta_Lambda, size = (K,d))
Mu = np.random.normal(mu_Mu, np.sqrt(sigma2_Mu/Lambda), (K,d))
Tau = Mu * Lambda

print("the value of a:", a)
print("the value of rho:", rho)

# generate dataset
X, Y, G, U = document_generator(a, rho, T, Lambda, Tau, Ntrain)
training_set = {'X': X, "Y": Y, "G": G, "U":U}
X, Y, G, U = document_generator(a, rho, T, Lambda, Tau, Ntest)
test_set = {'X': X, "Y": Y, "G": G, "U":U}

# supervised learning
res1, estimation1 = CV_supervised_Normal(training_set["X"], training_set["Y"], Nfold, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, alpha, ntrace, nchain, nskip)
prediction1 = predict_supervised(test_set["X"], estimation1, T, ntrace, nchain, nskip)
accuracy1 = sum(test_set["Y"] == prediction1["Y"])/Ntest

# unsupervised learning
res2, estimation2, classifier = CV_unsupervised_Normal(training_set["X"], training_set["Y"], Nfold, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, K, b, np.ones(K), ntrace, nchain, nskip)
prediction2 = predict_unsupervised(test_set["X"], estimation2, classifier, ntrace, nchain, nskip)
accuracy2 = sum(test_set["Y"] == prediction2["Y"])/Ntest

# write the result in the output file
res = [res, str(a), str(accuracy1), str(accuracy2)] + [str(acc) for acc in res1] + [str(acc) for acc in res2]
res = ",".join(res)
header = "mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, d, k0, k1, nlabel, ntrace, nchain, nskip, Ntrain, Ntest, Nfold, a, supervised, unsupervised, " + ", ".join([j+str(i) for j in ["supervised_", "unsupervised_"] for i in range(1, Nfold+1)])
filename = res_file + str(index) + ".txt"
f = open(filename,"w")
f.write(header)
f.write("\n")
f.write(res)
f.write("\n")
f.close()
