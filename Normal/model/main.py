# import packages
import numpy as np
import json
from data_generator_Normal import document_generator_Normal
from train_supervised_Normal import train_supervised_Normal

# read parameters
id = 1
k0 = 2
k1 = 3
nlabel = 3
d = 20
mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda = 0, 10, 2, 4

Ntrain = 200
Ntest = 100
nchain = 2
ntrace = 2000
nskip = 2


# generate  parameters
np.random.seed(1)
dg = k0 + k1 
K = ntopic = nlabel * k0 + k1

b = 0.1
alpha = np.ones(dg)

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

rho = np.random.dirichlet(alpha, 1)[0]
a = np.random.exponential(1 / b, 1)[0]
Lambda = np.random.gamma(shape = alpha_Lambda, scale = 1 / beta_Lambda, size = (K, d))
Mu = np.random.normal(mu_Mu, np.sqrt(sigma2_Mu / Lambda), (K, d))
Tau = Mu * Lambda
# generate dataset, split for CV
X, Y, G, U = document_generator_Normal(a, rho, T, Lambda, Tau, Ntrain)
training_set = {'X': X, "Y": Y, "G": G, "U":U}
X, Y, G, U = document_generator_Normal(a, rho, T, Lambda, Tau, Ntest)
test_set = {'X': X, "Y": Y, "G": G, "U":U}

# train, save results
estimate = train_supervised_Normal(training_set["X"], training_set["Y"], K, b, alpha, T, mu_Mu, sigma2_Mu, alpha_Lambda, beta_Lambda, ntrace, nchain, nskip)

for i in estimate.keys():
    if type(estimate[i]) is type(np.array([1])):
        estimate[i] = estimate[i].tolist()

filename = "test_supervised_train.txt"
f = open(filename,"w")
f.write(json.dumps(estimate))
f.close()