import numpy as np
import sys
from data_generator_Bernoulli import document_generator 
from supervised_Bernoulli import CV_supervised_Bernoulli
from unsupervised_Bernoulli import CV_unsupervised_Bernoulli

confi_file = "configurations.csv"

# read parameters
f = open(confi_file,"r")
x = f.readlines()
f.close()

index = int(sys.argv[1]) + 1 # 1st line is header
res = x[index].strip()
para = res.split(",")
alpha_p, beta_p = [float(i) for i in para[:2]]
d, k0, k1, nlabel, ntrace, nchain, nskip, N, Nfold = [int(i) for i in para[2:]]

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

X, Y, G, U = document_generator(a, rho, T, P, N)

res1 = CV_supervised_Bernoulli(X, Y, Nfold, T, alpha_p, beta_p, K, b, alpha, ntrace, nchain, nskip)
print("supervised:")
print(res1)
res2 = CV_unsupervised_Bernoulli(X, Y, Nfold, alpha_p, beta_p, K, b, np.ones(K), ntrace, nchain, nskip)
print("unsupervised:")
print(res2)

res = [res, str(a)] + [str(acc) for acc in res1] + [str(acc) for acc in res2]
res = ",".join(res)
header = "alpha_p, beta_p, d, k0, k1, nlabel, ntrace, nchain, nskip, N, Nfold, a, " + ", ".join([j+str(i) for j in ["supervised_", "unsupervised_"] for i in range(1, Nfold+1)])


filename = "output{}.txt".format(str(index))
f = open(filename,"w")
f.write(header)
f.write("\n")
f.write(res)
f.write("\n")
f.close()