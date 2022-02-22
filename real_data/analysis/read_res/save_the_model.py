import json
import numpy as np

indexes = [[1, 15, 13, 12,  8, 25, 28, 50, 41, 35, 34, 54],
[16,  3, 18, 19, 21, 30, 24 ,51, 47, 46, 33, 55],
[17,  0, 11 , 6 ,29 ,23 ,44 ,39 ,43 ,40, 53],
[10,  4,  9 , 2 ,26 ,27 ,38 ,36, 32, 42 ,52],
[5 ,22 ,20, 14 , 7 ,31 ,45, 37, 48, 49, 56]]

train_indexes = [[] for i in range(5)]

for i in range(57):
    for j in range(5):
        if i not in set(indexes[j]):
            train_indexes[j].append(i)

nlabel = 4

# read result
with open("./res.txt", "r") as f:
    x = f.read()
res = json.loads(x)

# read real data
data_file  = "../data/cleaned_data.csv"
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

# save the model and memberships
# The result can be found in method_k0_noFold_parameter.csv

# supervised
meth = "supervised"
for k0 in range(1,16):
    k1 = k0
    T = []
    cur = res[meth][str(k0)]
    for i in range(nlabel):
        tem = np.block([
            [np.zeros((k0*i,k0+k1))],
            [np.eye(k0), np.zeros((k0, k1))],
            [np.zeros((k0*(nlabel-i-1),k0+k1))],
            [np.zeros((k1,k0)), np.eye(k1)]
        ])
        T.append(tem)    

    for no_fold in range(5):
        no = np.array([[i] for i in indexes[no_fold] + train_indexes[no_fold]])
        filename = "./model/"+"_".join([meth,str(k0),str(no_fold)])+"_{}.txt"

        G1 = np.array(cur[str(no_fold)]["prediction"]["G"]) #G for testing
        G2 = np.array(cur[str(no_fold)]["estimation"]["G"]) #G for training
        G = np.concatenate((G1,G2), axis=0)
        Y1 = cur[str(no_fold)]["prediction"]["Y"]
        Y2 = Y[train_indexes[no_fold]]
        Ys = np.concatenate((Y1,Y2), axis=0)
        U = np.array([np.dot(T[Ys[i]], G[i]) for i in range(57)])

        U = np.concatenate((no,U), axis=1)
        np.savetxt(filename.format("U"), U, delimiter=',', header='',fmt="%.5f")

        estimate = cur[str(no_fold)]["estimation"]
        Mu, Lambda = [estimate["Mu1"], estimate["Mu2"], estimate["Mu3"]], [estimate["Lambda1"], estimate["Lambda2"], estimate["Lambda3"]]
        Mu, Lambda = [np.array(mus) for mus in Mu], [np.array(lambdas) for lambdas in Lambda]
        Mu = np.concatenate((Mu[0], Mu[1], Mu[2]), axis=1)
        Lambda = np.concatenate((Lambda[0], Lambda[1], Lambda[2]), axis=1)
        S = 1/Lambda
        np.savetxt(filename.format("Mu"), Mu, delimiter=',', header='',fmt="%.5f")
        np.savetxt(filename.format("S"), S, delimiter=',', header='',fmt="%.5f")
            


meth = "unsupervised"
for k0 in range(1,16):
    k1 = k0
    cur = res[meth][str(k0)]
 
    for no_fold in range(5):
        no = np.array([[i] for i in indexes[no_fold] + train_indexes[no_fold]])
        filename = "./model/"+"_".join([meth,str(k0),str(no_fold)])+"_{}.txt"

        U1 = np.array(cur[str(no_fold)]["prediction"]["U"]) #U for testing
        U2 = np.array(cur[str(no_fold)]["estimation"]["U"]) #U for training
        U = np.concatenate((U1,U2), axis=0)
        U = np.concatenate((no,U), axis=1)
        np.savetxt(filename.format("U"), U, delimiter=',', header='',fmt="%.5f")

        estimate = cur[str(no_fold)]["estimation"]
        Mu, Lambda = [estimate["Mu1"], estimate["Mu2"], estimate["Mu3"]], [estimate["Lambda1"], estimate["Lambda2"], estimate["Lambda3"]]
        Mu, Lambda = [np.array(mus) for mus in Mu], [np.array(lambdas) for lambdas in Lambda]
        Mu = np.concatenate((Mu[0], Mu[1], Mu[2]), axis=1)
        Lambda = np.concatenate((Lambda[0], Lambda[1], Lambda[2]), axis=1)
        S = 1/Lambda
        np.savetxt(filename.format("Mu"), Mu, delimiter=',', header='',fmt="%.5f")
        np.savetxt(filename.format("S"), S, delimiter=',', header='',fmt="%.5f")
            


