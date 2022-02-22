import json
import numpy as np
import sys

'''
This file is used to generated x based on the result of learning.
We will use the learned parameters for topics and the learned memberships for each record to simulate new samples.
This file require one parameter sim_N, which is the sample size to simulate for each x.

For supervised model, the simulated data will be saved in ./simulated/sim_x_su_k0.csv
For unsupervised model, the simulated data will be saved in ./simulated/sim_x_un_k0.csv
The data has 241 columns, the first 240 columns are simulated xs, the last column is the record of data.
'''

sim_N = int(sys.argv[1])
nlabel = 4

with open("./real_data/res/res.txt", "r") as f:
    x = f.read()
res = json.loads(x)

indexes = [[1, 15, 13, 12,  8, 25, 28, 50, 41, 35, 34, 54],
[16,  3, 18, 19, 21, 30, 24 ,51, 47, 46, 33, 55],
[17,  0, 11 , 6 ,29 ,23 ,44 ,39 ,43 ,40, 53],
[10,  4,  9 , 2 ,26 ,27 ,38 ,36, 32, 42 ,52],
[5 ,22 ,20, 14 , 7 ,31 ,45, 37, 48, 49, 56]]

for k0 in range(1, 16):
    k1 = k0

    # supervised
    T = []
    for i in range(nlabel):
        tem = np.block([
            [np.zeros((k0*i,k0+k1))],
            [np.eye(k0), np.zeros((k0, k1))],
            [np.zeros((k0*(nlabel-i-1),k0+k1))],
            [np.zeros((k1,k0)), np.eye(k1)]
        ])
        T.append(tem)
    
    sim_x = np.array([list(range(241))]) # len(each row) = 241
    cur = res["supervised"][str(k0)]

    for i in range(0,5):
        # get the topic parameters: Tau and Lambda
        estimate = cur[str(i)]["estimation"]
        Mu, Lambda = [estimate["Mu1"], estimate["Mu2"], estimate["Mu3"]], [estimate["Lambda1"], estimate["Lambda2"], estimate["Lambda3"]]
        Mu, Lambda = [np.array(mus) for mus in Mu], [np.array(lambdas) for lambdas in Lambda]
        Mu = np.concatenate((Mu[0], Mu[1], Mu[2]), axis=1)
        Lambda = np.concatenate((Lambda[0], Lambda[1], Lambda[2]), axis=1)
        Tau = Mu * Lambda
        # get the estimated membership
        G = np.array(cur[str(i)]["prediction"]["G"])
        Ybar = cur[str(i)]["prediction"]["Y"]
        # Ybar = Y[indexes[i]]
        U = np.array([np.dot(T[Ybar[i]], G[i]) for i in range(len(Ybar))])

        for j in range(len(indexes[i])):
            u = U[j]
            tem = np.zeros((sim_N, 241)) + indexes[i][j]
            for k in range(0,240):
                lambdax = np.dot(u, Lambda[:, k])
                taux = np.dot(u, Tau[:, k])
                sx = 1/lambdax
                mux = sx*taux
                tem[:,k] = np.random.normal(mux, np.sqrt(sx),sim_N)
            sim_x = np.concatenate((sim_x, tem), axis = 0)
    np.savetxt("./real_data/res/simulated/sim_x_su_{}.csv".format(str(k0)), sim_x[1:,:], delimiter=",",fmt="%.2f")

    sim_x = np.array([list(range(241))]) # len(each row) = 241
    cur = res["unsupervised"][str(k0)]

    for i in range(0,5):
        # get the topic parameters: Tau and Lambda
        estimate = cur[str(i)]["estimation"]
        Mu, Lambda = [estimate["Mu1"], estimate["Mu2"], estimate["Mu3"]], [estimate["Lambda1"], estimate["Lambda2"], estimate["Lambda3"]]
        Mu, Lambda = [np.array(mus) for mus in Mu], [np.array(lambdas) for lambdas in Lambda]
        Mu = np.concatenate((Mu[0], Mu[1], Mu[2]), axis=1)
        Lambda = np.concatenate((Lambda[0], Lambda[1], Lambda[2]), axis=1)
        Tau = Mu * Lambda
        # get the estimated membership
        U = np.array(cur[str(i)]["prediction"]["U"])
        Ybar = cur[str(i)]["prediction"]["Y"]
        # Ybar = Y[indexes[i]]

        for j in range(len(indexes[i])):
            u = U[j]
            tem = np.zeros((sim_N, 241)) + indexes[i][j]
            for k in range(0,240):
                lambdax = np.dot(u, Lambda[:, k])
                taux = np.dot(u, Tau[:, k])
                sx = 1/lambdax
                mux = sx*taux
                tem[:,k] = np.random.normal(mux, np.sqrt(sx),sim_N)
            sim_x = np.concatenate((sim_x, tem), axis = 0)
    np.savetxt("./real_data/res/simulated/sim_x_un_{}.csv".format(str(k0)), sim_x[1:,:], delimiter=",",fmt="%.2f")
    