import json
import numpy as np
'''
This file is used to read the result.
All results will be saved in a directory res.
for k0 in 1:15:
    for method in ["supervised", "unsupervised"]:
        for no_fold in 0:4:
            the result can be obtained by: res[method]["k0"]["no_fold"]

For supervised model, the result will contain:
estimation = {a, rho, G, Mu1, Mu2, Mu3, Lambda1, Lambda2, Lambda3}
prediction = {G, Y}

For unsupervised model, the result will contain:
estimation = {a, rho, U, Mu1, Mu2, Mu3, Lambda1, Lambda2, Lambda3}
prediction = {U, Y}
'''

with open("./real_data/res/res.txt", "r") as f:
    x = f.read()
res = json.loads(x)

# example to get the estimate in first fold when k0=k1=2 under supervised model:
# res["supervised"]["2"]["1"]
