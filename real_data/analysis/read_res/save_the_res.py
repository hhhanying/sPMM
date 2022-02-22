import json
import numpy as np

'''
Running this scripts can collect all the results and save them in res.txt.
To read the result, use the codes in read_res.py
'''

nlabel = 4

supervised = {}
for k0 in range(1, 16):
    k1 = k0

    T = []
    for i in range(nlabel):
        tem = np.block([
            [np.zeros((k0*i,k0+k1))],
            [np.eye(k0), np.zeros((k0, k1))],
            [np.zeros((k0*(nlabel-i-1),k0+k1))],
            [np.zeros((k1,k0)), np.eye(k1)]
        ])
        T.append(tem)
    
    cur= {}
    start_file = (k0 - 1) * 5 + 1
    for i in range(0,5):
        with open("./real_data/res/supervised/MOFA{}.txt".format(str(start_file+i))) as f:
            x = f.read()
        cur[i] = json.loads(x)
        # cur[i]["accuracy"] = sum(cur[i]["prediction"]["Y"] == Y[indexes[i]])/len(indexes[i])
    supervised[k0] = cur


unsupervised = {}
for k0 in range(1, 16):
    k1 = k0

    # read the result
    cur= {}
    start_file = (k0 - 1) * 5 + 1
    for i in range(0,5):
        with open("./real_data/res/unsupervised/MOFA_un{}.txt".format(str(start_file+i))) as f:
            x = f.read()
        cur[i] = json.loads(x)
        # cur[i]["accuracy"] = sum(cur[i]["prediction"]["Y"] == Y[indexes[i]])/len(indexes[i])
    unsupervised[k0] = cur

res = {"supervised": supervised, "unsupervised": unsupervised}
with open("./real_data/res/res.txt", "w") as f:
    f.write(json.dumps(res))
