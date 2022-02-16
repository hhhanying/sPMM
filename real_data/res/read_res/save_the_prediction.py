import json
import numpy as np

indexes = [[1, 15, 13, 12,  8, 25, 28, 50, 41, 35, 34, 54],
[16,  3, 18, 19, 21, 30, 24 ,51, 47, 46, 33, 55],
[17,  0, 11 , 6 ,29 ,23 ,44 ,39 ,43 ,40, 53],
[10,  4,  9 , 2 ,26 ,27 ,38 ,36, 32, 42 ,52],
[5 ,22 ,20, 14 , 7 ,31 ,45, 37, 48, 49, 56]]

nlabel = 4

# read result
with open("./real_data/res/res.txt", "r") as f:
    x = f.read()
res = json.loads(x)

# read real data
data_file  = "../real_data/data/cleaned_data.csv"
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

# save the prediction in file prediction.csv
# it stores index(1-57), k0(1-15), true label, predicted label, method("supervised", "unsupervised")
header_label = "index, k0, true, prediction, method\n"
for meth in ["supervised", "unsupervised"]:
    for i in range(1,16):
        for no_fold in range(5):
            for j in range(len(indexes[no_fold])):
                ind_sample = indexes[no_fold][j]
                infos = [ind_sample, i, Y[ind_sample], res[meth][str(i)][str(no_fold)]["prediction"]["Y"][j],meth]
                header_label += ",".join([str(info) for info in infos]) + "\n"

with open("./real_data/res/accuracy/prediction.csv", "w") as f:
    f.write(header_label)            
