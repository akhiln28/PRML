import numpy as np
import matplotlib.pyplot as plt

# tree is represented as dictionary of all tuples
global tree
tree = {}
data = np.load('Archive/dataset_D.npz')
value = data['arr_1']

N = data['arr_0'].shape[0]
d = data['arr_0'].shape[1]
min_node_size = 50
X_test = data['arr_2']
Y_test = data['arr_3']
data = np.transpose(data['arr_0'])

# sort each column for calculating accuracy
# sortedData = [np.sort(data['arr_0'][:, i]) for i in range(d)]

# find max and min for each x_i in data
dataMinMax = [[min(data[i, :]), max(data[i, :])]
              for i in range(d)]

# data is the list of indices in data3


def DecisionTree(dataIndices, index):
    # print(len(dataIndices),index)
    plusCount = 0
    for i in dataIndices:
        if value[i] == 1:
            plusCount += 1
    if len(dataIndices) <= min_node_size:
        tree[index] = [index, -1, -1, -1]
        if plusCount/len(dataIndices) > 0.5:
            tree[index][3] = 1
        else:
            tree[index][3] = 0
        return

    dataIndicesTrue = []
    dataIndicesFalse = []
    maxAccuracy = 0
    idx = 0
    maxcur = 0
    for i in range(d):
        min = dataMinMax[i][0]
        max = dataMinMax[i][1]
        cur = min + (max - min)/10
        # it=0
        while(cur < max):
            # it+=1
            temp1 = 0
            temp2 = 0
            truesize = 0
            for j in dataIndices:
                if data[i][j] >= cur:
                    truesize += 1
                    if value[j] == 1:
                        temp1 += 1
                else:
                    if value[j] == 1:
                        temp2 += 1
            accuracy = (temp1 + len(dataIndices) -
                        truesize - temp2)/len(dataIndices)
            if accuracy < 1 - accuracy:
                accuracy = 1 - accuracy
            # for best split
            if maxAccuracy <= accuracy:
                maxAccuracy = accuracy
                maxcur = cur
                idx = i
            cur += (max - min)/10
    for j in dataIndices:
        if data[idx][j] >= maxcur:
            dataIndicesTrue.append(j)
        else:
            dataIndicesFalse.append(j)

    tree[index] = [index, idx, maxcur, -1]

    # stopping further recursion
    if len(dataIndices) > min_node_size:
        # mistake: i made mistake by putting only one condition here
        if len(dataIndicesTrue) > 0 and len(dataIndicesFalse) > 0:
            DecisionTree(dataIndicesTrue, 2 * index + 1)
        if len(dataIndicesTrue) > 0 and len(dataIndicesFalse) > 0:
            DecisionTree(dataIndicesFalse, 2 * index)
        elif len(dataIndicesTrue) == 0 or len(dataIndicesFalse) == 0:
            tree[index] = [index, -1, -1, -1]
            if plusCount/len(dataIndices) > 0.5:
                tree[index][3] = 1
            else:
                tree[index][3] = 0
    return

# now testing


def DecisionTreeTest(X_test, Y_test):
    count = 0
    Y_test_pred = np.zeros(Y_test.shape[0])

    for i in range(X_test.shape[0]):
        t = -1
        index = 1
        while(t == -1):
            t = tree[index][3]
            if X_test[i][tree[index][1]] >= tree[index][2]:
                index = 2 * index + 1
            else:
                index = 2 * index
        # print(t,Y_test[i])
        Y_test_pred[i] = 2*(t - 0.5)
        if Y_test[i] == t or (Y_test[i] == -1 and t == 0):
            count += 1
    print(count/X_test.shape[0])
    return Y_test_pred


dataIndices = [i for i in range(N)]
# mistake: i passed 0 initially (infinite loop)
DecisionTree(dataIndices, 1)
DecisionTreeTest(X_test, Y_test)

datasets = ['Archive/dataset_A.npz','Archive/dataset_B.npz','Archive/dataset_C.npz','Archive/dataset_D.npz']

# plotting for A,B datasets
# X = X_test[:, 0]
# Y = X_test[:, 1]

# plt.figure(figsize=(6, 6))
# plt.scatter(X, Y, c=Y_test, s=6)
# plt.title('datasetB True')
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.scatter(X, Y, c=DecisionTreeTest(X_test,Y_test), s=6)
# plt.title('datasetB Pred')
# plt.show()
