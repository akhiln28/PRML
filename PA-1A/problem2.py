import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def classify(ita):
    L = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    min = np.dot(L[0], ita)
    ans = 0
    for i in range(1, 4):
        temp = np.dot(L[i], ita)
        if temp < min:
            min = temp
            ans = i
    return ans + 1


def Bayes2a(X_train, Y_train, X_test):
    """ Give Bayes classifier prediction for test instances 
    using assumption 2a.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: {1,2,3,4} numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : {1,2,3,4} numpy array of shape (m,)

    """
    n = X_train.shape[0]
    d = X_train[0].shape[0]  # dimension
    num_0 = (Y_train == 1.0).sum()
    num_1 = (Y_train == 2.0).sum()
    num_2 = (Y_train == 3.0).sum()
    num_3 = n - num_0 - num_1 - num_2
    num = np.array([num_0, num_1, num_2, num_3])
    mu = np.zeros((4, d))
    a = np.zeros(4)

    for i in range(4):
        a[i] = num[i]/n

    for i in range(n):
        mu[int(Y_train[i] - 1.0)] += X_train[i]

    for i in range(4):
        mu[i] /= num[i]

    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    for i in range(m):
        denom = 0
        for j in range(4):
            denom += np.exp(-0.5 *
                            np.dot(X_test[i] - mu[j], X_test[i] - mu[j])) * a[j]
        ita = np.zeros(4)
        for j in range(4):
            ita[j] = np.exp(-0.5 * np.dot(X_test[i] - mu[j],
                                          X_test[i] - mu[j])) * a[j]/denom
        Y_test_pred[i] = classify(ita)
    return Y_test_pred


def Bayes2b(X_train, Y_train, X_test):
    """ Give Bayes classifier prediction for test instances 
    using assumption 2b.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: {1,2,3,4} numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : {1,2,3,4} numpy array of shape (m,)

    """
    n = X_train.shape[0]
    d = X_train[0].shape[0]  # dimension
    num_0 = (Y_train == 1.0).sum()
    num_1 = (Y_train == 2.0).sum()
    num_2 = (Y_train == 3.0).sum()
    num_3 = n - num_0 - num_1 - num_2
    num = np.array([num_0, num_1, num_2, num_3])
    mu = np.zeros((4, d))
    a = np.zeros(4)
    cov = np.zeros((d, d))

    for i in range(4):
        a[i] = num[i]/n

    for i in range(n):
        mu[int(Y_train[i] - 1.0)] += X_train[i]

    for i in range(4):
        mu[i] /= num[i]

    for i in range(n):
        j = int(Y_train[i] - 1.0)
        temp = np.reshape(X_train[i] - mu[j], (1, d))
        cov += np.dot(temp.T, temp)

    cov /= n

    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    for i in range(m):
        denom = 0
        ita = np.zeros(4)
        for j in range(4):
            ita[j] = np.exp(-0.5 * np.dot(np.dot(X_test[i] - mu[j],
                                                 np.linalg.inv(cov)), X_test[i] - mu[j])) * a[j]
            denom += ita[j]

        for j in range(4):
            ita[j] /= denom
        Y_test_pred[i] = classify(ita)
    return Y_test_pred


def Bayes2c(X_train, Y_train, X_test):
    """ Give Bayes classifier prediction for test instances 
    using assumption 2c.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: {1,2,3,4} numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : {1,2,3,4} numpy array of shape (m,)

    """
    n = X_train.shape[0]
    d = X_train[0].shape[0]  # dimension
    num_0 = (Y_train == 1.0).sum()
    num_1 = (Y_train == 2.0).sum()
    num_2 = (Y_train == 3.0).sum()
    num_3 = n - num_0 - num_1 - num_2
    num = np.array([num_0, num_1, num_2, num_3])
    mu = np.zeros((4, d))
    a = np.zeros(4)
    cov = np.zeros((4, d, d))

    for i in range(4):
        a[i] = num[i]/n

    for i in range(n):
        mu[int(Y_train[i] - 1.0)] += X_train[i]

    for i in range(4):
        mu[i] /= num[i]

    for i in range(n):
        j = int(Y_train[i] - 1.0)
        temp = np.reshape(X_train[i] - mu[j], (1, d))
        cov[j] += np.dot(temp.T, temp)

    for i in range(4):
        cov[i] /= num[i]

    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    for i in range(m):
        denom = 0
        ita = np.zeros(4)
        for j in range(4):
            ita[j] = np.exp(-0.5 * np.dot(np.dot(X_test[i] - mu[j],
                                                 np.linalg.inv(cov[j])), X_test[i] - mu[j])) * a[j]
            denom += ita[j]

        for j in range(4):
            ita[j] /= denom
        Y_test_pred[i] = classify(ita)
    return Y_test_pred


# Cell type : Convenience
# Testing the functions above
# Data 1
mat1 = np.array([[1., 0.], [0., 1.]])
mat2 = np.array([[1., 0.], [0., 1.]])
mat3 = np.array([[1., 0.], [0., 1.]])
mat4 = np.array([[1., 0.], [0., 1.]])

X_train_1 = np.dot(np.random.randn(1000, 2), mat1)+np.array([[0., 0.]])
X_train_2 = np.dot(np.random.randn(1000, 2), mat2)+np.array([[0., 2.]])
X_train_3 = np.dot(np.random.randn(1000, 2), mat3)+np.array([[2., 0.]])
X_train_4 = np.dot(np.random.randn(1000, 2), mat4)+np.array([[2., 2.]])

X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4), axis=0)
Y_train = np.concatenate(
    (np.ones(1000), 2*np.ones(1000), 3*np.ones(1000), 4*np.ones(1000)))


X_test_1 = np.dot(np.random.randn(1000, 2), mat1)+np.array([[0., 0.]])
X_test_2 = np.dot(np.random.randn(1000, 2), mat2)+np.array([[0., 2.]])
X_test_3 = np.dot(np.random.randn(1000, 2), mat3)+np.array([[2., 0.]])
X_test_4 = np.dot(np.random.randn(1000, 2), mat4)+np.array([[2., 2.]])

X_test = np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4), axis=0)
Y_test = np.concatenate((np.ones(1000), 2*np.ones(1000),
                         3*np.ones(1000), 4*np.ones(1000)))


Y_pred_test_2a = Bayes2a(X_train, Y_train, X_test)

# for i in range(X_test.shape[0]):
#     print(X_test[i], " ans = ", Y_pred_test_2a[i])
# Y_pred_test_2c = Bayes2c(X_train, Y_train, X_test)
# for i in range(X_test.shape[0]):
#     print(X_test[i], " ans = ", Y_pred_test_2c[i])
# Y_pred_test_2c = Bayes2c(X_train, Y_train, X_test)

"""**Cell type : TextRead**

# Problem 2

2d) Run the above three algorithms (Bayes2a,2b and 2c), for the two datasets given (dataset2_1.npz, dataset2_2.npz) in the cell below.


In the next CodeWrite cell, Plot all the classifiers (3 classification algos on 2 datasets = 6 plots) on a 2d plot (color the 4 areas classified as 1,2,3 and 4 differently). Add the training data points also on the plot. Plots to be organised as follows: One plot for each dataset, with three subplots in each for the three classifiers. Label the 6 plots appropriately. 

In the next Textwrite cell, summarise your observations regarding the six learnt classifiers. Give the *expected loss* (use the Loss matrix given in the problem.) of the three classifiers on the two datasets as 2x3 table, with appropriately named rows and columns. Also, give the 4x4 confusion matrix of the final classifier for all three algorithms and both datasets.
"""

datasets = ['archive/dataset2_1.npz', 'archive/dataset2_2.npz']

for dataset in datasets:
    data = np.load(dataset)
    Y_pred_test_2a = Bayes2a(data['arr_0'], data['arr_1'], data['arr_2'])
    Y_pred_test_2b = Bayes2b(data['arr_0'], data['arr_1'], data['arr_2'])
    Y_pred_test_2c = Bayes2c(data['arr_0'], data['arr_1'], data['arr_2'])

    X = np.concatenate((data['arr_0'][:,0],data['arr_2'][:,0]))
    Y = np.concatenate((data['arr_0'][:,1],data['arr_2'][:,1]))

    plt.figure(figsize=(10, 10))
    plt.scatter(X,Y,c=np.concatenate((data['arr_1'],Y_pred_test_2a)))
    plt.title(dataset + ' 2a')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(X,Y,c=np.concatenate((data['arr_1'],Y_pred_test_2b)))
    plt.title(dataset + ' 2b')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(X,Y,c=np.concatenate((data['arr_1'],Y_pred_test_2c)))
    plt.title(dataset + ' 2b')
    plt.show()
    temp1 = 0
    temp2 = 0
    temp3 = 0
    for i in range(Y_pred_test_2a.shape[0]):
        if Y_pred_test_2a[i] == data['arr_3'][i]:
            temp1 += 1
    for i in range(Y_pred_test_2b.shape[0]):
        if Y_pred_test_2b[i] == data['arr_3'][i]:
            temp2 += 1
    for i in range(Y_pred_test_2c.shape[0]):
        if Y_pred_test_2c[i] == data['arr_3'][i]:
            temp3 += 1

    accuracy1a = temp1/Y_pred_test_2a.shape[0]
    accuracy1b = temp2/Y_pred_test_2b.shape[0]
    accuracy1c = temp3/Y_pred_test_2c.shape[0]
    print(accuracy1a, accuracy1b, accuracy1c)

# plt.figure(figsize=(10,10))
# plt.axis('equal')
# colors = ['red', 'blue', 'green','yellow']

# Y_pred_test_2a_modified = list(map(int, Y_pred_test_2a - 1.0))

# plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred_test_2a_modified,
#             cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()