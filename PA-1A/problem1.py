# Cell type : CodeRead

import numpy as np
import matplotlib.pyplot as plt

def Bayes1a(X_train, Y_train, X_test):
    """ Give prediction for test instance using assumption 1a.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: +1/-1 numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : +1/-1 numpy array of shape (m,)

    """
    n = X_train.shape[0]
    num_pos = (Y_train == 1).sum()
    num_neg = n - num_pos
    X_train_pos = np.zeros((num_pos, 2))
    X_train_neg = np.zeros((num_neg, 2))
    a = num_pos/(num_neg + num_pos)

    p_c = 0
    n_c = 0
    for i in range(n):
        if Y_train[i] == 1:
            X_train_pos[p_c] = X_train[i]
            p_c += 1
        else:
            X_train_neg[n_c] = X_train[i]
            n_c += 1

    mu_pos = np.sum(X_train_pos, axis=0)/num_pos
    mu_neg = np.sum(X_train_neg, axis=0)/num_neg

    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    w = mu_neg - mu_pos
    b = (np.dot(mu_pos, mu_pos) - np.dot(mu_neg, mu_neg))*0.5

    for i in range(m):
        temp = np.dot(w, X_test[i]) + b
        Y_test_pred[i] = np.sign(a/(a + (1 - a)*np.exp(temp)) - 0.5)

    return Y_test_pred


def Bayes1b(X_train, Y_train, X_test):
    """ Give prediction for test instance using assumption 1b.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: +1/-1 numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : +1/-1 numpy array of shape (m,)

    """
    n = X_train.shape[0]
    d = X_train[0].shape[0]  # dimension
    num_pos = (Y_train == 1).sum()
    num_neg = n - num_pos
    X_train_pos = np.zeros((num_pos, 2))
    X_train_neg = np.zeros((num_neg, 2))
    a = num_pos/(num_neg + num_pos)

    p_c = 0
    n_c = 0
    for i in range(n):
        if Y_train[i] == 1:
            X_train_pos[p_c] = X_train[i]
            p_c += 1
        else:
            X_train_neg[n_c] = X_train[i]
            n_c += 1

    mu_pos = np.sum(X_train_pos, axis=0)/num_pos
    mu_neg = np.sum(X_train_neg, axis=0)/num_neg

    # initialize covarience matrices
    cov = np.zeros((X_train[0].shape[0], X_train[0].shape[0]))

    for i in range(num_pos):
        temp = np.reshape(X_train_pos[i] - mu_pos, (1, d))
        cov += np.dot(temp.T, temp)

    for i in range(num_neg):
        temp = np.reshape(X_train_neg[i] - mu_neg, (1, d))
        cov += np.dot(temp.T, temp)
    cov /= n

    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    for i in range(m):
        temp_pos = np.dot(
            np.dot((X_test[i] - mu_pos).T, np.linalg.inv(cov)), X_test[i] - mu_pos)
        temp_neg = np.dot(
            np.dot((X_test[i] - mu_neg).T, np.linalg.inv(cov)), X_test[i] - mu_neg)
        Y_test_pred[i] = np.sign(
            a/(a + (1-a)*np.exp(-0.5*(temp_neg - temp_pos))) - 0.5)

    return Y_test_pred


def Bayes1c(X_train, Y_train, X_test):
    """ Give prediction for test instance using assumption 1c.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: +1/-1 numpy array of shape (n,)
    X_test : numpy array of shape (m,d)

    Returns:
    Y_test_pred : +1/-1 numpy array of shape (m,)

    """
    n = X_train.shape[0]
    d = X_train[0].shape[0]  # dimension
    num_pos = (Y_train == 1).sum()
    num_neg = n - num_pos
    X_train_pos = np.zeros((num_pos, 2))
    X_train_neg = np.zeros((num_neg, 2))
    a = num_pos/(num_neg + num_pos)

    p_c = 0
    n_c = 0
    for i in range(n):
        if Y_train[i] == 1:
            X_train_pos[p_c] = X_train[i]
            p_c += 1
        else:
            X_train_neg[n_c] = X_train[i]
            n_c += 1

    mu_pos = np.sum(X_train_pos, axis=0)/num_pos
    mu_neg = np.sum(X_train_neg, axis=0)/num_neg

    # initialize covarience matrices
    cov1 = np.zeros((X_train[0].shape[0], X_train[0].shape[0]))
    cov2 = cov1

    for i in range(num_pos):
        temp = np.reshape(X_train_pos[i] - mu_pos, (1, d))
        cov1 += np.dot(temp.T, temp)
    cov1 /= num_pos

    for i in range(num_neg):
        temp = np.reshape(X_train_neg[i] - mu_neg, (1, d))
        cov2 += np.dot(temp.T, temp)
    cov2 /= num_neg

    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    for i in range(m):
        temp_pos = np.dot(
            np.dot((X_test[i] - mu_pos).T, np.linalg.inv(cov1)), X_test[i] - mu_pos)
        temp_neg = np.dot(
            np.dot((X_test[i] - mu_neg).T, np.linalg.inv(cov2)), X_test[i] - mu_neg)
        Y_test_pred[i] = np.sign(
            a/(a + (1-a)*np.exp(-0.5*(temp_neg - temp_pos))) - 0.5)

    return Y_test_pred

# Cell type : Convenience

# Testing the functions above

# To TAs: Replace this cell with the testing cell developed.

# To students: You may use the example here for testing syntax issues
# with your functions, and also as a sanity check. But the final evaluation
# will be done for different inputs to the functions. (So you can't just
# solve the problem for this one example given below.)

# testing the algorithms
# X_train_pos = np.random.randn(1000, 2)+np.array([[1., 2.]])
# X_train_neg = np.random.randn(1000, 2)+np.array([[2., 4.]])
# X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
# Y_train = np.concatenate((np.ones(1000), -1*np.ones(1000)))
# X_test_pos = np.random.randn(1000, 2)+np.array([[1., 2.]])
# X_test_neg = np.random.randn(1000, 2)+np.array([[2., 4.]])
# X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)
# Y_test = np.concatenate((np.ones(1000), -1*np.ones(1000)))

# Y_pred_test_1a = Bayes1a(X_train, Y_train, X_test)
# Y_pred_test_1c = Bayes1c(X_train, Y_train, X_test)


# for i in range(X_test.shape[0]):
#     print(X_test[i], " ans = ", Y_pred_test_1a[i])

# for i in range(X_test.shape[0]):
#     print(X_test[i], " ans = ", Y_pred_test_1c[i])
# Y_pred_test_1c = Bayes1c(X_train, Y_train, X_test)

"""**Cell type : TextRead**

# Problem 1

1d) Run the above three algorithms (Bayes1a,1b and 1c), for the three datasets given (dataset1_1.npz, dataset1_2.npz, dataset1_3.npz) in the cell below.

In the next CodeWrite cell, Plot all the classifiers (3 classification algos on 3 datasets = 9 plots) on a 2d plot (color the positively classified area light green, and negatively classified area light red). Add the training data points also on the plot. Plots to be organised into 3 plots follows: One plot for each dataset, with three subplots in each for the three classifiers. Label the 9 plots appropriately. 

In the next Textwrite cell, summarise (use the plots of the data and the assumptions in the problem to explain) your observations regarding the six learnt classifiers, and also give the error rate of the three classifiers on the three datasets as 3x3 table, with appropriately named rows and columns.
"""

datasets = ['archive/dataset1_1.npz', 'archive/dataset1_2.npz', 'archive/dataset1_3.npz']

for dataset in datasets:
    data = np.load(dataset)
    Y_pred_test_1a = Bayes1a(data['arr_0'], data['arr_1'], data['arr_2'])
    Y_pred_test_1b = Bayes1b(data['arr_0'], data['arr_1'], data['arr_2'])
    Y_pred_test_1c = Bayes1c(data['arr_0'], data['arr_1'], data['arr_2'])
    X = np.concatenate((data['arr_0'][:, 0], data['arr_2'][:, 0]))
    Y = np.concatenate((data['arr_0'][:, 1], data['arr_2'][:, 1]))
    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, c=np.concatenate((data['arr_1'], Y_pred_test_1a)), s=6)
    plt.title(dataset + ' 1a')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, c=np.concatenate((data['arr_1'], Y_pred_test_1b)), s=6)
    plt.title(dataset + ' 1b')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, c=np.concatenate((data['arr_1'], Y_pred_test_1c)), s=6)
    plt.title(dataset + ' 1c')
    plt.show()

    temp1 = 0
    temp2 = 0
    temp3 = 0
    for i in range(Y_pred_test_1a.shape[0]):
        if Y_pred_test_1a[i] == data['arr_3'][i]:
            temp1 += 1
    for i in range(Y_pred_test_1b.shape[0]):
        if Y_pred_test_1b[i] == data['arr_3'][i]:
            temp2 += 1
    for i in range(Y_pred_test_1c.shape[0]):
        if Y_pred_test_1c[i] == data['arr_3'][i]:
            temp3 += 1

    accuracy1a = temp1/Y_pred_test_1a.shape[0]
    accuracy1b = temp2/Y_pred_test_1b.shape[0]
    accuracy1c = temp3/Y_pred_test_1c.shape[0]
    print(accuracy1a, accuracy1b, accuracy1c)


# plt.figure(figsize=(10, 10))
# plt.axis('equal')
# colors = ['blue', 'red']
# Y_pred_test_1a_modified = list(map(int,(Y_pred_test_1a + 1)/2))

# plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred_test_1a_modified)
# plt.show()

# plt.figure(figsize=(10, 10))
# Y_pred_test_1c_modified = list(map(int,(Y_pred_test_1c + 1)/2))

# plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred_test_1c_modified)
# plt.show()

# Cell type : CodeWrite
# write the code for loading the data, running the three algos, and plotting here.
# (Use the functions written previously.)

"""** Cell type : TextWrite ** 
(Write your observations and table of errors here)"""
