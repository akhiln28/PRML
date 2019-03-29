from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

def randomForest(fractionData, fractionFeatures, X_train, Y_train, X_test, Y_test):
    Y_test_pred = np.zeros(Y_test.shape[0])
    for i in range(10):
        # pick random indices from X_train
        randomIndices = np.random.choice(range(X_train.shape[0]),int(fractionData*X_train.shape[0]))
        X_train_sample = X_train[randomIndices]
        Y_train_sample = Y_train[randomIndices]
        X_test_sample = X_test
        # remove (1-f2) * d features from X_train_sample
        removefeatures = np.random.choice(range(X_train.shape[1]),int((1-fractionFeatures)*X_train.shape[1]))
        removefeatures[::-1].sort()
        # remove selected features
        for j in removefeatures:
            if j < X_train_sample.shape[1]:
                X_train_sample = np.delete(X_train_sample,j,1)
                X_test_sample = np.delete(X_test_sample,j,1)
        
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train_sample,Y_train_sample)
        Y_test_pred = np.add(clf.predict(X_test_sample),Y_test_pred)
    for i in range(Y_test_pred.shape[0]):
        Y_test_pred[i] = np.sign(Y_test_pred[i])
    return accuracy_score(Y_test,Y_test_pred)


datasets = ['Archive/dataset_A.npz','Archive/dataset_B.npz','Archive/dataset_C.npz','Archive/dataset_D.npz']
for dataset in datasets:
    data = np.load(dataset)
    print(randomForest(0.5,0.5,data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']))


