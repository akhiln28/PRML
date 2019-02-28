import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""** Cell type : TextWrite ** 
(Write your observations and table of errors here)

**Cell type : TextRead **

# Problem 3 : Bias-Variance analysis in regression

Do bias variance analysis for the following setting: 

$X \sim Unif([-1,1]\times[-1,1])$

$Y=\exp(-4*||X-a||^2) + \exp(-4*||X-b||^2) + \exp(-4*||X-c||^2)$

where $a=[0.5,0.5], b=[-0.5,0.5], c=[0.5, -0.5]$.

Regularised Risk = $\frac{1}{m} \sum_{i=1}^m (w^\top \phi(x_i) - y_i)^2 + \frac{\lambda}{2} ||w||^2 $ 

Sample 50 (X,Y) points from above distribution, and do ridge regularised polynomial regression with degrees=[1,2,4,8,16] and regularisation parameters ($\lambda$) = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1]. Repeat for 100 times, and estimate the bias and variance for all 30 algorithms. You may approximate the distribution over X by discretising the $[-1,1]\times[-1,1]$ space into 10000 points. (Both expectations over S and (x,y) are simply estimates due to the finiteness of our experiments and sample)
 
3a) For each of the 30 algorithms (corresponding to 5 degrees and 6 lambda values) analyse the contour plot of the estimated $f_S$ for 3 different training sets. And the average $g(x) = E_S [f_S(x)]$. Write one function for doing everything in the code cell below. 

3b) In the next text cell, give the Bias and Variance computed as a $5\times 6$ matrix, appropriately label the rows and columns. And give your conclusion in one or two sentences.
"""

# Cell type : CodeWrite

# helper function for phi


def phi(x, degree):
    phi = np.zeros(int((degree + 1)*(degree + 2)/2))
    count = 0
    for d in range(degree + 1):
        for i in range(d + 1):
            phi[count] = x[0]**i * x[1]**(d - i)
    return phi


def f(x):
    a = np.array([0.5, 0.5])
    b = np.array([-0.5, 0.5])
    c = np.array([0.5, -0.5])

    return np.exp(-4 * np.dot(x - a, x - a)) + np.exp(-4 * np.dot(x - b, x - b)) + np.exp(-4 * np.dot(x - c, x - c))


def polynomial_regression_ridge_pred(X_test, wt_vector, degree=1):
    """ Give the value of the learned polynomial function, on test data.

    Arguments:
    X_test: numpy array of shape (n,d)
    wt_vec: numpy array of shape (d',)

    Returns:
    Y_test_pred : numpy array of shape (n,)

    """
    m = X_test.shape[0]
    Y_test_pred = np.zeros(m)

    for i in range(m):
        Y_test_pred[i] = np.dot(wt_vector, phi(X_test[i], degree))

    return Y_test_pred


def visualise_polynomial_2d(wt_vector, degree, title=""):
    """
    Give a contour plot over the 2d-data domain for the learned polynomial given by the weight vector wt_vector.

    """
    # X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

    # plt.contourf(X, Y, Z, levels=np.linspace(0., 1.2, 20))
    # plt.title('learned function : degree= ' + str(degree) + title)
    # plt.colorbar()


def polynomial_regression_ridge_train(X_train, Y_train, degree=1, reg_param=0.01):
    """ Give best polynomial fitting data, based on empirical squared error minimisation.

    Arguments:
    X_train: numpy array of shape (n,d)
    Y_train: numpy array of shape (n,)

    Returns:
    w : numpy array of shape (d',) with appropriate d'

    """
    n = X_train.shape[0]
    Phi = np.zeros((n, int((degree + 1)*(degree + 2)/2)))

    for i in range(n):
        Phi[i] = phi(X_train[i], degree)

    temp1 = np.linalg.inv(
        np.dot(Phi.T, Phi) + n/2 * reg_param * np.identity(int((degree + 1)*(degree + 2)/2)))
    temp2 = np.dot(Phi.T, Y_train)
    wt_vector = np.dot(temp1, temp2)
    return wt_vector


def compute_BV_error_sample_plot(degree, reg_param, num_training_samples=50):
    """Write code for generating data, fitting polynomial for given degree and reg_param. 
    Use num_training_samples samples for training.

    Compute the $f_S$ of 100 runs. 

    Plot 3 examples of learned function to illustrate how learned function varies 
    with different training samples. Also plot the average $f_S$ of all 100 runs.

    In total 4 subplots in one plot with appropriate title including degree and lambda value.

    Fill code to compute bias and variance, and average mean square error using the computed 100 $f_S$ functions.

    All contourplots are to be drawn with levels=np.linspace(0,1.2,20)

    Also return bias, variance, mean squared error. """
    # g is expectation of f_s
    g = np.zeros(int((degree + 1)*(degree + 2)/2))
    X_train = 2 * np.random.rand(num_training_samples, 2) - 1
    Y_train = list(map(f, X_train))
    fs = polynomial_regression_ridge_train(X_train, Y_train, degree, reg_param)

    for i in range(100):
        X_train = 2 * np.random.rand(num_training_samples, 2) - 1
        Y_train = list(map(f, X_train))
        fs = polynomial_regression_ridge_train(
            X_train, Y_train, degree, reg_param)
        g += fs
    g /= 100

    bias_sqr = 0
    for i in range(num_training_samples):
        temp = f(X_train[i]) - np.dot(g, phi(X_train[i], degree))
        bias_sqr += temp * temp
    bias_sqr /= num_training_samples

    var = 0
    for i in range(num_training_samples):
        temp = np.dot(g, phi(X_train[i], degree)) - np.dot(fs, phi(X_train[i], degree))
        var += temp * temp
    var /= num_training_samples

    mse = 0
    for i in range(num_training_samples):
        temp = f(X_train[i]) - np.dot(fs, phi(X_train[i], degree))
        mse += temp * temp
    mse /= num_training_samples
    return np.sqrt(bias_sqr), var, mse - bias_sqr - var


for degree in [1, 2, 4, 8, 16]:
    for reg_param in [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1]:
        plt.figure()
        b, v, e = compute_BV_error_sample_plot(degree, reg_param)
        print('================================')
        print('Degree= '+str(degree)+' lambda= '+str(reg_param))
        print('Bias = '+str(b))
        print('Variance = '+str(v))
        print('MSE = '+str(e))
