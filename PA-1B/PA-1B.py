#%% [markdown]
# # General Instructions to students:
# 
# 1. There are 5 types of cells in this notebook. The cell type will be indicated within the cell.
#     1. Markdown cells with problem written in it. (DO NOT TOUCH THESE CELLS) (**Cell type: TextRead**)
#     2. Python cells with setup code for further evaluations. (DO NOT TOUCH THESE CELLS) (**Cell type: CodeRead**)
#     3. Python code cells with some template code or empty cell. (FILL CODE IN THESE CELLS BASED ON INSTRUCTIONS IN CURRENT AND PREVIOUS CELLS) (**Cell type: CodeWrite**)
#     4. Markdown cells where a written reasoning or conclusion is expected. (WRITE SENTENCES IN THESE CELLS) (**Cell type: TextWrite**)
#     5. Temporary code cells for convenience and TAs. (YOU MAY DO WHAT YOU WILL WITH THESE CELLS, TAs WILL REPLACE WHATEVER YOU WRITE HERE WITH OFFICIAL EVALUATION CODE) (**Cell type: Convenience**)
#     
# 2. You are not allowed to insert new cells in the submitted notebook.
# 
# 3. You are not allowed to import any extra packages.
# 
# 4. The code is to be written in Python 3.6 syntax. Latest versions of other packages maybe assumed.
# 
# 5. In CodeWrite Cells, the only outputs to be given are plots asked in the question. Nothing else to be output/print. 
# 
# 6. If TextWrite cells ask you to give accuracy/error/other numbers you can print them on the code cells, but remove the print statements before submitting.
# 
# 7. The convenience code can be used to check the expected syntax of the functions. At a minimum, your entire notebook must run with "run all" with the convenience cells as it is. Any runtime failures on the submitted notebook as it is will get zero marks.
# 
# 8. All code must be written by yourself. Copying from other students/material on the web is strictly prohibited. Any violations will result in zero marks.
# 
# 9. All datasets will be given as .npz files, and will contain data in 4 numpy arrays :"X_train, Y_train, X_test, Y_test". In that order. The meaning of the 4 arrays can be easily inferred from their names.
# 
# 10. All plots must be labelled properly, all tables must have rows and columns named properly.

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#%% [markdown]
# # 1. Logistic Regression 
# 
# Write code for doing logistic regression below. Also write code for choosing best hyperparameters for each kernel type (use a part of training set as validation set).
# 
# Write code for running in the cell after (You may be asked to demonstrate your code during the viva using this cell.)
# 
# In text cell after that report the following numbers you get by running appropriate code.
# 
# Split the X_train, Y_train into train and validation based on an 80:20 split. For a given dataset, kernel parameter and regularisation value run gradient descent on the regularised kernel logistic loss on training set, with some learning rate eta. Try different etas, and choose the best eta (the eta that achieves the lowest R(alpha)), based on the alpha  you get after 1000 iterations.
# 
# For the classification data sets A and B, report the best kernel and regularisation parameters for the RBF kernel.  Choose the best kernel and regularisation parameter based on the accuracy of the model given by the optimisation procedure. Report the training and test zero-one error (or 1-accuracy) for those hyperparameters. 
# 
# For both  the synthetic classification datasets (dataset_A and dataset_B) in 2-dimensions, also illustrate the learned classifier. Do this in the last codeWrite cell for this question.
# 

#%%
# CodeWrite 
#Write logistic regression code from scratch. Use gradient descent.
# Only write functions here

def train_pred_logistic_regression(X_train, Y_train, X_test, kernel='linear', reg_param=0., 
                                   kernel_param=1., num_iter_gd=100):
"""
Arguments:
X_train : (n,d) shape numpy array
Y_train : (n,)  shape numpy array
X_test : (m,d) shape numpy array
kernel = 'linear' or 'rbf' or 'poly' 
reg_param = $\lambda$

Returns the prediction of logistic regression :
$ \min \sum_{i=1}^n \log(1+\exp(-y_i* \w^\top \phi(\x_i)))  + \frac{\lambda}{2} ||\w||^2 $
where $\phi$ is the feature got by the kernel.

The kernel is defined by the kernel_param:
If kernel=linear: K(\u,\v) = \u^\top \v  
If kernel=poly:  K(\u,\v) = (1+\u^\top \v)^(kernel_param)
If kernel=rbf:  K(\u,\v) = \exp(-kernel_param*||\u-\v||^2)

Returns:
Y_test_pred: (m,) shape numpy array

"""
    
    
def return_best_hyperparam( ): # give appropriate arguments, return appropriate variables
    


#%%
# CodeWrite : Use the functions above to get the numbers you report below. 

#%% [markdown]
# TextWrite Cell: Give your observations and the list of hyperparameter choices and train zero-one error  and test zero-one error for all three kernel choices, for all 4 datasets (2 real world and 2 synthetic).  
# 
# 
# 

#%%
# Codewrite cell: Generate plots of learned classifier for all three kernel types, on dataset_A and datasset_B.
# Plots should give both the learned classifier and the train data. 
# Similar to  Bishop Figure 4.5 (with just two classes here.)
# Total number of plots = 3 * 2 = 6

#%% [markdown]
# # 2. SVM
# 
# Write code for learning SVM below. Also write code for choosing best hyperparameters for each kernel type. You may use sklearn.svm for this purpose. (use a part of training set as validation set)
# 
# Write code for running in the cell after (You may be asked to demonstrate your code during the via using this cell.)
# 
# In text cell after that report the following numbers you get by running appropriate code:
# 
# For each classification data set (dataset A,B,C,D) report the best kernel and regularisation parameters for linear, RBF and Poly kernels. (Linear has no kernel parameter.) Report the training and test zero-one error for those hyperparameters.
# 
# For the synthetic classification datasets in 2-dimensions, also illustrate the learned classifier for each kernel setting. Do this in the last codeWrite cell for this question.

#%%
# CodeWrite cell
# Write SVM classifier using SKlearn, write code for choosing best hyper parameters.
# write only functions here


#%%
# CodeWrite cell
# Write code here for generating the numbers that you report below.

#%% [markdown]
# TextWrite Cell: Give your observations and the list of hyperparameter choices and train zero-one error  and test zero-one error for all three kernel choices, for all 4 datasets (2 real world and 2 synthetic).  
# 

#%%
# Codewrite cell: Generate plots of learned classifier for all three kernel types, on dataset_A and datasset_B.
# Plots should give both the learned classifier and the train data. 
# Similar to  Bishop Figure 4.5 (with just two classes here.)
# Total number of plots = 3 * 2 = 6


#%% [markdown]
# # 3. Decision Tree
# 
# Write code for learning decision tree below. Take as an argument a hyperparameter on what size node to stop splitting. Choose the number of training points at which you stop splitting the node further between 1,10 and 50. You are NOT allowed to use sklearn modules for this.)
# 
# Write code for running in the cell after (You may be asked to demonstrate your code during the viva using this cell.)
# 
# In text cell after that report the following numbers you get by running appropriate code:
# 
# For the classification data sets A and B report the best node size to stop splitting. Report the training and test zero-one error for those hyperparameters.
# 
# Also illustrate the learned classifier. Do this in the last codeWrite cell for this question.
# 
# Important: Think about how you will represent a decision tree. (Possible soln: Store as a list of tuples containing node position, attribute to split, threshold, class to classifiy (if leaf node) )
# 

#%%
# CodeWrite cell
# Write Decision tree classifier from scratch, write code for choosing best node size to stop splitting.
# write only functions here



#%%
# CodeWrite cell
# Write code here for generating the numbers that you report below.

#%% [markdown]
# TextWrite cell: Give your observations and the list of hyperparameter choices and train zero-one error  and test zero-one error, for all 4 datasets (2 real world and 2 synthetic).  
# 
# 

#%%
## Codewrite cell: Generate plots of learned decision tree classifier on dataset_A and datasset_B.
# Plots should give both the learned classifier and the train data. 
# Similar to  Bishop Figure 4.5 (with just two classes here.)
# Total number of plots = 2 

#%% [markdown]
# # 4 Random Forest classifier
# 
# Write code for learning RandomForests below. Fix the following hyper parameters: (Fraction of data to learn tree=0.5, Fraction of number of features taken per data=0.5).  Choose the number of trees to add in the forest by using a validation set. (You may use sklearn decision tree function, if you want)
# 
# Write code for running in the cell after the nest. (You may be asked to demonstrate your code during the via using this cell.)
# 
# In text cell after that report the following numbers you get by running appropriate code:
# 
# For each classification data set (A,B,C,D) report the best number of trees found. Report the training and test zero-one error for those hyperparameters.
# 
# For the synthetic classification datasets in 2-dimensions (datasets A,B), also illustrate the learned classifier. Do this in the last codeWrite cell for this question.

#%%
# CodeWrite cell
# Write Random Forest classifier assuming access to a decision tree learner, 
# write code for choosing best node size to stop splitting.
# write only functions here



#%%
# CodeWrite cell
# Write code here for generating the numbers that you report below.

#%% [markdown]
# TextWrite cell: Give your observations and the list of hyperparameter choices and train zero-one error  and test zero-one error, for all 4 datasets (2 real world and 2 synthetic).  
# 

#%%
## Codewrite cell: Generate plots of learned Random Forest classifier on dataset_A and datasset_B.
# Plots should give both the learned classifier and the train data. 
# Similar to  Bishop Figure 4.5 (with just two classes here.)
# Total number of plots = 2 


