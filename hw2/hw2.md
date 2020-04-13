# Homework 2
1. We mentioned that the covariance matrix may be ill-conditioned. Find the (sample) covariance matrices for the three classes of the Iris dataset and compute the condition numbers for the covariance matrices. For simplicity, use the *λ* following as the condition number: κ(A) = | *λ_max* / *λ_min* | , where *λ_max* and *λ_min* are the largest and smallest eigenvalues of matrix A  

2. In this problem, you are asked to use the Iris dataset to perform PCA dimensionality reduction before classification. Randomly draw 35 samples in
each class to find the vectors w (j) for the largest two principal components.  
Recall that PCA is unsupervised; therefore, you need to use 35×3 = 105 data points to find the parameters of the PCA.  
Implement the 3-NN classifier to test the rest 15 samples in each class and record the accuracy. Repeat the drawing and the k-NN classification 10 times and compute the average accuracy and variance.  
For simplicity, use the Euclidean distance in the k-NN computation.  

3. Following the general steps of problem 1, but use the FA approach for dimensionality reduction. For simplicity, you may assume *ψ* = 0 and use the LS solutions.  


4. Repeat problem 1 by using LDA as the reduction method. Remember to compute the parameters for each class in order to use LDA.  