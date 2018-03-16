#Homework 1
1. UC Irvine has a large repository for various kinds of data. In this problem, you are asked to use the [iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris) to perform the experiments. Implement the k-NN classifier for the classification task. To begin one experiment, randomly draw 70 % of the instances for training and the rest for testing. Repeat the drawing and the k-NN classification 10 times and compute the average accuracy. Then, plot the curve of k versus accuracy for k = 1, 3, ..., 15. For simplicity, use the Euclidean distance in your computation.

2. Following previous problem, if you do not have the test dataset (i.e., you have only the 70 % of dataset), how do you determine the optimal value of k? Use your own approach to find such a value and compare the results you have in problem 2.

3. In the class, we covered the naive Bayes classifier, but only with discrete-type features. Consult any paper to learn how to extend this approach to continuous-type features. Repeat the first problem with your algorithm. Compare the accuracy of naive Bayes classifier with the k-NN.

~Note: Iris dataset is provided as data.csv~