import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

input_data = pd.read_csv('data.csv')

def classify(data, model, result_train, result_test):
	X = data.iloc[:,0:4].values
	Y = data.iloc[:,4].values
	sum_acc_train = 0.0
	sum_acc_test = 0.0
	iterate = 1
	while iterate <= 10:
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
		model.fit(X_train, Y_train)
		train_predict = model.predict(X_train)
		# prediction
		test_predict = model.predict(X_test)
		sum_acc_train += accuracy_score(Y_train, train_predict)
		sum_acc_test += accuracy_score(Y_test, test_predict)
		iterate += 1
	result_train.append(sum_acc_train/10)
	result_test.append(sum_acc_test/10)
	
#K-NN and K-NN with less data
k_avgacc_train = []
k_avgacc_test = []
kless_avgacc_train = []
kless_avgacc_test = []
neighbors = [1, 3, 5, 7, 9, 11, 13, 15]
for nbr in neighbors:
	knn = KNeighborsClassifier(n_neighbors=nbr, p=2, metric="minkowski")
	classify(input_data, knn, k_avgacc_train, k_avgacc_test)
	classify(input_data.sample(frac=0.7, random_state=99), knn, kless_avgacc_train, kless_avgacc_test)

# Naive Bayes (Gaussian distribution)
nb_avgacc_train = []
nb_avgacc_test = []
gnb = GaussianNB()
classify(input_data, gnb, nb_avgacc_train, nb_avgacc_test)
itr = 1
while itr <= 7:
	nb_avgacc_train.append(nb_avgacc_train[0])
	nb_avgacc_test.append(nb_avgacc_test[0])
	itr += 1

plt.plot(range(1, 16, 2), k_avgacc_train, 'or-', label = 'Train acc.')
plt.plot(range(1, 16, 2), k_avgacc_test, 'ob-', label = 'Test acc.')
plt.plot(range(1, 16, 2), kless_avgacc_train, 'sc--', label = 'Less Train acc.')
plt.plot(range(1, 16, 2), kless_avgacc_test, 'sm--', label = 'Less Test acc.')
plt.plot(range(1, 16, 2), nb_avgacc_test, 'g-', label = 'N.Bayes Test Avg. Acc.')
plt.title('Accuracy of K-NN and Naive Bayes')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
