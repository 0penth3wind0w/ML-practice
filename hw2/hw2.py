import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

input_data = pd.read_csv('data.csv',header=None)

# Problem 1
# Covariance matrix: setosa
setosa_X = input_data.iloc[:50,0:4].values
setosa_Y = input_data.iloc[:50,4].values

setosa_X_first_total = 0
setosa_X_second_total = 0
setosa_X_third_total = 0
setosa_X_fourth_total = 0
for data in setosa_X:
	setosa_X_first_total = setosa_X_first_total + data[0]
	setosa_X_second_total = setosa_X_second_total + data[1]
	setosa_X_third_total = setosa_X_third_total + data[2]
	setosa_X_fourth_total = setosa_X_fourth_total + data[3]

setosa_X_mean = np.array([setosa_X_first_total/50,setosa_X_second_total/50,setosa_X_third_total/50,setosa_X_fourth_total/50])
setosa_X_array = np.array(([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]))
for each in setosa_X:
	tmp1 = each-setosa_X_mean
	tmp2 = each-setosa_X_mean
	setosa_X_array = setosa_X_array + np.transpose(np.mat(tmp1)).dot(np.mat(tmp2))/50
	
# Covariance matrix: versicolor
versicolor_X = input_data.iloc[50:100,0:4].values
versicolor_Y = input_data.iloc[50:100,4].values

versicolor_X_first_total = 0
versicolor_X_second_total = 0
versicolor_X_third_total = 0
versicolor_X_fourth_total = 0
for data in versicolor_X:
	versicolor_X_first_total = versicolor_X_first_total + data[0]
	versicolor_X_second_total = versicolor_X_second_total + data[1]
	versicolor_X_third_total = versicolor_X_third_total + data[2]
	versicolor_X_fourth_total = versicolor_X_fourth_total + data[3]

versicolor_X_mean = np.array([versicolor_X_first_total/50,versicolor_X_second_total/50,versicolor_X_third_total/50,versicolor_X_fourth_total/50])
versicolor_X_array = np.array(([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]))
for each in versicolor_X:
	tmp5 = each-versicolor_X_mean
	tmp6 = each-versicolor_X_mean
	versicolor_X_array = versicolor_X_array + np.transpose(np.mat(tmp5)).dot(np.mat(tmp6))/50

# Covariance matrix: virginica
virginica_X = input_data.iloc[100:150,0:4].values
virginica_Y = input_data.iloc[100:150,4].values

virginica_X_first_total = 0
virginica_X_second_total = 0
virginica_X_third_total = 0
virginica_X_fourth_total = 0
for data in virginica_X:
	virginica_X_first_total = virginica_X_first_total + data[0]
	virginica_X_second_total = virginica_X_second_total + data[1]
	virginica_X_third_total = virginica_X_third_total + data[2]
	virginica_X_fourth_total = virginica_X_fourth_total + data[3]

virginica_X_mean = np.array([virginica_X_first_total/50,virginica_X_second_total/50,virginica_X_third_total/50,virginica_X_fourth_total/50])
virginica_X_array = np.array(([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]))
for each in virginica_X:
	tmp3 = each-virginica_X_mean
	tmp4 = each-virginica_X_mean
	virginica_X_array = virginica_X_array + np.transpose(np.mat(tmp3)).dot(np.mat(tmp4))/50

# eigen value
setosa_i = LA.eigvals(setosa_X_array)
versicolor_i =LA.eigvals(versicolor_X_array)
virginica_i =LA.eigvals(virginica_X_array)

# kappa
kappa_setosa = np.absolute(max(setosa_i)/min(setosa_i))
kappa_versicolor = np.absolute(max(versicolor_i)/min(versicolor_i))
kappa_virginica = np.absolute(max(virginica_i)/min(virginica_i))

# Result of problem 1
print(setosa_X_array)
print(versicolor_X_array)
print(virginica_X_array)

print(setosa_i)
print(versicolor_i)
print(virginica_i)

print(kappa_setosa)
print(kappa_versicolor)
print(kappa_virginica)

# Problem 2
results = []
for i in range(10):
	X_setosa_train, X_setosa_test, Y_setosa_train, Y_setosa_test = train_test_split(setosa_X, setosa_Y, train_size=0.7)
	X_versicolor_train, X_versicolor_test, Y_versicolor_train, Y_versicolor_test = train_test_split(versicolor_X, versicolor_Y, train_size=0.7)
	X_virginica_train, X_virginica_test, Y_virginica_train, Y_virginica_test = train_test_split(virginica_X, virginica_Y, train_size=0.7)
	X_train = np.concatenate((X_setosa_train, X_versicolor_train,X_virginica_train), axis=0)
	X_test = np.concatenate((X_setosa_test, X_versicolor_test,X_virginica_test), axis=0)
	Y_train = np.concatenate((Y_setosa_train, Y_versicolor_train,Y_virginica_train), axis=0)
	Y_test = np.concatenate((Y_setosa_test, Y_versicolor_test,Y_virginica_test), axis=0)
	pca = PCA(n_components=2)
	X_tra_train = pca.fit_transform(X_train)
	X_tra_test = pca.transform(X_test)
	knn = KNeighborsClassifier(n_neighbors=3, p=2, metric="minkowski")
	knn.fit(X_tra_train,Y_train)
	predict = knn.predict(X_tra_test)
	results.append(accuracy_score(Y_test, predict))

# average accuracy
print(np.mean(results))
# variance
print(np.var(results))

# Problem 3: FA
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(np.concatenate((setosa_X, versicolor_X, virginica_X)))

# setosa
X_set_fa = X_fa[:50,0:2]

set_X_first_total = 0
set_X_second_total = 0
for data in X_set_fa:
	set_X_first_total = set_X_first_total + data[0]
	set_X_second_total = set_X_second_total + data[1]

set_X_mean = np.array([set_X_first_total/50,set_X_second_total/50])
set_X_array = np.array(([0,0],[0,0]))
for each in X_set_fa:
	ftmp1 = each-set_X_mean
	ftmp2 = each-set_X_mean
	set_X_array = set_X_array + (np.transpose(np.mat(ftmp1)).dot(np.mat(ftmp2)))/50

# versicolor
fa = FactorAnalysis(n_components=2)

X_ver_fa = X_fa[50:100,0:2]

ver_X_first_total = 0
ver_X_second_total = 0
for data in X_ver_fa:
	ver_X_first_total = ver_X_first_total + data[0]
	ver_X_second_total = ver_X_second_total + data[1]

ver_X_mean = np.array([ver_X_first_total/50,ver_X_second_total/50])
ver_X_array = np.array(([0,0],[0,0]))
for each in X_ver_fa:
	ftmp3 = each-ver_X_mean
	ftmp4 = each-ver_X_mean
	ver_X_array = ver_X_array + (np.transpose(np.mat(ftmp3)).dot(np.mat(ftmp4)))/50

# virginica
fa = FactorAnalysis(n_components=2)

X_vir_fa = X_fa[100:150,0:2]

vir_X_first_total = 0
vir_X_second_total = 0
for data in X_vir_fa:
	vir_X_first_total = vir_X_first_total + data[0]
	vir_X_second_total = vir_X_second_total + data[1]

vir_X_mean = np.array([vir_X_first_total/50,vir_X_second_total/50])
vir_X_array = np.array(([0,0],[0,0]))
for each in X_vir_fa:
	ftmp5 = each-vir_X_mean
	ftmp6 = each-vir_X_mean
	vir_X_array = vir_X_array + (np.transpose(np.mat(ftmp5)).dot(np.mat(ftmp6)))/50

# eigen value
set_i = LA.eigvals(set_X_array)
ver_i =LA.eigvals(ver_X_array)
vir_i =LA.eigvals(vir_X_array)
# kappa
kappa_set = np.absolute(max(set_i)/min(set_i))
kappa_ver = np.absolute(max(ver_i)/min(ver_i))
kappa_vir = np.absolute(max(vir_i)/min(vir_i))

print(set_X_array)
print(ver_X_array)
print(vir_X_array)
print(set_i)
print(ver_i)
print(vir_i)
print(kappa_set)
print(kappa_ver)
print(kappa_vir)

# Problem 4: LDA
lda=LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(np.concatenate((setosa_X, versicolor_X, virginica_X)),np.concatenate((setosa_Y, versicolor_Y, virginica_Y)))

# setosa
X_set_lda = X_lda[:50,0:2]

set_X_fir_total = 0
set_X_sec_total = 0
for data in X_set_lda:
	set_X_fir_total = set_X_fir_total + data[0]
	set_X_sec_total = set_X_sec_total + data[1]

set_X_mea = np.array([set_X_fir_total/50,set_X_sec_total/50])
set_X_arr = np.array(([0,0],[0,0]))
for each in X_set_lda:
	ltmp1 = each-set_X_mea
	ltmp2 = each-set_X_mea
	set_X_arr = set_X_arr + (np.transpose(np.mat(ltmp1)).dot(np.mat(ltmp2)))/50

# versicolor
X_ver_lda = X_lda[50:100,0:2]

ver_X_fir_total = 0
ver_X_sec_total = 0
for data in X_ver_lda:
	ver_X_fir_total = ver_X_fir_total + data[0]
	ver_X_sec_total = ver_X_sec_total + data[1]

ver_X_mea = np.array([ver_X_fir_total/50,ver_X_sec_total/50])
ver_X_arr = np.array(([0,0],[0,0]))
for each in X_ver_lda:
	ltmp3 = each-ver_X_mea
	ltmp4 = each-ver_X_mea
	ver_X_arr = ver_X_arr + (np.transpose(np.mat(ltmp3)).dot(np.mat(ltmp4)))/50

# virginica
X_vir_lda = X_lda[100:150,0:2]

vir_X_fir_total = 0
vir_X_sec_total = 0
for data in X_vir_lda:
	vir_X_fir_total = vir_X_fir_total + data[0]
	vir_X_sec_total = vir_X_sec_total + data[1]

vir_X_mea = np.array([vir_X_fir_total/50,vir_X_sec_total/50])
vir_X_arr = np.array(([0,0],[0,0]))
for each in X_vir_lda:
	ltmp5 = each-vir_X_mea
	ltmp6 = each-vir_X_mea
	vir_X_arr = vir_X_arr + (np.transpose(np.mat(ltmp5)).dot(np.mat(ltmp6)))/50

# eigen value
set_li = LA.eigvals(set_X_arr)
ver_li =LA.eigvals(ver_X_arr)
vir_li =LA.eigvals(vir_X_arr)
# kappa
ka_set = np.absolute(max(set_li)/min(set_li))
ka_ver = np.absolute(max(ver_li)/min(ver_li))
ka_vir = np.absolute(max(vir_li)/min(vir_li))

print(set_X_arr)
print(ver_X_arr)
print(vir_X_arr)
print(set_li)
print(ver_li)
print(vir_li)
print(ka_set)
print(ka_ver)
print(ka_vir)