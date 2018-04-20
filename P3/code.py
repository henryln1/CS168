import random
import numpy as np

#PART 1

d = 100 # dimensions of data
n = 1000 # number of data points
X = np.random.normal(0,1, size=(n,d))
a_true = np.random.normal(0,1, size=(d,1))
y = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1)) #(1000, 1)
print("shape of y: ", y.shape)


learning_rates = [0.00005, 0.0005, 0.0007]
iterations = 20
#a = np.zeros(shape=(d, 1))


##print("shape of a: ", a.shape)
#print("shape of X: ", X.shape)
#print("shape of y: ", y.shape)

#print(a)

gd_a = {}
def gradientDescent(): #does it work or not? hmmmmmm
	for lr in learning_rates:
		a = np.zeros(shape=(d, 1))

		for i in range(iterations):
			totalGradient = 0.0
			for point in range(n): #loop through all the datapoints and calculate loss?
				curr = X[point].reshape((d, 1))
				totalGradient += 2 * curr * (np.matmul(a.T, curr) - y[point])

			a -= lr * totalGradient

		gd_a[lr] = a

	return
#gradientDescent()
#print(gd_a)

sgd_iterations = 1000

sgd_a = {}

sgd_lr = [0.0005, 0.005, 0.01]
def SGD(): #I think this is working?
	for lr in sgd_lr:
		a = np.zeros(shape=(d, 1))

		for i in range(sgd_iterations):
			random_point = random.randint(0, n - 1)
			curr = X[random_point].reshape((d, 1))
			gradient = 2 * curr * (np.matmul(a.T, curr) - y[random_point])

			a -= lr * gradient
		sgd_a[lr] = a
	return

SGD()
print(sgd_a)

#print(a)


#a = np.linalg.inv(np.matmul(X.T, X))
#a = np.matmul(a, X.T)
#a = np.matmul(a, y)
#print(a)



#PART 2

train_n = 100
test_n = 1000
d = 100
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))


#PART 3

train_n = 100
test_n = 10000
d = 200
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))