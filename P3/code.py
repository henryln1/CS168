import random
import numpy as np
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt

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

def part_a(zeros = False):
	a = solve_for_a(zeros)
	sqrd_error = obj_fn(a)
	return sqrd_error

def solve_for_a(zeros = False):
	if zeros:
		return np.zeros(shape=(d, 1))
	X_t = X.T
	X_t_X = np.matmul(X_t, X) # (XTX)
	inv = np.linalg.inv(X_t_X) # (XTX)^-1
	inv_X_t = np.matmul(inv, X_t) # (XTX)^-1 * XT
	a = np.matmul(inv_X_t, y) # (XTX)^-1 * XTy
	return a

def obj_fn(a):
	sqrd_error = 0
	a_t = a.T
	for i in range(n):
		curr = X[i].reshape((d, 1))
		error = a_t.dot(curr) - y[i]
		sqrd_error += np.power(error, 2)
	return sqrd_error[0]

# print("1A")
# print("part a value: ", part_a())
# print("part a w 0s: ", part_a(True))
#normal a: 220.34264027, a w all 0s: 91827.65959497

gd_a = defaultdict(list)
def gradientDescent(): #does it work or not? hmmmmmm
	for lr in learning_rates:
		a = np.zeros(shape=(d, 1))

		for i in range(iterations):
			totalGradient = 0.0
			for point in range(n): #loop through all the datapoints and calculate loss?
				curr = X[point].reshape((d, 1))
				totalGradient += 2 * curr * (a.T.dot(curr) - y[point])

			a -= lr * totalGradient
			loss = obj_fn(a)
			gd_a[lr].append(loss)

def makePlot(objectiveFnValues, lr, numIterations, separate, outputFileName):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.title("Objective fn Value vs Iteration #")
        # plt.axis([0, 1000, 0.5, 0.75])
        iterations = [i for i in range(1, numIterations + 1)]
        plt.plot(iterations, objectiveFnValues[lr[0]], 'r--', label=lr[0])
        plt.plot(iterations, objectiveFnValues[lr[1]], 'bs', label=lr[1])
        if not separate:
        	plt.plot(iterations, objectiveFnValues[lr[2]], 'g^', label=lr[2])
        plt.xlabel("Iteration #")
        plt.ylabel("Objective fn Value")
        plt.legend(shadow=True, fontsize='x-large', title="Step Sizes", loc = 0)
        plt.savefig(outputFileName + ".png", format = 'png')
        plt.close()
        if separate:
	        plt.title("Objective fn Value vs Iteration #")
	        plt.plot(iterations, objectiveFnValues[lr[2]], 'g^', label='0.0007')
	        plt.xlabel("Iteration #")
	        plt.ylabel("Objective fn Value")
	        plt.legend(shadow=True, fontsize='x-large', title="Step Sizes", loc = 0)
	        plt.savefig(outputFileName + "_2.png", format = 'png')
	        plt.close()

print("1B")
gradientDescent()
makePlot(gd_a, learning_rates, iterations, True, "1b")
for lr in learning_rates:
	print(lr, " final value: ", gd_a[lr][iterations - 1]) 	# 2092.7466054, 233.09586745, 6.11484504+08 



sgd_iterations = 1000

sgd_a = defaultdict(list)

sgd_lr = [0.0005, 0.005, 0.01]
def SGD(): #I think this is working?
	for lr in sgd_lr:
		a = np.zeros(shape=(d, 1))

		for i in range(sgd_iterations):
			random_point = random.randint(0, n - 1)
			curr = X[random_point].reshape((d, 1))
			gradient = 2 * curr * (a.T.dot(curr) - y[random_point])

			a -= lr * gradient
			loss = obj_fn(a)
			sgd_a[lr].append(loss)

print("1C")
SGD()
makePlot(sgd_a, sgd_lr, sgd_iterations, False, "1c") 
for lr in sgd_lr:
	print(lr, " final value: ", sgd_a[lr][sgd_iterations - 1])



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