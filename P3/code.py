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
#print("shape of y: ", y.shape)


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

# print("1B")
# gradientDescent()
# makePlot(gd_a, learning_rates, iterations, True, "1b")
# for lr in learning_rates:
# 	print(lr, " final value: ", gd_a[lr][iterations - 1]) 	# 2092.7466054, 233.09586745, 6.11484504+08 



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

# print("1C")
# SGD()
# makePlot(sgd_a, sgd_lr, sgd_iterations, False, "1c") 
# for lr in sgd_lr:
# 	print(lr, " final value: ", sgd_a[lr][sgd_iterations - 1])



#PART 2

train_n = 100
test_n = 1000
d = 100
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
num_trials = 10
#import math

def part_2a():
	a = solve_for_a_2()
	train_error = train_fn(a)
	test_error = test_fn(a)
	return train_error, test_error

def solve_for_a_2():
	inv = np.linalg.inv(X_train)
	a = np.matmul(inv, y_train) 
	return a

def train_fn(a):
	sqrd_error = 0
	a_t = a.T
	for i in range(n):
		curr = X_train[i].reshape((d, 1))
		error = a_t.dot(curr) - y_train[i]
		sqrd_error += np.power(error, 2)
	return sqrd_error

def test_fn(a):
	error = 0
	for i in range(n):
		curr = X_test[i].reshape((d, 1))
		numerator = np.linalg.norm(np.matmul(X_test, a) - y_test)
		denom = np.linalg.norm(y_test)
		error += numerator/denom
	return error

train_err = 0
test_err = 0
for n in range(num_trials):
	curr_train_err, curr_test_err = part_2a()
	train_err += curr_train_err
	test_err += curr_test_err

train_err /= num_trials
test_err /= num_trials

# print("2A")
# print("Avg train err: ", train_err)
# print("Avg test err: ", test_err)

#Avg train err:  [[12951.94950903]]
#Avg test err:  28.456756214230317

#BBBBBBBBBB

def solve_for_a_2b(x, y, lmda):
	temp = np.matmul(x.T, x) + lmda * np.identity(d)
	temp = np.linalg.inv(temp)
	a = np.matmul(temp, x.T)
	a = np.matmul(a, y)
	return a

def regularized_l2_objective(x, y, lmda):
	summation = 0.0
	a = solve_for_a_2b(x, y, lmda)
	for i in range(x.shape[0]): #loop through all the datapoints
		summation += (np.matmul(a.T, x[i]) - y[i]) ** 2
	regularization = lmda * np.sum(np.square(a))
	return summation + regularization

lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]

lambda_train = []
lambda_test = []

def part_b():
	num_trials = 10
	for l in lambdas:
		training_error = 0.0
		test_error = 0.0
		for iteration in range(num_trials):

			#print("Current Lambda: ", l)
			train_error_curr = regularized_l2_objective(X_train, y_train, l)
			test_error_curr = regularized_l2_objective(X_test, y_test, l)
			#print("Training Error: ", train_error)
			#print("Testing Error: ", test_error)
			training_error += train_error_curr
			test_error += test_error_curr
		print("Current Lambda: ", l)
		print("Training Error: ", training_error/num_trials)
		print("Testing Error: ", test_error/num_trials)
		lambda_train.append(training_error/num_trials)
		lambda_test.append(test_error/num_trials)
	return


import math 

def partb_plot(lambdas, train, test, outputFileName):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		plt.title("Training and Test Errors vs. Log of Lambda Values")
		lambdas = [math.log(x) for x in lambdas]
		plt.plot(lambdas, train, 'r--', label = "Training")
		plt.plot(lambdas, test, 'bs', label = "Test")
		plt.xlabel("Log Lambda")
		plt.ylabel("Error")
		plt.legend(shadow=True, fontsize='x-large', loc = 0)
		plt.savefig(outputFileName + ".png", format = 'png')
		plt.close()


# part_b()
# partb_plot(lambdas, lambda_train, lambda_test, "2b")

#CCCCC
num_iterations = 1000000
step_sizes = [0.00005, 0.0005, 0.005]
num_trials = 10

#step_sizes_dict = {}

def normalized_error(x, a , y):
	numerator = np.matmul(x, a) - y
	numerator = math.sqrt(np.sum(np.square(numerator)))
	denominator = math.sqrt(np.sum(np.square(y)))
	return float(numerator) / denominator

training_sgd2_error = {
	0.00005: [],
	0.0005: [],
	0.005: []
}
test_sgd2_error = {
	0.00005: [],
	0.0005: [],
	0.005: []
}
def SGD_2(): #does for train and test simultaneously
	for step in step_sizes:
		#print(step)
		training_error = 0.0
		test_error = 0.0
		for trial in range(num_trials):
			print("Trial ", trial)
			a = np.zeros(shape=(d, 1))

			for i in range(num_iterations):
				random_point = random.randint(0, n - 1)
				curr_train = X_train[random_point].reshape((d, 1))
				gradient = 2 * curr_train * (a.T.dot(curr_train) - y_train[random_point])

				a -= step * gradient
				#loss = obj_fn(a)
				#sgd_a[lr].append(loss)
			curr_training_error = normalized_error(X_train, a, y_train)
			curr_test_error = normalized_error(X_test, a, y_test)
			training_error += curr_training_error
			test_error += curr_test_error
			training_sgd2_error[step].append(curr_training_error)
			#print(len(training_sgd2_error[step]))
			test_sgd2_error[step].append(curr_test_error)

		print("Step size: ", step)
		print("Average Training Error: ", training_error/ num_trials)
		print("Average Test Error: ", test_error / num_trials)

#SGD_2()

def partc_plot(lambdas, training, test, outputFileName): #Dunno how he wants this plotted
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		plt.title("Training and Test Errors each Iteration for Different Step Sizes")
		iterations = [x for x in range(10)]
		plt.plot(iterations, training[lambdas[0]], 'rs', label = "Training 0.00005")
		plt.plot(iterations, test[lambdas[0]], 'bs', label = "Test 0.00005")
		plt.plot(iterations, training[lambdas[1]], 'r--', label = "Training 0.0005")
		plt.plot(iterations, test[lambdas[1]], 'b--', label = "Test 0.0005")
		plt.plot(iterations, training[lambdas[2]], 'r^ ', label = "Training 0.005")
		plt.plot(iterations, test[lambdas[2]], 'b^', label = "Test 0.005")
		plt.xlabel("Iterations")
		plt.ylabel("Error")
		plt.legend(shadow=True, fontsize='x-large', loc = 0)
		plt.savefig(outputFileName + ".png", format = 'png')
		plt.close()	

SGD_2()
partc_plot(step_sizes, training_sgd2_error, test_sgd2_error, "2c") 
#this thing takes so long to run, idk if we can speed it up or its just like that

#C NOT DONE YET, should we do part about computing error corresponding to true coefficient vector f(a*)





#DDDDDDDDDD

total_iterations = 1000000
step_sizes = [0.00005, 0.005]

error_training_each_iteration = {
	0.00005: [],
	0.005: []
}

error_test_each100_iteration = {
	0.00005: [],
	0.005: []
}

def SGD_3():
	for step in step_sizes:

		a = np.zeros(shape=(d, 1))
		for i in range(total_iterations):
			random_point = random.randint(0, n - 1)
			curr_train = X_train[random_point].reshape((d, 1))
			gradient = 2 * curr_train * (a.T.dot(curr_train) - y_train[random_point])

			a -= step * gradient
			curr_training_error = normalized_error(X_train, a, y_train)
			error_training_each_iteration[step].append(curr_training_error)
			if i % 100 == 0:
				curr_test_error = normalized_error(X_test, a, y_test)
				error_test_each100_iteration[step].append(curr_test_error)
	#NOT FINISHED





#PART 3

train_n = 100
test_n = 10000
d = 200
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))