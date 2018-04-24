import random
import numpy as np
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import math

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

def normalized_error(x, a , y):
	numerator = np.matmul(x, a) - y
	numerator = math.sqrt(np.sum(np.square(numerator)))
	denominator = math.sqrt(np.sum(np.square(y)))
	return float(numerator) / denominator

def part_2a():
	a = solve_for_a_2()
	train_error = normalized_error(X_train, a, y_train) # idk if this is correct but i switched both to using the normalized error fn
	test_error = normalized_error(X_test, a, y_test)
	return train_error, test_error

def solve_for_a_2():
	inv = np.linalg.inv(X_train)
	a = np.matmul(inv, y_train) 
	return a

# def train_fn(a):
# 	sqrd_error = 0
# 	a_t = a.T
# 	for i in range(n):
# 		curr = X_train[i].reshape((d, 1))
# 		error = a_t.dot(curr) - y_train[i]
# 		sqrd_error += np.power(error, 2)
# 	return sqrd_error

# def test_fn(a):
# 	error = 0
# 	for i in range(n):
# 		curr = X_test[i].reshape((d, 1))
# 		numerator = np.linalg.norm(np.matmul(X_test, a) - y_test)
# 		denom = np.linalg.norm(y_test)
# 		error += numerator/denom
# 	return error

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

# Avg train err:  1.08482770211125681e-14
# Avg test err:  0.5823632339833625

#BBBBBBBBBB

def solve_for_a_2b(x, y, lmda):
	temp = np.matmul(x.T, x) + lmda * np.identity(d)
	temp = np.linalg.inv(temp)
	a = np.matmul(temp, x.T)
	a = np.matmul(a, y)
	return a

# def regularized_l2_objective(x, y, lmda):
# 	summation = 0.0
# 	a = solve_for_a_2b(x, y, lmda)
# 	for i in range(x.shape[0]): #loop through all the datapoints
# 		summation += (np.matmul(a.T, x[i]) - y[i]) ** 2
# 	regularization = lmda * np.sum(np.square(a))
# 	return summation + regularization

lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]

lambda_train = []
lambda_test = []

def part_b():
	num_trials = 10
	for l in lambdas:
		training_error = 0.0
		test_error = 0.0
		a = solve_for_a_2b(X_train, y_train, l)
		for iteration in range(num_trials):
			#print("Current Lambda: ", l)
			print("Iteration: ", iteration)
			train_error_curr = normalized_error(X_train, a, y_train)
			test_error_curr = normalized_error(X_test, a, y_test)
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


# def partc_plot(lambdas, training, test, outputFileName): #Dunno how he wants this plotted
# 	with warnings.catch_warnings(): # can't see the triangles on the graph - should we move 0.005 to its own graph
# 		warnings.simplefilter("ignore")
# 		plt.title("Training and Test Errors each Iteration for Different Step Sizes")
# 		iterations = [x for x in range(10)]
# 		plt.plot(iterations, training[lambdas[0]], 'rs', label = "Training 0.00005")
# 		plt.plot(iterations, test[lambdas[0]], 'bs', label = "Test 0.00005")
# 		plt.plot(iterations, training[lambdas[1]], 'r--', label = "Training 0.0005")
# 		plt.plot(iterations, test[lambdas[1]], 'b--', label = "Test 0.0005")
# 		plt.plot(iterations, training[lambdas[2]], 'r^ ', label = "Training 0.005")
# 		plt.plot(iterations, test[lambdas[2]], 'b^', label = "Test 0.005")
# 		plt.xlabel("Iterations")
# 		plt.ylabel("Error")
# 		plt.legend(shadow=True, fontsize='x-large', loc = 0)
# 		plt.savefig(outputFileName + ".png", format = 'png')
# 		plt.close()	

#SGD_2()

#Step size:  0.005
# Average Training Error:  0.945414598553814
# Average Test Error:  0.9704421367327896

# Step size:  0.0005
# Average Training Error:  0.9454145985538138
# Average Test Error:  0.9704421367327896

# Step size:  5e-05
# Average Training Error:  0.9454145985538143
# Average Test Error:  0.9704421367327909

# partc_plot(step_sizes, training_sgd2_error, test_sgd2_error, "2c") 
# print("Error for true coefficient vector")
# print("training: ", normalized_error(X_train, a_true, y_train)) # 0.04064638868719073
# print("test: ", normalized_error(X_test, a_true, y_test)) # 0.04741816119303346
#this thing takes so long to run, idk if we can speed it up or its just like that





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

l2_norms = {
	0.00005: [],
	0.005: []
}

def SGD_3():
	for step in step_sizes:
		print("Current Step: ", step)
		a = np.zeros(shape=(d, 1))
		for i in range(total_iterations):
			if i % 100000 == 0:
				print("Current Iteration: ", i)
			random_point = random.randint(0, n - 1)
			curr_train = X_train[random_point].reshape((d, 1))
			gradient = 2 * curr_train * (a.T.dot(curr_train) - y_train[random_point])

			a -= step * gradient
			curr_training_error = normalized_error(X_train, a, y_train)
			error_training_each_iteration[step].append(curr_training_error)
			if i % 100 == 0:
				curr_test_error = normalized_error(X_test, a, y_test)
				error_test_each100_iteration[step].append(curr_test_error)
			l2_norms[step].append(np.linalg.norm(a))


def plot_2d(outputFileName, y_dict, x_axis, title, y_label = "Error"):
	with warnings.catch_warnings(): 
		plt.title(title)
		plt.plot(x_axis, y_dict[step_sizes[0]], 'rs', label = "0.00005")
		plt.plot(x_axis, y_dict[step_sizes[1]], 'bs', label = "0.005")
		plt.xlabel("Iterations")
		plt.ylabel("Error")
		plt.legend(shadow=True, fontsize='x-large', loc = 0)
		plt.savefig(outputFileName + ".png", format = 'png')
		plt.close()	

# NEEDS TO BE RUN
SGD_3()
plot_2d("2d_1_1mill", error_training_each_iteration, [x for x in range(total_iterations)], "Training Error vs Iteration Number")
plot_2d("2d_2_1mill", error_test_each100_iteration, [x * 100 for x in range(1, 10001)], "Test Error vs Iteration Number")
plot_2d("2d_3_1mill", l2_norms, [x for x in range(total_iterations)], "SGD Solution l2 Norm vs Iteration Number", "l2 norm of SGD Solution")


#EEEEEEEEEEEE
step_size = 0.00005
radius_opts = [0, 0.1, 0.5, 1, 10, 20, 30]
training_errors = []
test_errors = []

def SGD_4():
	for radius in radius_opts:
		print("Radius: ", radius)
		a = np.random.uniform(size=(d, 1)) * radius
		total_train_err = 0.0
		total_test_err = 0.0
		for i in range(total_iterations):
			if i % 100000 == 0:
				print("Current Iteration: ", i)
			random_point = random.randint(0, n - 1)
			curr_train = X_train[random_point].reshape((d, 1))
			gradient = 2 * curr_train * (a.T.dot(curr_train) - y_train[random_point])

			a -= step_size * gradient
			curr_training_error = normalized_error(X_train, a, y_train)
			total_train_err += curr_training_error
			curr_test_error = normalized_error(X_test, a, y_test)
			total_test_err += curr_test_error
		training_errors.append(total_train_err/total_iterations)
		test_errors.append(total_test_err/total_iterations)

def plot_2e():
	with warnings.catch_warnings(): 
		plt.title("Average errors vs r")
		plt.plot(radius_opts, training_errors, 'rs', label = "Training Error")
		plt.plot(radius_opts, test_errors, 'bs', label = "Test Error")
		plt.xlabel("r")
		plt.ylabel("Error")
		plt.legend(shadow=True, fontsize='x-large', loc = 0)
		plt.savefig("2e.png", format = 'png')
		plt.close()		

# SGD_4()
# plot_2e()


#PART 3

train_n = 100
test_n = 10000
d = 200
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))




