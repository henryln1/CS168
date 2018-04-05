import numpy as np 
import matplotlib.pyplot as plt

def plot_histogram(bins, filename = None):
	"""
	This function wraps a number of hairy matplotlib calls to smooth the plotting 
	part of this assignment.

	Inputs:
	- bins: 	numpy array of shape max_bin_population X num_strategies numpy array. For this 
				assignment this must be 200000 X 4. 
				WATCH YOUR INDEXING! The element bins[i,j] represents the number of times the most 
				populated bin has i+1 balls for strategy j+1. 
	
	- filename: Optional argument, if set 'filename'.png will be saved to the current 
				directory. THIS WILL OVERWRITE 'filename'.png
	"""
	assert bins.shape == (200000,4), "Input bins must be a numpy array of shape (max_bin_population, num_strategies)"
	assert np.array_equal(np.sum(bins, axis = 0),(np.array([30,30,30,30]))), "There must be 30 runs for each strategy"

	thresh =  max(np.nonzero(bins)[0])+3
	n_bins = thresh
	bins = bins[:thresh,:]
	print("\nPLOTTING: Removed empty tail. Only the first non-zero bins will be plotted\n")

	ind = np.arange(n_bins) 
	width = 1.0/6.0

	fig, ax = plt.subplots()
	rects_strat_1 = ax.bar(ind + width, bins[:,0], width, color='yellow')
	rects_strat_2 = ax.bar(ind + width*2, bins[:,1], width, color='orange')
	rects_strat_3 = ax.bar(ind + width*3, bins[:,2], width, color='red')
	rects_strat_4 = ax.bar(ind + width*4, bins[:,3], width, color='k')

	ax.set_ylabel('Number Occurrences in 30 Runs')
	ax.set_xlabel('Number of Balls In The Most Populated Bin')
	ax.set_title('Histogram: Load on Most Populated Bin For Each Strategy')

	ax.set_xticks(ind)
	ax.set_xticks(ind+width*3, minor = True)
	ax.set_xticklabels([str(i+1) for i in range(0,n_bins)], minor = True)
	ax.tick_params(axis=u'x', which=u'minor',length=0)

	ax.legend((rects_strat_1[0], rects_strat_2[0], rects_strat_3[0], rects_strat_4[0]), ('Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'))
	plt.setp(ax.get_xmajorticklabels(), visible=False)
	
	if filename is not None: plt.savefig(filename+'.png', bbox_inches='tight')

	plt.show()


import random

# def oneRandomBin(N):
# 	bins = [0 for x in range(N)]

# 	for ball in range(N):
# 		bins[random.randint(0, N - 1)] += 1

# 	print(bins)
# 	return max(bins)

def oneRandomBin(N):



	bins1 = [0 for x in range(N)]

	for ball in range(N):
		bins1[random.randint(0, N - 1)] += 1

	bins1Array = np.array(bins1)
	print(sum(bins1))
	return max(bins1Array)

def twoRandomBin(N):

	bins2 = [0 for x in range(N)]

	for ball in range(N):
		choice1Index = random.randint(0, N - 1)
		choice2Index = random.randint(0, N - 1)

		if bins2[choice1Index] > bins2[choice2Index]:
			bins2[choice2Index] += 1
		elif bins2[choice1Index] < bins2[choice2Index]:
			bins2[choice1Index] += 1
		else:
			bins2[random.choice([choice1Index, choice2Index])] += 1

	bins2Array = np.array(bins2)
	print(sum(bins2))
	return max(bins2Array)

def threeRandomBin(N):

	bins3 = [0 for x in range(N)]
	for ball in range(N):
		choice1Index = random.randint(0, N - 1)
		choice2Index = random.randint(0, N - 1)
		choice3Index = random.randint(0, N - 1)

		#print(choice3Index)
		if bins3[choice1Index] < bins3[choice2Index] and bins3[choice1Index] < bins3[choice2Index]:
			bins3[choice1Index] += 1
		elif bins3[choice2Index] < bins3[choice1Index] and bins3[choice2Index] < bins3[choice3Index]:
			bins3[choice2Index] += 1
		elif bins3[choice3Index] < bins3[choice1Index] and bins3[choice3Index] < bins3[choice2Index]:
			bins3[choice3Index] += 1


		elif bins3[choice1Index] < bins3[choice3Index] and bins3[choice2Index] < bins3[choice3Index] and bins3[choice1Index] == bins3[choice2Index]:
			bins3[random.choice([choice1Index, choice2Index])] += 1 #choose randomly between 1 and 2
		elif bins3[choice3Index] < bins3[choice1Index] and bins3[choice2Index] < bins3[choice1Index] and bins3[choice2Index] == bins3[choice3Index]: # 2 = 3 and 2 < 1 and 3 < 1

			bins3[random.choice([choice3Index, choice2Index])] += 1 #choose randomly between 2 and 3
		elif bins3[choice1Index] < bins3[choice2Index] and bins3[choice3Index] < bins3[choice2Index] and bins3[choice1Index] == bins3[choice3Index]:
			bins3[random.choice([choice3Index, choice1Index])] += 1 #choose randomly between 1 and 3

		elif bins3[choice1Index] == bins3[choice2Index] and bins3[choice1Index] == bins3[choice2Index]:
			bins3[random.choice([choice1Index, choice2Index, choice3Index])] += 1

	print(len(bins3))
	print(sum(bins3))

	bins3Array = np.array(bins3)
	return max(bins3Array)

def halfRandomBin(N):
	bins4 = [0 for x in range(N)]


	for ball in range(N):
		firstHalfChoiceIndex = random.randint(0, N/2 - 1)
		secondHalfChoiceIndex = random.randint(N/2, N - 1)
		if bins4[firstHalfChoiceIndex] <= bins4[secondHalfChoiceIndex]:
			bins4[firstHalfChoiceIndex] += 1
		else:
			bins4[secondHalfChoiceIndex] += 1

	bins4 = np.array(bins4)

	print(sum(bins4))
	return max(bins4)


def trialTime(N, numberOfRuns, numberOfStrategies):
	stacked = np.zeros((N, numberOfStrategies))

	for i in range(numberOfRuns):
		first = oneRandomBin(N)
		stacked[first - 1, 0] += 1

		second = twoRandomBin(N)
		stacked[second - 1, 1] += 1

		third = threeRandomBin(N)
		stacked[third - 1, 2] += 1

		fourth = halfRandomBin(N)
		stacked[fourth - 1, 3] += 1

	plot_histogram(stacked, "test30")

#trialTime(200000, 30, 4)





#QUESTION 2 CODE

import hashlib
import random as r

def dataStream(): 

	numberBuckets = 256
	numTables = 4
	numberTrials = 10

	array = np.zeros((numTables, numberBuckets))
	print("shape of array: ", array.shape)

	def increment(x, hashV):
		for j in range(numTables):
			array[j][hashV] += 1

	def count(trial, x):
		stringForm = str(x) + str(trial - 1)
		MD5Score = hashlib.md5(stringForm.encode("utf-8")).hexdigest()

		minValue = float('inf')
		for j in range(numTables):
			hashHex = MD5Score[:2]
			MD5Score = MD5Score[2:]
			hashValue = int(hashHex, 16)
			if array[j][hashValue] < minValue:
				minValue = array[j][hashValue]
		return minValue


	def countMinSketch(trial, x):

		stringForm = str(x) + str(trial - 1)
		MD5Score = hashlib.md5(stringForm.encode("utf-8")).hexdigest()

		for j in range(numTables):

			hashHex = MD5Score[:2]
			MD5Score = MD5Score[2:]
			hashValue = int(hashHex, 16)
			increment(x, hashValue)


	# forward = []
	# for i in range(1, 10):
	# 	lowerBound = 1000 * (i - 1) + 1
	# 	upperBound = 1000 * i
	# 	integer = r.randint(lowerBound, upperBound)
	# 	for j in range(i):
	# 		forward.append(integer)


	forward = []
	for i in range(1, 10):
		lowerBound = 1000 * (i - 1) + 1
		upperBound = 1000 * i
		for j in range(lowerBound, upperBound + 1):
			forward += [j] * i

	#9001 to 9050
	for i in range(1, 51):
		for j in range(i ** 2):
			forward.append(9000 + i)
	print("Number of elements total: ", len(forward))

	reverse = list(reversed(forward))
	randomized = forward[:]
	random.shuffle(randomized)
	def findHeavyHitters(trial):
		heavy = []
		for val in randomized:
			if count(trial, val) >= len(reverse) * 0.01:
				heavy.append(val)
		return list(set(heavy))

	numberHeavyHitters = 0
	freq9050 = 0
	for i in range(1, numberTrials + 1):

		for value in randomized:
			countMinSketch(i, value)
		freq9050 += count(i, max(reverse))
		#print("freq9050: ", freq9050)
		#print("heavy hitters", findHeavyHitters(i))
		print(len(findHeavyHitters(i)))
		numberHeavyHitters += len(findHeavyHitters(i))
		array = np.zeros((numTables, numberBuckets)) #reset array for next trial

	print("Average number of heavy hitters average per trial: ", numberHeavyHitters/numberTrials)
	print("Average frequency of element 9050 average per trial: ", freq9050/numberTrials)

#print("forward")
#dataStream()


#RESULTS
#forward
# Average number of heavy hitters average per trial:  403.6
# Average frequency of element 9050 average per trial:  3215.5

#reverse
# Average number of heavy hitters average per trial:  403.6
# Average frequency of element 9050 average per trial:  3215.5

# random
# Average number of heavy hitters average per trial:  403.6
# Average frequency of element 9050 average per trial:  3215.5




def conservative(): 
	print("conservative")

	numberBuckets = 256
	numTables = 4
	numberTrials = 10

	array = np.zeros((numTables, numberBuckets))
	print("shape of array: ", array.shape)

	def increment(x, hashV):
		for j in range(numTables):
			array[j][hashV] += 1

	def count(trial, x):
		stringForm = str(x) + str(trial - 1)
		MD5Score = hashlib.md5(stringForm.encode("utf-8")).hexdigest()

		minValue = float('inf')
		for j in range(numTables):
			hashHex = MD5Score[:2]
			MD5Score = MD5Score[2:]
			hashValue = int(hashHex, 16)
			if array[j][hashValue] < minValue:
				minValue = array[j][hashValue]
		return minValue


	def countMinSketch(trial, x):

		stringForm = str(x) + str(trial - 1)
		MD5Score = hashlib.md5(stringForm.encode("utf-8")).hexdigest()

		currCount = []
		hashValues = []
		js = []
		for j in range(numTables):

			hashHex = MD5Score[:2]
			MD5Score = MD5Score[2:]
			hashValue = int(hashHex, 16)
			currCount.append(array[j][hashValue])
			hashValues.append(hashValue)
			js.append(j)
			#increment(x, hashValue)
		smallest = min(currCount)
		for i in range(len(currCount)):
			if currCount[i] == smallest:
				increment(js[i], hashValues[i])

	# forward = []
	# for i in range(1, 10):
	# 	lowerBound = 1000 * (i - 1) + 1
	# 	upperBound = 1000 * i
	# 	integer = r.randint(lowerBound, upperBound)
	# 	for j in range(i):
	# 		forward.append(integer)


	forward = []
	for i in range(1, 10):
		lowerBound = 1000 * (i - 1) + 1
		upperBound = 1000 * i
		for j in range(lowerBound, upperBound + 1):
			forward += [j] * i

	#9001 to 9050
	for i in range(1, 51):
		for j in range(i ** 2):
			forward.append(9000 + i)
	print("Number of elements total: ", len(forward))

	reverse = list(reversed(forward))
	randomized = forward[:]
	random.shuffle(randomized)
	def findHeavyHitters(trial):
		heavy = []
		for val in forward:
			if count(trial, val) >= len(reverse) * 0.01:
				heavy.append(val)
		return list(set(heavy))

	numberHeavyHitters = 0
	freq9050 = 0
	for i in range(1, numberTrials + 1):

		for value in forward:
			countMinSketch(i, value)
		freq9050 += count(i, max(reverse))
		#print("freq9050: ", freq9050)
		#print("heavy hitters", findHeavyHitters(i))
		print(len(findHeavyHitters(i)))
		numberHeavyHitters += len(findHeavyHitters(i))
		array = np.zeros((numTables, numberBuckets)) #reset array for next trial

	print("Average number of heavy hitters average per trial: ", numberHeavyHitters/numberTrials)
	print("Average frequency of element 9050 average per trial: ", freq9050/numberTrials)

conservative()


#RESULTS Conversative
#random
# Average number of heavy hitters average per trial:  77.6
# Average frequency of element 9050 average per trial:  2500.0

#reverse
# Average number of heavy hitters average per trial:  79.2
# Average frequency of element 9050 average per trial:  2500.0

#forward
# Average number of heavy hitters average per trial:  159.9
# Average frequency of element 9050 average per trial:  2864.9
