import numpy as np
from collections import Counter


def process_text_file(fileName):
	identifier = []
	sex = []
	population = []
	nucleobases = []
	with open(fileName) as f:
		for line in f:
			line_list_form = line.split(' ')
			identifier.append(line_list_form.pop(0))
			sex.append(line_list_form.pop(0))
			population.append(line_list_form.pop(0))
			line_list_form[-1] = line_list_form[-1][:-1]
			#remove the non-genetic information
			#print(line_list_form)
			nucleobases.append(line_list_form) #add it to a list of lists that we can convert to np.array
			#print(len(nucleobases))
			#break

	nucleobases_array = np.array(nucleobases)

	return identifier, sex, population, nucleobases_array



def most_common_element_list(lst):
	c = Counter(lst)
	#print(c.most_common(1))
	element, count = c.most_common(1)[0]
	return element


def convert_array_to_binary(nucleobases):
	#get most common element in each column and store as list of elements
	most_common_list = []
	for j in range(nucleobases.shape[1]):
		list_form_curr_column = nucleobases[:, j].tolist()
		most_common_list.append(most_common_element_list(list_form_curr_column))

	#print(len(most_common_list))

	binary_array = np.zeros_like(nucleobases)
	for j in range(nucleobases.shape[1]): #we want to go down each column and replace the value and I think this is how you do it
		most_common_curr = most_common_list[j]
		for i in range(nucleobases.shape[0]):
			if nucleobases[i][j] != most_common_curr:
				binary_array[i][j] = 1
			else:
				binary_array[i][j] = 0

	#print(binary_array)
	#print(binary_array.shape)
	return binary_array



from sklearn.decomposition import PCA

def run_PCA(nucleobases, comp_number):
	pca = PCA(comp_number)
	dim_reduced_nucleobases = pca.fit_transform(nucleobases)
	#print(dim_reduced_nucleobases.shape)
	return dim_reduced_nucleobases

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
import matplotlib.cm as cm


#1B

def make_plot_PCA(nucleobases_PCA, population_tags, file_name): #assumes that the number of components is 2
	#print(population_tags)

	population_tags_dict = defaultdict(int)
	for popIndex in range(len(population_tags)):
		population_tags_dict[population_tags[popIndex]] = popIndex

	inv_dict = {v: k for k, v in population_tags_dict.items()}
	population_tags_int = [population_tags_dict[x] for x in population_tags]
	colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'cyan']
	cdict = {}
	unique_pop_tags_int = list(set(population_tags_int))
	print(len(unique_pop_tags_int))
	for pop in range(len(unique_pop_tags_int)): #assign each group a color
		cdict[unique_pop_tags_int[pop]] = colors[pop] 


	v1_components = np.array(nucleobases_PCA[:, 0].tolist())
	v2_components = np.array(nucleobases_PCA[:, 1].tolist())
	populations = np.array(population_tags_int)

	fig, ax = plt.subplots()
	for g in np.unique(populations):
		ix = np.where(populations == g)
		ax.scatter(v1_components[ix], v2_components[ix], c = cdict[g], label = inv_dict[g])
	ax.legend()
	ax.set_title("PCA Analysis of 1000 Genomes Project, Part 1B")
	ax.set_xlabel('v1')
	ax.set_ylabel('v2')
	#plt.show()
	plt.savefig(file_name + ".png", format = 'png')
	plt.close()


identifiers, sexes, population_tag, nucleobases = process_text_file('p4dataset2018.txt')
nucleobases_binary = convert_array_to_binary(nucleobases)



#1B
# principal_components = run_PCA(nucleobases_binary, 2)
# make_plot_PCA(principal_components, population_tag, "1b")



#1DDDDDDDD
def make_plot_PCA_1D(nucleobases_PCA, population_tags, file_name): #assumes that the number of components is 3
	#print(population_tags)

	population_tags_dict = defaultdict(int)
	for popIndex in range(len(population_tags)):
		population_tags_dict[population_tags[popIndex]] = popIndex

	inv_dict = {v: k for k, v in population_tags_dict.items()}
	population_tags_int = [population_tags_dict[x] for x in population_tags]
	colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'cyan']
	cdict = {}
	unique_pop_tags_int = list(set(population_tags_int))
	print(len(unique_pop_tags_int))
	for pop in range(len(unique_pop_tags_int)): #assign each group a color
		cdict[unique_pop_tags_int[pop]] = colors[pop] 


	v1_components = np.array(nucleobases_PCA[:, 0].tolist())
	v2_components = np.array(nucleobases_PCA[:, 1].tolist())
	v3_components = np.array(nucleobases_PCA[:, 2].tolist())
	populations = np.array(population_tags_int)

	fig, ax = plt.subplots()
	for g in np.unique(populations):
		ix = np.where(populations == g)
		ax.scatter(v1_components[ix], v3_components[ix], c = cdict[g], label = inv_dict[g])
	ax.legend()
	ax.set_title("PCA Analysis of 1000 Genomes Project, Part 1D")
	ax.set_xlabel('v1')
	ax.set_ylabel('v3')
	#plt.show()
	plt.savefig(file_name + ".png", format = 'png')
	plt.close()

#1DDDD
# principal_components = run_PCA(nucleobases_binary, 3)
# make_plot_PCA_1D(principal_components, sexes, "1d_sex")


#1FFFFF
def run_PCA_get_v(nucleobases, comp_number, interesting_v):
	pca = PCA(comp_number)
	dim_reduced_nucleobases = pca.fit_transform(nucleobases)
	v_vector = pca.components_[interesting_v - 1:]
	print(v_vector.shape)

	return v_vector

import math

def make_plot_1F(nucleobases_size, interesting_vector, file_name):
	interesting_vector = list(interesting_vector.T)
	interesting_vector = [abs(x) for x in interesting_vector]
	size_list = range(0, nucleobases_size)
	plt.plot(size_list, interesting_vector)
	plt.xlabel('Nucleobase Index')
	plt.ylabel('Absolute Value of v3')
	plt.title('Nucleobases Analysis with PCA, Part 1F')
	#plt.show()
	plt.savefig(file_name + ".png", format = 'png')
	plt.close()

# interesting_vector = run_PCA_get_v(nucleobases_binary, 3, 3)
# make_plot_1F(10101, interesting_vector, "1F")

# 2AAAAAAAAAAAv - can use for 2b if needed
def pca_recover(X, Y):
	vector_list = [X, Y]
	vector_array = np.array(vector_list)
	#print("shape of vector array: ", vector_array.shape)
	pca = run_PCA_get_v(vector_array, 1, 1)
	slope = pca[0][1] / pca[0][0]
	return slope

def ls_recover(X, Y):
	X_mean = np.mean(X)
	Y_mean = np.mean(Y)
	numerator = np.dot(X - X_mean, Y - Y_mean)
	denominator = ((X - X_mean)**2).sum()
	#print(numerator/denominator)
	return numerator/denominator

# X_a = [x * 0.001 for x in range(1, 1001)]
# y_a = [2 * x for x in X_a]

# print("pca_recover: ", pca_recover(X_a, y_a))
# print("ls recover: ", ls_recover(X_a, y_a)) 

from numpy.random import randn

# 2CCCCCCCC
cs = [c * .05 for c in range(11)]

def make_Y(c):
	Y = np.array([2 * i * .001 for i in range(1, 1001)])
	noise = randn(1000) * np.sqrt(c)
	Y += noise
	return Y

def make_X(c):
	X = np.array([i * .001 for i in range(1, 1001)])
	noise = randn(1000) * np.sqrt(c)
	X += noise
	#print("shape of X: ", X.shape)
	return X


def make_plot_2(filename, X = None):
	# c on horizontal axis
	# pca-recover on vertical - red dot
	# ls-recover on vertical - blue dot
	plt.title("PCA-Recover & LS-Recover vs c")

	for i in range(30):
		for c in cs:
			Y = make_Y(c)
			if X is None:
				X = make_X(c)
			pca = pca_recover(X, Y)
			ls = ls_recover(X, Y)
			plt.plot(c, pca, 'rs', label="PCA-Recover")
			plt.plot(c, ls, 'bs', label="LS-Recover")
	plt.savefig(filename + ".png", format = 'png')
	plt.close()

X = [x * .001 for x in range(1, 1001)]
make_plot_2("2c", X)
make_plot_2("2d")




	


# identifiers, sexes, population_tag, nucleobases = process_text_file('p4dataset2018.txt')
# nucleobases_binary = convert_array_to_binary(nucleobases)

# 	