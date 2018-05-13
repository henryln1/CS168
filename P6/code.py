import pandas as pd 
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import eigs
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# QUESTION 1
n = 100

def get_ordered_evals(vals):
	enumerated = dict(enumerate(vals))
	counter = Counter(enumerated)
	ordered = counter.most_common()
	return ordered

def plot_eigenvecs(vals, vecs, title, filename):
	ordered = get_ordered_evals(vals)
	biggest_vals = ordered[0:2]
	smallest_vals = ordered[-2:]
	print(title)
	print("biggest: ", biggest_vals)
	print("smallest: ", smallest_vals)
	X = [i for i in range(n)]
	plt.plot(X, vecs[:,biggest_vals[0][0]], 'rs', label="Largest Eigenval")
	plt.plot(X, vecs[:,biggest_vals[1][0]], 'bs', label="2nd Largest")
	plt.plot(X, vecs[:,smallest_vals[0][0]], 'gs', label="2nd Smallest")
	plt.plot(X, vecs[:,smallest_vals[1][0]], 'ys', label="Smallest Eigenval")

	plt.title(title)
	plt.xlabel("i")
	plt.ylabel("v_i")
	plt.legend(shadow=True, loc = 0)
	plt.savefig(filename + ".png", format = 'png')
	plt.close()

def plot_embeddings_c(vals, vecs, title, filename):
	ordered = get_ordered_evals(vals)
	v2 = ordered[-2]
	v3 = ordered[-3]

	plt.plot(vecs[:,v2[0]], vecs[:,v3[0]], '-o')
	plt.title(title)
	plt.xlabel("v_2(i)")
	plt.ylabel("v_3(i)")
	plt.savefig(filename + ".png", format = 'png')
	plt.close()

def plot_embeddings_d(vals, vecs, points, title, filename):
	ordered = get_ordered_evals(vals)
	v2 = ordered[-2]
	v3 = ordered[-3]

	plt.title(title)
	plt.xlabel("v_2(i)")
	plt.ylabel("v_3(i)")

	v2_vec = vecs[:,v2[0]]
	v3_vec = vecs[:,v3[0]]
	plt.plot(v2_vec, v3_vec, 'rs')
	for i in range(len(points)):
		if points[i][0] < 0.5 and points[i][1] < 0.5:
			plt.plot(v2_vec[i], v3_vec[i], 'bs')
	plt.savefig(filename + ".png", format = 'png')
	plt.close()


#B
# line_a = np.zeros((n, n))
# for i in range(n - 1):
# 	line_a[i][i + 1] = 1
# 	line_a[i + 1][i] = 1
# line_d_vec = [1 if i == 0 or i == n - 1 else 2 for i in range(n)]
# line_d = np.diag(line_d_vec)
# line_l = line_d - line_a
# line_l_eig_vals, line_l_eig_vecs = np.linalg.eig(line_l)
# line_a_eig_vals, line_a_eig_vecs = np.linalg.eig(line_a)
# plot_eigenvecs(line_l_eig_vals, line_l_eig_vecs, "Graph A, Laplacian", "1b_a_i")
# plot_eigenvecs(line_a_eig_vals, line_a_eig_vecs, "Graph A, Adjacency", "1b_a_ii")

# line_add_a = np.zeros((n, n))
# for i in range(n - 1):
# 	line_add_a[i][i + 1] = 1
# 	line_add_a[i + 1][i] = 1
# 	line_add_a[n - 1][i] = 1
# 	line_add_a[i][n - 1] = 1
# line_add_d_vec = [2 if i == 0 or i == n - 2 else 3 for i in range(n)]
# line_add_d_vec[n - 1] = n - 1
# line_add_d = np.diag(line_add_d_vec)
# line_add_l = line_add_d - line_add_a
# line_add_l_eig_vals, line_add_l_eig_vecs = np.linalg.eig(line_add_l)
# line_add_a_eig_vals, line_add_a_eig_vecs = np.linalg.eig(line_add_a)
# plot_eigenvecs(line_add_l_eig_vals, line_add_l_eig_vecs, "Graph B, Laplacian", "1b_b_i")
# plot_eigenvecs(line_add_a_eig_vals, line_add_a_eig_vecs, "Graph B, Adjacency", "1b_b_ii")

# circle_a = np.zeros((n, n))
# for i in range(1, n - 1):
# 	circle_a[i][i + 1] = 1
# 	circle_a[i][i - 1] = 1
# 	circle_a[i + 1][i] = 1
# 	circle_a[i - 1][i] = 1
# circle_a[n - 1][0] = 1
# circle_a[0][n - 1] = 1
# circle_d = np.diag([2 for i in range(n)])
# circle_l = circle_d - circle_a
# circle_l_eig_vals, circle_l_eig_vecs = np.linalg.eig(circle_l)
# circle_a_eig_vals, circle_a_eig_vecs = np.linalg.eig(circle_a)
# plot_eigenvecs(circle_l_eig_vals, circle_l_eig_vecs, "Graph C, Laplacian", "1b_c_i")
# plot_eigenvecs(circle_a_eig_vals, circle_a_eig_vecs, "Graph C, Adjacency", "1b_c_ii")

# circle_add_a = np.zeros((n, n))
# for i in range(1, n - 2):
# 	circle_add_a[i][i + 1] = 1
# 	circle_add_a[i][i - 1] = 1
# 	circle_add_a[i + 1][i] = 1
# 	circle_add_a[i - 1][i] = 1
# 	circle_add_a[i][n - 1] = 1
# 	circle_add_a[n - 1][i] = 1
# circle_add_a[n - 2][0] = 1
# circle_add_a[0][n - 2] = 1
# circle_add_a[0][n - 1] = 1
# circle_add_a[n - 1][0] = 1
# circle_add_a[n - 2][n - 1] = 1
# circle_add_a[n - 1][n - 2] = 1
# circle_add_d = np.diag([3 if i != n - 1 else n - 1 for i in range(n)])
# circle_add_l = circle_add_d - circle_add_a
# circle_add_l_eig_vals, circle_add_l_eig_vecs = np.linalg.eig(circle_add_l)
# circle_add_a_eig_vals, circle_add_a_eig_vecs = np.linalg.eig(circle_add_a)
# plot_eigenvecs(circle_add_l_eig_vals, circle_add_l_eig_vecs, "Graph D, Laplacian", "1b_d_i")
# plot_eigenvecs(circle_add_a_eig_vals, circle_add_a_eig_vecs, "Graph D, Adjacency", "1b_d_ii")

# C
# plot_embeddings_c(line_l_eig_vals, line_l_eig_vecs, "Embedding of Graph A", "1c_a")
# plot_embeddings_c(line_add_l_eig_vals, line_add_l_eig_vecs, "Embedding of Graph B", "1c_b")
# plot_embeddings_c(circle_l_eig_vals, circle_l_eig_vecs, "Embedding of Graph C", "1c_c")
# plot_embeddings_c(circle_add_l_eig_vals, circle_add_l_eig_vecs, "Embedding of Graph D", "1c_d")

# D
# rand_n = 500
# rand_points = np.random.uniform(size = (rand_n, 2))
# print(rand_points)
# rand_a = np.zeros((rand_n, rand_n))
# for i in range(rand_n):
# 	for j in range(i + 1, rand_n):
# 		dist = np.linalg.norm(rand_points[i] - rand_points[j])
# 		if dist <= 0.25:
# 			rand_a[i][j] = 1
# 			rand_a[j][i] = 1
# rand_d = np.diag([sum(rand_a[k]) for k in range(rand_n)])
# rand_l = rand_d - rand_a
# rand_l_eig_vals, rand_l_eig_vecs = np.linalg.eig(rand_l)
# plot_embeddings_d(rand_l_eig_vals, rand_l_eig_vecs, rand_points, "Embedding of Random Graph", "1d")


#QUESTION 2

#A

def read_csv(csv_name):
	df = pd.read_csv(csv_name, header = None)
	return df.as_matrix()


friendship_array = read_csv("cs168mp6.csv")
print("shape of array for 2: ", friendship_array.shape)


#B

unique_people = 1495
def process_array_into_D_and_A(array):
	D = np.zeros((unique_people, unique_people)) #degree matrixx
	A = np.zeros_like(D) #adjacency matrix
	for row in range(array.shape[0]): #iterating through each row
		person = array[row][0]
		#print("person: ", person)
		friend = array[row][1]
		D[person - 1, person - 1] += 1
		A[person - 1, friend - 1] = 1


	return D, A

def plot_two_eigenvectors(vector1, vector2, vector1_name, vector2_name, title, filename):
	#vector 1 should be bigger eigenvector
	plt.scatter(vector1, vector2)
	plt.xlabel(vector1_name)
	plt.ylabel(vector2_name)
	plt.title(title)
	plt.savefig(filename + ".png", format = 'png')
	plt.close()

def plot_eigenvector_vs_person(eigenvector, title, filename):
	number_people = eigenvector.shape[0]
	people_list = [x for x in range(1, number_people + 1)]
	plt.scatter(people_list, eigenvector)
	plt.xlabel("Person ID")
	plt.ylabel("Corresponding Eigenvector Value")
	plt.title(title)
	plt.savefig(filename + ".png", format = 'png')
	plt.close()

#print("Shape of friendship array: ", friendship_array.shape)
D, A = process_array_into_D_and_A(friendship_array)

Laplacian = D - A
#print("Laplacian: ", Laplacian)

eigenvalues, eigenvectors = np.linalg.eig(Laplacian)
#eigenvalues_list = sorted(eigenvalues.tolist())
idx = eigenvalues.argsort() #[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


# eigenvectors = np.log(eigenvectors)
# smallest_eigenvector = eigenvectors[0, :]
# second_smallest_eigenvector = eigenvectors[1, :]
# third_smallest_eigenvector = eigenvectors[2, :]

#plot_two_eigenvectors(smallest_eigenvector, second_smallest_eigenvector, "1st eigenvector", "2nd eigenvector", "Smallest Eigenvectors", "2b")
#plot_two_eigenvectors(second_smallest_eigenvector, third_smallest_eigenvector, "2nd eigenvector", "3rd eigenvector", "Smallest Eigenvectors", "2b_2")
#plot_two_eigenvectors(eigenvectors[2, :], eigenvectors[3, :], "3rd eigenvector", "4th eigenvector", "Smallest Eigenvectors", "2b_3")
#plot_two_eigenvectors(eigenvectors[6, :], eigenvectors[7, :], "7th eigenvector", "8th eigenvector", "Smallest Eigenvectors", "2b_5")
#plot_two_eigenvectors(eigenvectors[7, :], eigenvectors[8, :], "8th eigenvector", "9th eigenvector", "Smallest Eigenvectors", "2b_6")

#plot_eigenvector_vs_person(eigenvectors[7, :], "2nd Smallest Eigenvector", "2b_11")
#plot_eigenvector_vs_person(eigenvectors[14, :], "15th Eigenvector", "2b_15th")

#list of smallest eigenvalues
#List of eigenvalues:  
#[-8.176427170721321e-14, -2.2035097765727694e-14, -8.880769415071853e-15, 
#4.355158870870788e-15, 6.732637604348797e-14, 8.577736296227727e-14, 
#0.014304016619435813, 0.05379565273704709, 0.07390297669255014, 0.0812896697122266, 
#0.12022393183749233, 0.13283886699780015]


#2D

def calculate_conductance(A, S):
	#A is adjacency matrixx
	#S is list of nodes that we are considering for calculating conductance (should be >= 150)
	A_without_S_nodes = A.copy()
	A_with_S_nodes = np.zeros_like(A)
	for node in S:
		A_without_S_nodes[node] = 0
		A_without_S_nodes[:, node] = 0
		A_with_S_nodes[node] = A[node]
		A_with_S_nodes[:, node] = A[:, node]

	A_without_S_sum = np.sum(A_without_S_nodes)
	A_with_S_sum = np.sum(A_with_S_nodes)
	#print("A without S sum: ", A_without_S_sum)
	#print("A with S sum: ", A_with_S_sum)
	denominator = min(A_with_S_sum, A_without_S_sum)

	numerator = 0
	A = np.tril(A)
	for row in range(A.shape[0]):
		for column in range(A.shape[1]):
			if row in S and column not in S:
				numerator += A[row, column]
			elif row not in S and column in S:
				numerator += A[row, column]
	print("numerator: ", numerator)
	print("denominator: ", denominator)
	return numerator / denominator

import random 

#test_S = [x for x in range(50, 200)]
#2D NOT DONE


#2E

random_S = random.sample(range(0, 1495), 150)
print("Len of Test S: ", len(random_S))
cond = calculate_conductance(A, random_S)
print("conductance: ", cond)

#conductance:  0.47379285586070724
