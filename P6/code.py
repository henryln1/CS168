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

def plot_eigenvecs(vals, vecs, title, filename):
	enumerated = dict(enumerate(vals))
	counter = Counter(enumerated)
	ordered = counter.most_common()
	biggest_vals = ordered[0:2]
	smallest_vals = ordered[-2:]
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

circle_add_a = np.zeros((n, n))
for i in range(1, n - 2):
	circle_add_a[i][i + 1] = 1
	circle_add_a[i][i - 1] = 1
	circle_add_a[i + 1][i] = 1
	circle_add_a[i - 1][i] = 1
	circle_add_a[i][n - 1] = 1
	circle_add_a[n - 1][i] = 1
circle_add_a[n - 2][0] = 1
circle_add_a[0][n - 2] = 1
circle_add_a[0][n - 1] = 1
circle_add_a[n - 1][0] = 1
circle_add_a[n - 2][n - 1] = 1
circle_add_a[n - 1][n - 2] = 1
circle_add_d = np.diag([3 if i != n - 1 else n - 1 for i in range(n)])
circle_add_l = circle_add_d - circle_add_a
circle_add_l_eig_vals, circle_add_l_eig_vecs = np.linalg.eig(circle_add_l)
circle_add_a_eig_vals, circle_add_a_eig_vecs = np.linalg.eig(circle_add_a)
plot_eigenvecs(circle_add_l_eig_vals, circle_add_l_eig_vecs, "Graph D, Laplacian", "1b_d_i")
plot_eigenvecs(circle_add_a_eig_vals, circle_add_a_eig_vecs, "Graph D, Adjacency", "1b_d_ii")


#QUESTION 2

#A

# def read_csv(csv_name):
# 	df = pd.read_csv(csv_name, header = None)
# 	return df.as_matrix()


# friendship_array = read_csv("cs168mp6.csv")
# print("shape of array for 2: ", friendship_array.shape)


# #B

# unique_people = 1495
# def process_array_into_D_and_A(array):
# 	D = np.zeros((unique_people, unique_people)) #degree matrixx
# 	A = np.zeros_like(D) #adjacency matrix
# 	for row in range(array.shape[0]): #iterating through each row
# 		person = array[row][0]
# 		friend = array[row][1]
# 		D[person - 1][person - 1] += 1
# 		A[person - 1][friend - 1] = 1

# 	return D, A

# D, A = process_array_into_D_and_A(friendship_array)

# Laplacian = D - A

# eigenvalues, eigenvectors = eigs(Laplacian, k = 1493)
# print("List of eigenvalues: ", eigenvalues[-12:])

# #this is what I got put idk if its right
# #List of eigenvalues:  
# # [10.00560409+0.j  9.        +0.j 10.        +0.j 10.        +0.j
#  # 24.        +0.j 10.        +0.j  9.        +0.j 10.        +0.j
#  #  9.        +0.j  9.        +0.j  9.        +0.j  9.        +0.j]