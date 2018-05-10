import pandas as pd 
import numpy as np

from scipy.sparse import identity
from scipy.sparse.linalg import eigs


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
		friend = array[row][1]
		D[person - 1][person - 1] += 1
		A[person - 1][friend - 1] = 1

	return D, A

D, A = process_array_into_D_and_A(friendship_array)

Laplacian = D - A

eigenvalues, eigenvectors = eigs(Laplacian, k = 1493)
print("List of eigenvalues: ", eigenvalues[-12:])

#this is what I got put idk if its right
#List of eigenvalues:  
# [10.00560409+0.j  9.        +0.j 10.        +0.j 10.        +0.j
 # 24.        +0.j 10.        +0.j  9.        +0.j 10.        +0.j
 #  9.        +0.j  9.        +0.j  9.        +0.j  9.        +0.j]