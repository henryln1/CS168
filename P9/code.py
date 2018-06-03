import numpy as np 
import random
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cvx


#YAYYYYY FINAL PSET :) :) :) :D :D :D

#Question 1
wonderland_tree = 'p9_images/wonderland-tree.txt'

#Part A
def load_text_file_as_matrix(file_location):
	array = []
	with open(file_location) as f:
		for line in f:
			#line = line.split()
			temp = []
			for index in range(len(line) - 1):
				temp.append(int(line[index]))
			#array.append(line)
			array.append(temp)
			#print(temp)
	array = np.asarray(array)
	print("array shape: ", array.shape)
	return array

def load_text_file_as_array(file_location):
	array = []
	with open(file_location) as f:
		for line in f:
			for index in range(len(line) - 1):
				array.append(int(line[index]))
	array = np.asarray(array)
	print("array shape: ", array.shape)
	return array

image = load_text_file_as_matrix(wonderland_tree)
print("Number of ones (k): ", np.sum(image))
print("Total number of pixels (n): ", image.shape[0] * image.shape[1])
print("k / n: ", np.sum(image) / (image.shape[0] * image.shape[1]))


# #Part B
# n = 1200
# r = 600
# A = np.random.normal(size=(n, n))
# x = load_text_file_as_array(wonderland_tree)
# b = cvx.Variable(r)
# objective = cvx.Minimize(cvx.norm(x, 1))
# constraints = [b == np.dot(A[0:r], x), x >= 0]
# prob = cvx.Problem(objective, constraints)

# print("prob.solve(): ", prob.solve())
# print("b.value: ", b.value)


#Part C

#Part D


#Question 2

def load_image_get_good_pixels(file_location):
	image = np.array(Image.open(file_location), dtype = int)[:, :, 0]
	known = (image > 0).astype(int)
	print(known)
	return image, known

def naive_reconstruction(image, mask):
	image_shape = image.shape
	new_image = np.zeros(image_shape)
	for row in range(image_shape[0]):
		for column in range(image_shape[1]):
			if mask[row][column] == 1:
				new_image[row][column] = image[row][column]
				#continue
			else: #get four average pixels
				neighbor_pixels = []
				if row > 0 and row < image_shape[0] - 1 and column > 0 and column < image_shape[1] - 1:
					#This is a pixel that is not on any edge, so we have 4 neighbors
					neighbor_pixels.append(image[row - 1][column])
					neighbor_pixels.append(image[row + 1][column])
					neighbor_pixels.append(image[row][column - 1])
					neighbor_pixels.append(image[row][column + 1])
				elif row == 0: #top edge
					if column == 0: #top left pixel
						neighbor_pixels.append(image[row + 1][column])
						neighbor_pixels.append(image[row][column + 1])
					elif column == image_shape[1] - 1: #top right pixel
						neighbor_pixels.append(image[row + 1][column])
						neighbor_pixels.append(image[row][column - 1])
					else: #any other pixel on the top edge
						neighbor_pixels.append(image[row][column - 1])
						neighbor_pixels.append(image[row][column + 1])
						neighbor_pixels.append(image[row + 1][column])
				elif row == image_shape[0] - 1: 
					if column == 0: #bottom left pixel
						neighbor_pixels.append(image[row][column + 1])
						neighbor_pixels.append(image[row- 1][column])
					elif column == image_shape[1] - 1: #bottom right pixel
						neighbor_pixels.append(image[row][column - 1])
						neighbor_pixels.append(image[row - 1[column]])
					else: #any other pixel on the bottom edge
						neighbor_pixels.append(image[row - 1][column])
						neighbor_pixels.append(image[row][column - 1])
						neighbor_pixels.append(image[row][column + 1])
				else: #any row that isn't the top or bottom
					if column == 0: #a pixel on the left edge
						neighbor_pixels.append(image[row - 1][column])
						neighbor_pixels.append(image[row + 1][column])
						neighbor_pixels.append(image[row][column + 1])
					elif column == image_shape[1] - 1: #a pixel on the right edge
						neighbor_pixels.append(image[row - 1][column])
						neighbor_pixels.append(image[row + 1][column])
						neighbor_pixels.append(image[row][column - 1])												
					#if column > 0 and column < image_shape[1] - 1: #any pixel that is not on an edge
					#	pass
				#print("Length of neighbor pixels: ", len(neighbor_pixels))
				new_image[row][column] = sum(neighbor_pixels) / len(neighbor_pixels)

	return new_image



original_image, good_pixel_mask = load_image_get_good_pixels("p9_images/corrupted.png")



#Part B
new_image = naive_reconstruction(original_image, good_pixel_mask)
plt.imshow(new_image)
plt.savefig("2b.png", format = 'png')

#Part C

from cvxpy import Variable, Minimize, Problem, multiply, tv
U = Variable(original_image.shape)
obj = Minimize(tv(U))
constraints = [multiply(good_pixel_mask, U) == multiply(good_pixel_mask, original_image)]
prob = Problem(obj, constraints)
prob.solve(verbose = True)

#U = U.astype(float)
plt.imshow(U.value)
plt.savefig("2c.png", format = "png")


#print(good_image)
#plt.imshow(good_image)
#plt.savefig("temp.png", format = "png")