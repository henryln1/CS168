import pandas as pd
import numpy as np

#question 1

def read_csv(file_path):
	#read csv and return it as a numpy matrix
	df = pd.read_csv(file_path, header = None)
	return df.as_matrix()



def normalize(matrix):
	return np.log(matrix + 1)



# co_occurence_matrix = read_csv('../co_occur.csv')
# normalized_matrix = normalize(co_occurence_matrix)
# print("shape of normalized_matrix: ", normalized_matrix.shape)





#question 2

from scipy.ndimage import imread
import matplotlib.pyplot as plt

im_array = imread("p5_image.gif", flatten=True)

#visualizes the image
# plt.imshow(im_array, cmap="gray")
# plt.show()


