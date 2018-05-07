import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

#question 1

def read_csv(file_path):
	#read csv and return it as a numpy matrix
	df = pd.read_csv(file_path, header = None)
	return df.as_matrix()

def normalize(matrix):
	return np.log(matrix + 1)

def plot(title, X, Y, X_label, Y_label, filename, X_ticks = None):
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.plot(X, Y, 'bs')
    if X_ticks:
        plt.xticks(X, X_ticks)
    plt.savefig(filename + ".png", format = 'png')
    plt.close()

def plot_numberline(title, X, labels, filename): # I don't think this is what they want but I'm not sure what else to do :s
    plt.title(title)
    plt.scatter(X, np.zeros_like(X))
    plt.yticks([])
    for i, label in enumerate(labels):
        plt.annotate(label, (X[i], 0), rotation = 'vertical')
    plt.savefig(filename + ".png", format = 'png')
    plt.close()

# scalar projection of B onto A
def projection(A, B):
    dot_prod = np.dot(A, B)
    mag_a = np.linalg.norm(A)
    return dot_prod/mag_a 



# co_occurence_matrix = read_csv('../co_occur.csv')
# with open("dictionary.txt") as file:
#     word_dictionary = file.readlines()
# word_dictionary = list(map(str.strip, word_dictionary))
# normalized_matrix = normalize(co_occurence_matrix)
# print("shape of normalized_matrix: ", normalized_matrix.shape)

# 1B
# U, singular_vals, VT = randomized_svd(normalized_matrix, n_components = 100, random_state=None)
# plot("Singular Values of rank-100 approx of M", [x for x in range(100)], singular_vals, "Ranking", "Singular Value", "1b")

# 1C - not sure how we want to choose vectors
# print("U: ", U)
# print("VT: ", VT)

# 1D
# embeddings = preprocessing.normalize(U, norm='l2')
# embeddings_len = len(embeddings)
# woman_idx = word_dictionary.index("woman")
# man_idx = word_dictionary.index("man")
# v = embeddings[woman_idx] - embeddings[man_idx]
# d_i_words = ["boy", "girl", "brother", "sister", "king", "queen", "he", "she", "john", "mary", "wall", "tree"]
# i_projections = []
# for word in d_i_words:
#     curr_idx = word_dictionary.index(word)
#     curr_projection = projection(v, embeddings[curr_idx])
#     i_projections.append(curr_projection)
# plot_numberline("Projections onto v", i_projections, d_i_words, "1d_i")
# #plot("Projections onto v", [x for x in range(len(d_i_words))], i_projections, "Word", "Projection", "1d_i", X_ticks = d_i_words)

# d_ii_words = ["math", "matrix", "history", "nurse", "doctor", "pilot", "teacher", "engineer", "science", "arts", "literature", "bob", "alice"]
# ii_projections = []
# for word in d_ii_words:
#     curr_idx = word_dictionary.index(word)
#     curr_projection = projection(v, embeddings[curr_idx])
#     ii_projections.append(curr_projection)
# plot_numberline("Projections onto v", ii_projections, d_ii_words, "1d_ii")
# #plot("Projections onto v",[x for x in range(len(d_ii_words))], ii_projections, "Word", "Projection", "1d_ii", X_ticks =  d_ii_words)

# # 1E
# cosine_similarities = {}
# stanford_idx = word_dictionary.index("stanford")
# for i in range(embeddings_len):
#     cosine_similarities[word_dictionary[i]] = np.dot(embeddings[i], embeddings[stanford_idx])
# similarities_counter = Counter(cosine_similarities)
# top_10 = similarities_counter.most_common(10)
# print("Top 10 closest words to Stanford: ", top_10) # stanford, harvard, cornell, ucla, yale, princeton, penn, auburn, mit, berkeley...

# with open("analogy_task.txt") as f:
#     analogies = f.readlines()
# num_correct_analogies = 0
# for analogy in analogies:
#     words = analogy.split()
#     indices = [word_dictionary.index(word) for word in words]
#     vec = embeddings[indices[1]] - embeddings[indices[0]] + embeddings[indices[2]]
#     target = vec / np.linalg.norm(vec)
#     best_word = None
#     best_similarity = -1
#     for i in range(embeddings_len):
#         similarity = np.dot(target, embeddings[i])
#         if similarity > best_similarity:
#             best_word = word_dictionary[i]
#             best_similarity = similarity
#     if best_word == words[3]:
#         num_correct_analogies += 1
# accuracy = float(num_correct_analogies) / len(analogies)
# print("Analogy Accuracy: ", accuracy)

#question 2
#CODE DOES NOT WORK YET

from scipy.ndimage import imread

im_array = imread("p5_image.gif", flatten=True)

#2B

U, S, V = np.linalg.svd(im_array)
# print("shape of U :", U.shape)
# print("shape of S: ", S)
# print("shape of V: ", V.shape)


#ranks of k we care about
k = [1, 3, 10, 20, 50, 100, 150, 200, 400, 800, 1170]
k = [100]

def calculate_k_approximation(k_rank, U, S, V):


    S = np.diag(S) #1170 by 1170
    print(S.shape)
    S = S[:k_rank, :k_rank] #take top k components
    print("shape of S", S.shape)

    V = V[:, :k_rank] #take top k vectors
    V_transposed = V.T #transpose
    print(V_transposed.shape) # k by 1170
    U = U[:, :k_rank] #take top k vectors
    print(U.shape) # 1600 by k

    A = np.matmul(U, S)
    A = np.matmul(A, V_transposed)

    return A

for curr_k in k:
    curr = calculate_k_approximation(curr_k, U, S, V)

print(curr.shape)
plt.imshow(curr)
plt.show()   
plt.savefig("testing.png", format = 'png')

# temp = U * S * V
# plt.imshow(temp)
# plt.show()
#visualizes the image
# plt.imshow(im_array, cmap="gray")
# plt.show()


