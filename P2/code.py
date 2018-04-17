import csv
from functools import reduce 
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.spatial as sp

# QUESTION 1

# groups - groups[i] is name of group i + 1
groups = []
with open('p2_data/groups.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        groups.append(row[0])

# labels - labels[i] has list of all articles (0 indexed instead of 1 indexed) in group i + 1
# labels_reversed[i] gives group id for article i + 1
labels = []
labels_reversed = []
curr_group = 1
with open('p2_data/label.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    articles = []
    index = 0
    for row in reader:
        group_id = int(row[0])
        if group_id != curr_group:
            labels.append(list(articles))
            articles = []
            curr_group = group_id
        articles.append(index)
        index += 1
        labels_reversed.append(group_id)
    labels.append(list(articles))

# articles - articles[i] has dict of (wordId, wordCount) pairs for articleId i + 1
articles = []
curr_article = 1
max_word_id = -1
with open('p2_data/data50.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    word_counts = {}
    for row in reader:
        article_id = int(row[0])
        if article_id != curr_article:
            articles.append(dict(word_counts))
            word_counts = {}
            curr_article = article_id
        word_id = int(row[1])
        if word_id > max_word_id:
            max_word_id = word_id
        count = row[2]
        word_counts[word_id] = int(count)
    articles.append(dict(word_counts))


def jaccard(x, y, word_ids):
    numerator = 0
    denominator = 0
    for word in word_ids:
        x_count = x.get(word, 0)
        y_count = y.get(word, 0)
        numerator += min(x_count, y_count)
        denominator += max(x_count,y_count)
    return float(numerator)/denominator

def l2sim(x, y, word_ids):
    similarity = 0
    for word in word_ids:
        x_count = x.get(word, 0)
        y_count = y.get(word, 0)
        similarity += math.pow(x_count - y_count, 2)
    similarity = math.sqrt(similarity)
    return -similarity

def cosine(x, y, word_ids):
    numerator = 0
    x_term = 0
    y_term = 0
    for word in word_ids:
        x_count = x.get(word, 0)
        y_count = y.get(word, 0)
        numerator += x_count * y_count
        x_term += math.pow(x_count, 2)
        y_term += math.pow(y_count, 2)
    x_term = math.sqrt(x_term)
    y_term = math.sqrt(y_term)
    return float(numerator)/(x_term * y_term)

def makeHeatMap(data, names, color, outputFileName):
    #to catch "falling back to Agg" warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
        fig, ax = plt.subplots()
        #create the map w/ color bar legend
        heatmap = ax.pcolor(data, cmap=color)
        cbar = plt.colorbar(heatmap)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(names,rotation=90)
        ax.set_yticklabels(names)

        plt.tight_layout()

        plt.savefig(outputFileName, format = 'png')
        plt.close()

def findSimilarity(func, filename):
    num_groups = len(groups)
    matrix = np.zeros((num_groups, num_groups))
    for i in range(num_groups):
        group1 = labels[i]
        for j in range(num_groups):
            group2 = labels[j]
            num_similarities = 0
            total_similarities = 0
            for article_id1 in group1:
                for article_id2 in group2:
                    x = articles[article_id1]
                    y = articles[article_id2]
                    word_ids = reduce(set.union, map(set, map(dict.keys, [x, y])))
                    total_similarities += func(x, y, word_ids)
                    num_similarities += 1
            matrix[i][j] = float(total_similarities)/num_similarities
    makeHeatMap(matrix, groups, plt.cm.Blues, filename)

# print("Jaccard Similarity...")
# findSimilarity(jaccard, "jaccard.png")
# print("Jaccard Done")
# print("L2 Similarity...")
# findSimilarity(l2sim, "l2sim.png")
# print("L2 Done")
# print("Cosine Similarity...")
# findSimilarity(cosine, "cosine.png")
# print("Cosine Done")

# QUESTION 2

num_articles = len(labels_reversed)
num_groups = len(groups)

def baseline(): 
    matrix = np.zeros((num_groups, num_groups))
    error_count = 0
    for i in range(num_articles):
        group_id = labels_reversed[i]
        best_similarity = float("-inf")
        nearest_article_group = -1 # what if more than one nearest article?
        for j in range(num_articles):
            if i == j:
                continue
            x = articles[i]
            y = articles[j]
            word_ids = reduce(set.union, map(set, map(dict.keys, [x, y])))
            similarity = cosine(x, y, word_ids)
            if similarity > best_similarity:
                best_similarity = similarity
                nearest_article_group = labels_reversed[j]
        matrix[group_id - 1][nearest_article_group - 1] += 1
        if group_id != nearest_article_group:
            error_count += 1

    print("Classification Error") # using error from piazza, not pset
    classification_error = float(error_count)/num_articles
    print(classification_error)
    makeHeatMap(matrix, groups, plt.cm.Blues, "baseline.png")

# print("Baseline...")
# baseline()
# print("Baseline Done")

def projection(d):
    new_articles = []
    matrix = np.random.normal(size = (d, max_word_id))
    for article in articles:
        full_vector = np.zeros(max_word_id,)
        for k, v in article.items():
            full_vector[k - 1] = v
        new_articles.append(np.inner(full_vector, matrix))
    return new_articles

def nearest_neighbor(projection, filename):
    matrix = np.zeros((num_groups, num_groups))
    error_count = 0
    for i in range(num_articles):
        group_id = labels_reversed[i]
        best_similarity = float("-inf")
        nearest_article_group = -1 # what if more than one nearest article?
        for j in range(num_articles):
            if i == j:
                continue
            x = projection[i]
            y = projection[j]
            similarity = cosine_arr(x, y)
            if similarity > best_similarity:
                best_similarity = similarity
                nearest_article_group = labels_reversed[j]
        matrix[group_id - 1][nearest_article_group - 1] += 1
        if group_id != nearest_article_group:
            error_count += 1

    print("Classification Error")
    classification_error = float(error_count)/num_articles
    print(classification_error)

    makeHeatMap(matrix, groups, plt.cm.Blues, filename)    

def cosine_arr(x, y):
    numerator = 0
    x_term = 0
    y_term = 0
    for i in range(len(x)):
        x_count = x[i]
        y_count = y[i]
        numerator += x_count * y_count
        x_term += math.pow(x_count, 2)
        y_term += math.pow(y_count, 2)
    x_term = math.sqrt(x_term)
    y_term = math.sqrt(y_term)
    return float(numerator)/(x_term * y_term)

# print("Projecting w d = 10...")
# projection1 = projection(10)
# print("Finding nearest neighbors...")
# nearest_neighbor(projection1, "projection1.png")
# print("Projecting w d = 25...")
# projection2 = projection(25)
# print("Finding nearest neigbors...")
# nearest_neighbor(projection2, "projection2.png")
# print("Projecting w d = 50...")
# projection3 = projection(50)
# print("Finding nearest neigbors...")
# nearest_neighbor(projection3, "projection3.png")
# print("Projecting w d = 100...")
# projection4 = projection(100)
# print("Finding nearest neighbors...")
# nearest_neighbor(projection4, "projection4.png")


# QUESTION 3

matrices = []
new_articles = []
l = 128
d = 5
combined_sq_size = 0

def create_matrices():
    for table in range(l):
        matrix = np.random.normal(size = (d, max_word_id))
        matrices.append(matrix)

def hyperplane_hashing():
    for article in articles:
        new_articles.append(hash_article(article))
    return new_articles

def hash_article(article):
    full_vector = np.zeros(max_word_id,)
    new_article = []
    for k, v in article.items():
        full_vector[k - 1] = v
    for i in range(l):
        hashvalue_full = np.inner(full_vector, matrices[i])
        hashvalue_i = [1 if i > 0 else 0 for i in hashvalue_full]
        new_article.append(hashvalue_i)
    return new_article

def classification(q):
    hashvalues = hash_article(q)
    best_similarity = float("-inf")
    best_group = -1
    for i in range(num_articles):
        group_id = labels_reversed[i]
        datapoint = new_articles[i]
        if datapoint == hashvalues:
            continue
        for j in range(l):
            if hashvalues[j] == datapoint[j]:
                print(hashvalues)
                print(datapoint)
                similarity = sp.distance.cdist(hashvalues, datapoint, 'cosine') # produces nan - can probably replace those w zeros?
                print(similarity)
                if similarity > best_similarity: # this doesn't work bc the above produces a matrix
                    best_similarity = similarity
                    best_group = group_id
                combined_sq_size += 1
                break
    return best_group  


def lsh(d):
    create_matrices()
    hyperplane_hashing()
    error_count = 0
    for i in range(num_articles):
        actual_group = labels_reversed[i]
        suggested_group = classification(articles[i])
        error_count += 1
    print("Classification Error")
    classification_error = float(error_count)/num_articles
    print(classification_error)
    print("Average Size of Sq")
    sq_size = float(combined_sq_size)/num_articles
    print(sq_size)

for d in range(5, 21):
    print("LSH for d value " + str(d))
    lsh(d)



