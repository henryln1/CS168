import random
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math


#Question 2

def read_csv(csv_name):
	df = pd.read_csv(csv_name, header = None)
	return df.as_matrix()

parks_info = read_csv('parks.csv')

print("parks info: ", parks_info)
parks_info = np.delete(parks_info, (0), axis=0)

#converting parks info from np array to dict for easier use key: park name, value: (longitude, latitude)
parks_dict = {}
for row in range(parks_info.shape[0]):
	park_name = parks_info[row][0]
	longitude = float(parks_info[row][1])
	latitude = float(parks_info[row][2])
	parks_dict[park_name] = (longitude, latitude)

print(parks_dict)

def calculate_distance_two_parks(park_1, park_2):
	longitude_1, latitude_1 = parks_dict[park_1]
	longitude_2, latitude_2 = parks_dict[park_2]

	return math.sqrt((longitude_1 - longitude_2) ** 2 + (latitude_1 - latitude_2) ** 2)

def calculate_route_total_distance(parks):
	total_distance = 0
	for i in range(len(parks) - 1):
		current_park = parks[i]
		next_park = parks[i + 1]
		total_distance += calculate_distance_two_parks(current_park, next_park)

	#forgot about going from last park to home, doing it outside for loop
	last_park = parks[-1]
	first_park = parks[0]
	total_distance += calculate_distance_two_parks(last_park, first_park)
	#print("hello")
	#print(total_distance)
	return total_distance



all_park_names = list(parks_dict.keys())

temp = all_park_names.copy()
temp.sort()
print("Distance between Acadia and Arches: ", calculate_distance_two_parks("Acadia", "Arches"))
print("Distance Alphabetical ", calculate_route_total_distance(temp))


def MCMC_algorithm(max_iterations, park_list, T, c = False):
	#print(park_list)
	random.shuffle(park_list) #creates random route
	#print("starting random route: ", park_list)
	#print("starting distance: ", calculate_route_total_distance(park_list))
	best_route = park_list
	#best_route_distance = calculate_route_total_distance(best_route)
	#route_distance_history = [calculate_route_total_distance(park_list)]
	route_distance_history = []
	for i in range(max_iterations):
		if c == False:
			random_park_index = random.randint(0, len(park_list) - 2)
			consec_park_index = random_park_index + 1
		else:
			random_park_index, consec_park_index = random.sample(range(len(park_list)), 2)
		random_park = park_list[random_park_index]
		consec_park = park_list[consec_park_index]
		curr_route_copy = park_list.copy()
		curr_route_copy[random_park_index] = consec_park 
		curr_route_copy[consec_park_index] = random_park

		change_distance_traveled = calculate_route_total_distance(curr_route_copy) - calculate_route_total_distance(park_list)
		route_distance_history.append(calculate_route_total_distance(curr_route_copy))
		if change_distance_traveled < 0 or (T > 0 and random.uniform(0, 1) < math.exp( - change_distance_traveled / T)):
			park_list = curr_route_copy
		if calculate_route_total_distance(park_list) < calculate_route_total_distance(best_route):
			best_route = park_list

	#print("Best Distance: ", calculate_route_total_distance(best_route))
	#print("Best Route: ", best_route)


	return best_route, route_distance_history #best route is best one we found, distance history is the distance of each route we tried

found_route, route_distance_history = MCMC_algorithm(1000, all_park_names, 0.1)

def plot_2b_c(X_axis, sets_route_histories, title, file_name):

	for history_index in range(len(sets_route_histories)):
		history = sets_route_histories[history_index]
		plt.scatter(X_axis, history, label = "Iteration " + str(history_index + 1))
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Route Distance")
	plt.legend(shadow=True, loc = 0)
	plt.savefig(file_name + ".png", format = "png")
	plt.close()


def part_b_c(c = False):
	list_of_T = [0, 1, 10, 100]
	number_iterations = 10000
	num_trials = 10

	iterations_list = [x for x in range(10000)]

	for T in list_of_T:
		history_distances_list = [] #list of route_distance_history lists
		for trial in range(num_trials):
			best_route, route_distance_history = MCMC_algorithm(number_iterations, all_park_names, T, c)
			history_distances_list.append(route_distance_history)
		if c == False:
			plot_2b_c(iterations_list, history_distances_list, "T = " + str(T), "2b_T=" + str(T))
		else:
			plot_2b_c(iterations_list, history_distances_list, "T = " + str(T), "2c_T=" + str(T))

#part_b_c()
#part_b_c(c = True)


