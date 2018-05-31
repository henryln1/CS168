import scipy.io.wavfile as wavfile 
import numpy as np
import matplotlib.pyplot as plt
import math

# question 2
def pad_arrays(x, y):
	x_len = len(x)
	y_len = len(y)
	x_padded = x
	y_padded = y
	if x_len < y_len:
		x_padded += [0 for i in range(y_len - x_len)]
	elif y_len < x_len:
		y_padded += [0 for i in range(x_len - y_len)]
	x_padded += [0 for i in range(len(x_padded))]
	y_padded += [0 for i in range(len(y_padded))]

	return x_padded, y_padded

def multiply(x, y):
	x_padded, y_padded = pad_arrays(x, y)
	x_fft = np.fft.fft(x_padded)
	y_fft = np.fft.fft(y_padded)
	x_y_mult = np.multiply(x_fft, y_fft)
	inv = np.fft.ifft(x_y_mult)
	values = []
	carry_over = 0
	for val in inv:
		print(val)
		curr = int(round(val.real, 0) + carry_over)
		if curr >= 10:
			carry_over = int(curr)//10
			curr %= 10
		else:
			carry_over = 0
		values.append(curr)
	while(values[-1] == 0):
		del values[-1]
	return values
	
x = [0,9,8,7,6,5,4,3,2,1,0,9,8,7,6,5,4,3,2,1]
y = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
print(multiply(x, y))










#Question 3
#with open(”laurel_yanny.wav”, ”rb”) as f: 
f = 'laurel_yanny.wav'
sampleRate, data = wavfile.read(f)

print("sample rate: ", sampleRate)
print("shape of data: ", data.shape)

#part b
def plot_time_vs_speaker(data, output_filename):
	time = data.shape[0]
	x_axis = [x for x in range(time)]
	plt.plot(x_axis, data)
	plt.title("Time versus Physical Position, 3B")
	plt.xlabel("Time")
	plt.ylabel("Phsyical Position/Displacement")
	plt.savefig(output_filename + '.png', format = 'png')
	plt.close()

#plot_time_vs_speaker(data, "3b")


#part c

#fourier_transformed_data = np.fft.fft(data)
# print("shape of transformed data: ", fourier_transformed_data.shape)
# print(fourier_transformed_data[0])
# x_axis = [x for x in range(fourier_transformed_data.shape[0])]

# plt.plot(x_axis, np.absolute(fourier_transformed_data))
# plt.title("Fourier Transform")
# plt.xlabel("Time")
# plt.ylabel("Fourier Transform Magnitude")
# plt.savefig("3c" + ".png", fomrat = 'png')
# plt.close()

#part d

def generate_heatmap(data, output_filename):
	chunk_size = 500
	num_fourier_coefficients = 80
	number_chunks = data.shape[0] // chunk_size

	fourier_coefficients_array = np.zeros((number_chunks, num_fourier_coefficients))
	for i in range(number_chunks):
		current_chunk = data[i * chunk_size : (i + 1) * chunk_size]
		curr_fourier = np.fft.fft(current_chunk)
		curr_fourier = np.absolute(curr_fourier)
		fourier_coefficients_array[i, ] = curr_fourier[:80]

	#fourier_coefficients_array = np.absolute(fourier_coefficients_array)
	fourier_coefficients_array = np.sqrt(fourier_coefficients_array)
	plt.imshow(fourier_coefficients_array, cmap = 'hot')
	plt.xlabel("Chunk Index")
	plt.ylabel("Magnitude Fourier Coefficients")
	plt.title("Spectrogram")
	plt.savefig(output_filename + '.png', format = 'png')
	plt.close()

#generate_heatmap(data, "3D_Square_Root")


#part e
def save_wav_file(data, threshold, low = False):
	data = (np.absolute(data) * 1.0 / np.max(np.absolute(data)) * 32767).astype(np.int16)
	with open(str(threshold) + ".wav", "wb") as f:
		sample_rate = sampleRate
		if low:
			sample_rate *= 1.3
		sample_rate = int(sample_rate)
		wavfile.write(f, sample_rate, data)


#print(np.absolute(np.fft.fft(data)))
def zero_out_frequencies(data, threshold):

	data = np.fft.fft(data)
	#data = np.absolute(data)
	print("fourier: ", data)
	#print(data)
	#if high == True: #zeroing out above threshold
	#data[data < high_threshold] = 0
		#print(data)
	#else: #zero out below certain threshold
	#data[data < low_threshold] = 0
	#for x in range(threshold, data.shape[0]):
	#	data[x] = 0
	#data[data > threshold] = 0
	print(data)
	inverse_fourier = np.fft.ifft(data)
	#inverse_fourier = np.absolute(inverse_fourier)
	save_wav_file(inverse_fourier, threshold)

#high_thresholds = [5, 10, 15, 20, 30, 100, 200]

#high_thresholds = [x for x in range(100000) if x % 1000 == 0]

#high_thresholds = [10]
#for threshold in high_thresholds:
#high = 400000
#ow = 42000
#zero_out_frequencies(data, low, high)
save_wav_file(data, "temp")
transformed_data = np.fft.fft(data)
# #high_threshold = 400000
# #low_threshold = 42000
# #transformed_data[transformed_data > high_threshold] = 0
# #transformed_data[transformed_data < low_threshold] = 0
# transformed_data[transformed_data > threshold] = 0
# reverted = np.fft.ifft(transformed_data)
# #transformed = np.fft.ifft(np.fft.fft(data))
# save_wav_file(reverted, "zero_below_42000_zero_above_400000")
# #print("diff: ", np.sum(transformed) - np.sum(data))
# print("sum", np.sum(data))
# zero_out_frequencies(data, 0)

#thresholds = [x * 40000 for x in range(1, 25)]

#thresholds = [2000000]
thresholds = [42000]
transformed_data = np.fft.fft(data)
for threshold in thresholds:
	curr_high_transformed = transformed_data.copy()
	#curr_high_transformed10urr_high_transformed > threshold] = 0
	curr_high_transformed[:threshold] = 0
	curr_high_transformed[:43008 - threshold] = 0
	curr_low_transformed = transformed_data.copy()
	#curr_low_transformed[curr_low_transformed < threshold] = 0
	curr_low_transformed[threshold:] = 0
	curr_low_transformed[:43008 - threshold] = 0
	curr_high_transformed_reverted = np.fft.ifft(curr_high_transformed)
	curr_low_transformed_reverted = np.fft.ifft(curr_low_transformed)
	save_wav_file(curr_high_transformed_reverted, "zero_above_sym_" + str(threshold))
	save_wav_file(curr_low_transformed_reverted, "zero_below_sym_" + str(threshold), low = True)



