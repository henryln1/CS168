import scipy.io.wavfile as wavfile 
import numpy as np
import matplotlib.pyplot as plt











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

#generate_heatmap(data, "2D_Square_Root")


#part e
def save_wav_file(data, threshold):
	data = (data * 1.0 / np.max(np.absolute(data)) * 32767).astype(np.int16)
	with open("output_threshold_low=" + str(threshold) + ".wav", "wb") as f:
		wavfile.write(f, sampleRate, data)


def zero_out_frequencies(data, threshold, high = True):

	data = np.fft.fft(data)
	if high == True: #zeroing out above threshold
		data[data > threshold] = 0
	else: #zero out below certain threshold
		data[data < threshold] = 0
	inverse_fourier = np.fft.ifft(data)
	inverse_fourier = np.absolute(inverse_fourier)
	save_wav_file(inverse_fourier, threshold)

high_thresholds = [5, 10, 15, 20, 30, 100, 200]

high_thresholds = [x for x in range(10000) if x % 100 == 0]


for threshold in high_thresholds:
	zero_out_frequencies(data, threshold, high = False)



