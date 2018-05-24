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

fourier_transformed_data = np.fft.fft(data)

plt.plot(data, fourier_transformed_data)
plt.title("Original vs Fourier Transformed")
plt.xlabel("Original")
plt.ylabel("Fourier Transform Value")
plt.savefig("3c" + ".png", fomrat = 'png')
plt.close()
