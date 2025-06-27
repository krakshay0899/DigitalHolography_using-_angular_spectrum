import os
import matplotlib.pyplot as plt
import matplotlib.image as mpi
import numpy as np
import time
import timeit
import numexpr as ne

ne.set_num_threads(os.cpu_count())


# Read the hologram image file
hologram = mpi.imread("ulf7.BMP")
hologram = hologram.astype(float) #Convert into float type. Crucial for non integer based mathematical operations
plt.figure(1)
plt.title("Hologram")
plt.imshow(hologram, cmap="viridis")

# DC term suppression. It involves subtraction of the mean value of hologram from itself
hologram2 = hologram - np.mean(hologram)
plt.figure(2)
plt.title("Hologram with DC suppression")
plt.imshow(hologram2, cmap="viridis")

# prepare the Convolution operand for the hologram
Nr,Nc = np.shape(hologram) #number of rows and columns in the hologram
print("Number of rows and columns in the hologram are: ", Nr, Nc)
wavelength = 632.8e-9 #HeNe laser wavelength in SI units i.e. meters
dx = 6.8e-6 #sensor pixel size in meters
d = 1.054 #object distance in meters
m = 1/14 #magnification factor
d2 = d*m #reconstruction distance in meters

Nr = np.linspace(0, Nr-1, Nr)-Nr/2
Nc = np.linspace(0, Nc-1, Nc)-Nc/2 
Fr = np.linspace(-0.5, 0.5-(1/(Nr.size)), Nr.size) #frequency in x direction
Fc = np.linspace(-0.5, 0.5-(1/(Nc.size)), Nc.size) #frequency in y direction
x, y = np.meshgrid(Nc,Nr)
fx, fy = np.meshgrid(Fr, Fc) #frequency in x and y direction
x = x*dx; y = y*dx; fx = fx/dx; fy = fy/dx 
k = 2 * np.pi / wavelength  # wave number
k_squared = k**2
four_pi_squared = 4 * np.pi**2
p=np.pi
LPF = np.zeros(hologram.shape)  # Low pass filter


# field2 = np.multiply(hologram2, L) #DC term suppressed hologram multiplied by conjugate field

# angular spectrum method
def compute_ASM_parameter():
    f = 1 / (1 / d + 1 / d2)  # lens responsible for imaging an object kept at d distance to an image at d2 distance
    L = ne.evaluate("exp(1j * p / (f * wavelength) * (x * x + y * y))") 
    alpha_squared = ne.evaluate("k_squared - four_pi_squared * (fx*fx + fy*fy)")
    alpha = np.sqrt(np.maximum(alpha_squared, 0))  # Ensure non-negative values for sqrt
    G = ne.evaluate("exp(-1j * alpha * d2)")
    f0 = (1/wavelength) * 1/np.sqrt(1 + (2*d2/(np.size(Nc)*dx))**2) #f0 is the cut-off frequency of the low pass filter
    LPF[fx**2 + fy**2 <= f0**2] = 1 # Low-pass filter
    G = ne.evaluate("G * LPF")  # Apply low-pass filter to G
    return L, G

time_ASM_parameter_calc = timeit.timeit(compute_ASM_parameter, number=100)
L, G = compute_ASM_parameter()
print(f"Average iteration time for ASM parameter computation: {time_ASM_parameter_calc / 100:.6f} seconds")

def angular_spectrum_method():
    field = ne.evaluate("hologram * L")  # Use numexpr for element-wise multiplication
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(field) * G))

time_angular = timeit.timeit(angular_spectrum_method, number=100)
reconstructed_field1 = angular_spectrum_method()
print(f"Average iteration time for angular spectrum method: {time_angular / 100:.6f} seconds")

# convolution method
def compute_convolution_parameters():
    f = 1/(1/d+1/d2) 
    L = ne.evaluate("exp(1j * p / (f * wavelength) * (x * x + y * y))")
    rho = ne.evaluate("d2**2 + x**2 + y**2")
    rho = np.sqrt(rho)
    g = ne.evaluate("1j / wavelength * exp(-1j * 2 * p / wavelength * rho) / (rho)")
    g1 = np.fft.fft2(g)
    return L, g1

time_convolution_params = timeit.timeit(compute_convolution_parameters, number=100)
L, g1 = compute_convolution_parameters()
print(f"Average iteration time for convolution parameters computation: {time_convolution_params / 100:.6f} seconds")

def convolution_method():
    field = ne.evaluate("hologram * L") #hologram multiplied by conjugate field
    return np.fft.fftshift(np.fft.ifft2(np.multiply(np.fft.fft2(field),g1 )))

time_convolution = timeit.timeit(convolution_method, number=100)
reconstructed_field3 = convolution_method()
print(f"Average iteration time for convolution method: {time_convolution / 100:.6f} seconds")


# save and plot the reconstructed field
I1 = np.abs(reconstructed_field1)/np.max(np.abs(reconstructed_field1)) #normalized intensity profile
I3 = np.abs(reconstructed_field3)/np.max(np.abs(reconstructed_field3)) #normalized intensity profile
plt.figure(3)
plt.title("Reconstruction with angular spectrum")
plt.imshow(I1, cmap="hot", clim=(0.0, 0.4))
plt.colorbar()
plt.figure(4)
plt.title("Reconstruction with convolution")
plt.imshow(I3, cmap="hot", clim=(0.0, 0.4))
plt.colorbar()
mpi.imsave('Angular_spectrum_reconstruction.png', I1, cmap="hot", vmin=0.0, vmax=0.3) #save reconstruction matrix as image
mpi.imsave('Convolution_reconstruction.png', I3, cmap="hot", vmin=0.0, vmax=0.3) #save reconstruction matrix as image
  

# save and plot DC suppressed reconstructed field
# I2 = np.abs(reconstructed_field2)/np.max(np.abs(reconstructed_field2)) #normalized intensity profile
# I4 = np.abs(reconstructed_field4)/np.max(np.abs(reconstructed_field4)) #normalized intensity profile    
# plt.figure(5)
# plt.title("Reconstruction(DC suppression) using angular spectrum")
# plt.imshow(I2, cmap="hot", clim=(0.0, 0.4))
# plt.colorbar()
# plt.figure(6)
# plt.title("Reconstruction(DC suppression) using convolution")
# plt.imshow(I4, cmap="hot", clim=(0.0, 0.4))
# plt.colorbar()
# mpi.imsave('Angular_spectrum_reconstruction_DCsuppressed.png', I2, cmap="hot", vmin=0.0, vmax=0.6) #save reconstruction matrix as image
# mpi.imsave('Convolution_reconstruction_DCsuppressed.png', I4, cmap="hot", vmin=0.0, vmax=0.6) #save reconstruction matrix as image

plt.show()