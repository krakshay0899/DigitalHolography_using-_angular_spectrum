import os
import matplotlib.pyplot as plt
import matplotlib.image as mpi
import numpy as np
import pyfftw
import timeit
import time
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
import numexpr as ne

if not os.path.exists(r"ulf7.BMP"):
    raise FileNotFoundError("The file 'ulf7.BMP' was not found.")

hologram = mpi.imread(r"ulf7.BMP")
hologram = hologram.astype(float)
plt.figure(1)
plt.title("Hologram")
plt.imshow(hologram, cmap="viridis")

# # DC term suppression
# hologram2 = hologram - np.mean(hologram)
# plt.figure(2)
# plt.title("Hologram with DC suppression")
# plt.imshow(hologram2, cmap="viridis")

# Hologram dimensions
Nr0, Nc0 = np.shape(hologram)
print("Number of rows and columns in the hologram are:", Nr0, Nc0)

# Physical parameters
wavelength = 632.8e-9  # HeNe laser wavelength (m)
dx = 6.8e-6  # sensor pixel size (m)
d = 1.054  # object distance (m)
m = 1/14
d2 = d * m
k = 2 * np.pi / wavelength
k_squared = k**2
four_pi_squared = 4 * np.pi**2
p=np.pi

# Coordinate and frequency grids
y_axis = np.linspace(0, Nr0 - 1, Nr0) - Nr0 / 2
x_axis = np.linspace(0, Nc0 - 1, Nc0) - Nc0 / 2
Fr = np.linspace(-0.5, 0.5 - 1/Nr0, Nr0)
Fc = np.linspace(-0.5, 0.5 - 1/Nc0, Nc0)

x, y = np.meshgrid(x_axis, y_axis)
fx, fy = np.meshgrid(Fc, Fr)
x *= dx
y *= dx
fx /= dx
fy /= dx
LPF = np.zeros(hologram.shape)  # Low pass filter

t3=time.perf_counter()
input_array = pyfftw.empty_aligned(hologram.shape, dtype='complex64')  # Dynamically set shape
output_array = pyfftw.empty_aligned(hologram.shape, dtype='complex64')

fft_object = pyfftw.FFTW(input_array, output_array, axes=(0, 1), direction='FFTW_FORWARD', threads=os.cpu_count())
ifft_object = pyfftw.FFTW(output_array, input_array, axes=(0, 1), direction='FFTW_BACKWARD', threads=os.cpu_count())

t4=time.perf_counter()

# Angular Spectrum Method using pyFFTW
def calculate_asm_parameters():
    f = 1 / (1 / d + 1 / d2)  # lens responsible for imaging an object kept at d distance to an image at d2 distance
    L = ne.evaluate("exp(1j * p / (f * wavelength) * (x * x + y * y))")
    alpha_squared = ne.evaluate("k_squared - four_pi_squared * (fx**2 + fy**2)")
    alpha = np.sqrt(np.maximum(alpha_squared, 0))  # Ensure non-negative values for sqrt
    G = ne.evaluate("exp(-1j * alpha * d2)")
    f0 = (1/wavelength) * 1/np.sqrt(1 + (4*d2**2/((np.size(x_axis)*dx)*(np.size(y_axis)*dx)))) #f0 is the cut-off frequency of the low pass filter
    LPF[fx**2 + fy**2 <= f0**2] = 1 # Low-pass filter
    G = ne.evaluate("G * LPF")  # Apply low-pass filter to G
    return L, G

# Timing the calculate_asm_parameters function using timeit
execution_time_asm = timeit.timeit(calculate_asm_parameters, number=100)
print(f"Average execution time for calculate_asm_parameters: {execution_time_asm / 100:.6f} seconds")

L, G = calculate_asm_parameters()


def angular_spectrum_method():
    input_array[:] = ne.evaluate("hologram * L")
    fft_object()  # output_array now contains FFT
    output_array[:] = output_array * G
    ifft_object()
    return pyfftw.interfaces.numpy_fft.fftshift(input_array)

# Timing the angular_spectrum_method using timeit
execution_time = timeit.timeit(angular_spectrum_method, number=100)
print(f"Average execution time for angular_spectrum_method: {execution_time / 100:.6f} seconds")


t0=time.perf_counter()
ftr=angular_spectrum_method()
t1=time.perf_counter()
field=np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(hologram)))
# ftr=angular_spectrum_method(hologram*2)
t2=time.perf_counter()
# print(f"{np.array_equal(hologram,field)}")
print(f"time for ASM:{t1-t0}, time for two numpy fft:{t2-t1} ,object prep time: {t4-t3}")

I =np.abs(ftr)/np.max(np.abs(ftr))

# Plotting
plt.figure(2)
plt.title("Reconstruction with angular spectrum")
plt.imshow(I,cmap="hot", clim=(0.0, 0.4))
plt.colorbar()
plt.show()