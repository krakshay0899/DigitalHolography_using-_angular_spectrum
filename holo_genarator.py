import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('bios.jpg', 0)
img2= cv2.imread('iitd.jpg', 0)
u1 = img1.astype(np.float32)
u2= img2.astype(np.float32)

u1 = (u1-np.min(u1)) /(np.max(u1)-np.min(u1)) # Normalize the image 
u2= (u2-np.min(u2)) /(np.max(u2)-np.min(u2)) # Normalize the image

# Zero padding of p% of the image size
def img_padding(u0):
    p = .5  # Padding percentage
    pad_y = int(p * u0.shape[0])
    pad_x = int(p * u0.shape[1])
    u0 = np.pad(u0, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')
    return u0,pad_y,pad_x

u1, pad_y, pad_x = img_padding(u1)
u2 = img_padding(u2)[0]


wavelength = 632.8e-9  # 633 nm
dx = 6.8e-6  # 10 microns
theta= 0.04  # angle of propagation
prop_dist= .1  # Propagation distance in meters
recon_dist = .1 # Reconstruction distance in meters
del_z=.1  
# m=1
# d2=prop_dist*m
# f = 1 / (1 / prop_dist + 1 / d2)

k = 2 * np.pi / wavelength
ny, nx = np.shape(u1)
x = np.linspace(0, nx-1, nx)-nx/2
y= np.linspace(0, ny-1, ny)-ny/2 
x, y = np.meshgrid(x, y)
x=x*dx; y=y*dx


fx = np.linspace(-0.5, 0.5-(1/(nx)), nx) 
fy = np.linspace(-0.5, 0.5-(1/(ny)), ny) 
FX, FY = np.meshgrid(fx, fy)
FX = FX / dx; FY = FY / dx
# print(FX.max(), FY.max(), FX.min(), FY.min())
def angular_spectrum_propagation(u0,z):

    H = np.exp(1j * z * np.sqrt(k**2 - 4*np.pi**2*((FX)**2 + (FY)**2)))
    U0 = fftshift(fft2(ifftshift(u0)))
    Uz = U0 * H
    uz = fftshift(ifft2(ifftshift(Uz)))
    uz = uz[pad_y:-pad_y, pad_x:-pad_x] 
    return uz

u2_0 = angular_spectrum_propagation(u2, del_z)  # Propagate the second image
u2_0= img_padding(u2_0)[0]  # Zero padding of the propagated image
uz = angular_spectrum_propagation(u1+u2_0, prop_dist)
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(np.abs(uz), cmap='gray')
plt.title('Magnitude')
plt.subplot(1, 2, 2)
plt.imshow(np.imag(uz), cmap='gray')
plt.title('Phase')  # Display the propagated field

x = x[pad_y:-pad_y, pad_x:-pad_x]
y = y[pad_y:-pad_y, pad_x:-pad_x]
L=np.exp(1j * np.pi / ((prop_dist/2) * wavelength) * (x * x + y * y))
# R = np.exp(1j * k* np.sqrt((x)**2 + (y-0.09)**2 + (1)**2)) / np.sqrt(x**2 + (y-0.09)**2 + (1)**2)
R=np.exp(1j*k*np.sin(theta*x))
R = R / np.max(np.abs(R))  # Normalize reference wave
hologram = np.abs(uz + R)**2  # Hologram generation
h=fftshift(fft2(ifftshift(hologram)))
h[int(h.shape[0]//2), int(h.shape[1]//2)] = 0  # Set central pixel to 0
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(np.abs(hologram), cmap='gray')
plt.title('Magnitude')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(np.abs(h), cmap='gray')
plt.title('FFT of Hologram')
plt.colorbar()  # Display the hologram
cv2.imwrite('hologram.bmp', (255 * (hologram - np.min(hologram)) / (np.max(hologram) - np.min(hologram))).astype(np.uint8))
# Reconstruct image from hologram
def reconstruct_from_hologram(hologram,recon_dist):
    hologram = hologram -np.mean(hologram)  # DC suppression
    reconstructed = hologram*np.conj(R) # Multiply hologram with conjugate of reference wave
    reconstructed = np.pad(reconstructed, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)
    recon_field = angular_spectrum_propagation(reconstructed, recon_dist)
    return recon_field


recon_field = reconstruct_from_hologram(hologram, 0.1)
recon_field2=reconstruct_from_hologram(hologram, -0.2)  # Reconstruct the field from hologram
plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(np.abs(recon_field)**2, cmap='gray')
plt.title('Reconstructed Magnitude at d1')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(recon_field2), cmap='gray')
plt.title('Reconstructed Magnitude at d2')
plt.show()