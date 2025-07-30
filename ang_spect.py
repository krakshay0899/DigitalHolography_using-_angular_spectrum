import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import cv2


def angular_spectrum_propagation(u0, wavelength, z, dx):

    k = 2 * np.pi / wavelength
    ny, nx = np.shape(u0)
    fx = np.linspace(-0.5, 0.5-(1/(nx)), nx) 
    fy = np.linspace(-0.5, 0.5-(1/(ny)), ny) 
    FX, FY = np.meshgrid(fx, fy)
    FX = FX / dx; FY = FY / dx
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))
    U0 = fftshift(fft2(ifftshift(u0)))
    Uz = U0 * H
    uz = ifftshift(ifft2(fftshift(Uz)))
    return uz,U0



img = cv2.imread('cm.png',0)
u0 = img.astype(np.float32)
wavelength = 632.8e-9  # 633 nm
z = 1  
dx = 6.8e-6  # 10 microns
uz,U0 = angular_spectrum_propagation(u0, wavelength, z, dx)
uz1, U1 = angular_spectrum_propagation(uz, wavelength, -z , dx)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('After 1st Propagation')
plt.imshow(np.abs(uz), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('After 2nd Propagation')
plt.imshow(np.abs(uz1), cmap='gray')
plt.axis('off')

plt.show()
# saved_image = np.abs(uz) / np.max(np.abs(uz)) * 255  # Normalize to [0, 255]
# cv2.imwrite('reconstructed_image.png', saved_image.astype(np.uint8))