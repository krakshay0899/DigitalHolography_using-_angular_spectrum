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
    U0 = fftshift(fft2(u0))
    Uz = U0 * H
    uz = ifftshift(ifft2(Uz))
    return uz,U0



img = cv2.imread('test.png',0)
u0 = img.astype(np.float32)
wavelength = 632.8e-9  # 633 nm
z = 1  
dx = 6.8e-6  # 10 microns
uz,U0 = angular_spectrum_propagation(u0, wavelength, z, dx)
plt.imshow(np.abs(uz), cmap='gray', norm=plt.Normalize(vmin=0, vmax=np.max(np.abs(U0))))
plt.show()