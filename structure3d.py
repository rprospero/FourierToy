#!/usr/bin/python
"""A quick program to show that the 3D fourier spectrum
of a sample does not necessarily match the spectrum of
the 2d slices"""

from scipy.fftpack import fftshift, ifftshift
import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import rand


def animate_plot(data):
    """Take a 3D array and return a 2d plot of the data,
    animated along the third axis"""
    fig2 = plt.figure()

    size = data.shape[2]
    ims = []
    for i in np.arange(size):
        ims.append((plt.imshow(np.abs(data[:, :, i])), ))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50,
                                       repeat_delay=3000,
                                       blit=True)

    plt.show()

    del im_ani


def animate_2d_ffts(data, filename=None):
    """Take a 3D array and return an animated 2d plot of
    the data fourier transforms of the slices along the
    third axis"""
    fig2 = plt.figure()

    size = data.shape[2]
    ims = []
    for i in np.arange(size):
        ims.append((plt.imshow(fftshift(np.abs(fftn(data[:, :, i])))), ))

    im_ani = animation.ArtistAnimation(fig2, ims,
                                       interval=50, repeat_delay=3000,
                                       blit=True)

    if filename:
        im_ani.save(filename)
    else:
        plt.show()


def imbin(image):
    """Take an n-dimensional array and return a 1D histogram
    of the radial integration"""
    dims = np.array(image.shape)
    dims /= 2

    values = np.zeros(sum(dims))
    pixels = np.zeros(sum(dims))

    indices = np.indices(tuple(image.shape))
    rs = np.zeros(image.shape, dtype=np.float32)
    for (index, dim) in zip(indices, dims):
        rs += (index-dim)**2
    rs = np.asarray(np.round(np.sqrt(rs)), dtype=np.int32)

    bins = np.arange(0, np.max(rs)+1)
    values, _ = np.histogram(rs, bins=bins, weights=image)
    pixels, _ = np.histogram(rs, bins=bins)

    values /= pixels

    x = np.arange(0, np.max(rs))

    return (x, values[:len(x)])


def main():
    """Take a 3D structure with a known spectrum and find
    out the expected spectrum from looking at the fourier
    transform of a single 2D slice"""
    size = 101
    tol = 5
    wavelength = 20

    spec = np.zeros((size, size, size), dtype=np.complex128)

    rs = sum([(index - (size-1)/2)**2
              for index in np.indices((size, size, size))])

    rs = np.round(np.sqrt(rs))

    mask = np.logical_and(
        rs >= (1-tol/100.0)*wavelength
        , rs <= (1+tol/100.0)*wavelength)

    spec[mask] = 1
    animate_plot(spec)

    (x, y) = imbin(spec)

    spec *= np.exp(2*np.pi*rand(size, size)*1.0j)

    # animate_plot(spec)
    spec = fftshift(spec)

    real = fftn(spec)
    # animate_plot(real)
    animate_2d_ffts(real)

    spec2 = ifftshift(ifftn(spec))
    (x2, y2) = imbin(np.abs(spec2))
    y2 /= np.max(y2)

    # for i in range(0, size, 10):
    #     (xa, ya) = imbin(real[:, :, i])
    #     ya /= np.max(ya)
    #     plt.plot(xa, ya)

    plt.plot(x, y)
    plt.plot(x2, y2)
    plt.show()

main()
