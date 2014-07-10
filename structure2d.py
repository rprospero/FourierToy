#!/usr/bin/python
"""This program takes a 2d structure with a known fourier
spectrum, extrudes it out to three dimensions, then looks
at the resulting spectrum."""

from scipy.fftpack import fftshift, ifftshift
import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from numpy.random import rand


def animate_plot(data, filename=None):
    """Take a 3D array and return a 2d plot of the data,
    animated along the third axis"""
    fig2 = plt.figure()

    size = data.shape[0]
    ims = []
    for i in np.arange(size):
        ims.append((plt.imshow(data[:, :, i],
                               cmap=cm.get_cmap("gray")),))
    plt.colorbar()

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=50,
                                       blit=True)

    if filename:
        im_ani.save(filename, writer='imagemagick')
    else:
        plt.show()
    del im_ani


def export_plot(data, filename):
    """Take a 3D array of data and save it as a
    series of 2d images"""
    size = data.shape[0]
    for i in np.arange(size):
        plt.imshow(data[:, :, i],
                   cmap=cm.get_cmap("gray"))
        plt.colorbar()
        plt.savefig(filename % i)
        plt.clf()


def animate_2d_ffts(data, filename=None):
    """Take a 3D array and return an animated 2d plot of
    the data fourier transforms of the slices along the
    third axis"""
    fig2 = plt.figure()

    size = data.shape[2]
    ims = []
    for i in np.arange(size):
        ims.append((plt.imshow(fftshift(np.abs(fftn(data[:, :, i])))),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=50,
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
    print(values[:len(x)])
    print(pixels[:len(x)])
    print(pixels[len(x)-1])

    return (x, values[:len(x)])


def main():
    """The main body of the program"""
    size = 101
    tol = 5
    wavelength = 20

    spec = np.zeros((size, size), dtype=np.complex128)

    rs = sum([(index - (size-1)/2)**2
              for index in np.indices((size, size))])

    rs = np.round(np.sqrt(rs))

    mask_lower = rs >= (1-tol/100.0)*wavelength
    mask_upper = rs <= (1+tol/100.0)*wavelength
    mask = np.logical_and(mask_upper, mask_lower)

    spec[mask] = 1

    (x, y) = imbin(spec)

    spec *= np.exp(2*np.pi*rand(size, size)*1.0j)

    spec = fftshift(spec)

    realim = fftn(spec)
    real = np.reshape(np.tile(realim, size),
                      (size, size, size),
                      order="F")
    # animate_plot(np.transpose(real,(2,0,1)))

    spec2 = ifftshift(ifftn(real))
    # print(spec2.shape)
    # animate_plot(np.abs(spec2))
    print(spec2.shape)
    (x2, y2) = imbin(np.abs(spec2))

    plt.plot(x, y, label="Desired")
    plt.plot(x2, y2, label="3D")
    plt.legend()
    plt.show()
    plt.clf()

main()
