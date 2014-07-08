from scipy import misc
from scipy.fftpack import fftshift, fftfreq, ifftshift
from scipy.ndimage.measurements import label
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import kmeans2
from functools import reduce
import numpy as np
import numpy.linalg as la
from numpy.fft import fftn,ifftn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import matplotlib.animation as animation
from numpy.random import rand,randn

size = 101
tol = 5
wavelength = 20

def animate_plot(data):
    fig2 = plt.figure()

    size = data.shape[2]
    ims = []
    for i in np.arange(size):
        ims.append((plt.imshow(np.abs(data[:,:,i])),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                       blit=True)

    plt.show()

def animate_2d_ffts(data,filename=None):
    fig2 = plt.figure()

    size = data.shape[2]
    ims = []
    for i in np.arange(size):
        ims.append((plt.imshow(fftshift(np.abs(fftn(data[:,:,i])))),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                       blit=True)

    if filename:
        im_ani.save(filename)
    else:
        plt.show()

def imbin(image):
    dims = np.array(image.shape)
    dims /= 2
    
    values = np.zeros(sum(dims))
    pixels = np.zeros(sum(dims))

    indices = np.indices(tuple(image.shape))
    rs = np.zeros(image.shape,dtype=np.float32)
    for (index,dim) in zip(indices,dims):
        rs += (index-dim)**2
    rs = np.asarray(np.round(np.sqrt(rs)),dtype=np.int32)


    bins = np.arange(0,np.max(rs)+1)
    values,_ = np.histogram(rs,bins=bins,weights=image)
    pixels,_ = np.histogram(rs,bins=bins)
    
    values /= pixels

    x = np.arange(0,np.max(rs))

    return (x,values[:len(x)])



spec = np.zeros((size,size,size),dtype = np.complex128)

xs,ys,zs = np.indices((size,size,size))

xs -= (size-1)/2
ys -= (size-1)/2
zs -= (size-1)/2

rs = np.round(np.sqrt(xs**2+ys**2+zs**2))

mask_lower = rs >= (1-tol/100.0)*wavelength
mask_upper = rs <= (1+tol/100.0)*wavelength
mask = np.logical_and(mask_upper,mask_lower)

spec[mask] = 1

(x,y) = imbin(spec)

spec *= np.exp(2*np.pi*rand(size,size)*1.0j)

#animate_plot(spec)
spec = fftshift(spec)

real = fftn(spec)
animate_2d_ffts(real)

spec2 = ifftshift(ifftn(spec))
(x2,y2) = imbin(np.abs(spec))
y2 /= np.max(y2)

for i in range(0,size,10):
    (xa,ya) = imbin(real[:,:,i])
    ya /= np.max(ya)
    plt.plot(xa,ya)


plt.plot(x,y)
plt.plot(x2,y2)
plt.show()
