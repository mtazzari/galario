#!/usr/bin/env python3

from galario import double
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pyDOE
import numpy

# Make an image.

grid = pyDOE.lhs(2, samples=400)

r = grid[:,0] * 2
phi = grid[:,1] * 2*numpy.pi

grid = pyDOE.lhs(2, samples=2000)

r = numpy.hstack((r, grid[:,0]*0.04 + 0.98))
phi = numpy.hstack((phi, grid[:,1]*2*numpy.pi))

x = r * numpy.cos(phi) * numpy.cos(numpy.pi/3)
y = r * numpy.sin(phi)

flux = numpy.where(r < 1., y - y.min(), 0.)

pa = numpy.pi/4

xp = x * numpy.cos(-pa) - y * numpy.sin(-pa)
yp = x * numpy.sin(-pa) + y * numpy.cos(-pa)

x = xp
y = yp

# Also make a traditional image to compare with.

xx, yy = numpy.meshgrid(numpy.linspace(15.,-15.,1024, endpoint=False), \
        numpy.linspace(15.,-15.,1024, endpoint=False))

xp = xx * numpy.cos(pa) - yy * numpy.sin(pa)
yp = xx * numpy.sin(pa) + yy * numpy.cos(pa)

rr = numpy.sqrt((xp/numpy.cos(numpy.pi/3))**2 + yp**2)

fflux = numpy.where(rr < 1., yp - y.min(), 0.)

# Plot the image.

triang = tri.Triangulation(x, y)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

ax[0].tripcolor(triang, flux, "ko-")
ax[0].triplot(triang, "k.-", linewidth=0.1, markersize=0.1)

ax[1].imshow(fflux, interpolation="nearest")

for i in range(1):
    ax[i].set_aspect("equal")

    ax[i].set_xlim(1.1,-1.1)
    ax[i].set_ylim(-1.1,1.1)

    ax[i].set_xlabel("x", fontsize=14)
    ax[i].set_ylabel("y", fontsize=14)

    ax[i].tick_params(labelsize=14)

plt.show()

# Do the Fourier transform with TrIFT

u, v = numpy.meshgrid(numpy.linspace(-3.,3.,100),numpy.linspace(-3.,3.,100))

u = u.reshape((u.size,))
v = v.reshape((v.size,))

vis = double.sampleUnstructuredImage(x, y, flux, 1024*4, 0.025/4, u, v, 0., 0.)

# Do the Fourier transform with GALARIO.

dxy = abs(xx[0,1] - xx[0,0])

vvis = double.sampleImage(fflux, dxy, u, v)

# Finally, plot the visibilities.

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

ax[0,0].scatter(u, v, c=vis.real/vis.real.max(), marker=".")
ax[0,1].scatter(u, v, c=vis.imag/vis.imag.max(), marker=".")

ax[1,0].scatter(u, v, c=vvis.real/vvis.real.max(), marker=".")
ax[1,1].scatter(u, v, c=vvis.imag/vvis.imag.max(), marker=".")

for a in ax.flatten():
    a.set_xlabel("u", fontsize=14)
    a.set_ylabel("v", fontsize=14)

    a.tick_params(labelsize=14)

plt.show()
