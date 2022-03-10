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

x = r * numpy.cos(phi)
y = r * numpy.sin(phi)

flux = numpy.where(r < 1., 1., 0.)

# Do the Fourier transform with TrIFT

u, v = numpy.meshgrid(numpy.linspace(-3.,3.,100),numpy.linspace(-3.,3.,100))

u = u.reshape((u.size,))
v = v.reshape((v.size,))

vis = double.sampleUnstructuredImage(x, y, flux, 4096, 0.02, u, v, 0.5, 0.25)

# Now shift the image manually.

x += 0.5
y += 0.25

vvis = double.sampleUnstructuredImage(x, y, flux, 4096, 0.02, u, v, 0., 0.)

# Plot the image to make sure we did the correct shifting.

triang = tri.Triangulation(x, y)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))

ax.tripcolor(triang, flux, "ko-")
ax.triplot(triang, "k.-", linewidth=0.1, markersize=0.1)

ax.set_aspect("equal")

ax.set_xlim(1.6,-1.6)
ax.set_ylim(-1.6,1.6)

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)

ax.tick_params(labelsize=14)

plt.show()

# Finally, plot the visibilities.

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

ax[0,0].scatter(u, v, c=vis.real, marker=".")
ax[0,1].scatter(u, v, c=vis.imag, marker=".")

ax[1,0].scatter(u, v, c=vvis.real, marker=".")
ax[1,1].scatter(u, v, c=vvis.imag, marker=".")

for a in ax.flatten():
    a.set_xlabel("u", fontsize=14)
    a.set_ylabel("v", fontsize=14)

    a.tick_params(labelsize=14)

plt.show()
