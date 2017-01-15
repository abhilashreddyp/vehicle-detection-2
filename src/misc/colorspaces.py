'''
Created on Jan 12, 2017

@author: safdar
'''

import cv2
import numpy as np
import matplotlib
import os
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(origimgs, scaledimgs, labels, limits):


# def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
def makegrid(axeslist):
    axs = np.array(axeslist)
    
    """Plot pixels in 3D."""
    h,v = None, None
    N = len(axeslist)
    if (N > 0):
        h=int(round(np.sqrt(N)))
        v=int(np.ceil(N/h))
        diff = (h*v) - N
        for i in range(diff):
            axs.append(None)
        axs = np.reshape(axs, (v,h))
        
    fig = plt.figure(figsize=plt.figaspect(0.5))
    for i in range (v):
        for j in range(h):
            fig.add_subplot(v, h, (i*h) + j)
            
        
    # Draw in a grid:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    text = fig.suptitle("", fontsize=14, fontweight='bold')
    axes_imgs = []
    for i in range(v):
#         axes_imgs.append([])
        for j in range(h):
            if j >= len(sections[i]):
                break # We've finished one section (horizontal plots). Goto next i.

            (image, cmap, title, stats) = sections[i][j]
            idx = (i*h) + j
            axes = fig.get_axes()[idx]
            font = min (max (int( 100 / (np.sqrt(len(title)) * v * h)), 7), 15)
            axes.set_title(title, fontsize=font)
            axes.set_xticks([])
            axes.set_yticks([])
            
            #TODO: Show stats?
            axes.imshow(image, cmap=cmap)
#             axes_imgs[i].append(axesimage)
    plt.ion()
    plt.show()
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
    
        # Set axis limits
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])
    
        # Set axis labels and sizes
        ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
        ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
        ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
        ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)
    
        # Plot pixel values with colors given in colors_rgb
        ax.scatter(
            pixels[:, :, 0].ravel(),
            pixels[:, :, 1].ravel(),
            pixels[:, :, 2].ravel(),
            c=colors_rgb.reshape((-1, 3)), edgecolors='none')
    
        return ax  # return Axes3D object for further manipulation


# Read a color image
print(os.getcwd())
img = cv2.imread('../imgs/000528.png')

# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

# Plot and show
plt.figure()
plot3d(img_small_RGB, img_small_rgb)
plt.show()
plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
plt.show()

# fig, _ = plt.subplots(2,2)
# fig.get_axes()[0].imshow(img_small_RGB, None)
# fig.get_axes()[1].imshow(img_small_HSV, None)
# plt.show()
