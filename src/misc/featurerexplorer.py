'''
Created on Jan 14, 2017

@author: safdar
'''
import matplotlib
from misc.spatialbinning import bin_spatial
from utils.plotter import Plotter
matplotlib.use('TKAgg')
import os
import matplotlib.image as mpimg
import numpy as np

# Read in the vehicle and non-vehicle images:
cars = []
notcars = []

carspath = '../data/imgs/vehicles'
cars = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(carspath)
    for f in files if f.endswith('.png')]

noncarspath = '../data/imgs/non-vehicles'
noncars = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(noncarspath)
    for f in files if f.endswith('.png')]

print ("Number of cars found: \t\t{}".format(len(cars)))
print ("Number of non-cars found: \t{}".format(len(noncars)))
mincount = min(len(cars), len(noncars))
print ("Comparing: {}".format(mincount))

# Now for each car & non-car image, explore features:
plotter = Plotter()
for (car, noncar) in zip(cars[:mincount], noncars[:mincount]):
    frame = plotter.nextframe()
    carimage = mpimg.imread(car)
    noncarimage = mpimg.imread(noncar)

    # Spatial binning:
    spatial = np.copy(carimage)
    spatial_rgb_features = bin_spatial(spatial, color_space='RGB', size=(32,32))
    
    
    # Color space:
    
    
    # HOG:
    
    
    # 