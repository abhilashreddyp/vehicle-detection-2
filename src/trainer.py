'''
Created on Dec 21, 2016

@author: safdar
'''
import matplotlib
from sklearn.preprocessing.data import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm.classes import LinearSVC
import time
from sklearn.utils import shuffle
matplotlib.use('TKAgg')
import matplotlib.image as mpimg
from extractors.spatialbinner import SpatialBinner
from extractors.hogextractor import HogExtractor
from extractors.colorhistogram import ColorHistogram
from extractors.featurecombiner import FeatureCombiner
import argparse
import numpy as np
import os
from sklearn.externals import joblib

CAR_FLAG = 1
NOTCAR_FLAG = 0

def getallpathsunder(path):
    if not os.path.isdir(path):
        raise "Folder {} does not exist, or is not a folder".format(path)
    cars = [os.path.join(dirpath, f)
            for dirpath, _, files in os.walk(path)
            for f in files if f.endswith('.png')]
    return cars

def appendXYs(imagefiles, extractor, label, Xs, Ys):
    for file in imagefiles:
        image = mpimg.imread(file)
        Xs.append(extractor.extract(image))
        Ys.append(label)

if __name__ == '__main__':
    print ("###############################################")
    print ("#                   TRAINER                   #")
    print ("###############################################")

    parser = argparse.ArgumentParser(description='Object Classifier')
    parser.add_argument('-vf', dest='vehicledir',    required=True, type=str, help='Path to folder containing vehicle images.')
    parser.add_argument('-nf', dest='nonvehicledir',    required=True, type=str, help='Path to folder containing non-vehicle images.')
    parser.add_argument('-o', dest='output',   required=True, type=str, help='File to store trainer parameters for later use')
    parser.add_argument('-tr', dest='testratio', default=0.10, type=float, help='% of training data held aside for testing.')
    parser.add_argument('-d', dest='dry', action='store_true', help='Dry run. Will not save anything to disk (default: false).')
    args = parser.parse_args()

    # Create all the extractors here:
    spatialex = SpatialBinner()
    colorhistex = ColorHistogram()
    hogex = HogExtractor()
    combiner = FeatureCombiner((spatialex, colorhistex, hogex))

    # Collect the image file names:
    cars = getallpathsunder(args.vehicledir)
    print ("Number of cars found: \t{}".format(len(cars)))
    assert len(cars)>0, "There should be at least one vehicle image to process. Found 0."
    notcars = getallpathsunder(args.nonvehicledir)
    print ("Number of non-cars found: \t{}".format(len(notcars)))
    assert len(notcars)>0, "There should be at least one non-vehicle image to process. Found 0."

    # Prepare feature vectors:
    Xs, Ys = [], []
    appendXYs(cars, combiner, CAR_FLAG, Xs, Ys)
    appendXYs(notcars, combiner, NOTCAR_FLAG, Xs, Ys)
    Xs = np.array(Xs, dtype=np.float64)

    # Prepare data:
    # - Normalize, shuffle and split:
    X_scaler = StandardScaler().fit(Xs)
    scaled_Xs = X_scaler.transform(Xs)
    rand_state = np.random.randint(0, 100)
    scaled_Xs, Ys = shuffle(scaled_Xs, Ys, random_state=rand_state)
    X_train, X_test, Y_train, Y_test = train_test_split(scaled_Xs, Ys, test_size=args.testratio, random_state=rand_state)

    # Train the classifier using a linear SVC 
    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, Y_train)
    t2 = time.time()
    print(t2-t, 'Seconds to train SVC...')
    print('Train Accuracy of SVC = ', svc.score(X_train, Y_train))
    print('Test Accuracy of SVC = ', svc.score(X_test, Y_test))
    t=time.time()
    prediction = svc.predict(X_test[0].reshape(1, -1))
    t2 = time.time()
    print(t2-t, 'Seconds to predict with SVC')
    
    if not args.dry:
        print ("Saving checkpoints to file: {}".format(args.output))
        joblib.dump(svc, args.output)
#         pickle.dump( dist_pickle, open( args.output, "wb" ) )
#         clf = joblib.load('filename.pkl')

    print ("Thank you. Come again!")


