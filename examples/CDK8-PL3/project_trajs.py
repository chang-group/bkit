#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pyemma
import sys


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
h = ('HDF5 file containing a saved PyEMMA PCA or TICA object '
     + 'equipped with a FeatureReader')
parser.add_argument('--model', metavar='MODELFILE', 
                    default='pca.h5', help=h)
h = 'NumPy .npz file containing the projected trajectories'
parser.add_argument('--output', metavar='OUTPUTFILE', 
                    default='ptrajs.npz', help=h)
h = ('trajectory file compatible with the FeatureReader of the '
     + 'transformation object saved in MODELFILE')
parser.add_argument('trajfiles', metavar='TRAJFILE', nargs='+', help=h)
args = parser.parse_args()

transformer = pyemma.load(args.model)

for i, trajfile in enumerate(args.trajfiles):
    data = pyemma.coordinates.load(
        trajfile, features=transformer.data_producer.featurizer)
    np.save(f'.{i}.npy', transformer.transform(data))

ptrajs = {trajfile: np.load(f'.{i}.npy') 
          for i, trajfile in enumerate(trajfiles)}
np.savez(args.output, **ptrajs)

for i in range(len(trajfiles)):
    os.remove(f'.{i}.npy')

