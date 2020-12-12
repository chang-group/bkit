#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pyemma
import sys


parser = argparse.ArgumentParser()

d = 'pca.h5'
h = ('HDF5 file containing a saved PyEMMA PCA or TICA object equipped with '
     + f'a FeatureReader (default: {d})')
parser.add_argument('--model', metavar='MODELFILE', default=d, help=h)

d = 'ptrajs.npz'
h = f'NumPy .npz file containing the projected trajectories (default: {d})'
parser.add_argument('--output', metavar='OUTPUTFILE', default=d, help=h)

h = ('trajectory file compatible with the FeatureReader of the '
     + 'transformation object saved in MODELFILE')
parser.add_argument('trajfiles', metavar='TRAJFILE', 
                    type=str, nargs='+', help=h)

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

