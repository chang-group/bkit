#!/usr/bin/env python3

import numpy as np
import os
import pyemma
import sys


transformer = pyemma.load('pca.h5')
trajfiles = sys.argv[1:]

for i, trajfile in enumerate(trajfiles):
    data = pyemma.coordinates.load(
        trajfile, features=transformer.data_producer.featurizer)
    np.save(f'.{i}.npy', transformer.transform(data))

ptrajs = {trajfile: np.load(f'.{i}.npy') 
          for i, trajfile in enumerate(trajfiles)}
np.savez('ptrajs.npz', **ptrajs)

for i in range(len(trajfiles)):
    os.remove(f'.{i}.npy')
 
