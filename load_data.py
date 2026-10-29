#!/usr/bin/python3

import sys
import time

import numpy as np
from numpy.lib import recfunctions as rfn


##################
### PARAMETERS ###
##################

N_SAMPLES = -1 # SET TO -1 TO USE ALL SAMPLES
N_PX = 256
PATH = f"/home/victor/Data/HRTEM_data/AgCo/avec_substrat/{N_PX}px/"


###############
### PROGRAM ###
###############

start_time = time.time()

data1 = np.load(PATH + "dataset1.npy", mmap_mode="r")
data2 = np.load(PATH + "dataset2.npy", mmap_mode="r")
data3 = np.load(PATH + "dataset3.npy", mmap_mode="r")
data4 = np.load(PATH + "dataset4.npy", mmap_mode="r")
data = np.concatenate((data1, data2, data3, data4), axis=0)
random_state = np.random.RandomState(seed=42)
random_state.shuffle(data)
if N_SAMPLES >= 0 and N_SAMPLES < len(data):
    data = data[:N_SAMPLES]
elif N_SAMPLES >= len(data):
    sys.exit(f"N_SAMPLES ({N_SAMPLES}) is greater than the number of samples in the dataset ({len(data)})")
np.save("all_data.npy", data)
np.save("meta.npy", rfn.repack_fields(data[[name for name in data.dtype.names if name != "image"]])) # descriptors only, same order: train.py reads them without touching the images

print("Time to load data: ", time.time() - start_time)
