import glob
import torch
import json,os
import pandas as pd
import numpy as np
#from models import LSTMTarget
import argparse
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

#file = 'France_0_2369854_all.pkl'
n_zone = 32
path = '../data-by-user/'
file_list = os.listdir(path)

histo_dict = {}
for i in range(n_zone):
    zone_data_list = []
    for file in  file_list:
        #print(file)
        country = file.split('_')[0]
        zone = file.split('_')[1]
        if int(zone)==i and country =="United States":
            fn = path + file
            with open(fn, 'rb') as pkl:
                data = pickle.load(pkl)
            print(data[0][1])
            exit()
            zone_data_list.append(data[0][1].flatten()) 
        #print(len(np.unique(zone_data_list[0])))                
    arr = np.concatenate(zone_data_list)
    print(len(arr))
    histo= plt.hist(arr)
    histo_dict[i] = histo[0]/sum(histo[0])
#    histo= plt.hist(data[0][1].flatten())

adj = np.zeros((n_zone, n_zone))
for i in range(n_zone):
    for j in range(n_zone):
        dist = np.linalg.norm(histo_dict[i] - histo_dict[j])
        if dist < 0.12:
#            print(dist)
            adj[i][j]=1