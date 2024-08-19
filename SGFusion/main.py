from tkinter import E
import torch
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from models import LSTMTarget
from coordinator_selfatt import ZoneCoordinator
from copy import deepcopy
import logging as log


# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
# DEVICE = torch.device("cuda")
if __name__ == '__main__':

    ### command line argument parser
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--country", type=str, default="United States")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_att", type=float, default=1e-2, help="attention learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay rate")
    parser.add_argument("--gstep", type=int, default=200, help="number of global rounds")
    parser.add_argument("--ustep", type=int, default=5, help="number of client epochs")
    parser.add_argument("--nclient_step", default='all', help="number of sampled clients per round")
    parser.add_argument("--n_workers", type=int, default=3, help="number of workers for parallel training")
    parser.add_argument("--load_ckpt", type=bool, default=False, help="whether to load checkpoint")
    parser.add_argument("--ckpt_round", type=int, default=0)
    args = parser.parse_args()


    ### hyper-params
    kwargs = {'lr': args.lr,
              'lr_att': args.lr_att,
              'wd': args.wd,
              'gstep': args.gstep,
              'ustep': args.ustep,
              'nclient_step': args.nclient_step,
              'device': DEVICE,
              'n_workers': args.n_workers,
              'load_ckpt': args.load_ckpt,
              'ckpt_round': args.ckpt_round,
              }


    ### load geopandas dataframe
    df = pd.read_csv('../processed_gdf/{}_provinces.csv'.format(args.country))
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')

    gdf['parent'] = None
    gdf['child'] = None

    ### initialize model prototypes for zone and client
    net = LSTMTarget(inputAtts=['distance','altitude','time_elapsed'],
                     targetAtts=['heart_rate'],
                     includeTemporal=True,
                     hidden_dim=64,
                     context_final_dim=32)


    init_weights = deepcopy(net.state_dict())
    # for k, v in init_weights.items():
    #     print(k, v.shape)
    # exit()

    ### init a zone coordinator
    coord = ZoneCoordinator(args.country, gdf, init_weights, args.ustep, args.lr, args.lr_att, kwargs['device'], kwargs['load_ckpt'])


    coord.train_all(net, **kwargs)
