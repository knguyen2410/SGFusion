import os
import csv
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from manager_selfatt import ZoneManager
from utils_selfatt import zone_adj, zone_adj_histo, softmax, flatten_weights,zone_adj_histo_top,getNeighbors
from copy import deepcopy
import geopandas as gpd
from shapely import wkt
from models import LSTMTarget


CLEAR_TMP = True  # clear temp memory
DEVICE = torch.device('cpu')
# DEVICE = torch.device("cuda")
n_neighbor = 1
th = 0.24

class ZoneCoordinator():
    def __init__(self, country, gdf, init_weights, ustep, lr, lr_att, device, load_ckpt=False):
        """
            args:
        """

        self.country = country
        self.device = device
        
        self.gdf = gdf[['geometry', 'uids']]

        ### init zone managers for all zones
        self.zmanagers = {zid: ZoneManager(country, zid, init_weights, active=True) for zid in range(len(self.gdf))}
        for zid in self.zmanagers:
            self.zmanagers[zid].set_uids(eval(self.gdf.loc[zid, 'uids']))

        self.num_zones_init = len(self.zmanagers)     # total number of zones at the beginning

        ### path to save models
        self.model_path = './models/zFL-noms/{}/'.format(self.country)

        # self.test_loss_fn = f'./results/test_top{n_neighbor}_{self.country}_selfatt_lr{lr}_ustep{ustep}.csv'
        
        # self.zone_loss_fn = f'./results/test_zone_top{n_neighbor}_{self.country}_selfatt_lr{lr}_ustep{ustep}.csv'

        self.test_loss_fn = f'./results/test_th{th}_{self.country}_selfatt_lr{lr}_ustep{ustep}.csv'
        
        self.zone_loss_fn = f'./results/test_zone_th{th}_{self.country}_selfatt_lr{lr}_ustep{ustep}.csv'
        

    #modify get_neighbors
    def __get_neighbors(self, zid):
        """ get all neighboring zones surrounding a zone (zid) """
        #adj = zone_adj(self.gdf)
        adj = getNeighbors(len(self.gdf), self.country) #change this
        #print(adj)
        #adj = zone_adj(self.gdf)
        #print(adj)
        #adj = zone_adj_histo_top('../data-by-user/', 16, n_neighbor)
        all_adj_zones = np.where(adj[zid]==1)[0]
        

        if len(all_adj_zones):
            neighbors = [neighbor for neighbor in all_adj_zones if self.zmanagers[neighbor].active]
            return neighbors
        else:
            return []
    

    def __save_active_models(self, path):
        ## save active zone models
        print("Saving active zone models...")
        for zm in self.zmanagers.values():
            if zm.active:   # save model if its zone is active
                save_path = path + f'zone{zm.zid}.pt'    # if save state dict, use extension .pt
                torch.save(zm.weights, save_path)

    @staticmethod
    def flatten_weights(weights, device):
        weights_concat = torch.tensor([]).to(device)
        if isinstance(weights, np.ndarray):
            for w in weights:
                w = torch.from_numpy(w).view(-1)
                weights_concat = torch.cat([weights_concat, w.to(device)], dim=-1)
        elif isinstance(weights, dict):
            for w in weights.values():
                w = w.view(-1)
                weights_concat = torch.cat([weights_concat, w.to(device)], dim=-1)
        return weights_concat


    def agg_neighbors(self,
                       zid: int,
                       lr: float,
                       ustep: int,
                       device: torch.device
                       ) -> None:
        """function to update a zone model using its neighbors' gradients"""
        neighbors = self.__get_neighbors(zid)
        if len(neighbors) == 0:
            print(f"zone {zid} has no neighbor, skip neighboring zone aggregations")
        else:
            e = {}
            grads = {}
            w_init = deepcopy(self.zmanagers[zid].weights)
            # train zone i
            uids = self.zmanagers[zid].uids
            self.zmanagers[zid].train_cross(zid, uids, lr, ustep)
            # train neighboring zones
            for nei in neighbors:
                uids = self.zmanagers[nei].uids
                self.zmanagers[zid].train_cross(nei, uids, lr, ustep)
            
            with torch.no_grad():
                #w1 = np.load(f'./tmp/{self.country}_lr{lr}_ustep{ustep}/weights_zone{zid}_cross{zid}.npy', allow_pickle=True)
                w1 = np.load(f'./tmp/{self.country}_lr{lr}_ustep{ustep}/weights_zone{zid}.npy', allow_pickle=True)
                w1_dict = {k: torch.tensor(v).to(device) for k, v in zip(w_init.keys(), w1)}
                grads[f'{zid}_{zid}'] = {k: w1_dict[k] - w_init[k].to(device) for k in w_init}
                g1_flat = flatten_weights(grads[f'{zid}_{zid}'], device)
                
                # mean = g1_flat.mean()
                # std = g1_flat.std()
                # g1_flat = (g1_flat - mean) / std

                for nei in neighbors:
                    #w2 = np.load(f'./tmp/{self.country}_lr{lr}_ustep{ustep}/weights_zone{zid}_cross{nei}.npy', allow_pickle=True)
                    w2 = np.load(f'./tmp/{self.country}_lr{lr}_ustep{ustep}/weights_zone{zid}.npy', allow_pickle=True)
                    w2_dict = {k: torch.tensor(v).to(device) for k, v in zip(w_init.keys(), w2)}
                    grads[f'{zid}_{nei}'] = {k: w2_dict[k] - w_init[k].to(device) for k in w_init}
                    g2_flat = flatten_weights(grads[f'{zid}_{nei}'], device)
                    # g2_flat = (g2_flat - mean) / std
                    e[f'{zid}_{nei}'] = torch.sigmoid( torch.matmul(g1_flat, g2_flat) )

                beta = softmax(e)

            for k in w_init:
                c = 0
                for nei in neighbors:
                    if c == 0:
                        sum_neighbors = beta[f'{zid}_{nei}'].to(device) * grads[f'{zid}_{nei}'][k]
                    else:
                        sum_neighbors += beta[f'{zid}_{nei}'].to(device) * grads[f'{zid}_{nei}'][k]
                        c += 1

                w_init[k] = w_init[k] + grads[f'{zid}_{zid}'][k] + sum_neighbors
                # make_dot(state_dict[k], params=dict(list(net.named_parameters()))).render("statedict", format="png")
            
            self.zmanagers[zid].weights = w_init


    def train_all(self, net, lr, lr_att, wd, gstep, ustep, nclient_step, device, n_workers, load_ckpt=False, ckpt_round=0):
        
        ### create directories to save results
        paths = ['./results/', f'./tmp/{self.country}_lr{lr}_ustep{ustep}',
                f'./models/zFL/{self.country}/prev_ckpt_lr{lr}_ustep{ustep}/',
                f'./models/zFL/{self.country}/best_ckpt_lr{lr}_ustep{ustep}/']
        
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # mp.set_start_method("spawn")
        best_global_loss = float(np.inf)
        if not load_ckpt:
            for round in range(gstep):
                print(f"\n******************** Global step: {round} ********************")
                start_time = time.time()

                # test_zones = [0]
                # for i in test_zones:
                for i in self.zmanagers:
                    print("self.zmanager: ", i)
                    if self.zmanagers[i].active:
                        self.agg_neighbors(i, lr, ustep, device)

                ### save test loss
                global_loss_train = eval_global(self.zmanagers, net, device, split='train')
                global_loss_test = eval_global(self.zmanagers, net, device, split='test')
                global_loss_val = eval_global(self.zmanagers, net, device, split='val')
                print(f'===> Round: {round}, train loss: {global_loss_train}, val loss: {global_loss_val}, test loss: {global_loss_test}')
                with open(self.test_loss_fn, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([round, global_loss_train, global_loss_val, global_loss_test])

                zone_loss_test = eval_zone(self.zmanagers, net, device, split='test')
                print(f'===> Round: {round}, test loss each zone: {zone_loss_test}')
                with open(self.zone_loss_fn, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(zone_loss_test)

                ### save best checkpoint
                if global_loss_test < best_global_loss:
                    best_global_loss = global_loss_test
                    path = f'./models/zFL/{self.country}/best_ckpt_lr{lr}_ustep{ustep}/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    for f in os.listdir(path):
                        os.remove(os.path.join(path, f))
                    self.__save_active_models(path)

                ### save previous checkpoint
                path = f'./models/zFL/{self.country}/prev_ckpt_lr{lr}_ustep{ustep}/'
                if not os.path.exists(path):
                    os.makedirs(path)
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                self.__save_active_models(path)

                ### empty temporary weights after the round
                CLEAR_TMP = True
                if CLEAR_TMP:
                    dir = f'./tmp/{self.country}_lr{lr}_ustep{ustep}'
                    for f in os.listdir(dir):
                        os.remove(os.path.join(dir, f))

                print(f"Finish one round in: {time.time() - start_time} seconds")
        else:
            self.zmanagers, _ = load_checkpoint(self.country, ckpt_round, ustep, lr, lr_att, device)
            for round in range(ckpt_round+1, gstep):
                print(f"\n******************** Global step: {round} ********************")
                start_time = time.time()

                # test_zones = [0]
                # for i in test_zones:
                for i in self.zmanagers:
                    print("self.zmanager: ", i)
                    if self.zmanagers[i].active:
                        self.agg_neighbors(i, lr, ustep, device)

                ### save test loss
                global_loss_train = eval_global(self.zmanagers, net, device, split='train')
                global_loss_test = eval_global(self.zmanagers, net, device, split='test')
                global_loss_val = eval_global(self.zmanagers, net, device, split='val')
                print(f'===> Round: {round}, train loss: {global_loss_train}, val loss: {global_loss_val}, test loss: {global_loss_test}')
                with open(self.test_loss_fn, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([round, global_loss_train, global_loss_val, global_loss_test])

                zone_loss_test = eval_zone(self.zmanagers, net, device, split='test')
                print(f'===> Round: {round}, test loss each zone: {zone_loss_test}')
                with open(self.zone_loss_fn, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(zone_loss_test)

                ### save best checkpoint
                if global_loss_test < best_global_loss:
                    best_global_loss = global_loss_test
                    path = f'./models/zFL/{self.country}/best_ckpt_lr{lr}_ustep{ustep}/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    for f in os.listdir(path):
                        os.remove(os.path.join(path, f))
                    self.__save_active_models(path)

                ### save previous checkpoint
                path = f'./models/zFL/{self.country}/prev_ckpt_lr{lr}_ustep{ustep}/'
                if not os.path.exists(path):
                    os.makedirs(path)
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                self.__save_active_models(path)

                ### empty temporary weights after the round
                CLEAR_TMP = True
                if CLEAR_TMP:
                    dir = f'./tmp/{self.country}_lr{lr}_ustep{ustep}/'
                    for f in os.listdir(dir):
                        os.remove(os.path.join(dir, f))

                print(f"Finish one round in: {time.time() - start_time} seconds")



def  eval_global(zmanagers, model, device, split='test'):
    """
    function to compute loss across all zones
    """
    print("Eval Global process")
    with torch.no_grad():
        all_sum_sq = []
        for zm in zmanagers.values():
            print("zm")
            if zm.active:
                print("active zm")
                _, sum_sq_error = zm.inquire_loss(model, device, split)
                all_sum_sq += sum_sq_error
        total_loss = sum([tup[0] for tup in all_sum_sq])
        total_samples = sum([tup[1] for tup in all_sum_sq])
    return torch.sqrt(total_loss/total_samples).cpu().item()

def eval_zone(zmanagers, model, device, split='test'):
    """
    function to compute loss across all zones
    """
    print("Eval Zone process")
    with torch.no_grad():
        each_sq = []
        for zm in zmanagers.values():
            if zm.active:
                zone_loss, sum_sq_error = zm.inquire_loss(model, device, split)
                each_sq.append(zone_loss)
        #         all_sum_sq += sum_sq_error
        # total_loss = sum([tup[0] for tup in all_sum_sq])
        # total_samples = sum([tup[1] for tup in all_sum_sq])
    # return torch.sqrt(total_loss/total_samples).cpu().item()
    return each_sq



def load_checkpoint(country, round, ustep, lr, lr_att, device):
    print(f"Loading check point round {round}...")
    df = pd.read_csv(f'../processed_gdf/{country}_provinces.csv')
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')


    net = LSTMTarget(inputAtts=['distance','altitude','time_elapsed'],
                     targetAtts=['heart_rate'],
                     includeTemporal=True,
                     hidden_dim=64,
                     context_final_dim=32)

    init_weights = deepcopy(net.state_dict())

    coord = ZoneCoordinator(country, gdf, init_weights, ustep, lr, lr_att, device, load_ckpt=True)

    for zid in coord.zmanagers:
        coord.zmanagers[zid].active = False
        if not len(coord.zmanagers[zid].uids):
            coord.zmanagers[zid].set_uids(eval(gdf.loc[zid, 'uids']))
            print(coord.zmanagers[zid].uids)

    model_path = f'./models/zFL/{country}/prev_ckpt_lr{lr}_ustep{ustep}/'

    files = os.listdir(model_path)
    files = [f for f in files if 'att' not in f]
    
    for f in files:
        saved_weights = torch.load(model_path + f)
        zid = int(f.split('one')[-1].split('.')[0])
        coord.zmanagers[zid].active = True
        coord.zmanagers[zid].weights = saved_weights

    return coord.zmanagers, coord.gdf