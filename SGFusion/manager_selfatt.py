import os
import glob
import random
import pickle
import torch
import numpy as np
from flower_utils import train_flower_cross, train_flower_diff
from typing import Optional, List

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cuda")
device = torch.device("cpu")
class ZoneManager():
    def __init__(self, country, zid, weights, active=True):
        self.country = country
        self.zid = zid
        self.weights = weights
        self.active = active
        files = glob.glob(f'../data-by-user/{country}_{zid}*_train*')
        uids = [f.split('_')[-2] for f in files]
        self.uids = [f'{country}_{zid}_{uid}' for uid in uids]


    def set_uids(self, uids):
        self.uids = uids
        # print(f'zid: {self.zid}, uids: {self.uids}')
    

    def train_diff(self,
                   net,
                   neighbors: List[int],
                   lr:float,
                   lr_att: float,
                   ustep: int,
                   nclient_step: Optional[int],
                   device: torch.device,
                   use_att: bool = True,
                   ) -> None:
        print(f'\n===> Training (diffusion) on country: {self.country}, zone: {self.zid}, neighbors: {neighbors}')

        if nclient_step == 'all':
            selected_uids = self.uids
        else:
            try:
                selected_uids = random.sample(self.uids, nclient_step)
            except ValueError:
                selected_uids = self.uids

        fed_dir = '../data-by-user/'
        train_flower_diff(self.country,
                          self.zid,
                          neighbors,
                          selected_uids,
                          fed_dir,
                          self.weights,
                          use_att=use_att,
                          lr=lr,
                          lr_att=lr_att,
                          ustep=ustep,
                          )


    def train_cross(self,
                    zid_cross: int,
                    uids: List[str],
                    lr: float,
                    ustep: int) -> None:
        print(f'Cross training on country: {self.country}, zone: {self.zid}, cross: {zid_cross}')
        fed_dir = '../data-by-user/'
        train_flower_cross(self.country,
                           self.zid,
                           zid_cross,
                           self.weights,
                           fed_dir,
                           uids,
                           lr=lr,
                           ustep=ustep,
                           num_rounds=1)


    @torch.no_grad()
    def inquire_loss(self, net, device, split='val'):
        net.eval()
        net.load_state_dict(self.weights)
        path = '../data-by-user/'
        loss_tuples = []
        for uid in self.uids:
            sum_sq = 0.
            total_samples = 0
            with open(path + uid + '_' + split + '.pkl', 'rb') as pkl:
                data = pickle.load(pkl)
            for batch in data:
                n_samples = batch[1].shape[0]
                total_samples += n_samples
                inputs, target = net.embed_inputs(batch)
                inputs = inputs.float().to(device)
                out = net(inputs, device)
                sum_sq += torch.mean((out - target.to(device))**2) * n_samples
            loss_tuples.append((sum_sq, total_samples))
        sum_loss = sum([tup[0] for tup in loss_tuples])
        sum_samples = sum([tup[1] for tup in loss_tuples])
        zone_loss = torch.sqrt(sum_loss/sum_samples)
        return zone_loss, loss_tuples


    def get_zone_size(self, split='val'):
        n_samples = 0
        for uid in self.uids:
            with open('../data-by-user/' + uid + '_' + split + '.pkl', 'rb') as pkl:
                data = pickle.load(pkl)
            for batch in data:
                n_samples += batch[1].shape[0]
        return n_samples