import glob
import torch
import json
import numpy as np
import pandas as pd
from models import LSTMTarget
import argparse
import pickle



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='preprocess country data, split data into zones and users')
    parser.add_argument('--country', default='Norway', type=str)
    args = parser.parse_args()

    # torch.manual_seed(11)

    net = LSTMTarget(inputAtts=['distance','altitude','time_elapsed'],
                     targetAtts=['heart_rate'],
                     includeTemporal=True,
                     hidden_dim=64,
                     context_final_dim=32)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    core_users_3 = json.load(open(f'../preprocess/{args.country}_core_users_3.json', 'r'))
    core_users_3 = {k: eval(v) for k, v in core_users_3.items()}

    core_users_10 = json.load(open(f'../preprocess/{args.country}_core_users_10.json', 'r'))
    core_users_10 = {k: eval(v) for k, v in core_users_10.items()}
        
    all_models = glob.glob(f'./models/zFL/{args.country}/best_ckpt_lr0.0025_ustep5/zone*')


    path = '../data-by-user/'
    sum_sq = 0.
    total_samples = 0
    with torch.no_grad():
        for model in all_models:
            zone = model.split('/')[-1].split('.')[0].split('ne')[-1]

            # load uids for zone    
            uids_10 = [uid for uid in core_users_10 if f'_{zone}_' in uid]
            uids_3 = [uid for uid in core_users_3 if f'_{zone}_' in uid]

            # load weights
            weights = torch.load(model, map_location=torch.device('cpu'))
            net.load_state_dict(weights)

            # load data
            for uid in uids_10:
                with open(path + uid + '_test' + '.pkl', 'rb') as pkl:
                    data = pickle.load(pkl)
                for batch in data:
                    n_samples = batch[1].shape[0]
                    total_samples += n_samples
                    inputs, target = net.embed_inputs(batch)
                    inputs = inputs.float().to(device)
                    out = net(inputs, device)
                    sum_sq += torch.mean((out - target.to(device))**2) * n_samples

            if len(uids_3) == 0:
                continue
            for uid in uids_3:
                with open(path + uid + '_all.pkl', 'rb') as pkl:
                    data = pickle.load(pkl)
                for batch in data:
                    n_samples = batch[1].shape[0]
                    total_samples += n_samples
                    inputs, target = net.embed_inputs(batch)
                    inputs = inputs.float().to(device)
                    out = net(inputs, device)
                    sum_sq += torch.mean((out - target.to(device))**2) * n_samples
            
        loss = torch.sqrt(sum_sq/total_samples)

    print(loss)


        

