import os
import torch
import numpy as np
import flwr as fl
from typing import List, Tuple, Union

from flwr.common import Metrics
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy

from pathlib import Path
from flwr.common.typing import Scalar
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List
from utils_selfatt import train, test, get_hr_dataset
from models import LSTMTarget


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')
#DEVICE = torch.device("cuda")
# DEVICE = "cpu"

net = LSTMTarget(inputAtts=['distance','altitude','time_elapsed'],
                 targetAtts=['heart_rate'],
                 includeTemporal=True,
                 hidden_dim=64,
                 context_final_dim=32)


class HrpRayClient(fl.client.NumPyClient):
    def __init__(self,
                 country: str,
                 zid: int,
                 uid: str,
                 fed_dir: str,
                 weights: Dict[str, torch.Tensor],
                 neighbors: List[int] = None,
                 use_att: bool = False):
        """
        args:
            country:

        """
        self.country = country
        self.zid = zid
        self.uid = uid
        self.fed_dir = Path(fed_dir)
        self.weights = weights
        self.neighbors = neighbors
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = DEVICE
        self.use_att = use_att


    def get_parameters(self):
        # para_list = []
        # for _, val in net.state_dict().items():
        #     if val.isnan().any():
        #         print ("nan found in this client weights. Set the weights to 0.")
        #         return 0
        #     else:
        #         para_list.append(val.cpu().numpy())
        # return para_list
        return [val.cpu().numpy() for val in net.state_dict().values()]

    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        print(f"fit() on client uid={self.uid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        trainloader = get_hr_dataset(
            self.fed_dir,
            self.uid,
            partition='train'
        )

        ### send model to device
        net.load_state_dict(self.weights)
        net.to(self.device)

        dataset_size = train(net,
                             trainloader,
                             lr=config["learning_rate"],
                             ustep=int(config["ustep"]),
                             device=self.device,
                             )

        # return local model and statistics
        # return self.get_parameters(), len(trainloader.dataset), {}
        para_list = self.get_parameters()
        for para_arr in para_list:
            if np.isnan(para_arr).any():
                print ("nan found in client %s's weights. Set the weights to 0."%self.cid)
                return 0,0,{}
        return para_list, dataset_size, {}


    def evaluate(self, parameters, config):
        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        ### load data for this client
        valloader = get_hr_dataset(
            self.fed_dir, self.uid, partition='val'
        )

        ### send model to device
        net.to(self.device)

        ### evaluate
        loss, dataset_size = test(net, valloader, device=self.device)
        return float(loss),  dataset_size, {"RMSE":float(loss)}



def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

class FedAvgWithStragglerDrop(fl.server.strategy.FedAvgZoneCross):
    """Custom FedAvg which discards updates from stragglers."""

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Discard all the models sent by the clients that were stragglers."""
        # Record which client was a straggler in this round
        country = self.config["country"]
        zid = self.config["zid"]
        lr = self.config["lr"]
        ustep = self.config["ustep"]
        zid_cross = self.config["zid_cross"]
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]

        # keep those results that are not from stragglers
        results = [res for i, res in enumerate(results) if not stragglers_mask[i]]

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        aggregated_params = super().aggregate_fit(rnd, results, failures)
        new_aggregated_params = aggregated_params[0]
        if new_aggregated_params is not None:
            ## save aggregated_weights
            print(f"Saving aggregated_weights, country: {country}, zone: {zid}, lr: {lr}, ustep: {ustep}...")
            if not os.path.exists(f"./tmp/{country}_lr{lr}_ustep{ustep}"):
                os.makedirs(f"./tmp/{country}_lr{lr}_ustep{ustep}")
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(new_aggregated_params)
            np.save(f"./tmp/{country}_lr{lr}_ustep{ustep}/weights_zone{zid}_cross{zid_cross}.npy", aggregated_weights, allow_pickle=True)
        else:
            aggregated_weights = 0
        return aggregated_weights

class SaveModelStrategyCross(fl.server.strategy.FedAvgZoneCross):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Weights], Dict[str, Scalar]]:
        country = self.config["country"]
        zid = self.config["zid"]
        lr = self.config["lr"]
        ustep = self.config["ustep"]
        zid_cross = self.config["zid_cross"]
        aggregated_params = super().aggregate_fit(rnd, results, failures)
        new_aggregated_params = aggregated_params[0]
        if new_aggregated_params is not None:
            ## save aggregated_weights
            print(f"Saving aggregated_weights, country: {country}, zone: {zid}, lr: {lr}, ustep: {ustep}...")
            if not os.path.exists(f"./tmp/{country}_lr{lr}_ustep{ustep}"):
                os.makedirs(f"./tmp/{country}_lr{lr}_ustep{ustep}")
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(new_aggregated_params)
            np.save(f"./tmp/{country}_lr{lr}_ustep{ustep}/weights_zone{zid}_cross{zid_cross}.npy", aggregated_weights, allow_pickle=True)
        else:
            aggregated_weights = 0
        return aggregated_weights

    def configure_evaluate(self, rnd, parameters, client_manager):
        """ disable client evaluation, comment out this method to recover default behavior """
        return None


class SaveModelStrategy(fl.server.strategy.FedAvgZone):
    """to be used for train_flower_diff"""
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Weights], Dict[str, Scalar]]:
        country = self.config["country"]
        zid = self.config["zid"]
        lr = self.config["lr"]
        ustep = self.config["ustep"]
        aggregated_params = super().aggregate_fit(rnd, results, failures)
        new_aggregated_params = aggregated_params[0]
        if new_aggregated_params is not None:
            ## save aggregated_weights
            print(f"Saving aggregated_weights, country: {country}, zone: {zid}, lr: {lr}, ustep: {ustep}...")
            if not os.path.exists(f"./tmp/{country}_lr{lr}_ustep{ustep}"):
                os.makedirs(f"./tmp/{country}_lr{lr}_ustep{ustep}")
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(new_aggregated_params)
            np.save(f"./tmp/{country}_lr{lr}_ustep{ustep}/weights_zone{zid}.npy", aggregated_weights, allow_pickle=True)
        else:
            aggregated_weights = 0
        return aggregated_weights

    def configure_evaluate(self, rnd, parameters, client_manager):
        """ disable client evaluation, comment out this method to recover default behavior """
        return None



def train_flower_cross(country: str,
                       zid: int,
                       zid_cross: int,
                       weights,
                       fed_dir: str,
                       uids: List[str],
                    #    num_client_gpus: float = 0,
                       num_client_cpus: float = 1,
                       num_rounds: int = 1,
                       ustep: int = 5,
                       lr: float = 0.001,
                       ) -> None:

    def client_fn(uid: str) -> HrpRayClient:
        # create a single client instance
        print("client_fn check")
        return HrpRayClient(country, zid, uid, fed_dir, weights, neighbors=None, use_att=False)


    def fit_config(rd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        print("fit_config check")
        config = {
            "epoch_global": str(rd),
            "ustep": str(ustep),  # number of local epochs
            "learning_rate": lr,
        }
        return config

    strategy = SaveModelStrategyCross(
        country=country,
        zid=zid,
        zid_cross=zid_cross,
        lr=lr,
        ustep=ustep,
        beta=None,
        fraction_fit=1,
        min_fit_clients=len(uids),
        min_available_clients=len(uids),  # All clients should be available
        fraction_eval=1,
        min_eval_clients=len(uids),
        on_fit_config_fn=fit_config,
    )
    
    # strategy = SaveModelStrategy(
    #     country=country,
    #     zid=zid,
    #     #zid_cross=zid_cross,
    #     lr=lr,
    #     ustep=ustep,
    #     #beta=None,
    #     fraction_fit=1,
    #     min_fit_clients=len(uids),
    #     min_available_clients=len(uids),  # All clients should be available
    #     fraction_eval=1,
    #     min_eval_clients=len(uids),
    #     on_fit_config_fn=fit_config,
    # )

    client_resources = {
        # "num_gpus": num_client_gpus,
        "num_cpus": num_client_cpus
    }

    ray_config = {"include_dashboard": False}
    print("flower cross check 1")
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=uids,
        client_resources=client_resources,
        num_rounds=num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )


def train_flower_diff(country: str,
                      zid: int,
                      neighbors: List[int],
                      uids: List[str],
                      fed_dir:str,
                      weights,
                      use_att: bool = True,
                      ustep: int = 5,
                      lr: float = 0.001,
                      lr_att: float = 0.001,
                      num_client_gpus: float = 1,
                      num_client_cpus: float = 4,
                      num_rounds: int = 1,
                      ) -> None:

    def client_fn(uid: str):
        # create a single client instance
        return HrpRayClient(country, zid, uid, fed_dir, weights, neighbors=neighbors, use_att=True)

    def fit_config(rd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rd),
            "ustep": ustep,  # number of local epochs
            "learning_rate": lr,
            "lr_att": lr_att,
            # "batch_size": bs
        }
        return config

    strategy = SaveModelStrategy(
        country=country,
        zid=zid,
        lr=lr,
        ustep=ustep,
        fraction_fit=1,
        min_fit_clients=len(uids),
        min_available_clients=len(uids),  # All clients should be available
        fraction_eval=1,
        min_eval_clients=len(uids),
        on_fit_config_fn=fit_config,
    )

    client_resources = {
        "num_gpus": num_client_gpus,
        "num_cpus": num_client_cpus
    }

    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        clients_ids=uids,
        client_resources=client_resources,
        num_rounds=num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )
