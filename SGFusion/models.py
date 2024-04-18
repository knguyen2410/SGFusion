from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Dict, List, Tuple
# from torchviz import make_dot 


use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
device = torch.device('cpu')
# device = torch.device("cuda")
Tensor = torch.Tensor


def softmax(e_ij: Dict[str, torch.DoubleTensor]) -> Dict[str, torch.DoubleTensor]:
    res = {}
    denom = [torch.exp(val.type(torch.DoubleTensor)) for val in e_ij.values()]
    denom = torch.stack(denom)
    denom = torch.sum(denom, dim=0)
    for k, v in e_ij.items():
        res[k] = torch.exp(v.type(torch.DoubleTensor)) / denom
    return res


class LSTMTarget(nn.Module):
    def __init__(self, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, hidden_dim=64, context_final_dim=32):
        super(LSTMTarget, self).__init__()

        self.dropout_rate = .2

        self.inputAtts = inputAtts
        self.targetAtts = 'tar_' + targetAtts[0]
        self.includeTemporal = includeTemporal
        self.trimmed_workout_len = 450

        self.input_dim = len(self.inputAtts)
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.context_final_dim = context_final_dim

        # build the context embedding for workout profile forecasting
        total_input_dim = self.input_dim
        if self.includeTemporal:
            self.context1_dim = self.input_dim + 1    # 4 
            self.context2_dim = 1                     # 1
 
            self.context_layer_1 = nn.LSTM(input_size = self.context1_dim, hidden_size = self.hidden_dim, batch_first=True).to(device)
            self.context_layer_2 = nn.LSTM(input_size = self.context2_dim, hidden_size = self.hidden_dim, batch_first=True).to(device)
            self.dropout_context = nn.Dropout(self.dropout_rate)
            self.project = nn.Linear(self.hidden_dim * 2, self.context_final_dim).to(device)

            total_input_dim += self.context_final_dim

        self.lstm_stacked = nn.LSTM(input_size=total_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=.2)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        self.lstm_stacked = self.lstm_stacked.to(device)
        self.linear = self.linear.to(device)


    def init_hidden(self, x, n_layers=1):
        return Variable(torch.zeros(n_layers, x.size(0), self.hidden_dim)) # dimension 0: batch


    def embed_inputs(self, batch):
        inputs = Variable(torch.tensor(batch[0]['input'])).to(device)   # size = [52, 500, 3]
        context_input_1 = batch[0]['context_input_1']
        context_input_2 = batch[0]['context_input_2']
        context_input_1 = Variable(torch.from_numpy(context_input_1).float()).to(device)
        context_input_2 = Variable(torch.from_numpy(context_input_2).float()).to(device)

        context_input_1 = self.dropout_context(context_input_1)   # [bs, 450, 4]
        context_input_2 = self.dropout_context(context_input_2)   # [bs, 450, 1]

        hidden_1 = self.init_hidden(inputs).to(device)  # [1, bs, 64]
        cell_1 = self.init_hidden(inputs).to(device)    # [1, bs, 64]
        hidden_2 = self.init_hidden(inputs).to(device)  # [1, bs, 64]
        cell_2 = self.init_hidden(inputs).to(device)    # [1, bs, 64]


        out1, (_, _) = self.context_layer_1(context_input_1, (hidden_1, cell_1)) # [bs, 450, 64] 
        out2, (_, _) = self.context_layer_2(context_input_2, (hidden_2, cell_2)) # [bs, 450, 64]

        out12 = torch.cat([out1, out2], dim=-1)  # [bs, 450, 128]
        all_outputs = self.project(out12)

        all_inputs = torch.cat([inputs, all_outputs], dim=-1)
        outputs = torch.tensor(batch[1]).float()
        return all_inputs, outputs


    def forward(self,
                embedded_inputs: Tensor,
                device: torch.device,
                ) -> Tensor:
        h_t = self.init_hidden(embedded_inputs, n_layers=2).float().to(device) #[num_layers(2), batch, hid(64)]
        c_t = self.init_hidden(embedded_inputs, n_layers=2).float().to(device) #[num_layers(2), batch, hid(64)]

        result, (h_t, c_t) = self.lstm_stacked(embedded_inputs, (h_t, c_t))   # result.shape = [bs, 450, hid]
        result = F.selu(self.linear(result))  # [bs, 450, 1]
        return result



class LSTM_custom(nn.Module):
    def __init__(self, inputAtts=['distance','altitude','time_elapsed'], targetAtts=['heart_rate'], includeTemporal=True, hidden_dim=64, context_final_dim=32):
        super(LSTM_custom, self).__init__()

        self.dropout_rate = .2

        self.inputAtts = inputAtts
        self.targetAtts = 'tar_' + targetAtts[0]
        self.includeTemporal = includeTemporal
        self.trimmed_workout_len = 450

        self.input_dim = len(self.inputAtts)
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.context_final_dim = context_final_dim

        # build the context embedding for workout profile forecasting
        total_input_dim = self.input_dim
        if self.includeTemporal:
            self.context1_dim = self.input_dim + 1   # 4
            self.context2_dim = 1

            self.context_layer_1 = nn.LSTM(input_size = self.context1_dim, hidden_size = self.hidden_dim, batch_first=True).to(device)
            self.context_layer_2 = nn.LSTM(input_size = self.context2_dim, hidden_size = self.hidden_dim, batch_first=True).to(device)
            self.dropout_context = nn.Dropout(self.dropout_rate)
            self.project = nn.Linear(self.hidden_dim * 2, self.context_final_dim).to(device)

            total_input_dim += self.context_final_dim

        self.lstm_stacked = nn.LSTM(input_size=total_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=.2)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        if use_cuda:
            self.lstm_stacked = self.lstm_stacked.to(device)
            self.linear = self.linear.to(device)


    def init_hidden(self, x, n_layers=1):
        return Variable(torch.zeros(n_layers, x.size(0), self.hidden_dim)) # dimension 0: batch


    def embed_inputs(self, batch):
        with torch.no_grad():
            inputs = Variable(torch.tensor(batch[0]['input'])).to(device)   # size = [52, 500, 3]
            outputs = torch.tensor(batch[1]).float()
            context_input_1 = batch[0]['context_input_1']
            context_input_2 = batch[0]['context_input_2']
            context_input_1 = Variable(torch.from_numpy(context_input_1).float()).to(device)
            context_input_2 = Variable(torch.from_numpy(context_input_2).float()).to(device)

            context_input_1 = self.dropout_context(context_input_1)
            context_input_2 = self.dropout_context(context_input_2)

        return inputs, outputs, (context_input_1, context_input_2)


    def lstm_forward(self, x, Wih, Whh, bih, bhh, h=None, c=None):
        if h is None:
            h = self.init_hidden(x).to(device)
        if c is None:
            c = self.init_hidden(x).to(device)
        seq_len = x.size(1)
        HS = self.hidden_dim
        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            gates = x_t @ Wih.T + h @ Whh.T + bih + bhh
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)
            hidden_seq.append(h.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h, c)



    def forward(self,
                inputs,
                context_inputs: Tuple[Tensor],
                device: torch.device,
                state_dict=None,
                ) -> Tensor:

        Wih_c1 = state_dict['context_layer_1.weight_ih_l0']
        Whh_c1 = state_dict['context_layer_1.weight_hh_l0']
        bih_c1 = state_dict['context_layer_1.bias_ih_l0']
        bhh_c1 = state_dict['context_layer_1.bias_hh_l0']

        Wih_c2 = state_dict['context_layer_2.weight_ih_l0']
        Whh_c2 = state_dict['context_layer_2.weight_hh_l0']
        bih_c2 = state_dict['context_layer_2.bias_ih_l0']
        bhh_c2 = state_dict['context_layer_2.bias_hh_l0']

        W_pro = state_dict['project.weight']
        b_pro = state_dict['project.bias']

        Wih_1 = state_dict['lstm_stacked.weight_ih_l0']
        Whh_1 = state_dict['lstm_stacked.weight_hh_l0']
        bih_1 = state_dict['lstm_stacked.bias_ih_l0']
        bhh_1 = state_dict['lstm_stacked.bias_hh_l0']
        
        Wih_2 = state_dict['lstm_stacked.weight_ih_l1']
        Whh_2 = state_dict['lstm_stacked.weight_hh_l1']
        bih_2 = state_dict['lstm_stacked.bias_ih_l1']
        bhh_2 = state_dict['lstm_stacked.bias_hh_l1']

        W_lin = state_dict['linear.weight']
        b_lin = state_dict['linear.bias']

        h1 = self.init_hidden(inputs).to(device)
        c1 = self.init_hidden(inputs).to(device)
        h2 = self.init_hidden(inputs).to(device)
        c2 = self.init_hidden(inputs).to(device)
        context_1, context_2 = context_inputs    # [bs, 450, 4]   [bs, 450, 1]

        out1, (_, _) = self.lstm_forward(context_1, Wih_c1, Whh_c1, bih_c1, bhh_c1, h1.squeeze(0), c1.squeeze(0))
        out2, (_, _) = self.lstm_forward(context_2, Wih_c2, Whh_c2, bih_c2, bhh_c2, h2.squeeze(0), c2.squeeze(0))

        out12 = torch.cat([out1, out2], dim=-1)
        context_features = out12 @ W_pro.T + b_pro

        all_inputs = torch.cat([inputs, context_features], dim=-1)    # [bs, 450, 32 + 3]

        seq_len = all_inputs.size(1)
        HS = self.hidden_dim

        h_t = self.init_hidden(all_inputs, n_layers=2).float().to(device) #[stack,batch,hid]
        c_t = self.init_hidden(all_inputs, n_layers=2).float().to(device)

        h_t_1 = h_t[0, :, :]
        h_t_2 = h_t[1, :, :]
        c_t_1 = c_t[0, :, :]
        c_t_2 = c_t[1, :, :]

        hidden_seq = []

        for t in range(seq_len):
            x_t = all_inputs[:, t, :]

            # forward through LSTM 1
            gates = x_t @ Wih_1.T + h_t_1 @ Whh_1.T + bih_1 + bhh_1
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t_1 = f_t * c_t_1 + i_t * g_t
            h_t_1 = o_t * torch.tanh(c_t_1)

            # forward through LSTM 2
            gates = h_t_1 @ Wih_2.T + h_t_2 @ Whh_2.T + bih_2 + bhh_2
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t_2 = f_t * c_t_2 + i_t * g_t   # [2, 128, 16]
            h_t_2 = o_t * torch.tanh(c_t_2)   # [2, 128, 16]

            hidden_seq.append(h_t_2.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        result = F.selu( hidden_seq @ W_lin.T + b_lin )
        return result