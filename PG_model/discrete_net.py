import torch
import time
import numpy as np
from torch import nn
from H3_battle import Battle
from H3_battle import actionType
from H3_battle import logger
from tianshou.data import ReplayBuffer
dist_fn = torch.distributions.Categorical


def unsparse(mask, size):
    value_size = len(mask)
    I = torch.tensor([[0] * value_size, mask])
    mask_targets = torch.sparse_coo_tensor(I, torch.ones(value_size), (1, size))
    return mask_targets


class my_reshape(nn.Module):
    def __init__(self,shape):
        super(my_reshape, self).__init__()
        self.shape = shape
    def forward(self,x):
        x = x.reshape(*self.shape)
        return x
class in_pipe(nn.Module):
    def __init__(self, device='cpu'):
        super(in_pipe,self).__init__()
        self.device = device
        self.id_emb = nn.Embedding(150, 16, padding_idx=122)
        self.stack_fc = nn.Sequential(nn.Linear(16 + 16, 64),
                                      my_reshape([-1,64 * 14]),
                                      nn.Linear(64 * 14, 512),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(512, 512),
                                      nn.ReLU(inplace=True))
        self.stack_plane_conv  = nn.Sequential(my_reshape([-1, 3, 11, 17]),
                                               nn.Conv2d(3, out_channels=8, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(kernel_size=2),
                                               my_reshape([-1, 14 * 8, 5, 8]),
                                               nn.Conv2d(14 * 8, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 32 * 5 * 8]))
        self.global_plane_conv = nn.Sequential(my_reshape([-1, 3, 11, 17]),
                                               nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(kernel_size=2),
                                               nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 32 * 5 * 8]))
        self.stack_plane_flat = nn.Linear(512 + 32 * 5 * 8 * 2, 512)
        #
        self.id_emb.to(self.device)
        self.stack_fc.to(self.device)
        self.stack_plane_conv.to(self.device)
        self.global_plane_conv.to(self.device)
        self.stack_plane_flat.to(self.device)
    def forward(self,ind,attri_stack,planes_stack,plane_glb):
        id_emb = self.id_emb(ind)
        stack_fc = self.stack_fc(torch.cat([id_emb,attri_stack],dim=-1))
        planes_conv = self.stack_plane_conv(planes_stack)
        glb_conv = self.global_plane_conv(plane_glb)
        all_fc = self.stack_plane_flat(torch.cat([stack_fc,planes_conv,glb_conv],dim=-1))
        return all_fc
class H3_net(nn.Module):
    def __init__(self, device='cpu'):
        super(H3_net,self).__init__()
        self.device = device
        self.inpipe = in_pipe(device=self.device)
        self.act_ids = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512, 5),nn.BatchNorm1d(5))
        self.position = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512, 11*17),nn.BatchNorm1d(11*17))
        self.targets = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512, 7),nn.BatchNorm1d(7))
        self.spells = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512,10),nn.BatchNorm1d(10))
        self.critic = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512,1),nn.BatchNorm1d(1))

        self.act_ids.to(self.device)
        self.position.to(self.device)
        self.targets.to(self.device)
        self.spells.to(self.device)
        self.critic.to(self.device)

    def forward(self,ind,attri_stack,planes_stack,plane_glb,critic_only = False):
        ind_t = torch.tensor(ind, device=self.device, dtype=torch.long)
        attri_stack_t = torch.tensor(attri_stack, device=self.device, dtype=torch.float)
        planes_stack_t = torch.tensor(planes_stack, device=self.device, dtype=torch.float)
        plane_glb_t = torch.tensor(plane_glb, device=self.device, dtype=torch.float)
        h = self.inpipe(ind_t,attri_stack_t,planes_stack_t,plane_glb_t)
        value = self.critic(h)
        if critic_only:
            return value
        act_logits = self.act_ids(h)
        targets_logits = self.targets(h)
        position_logits = self.position(h)
        spell_logits = self.spells(h)
        return act_logits,targets_logits,position_logits,spell_logits,value




