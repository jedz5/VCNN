import torch
from ding.torch_utils import to_tensor
from torch import nn
import numpy as np
# from H3_battle import logger
from typing import Dict,Union

from ENV.H3_battle import Battle

dist_fn = torch.distributions.Categorical
import copy

class my_reshape(nn.Module):
    def __init__(self,shape):
        super(my_reshape, self).__init__()
        self.shape = shape
    def forward(self,x):
        x = x.reshape(*self.shape)
        return x
id_emb_size = 16
act_emb_size = 8
speed_emb_size = 8
stack_emb_size = 32
x_emb_size = 8
y_emb_size = 8
stack_fc_size = 128
all_fc_size = 512
position_h_size = 32
target_h_size = 32
class H3Q_model(nn.Module):
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']
    def __init__(self, device='cpu'):
        super(H3Q_model,self).__init__()
        self.device = device
        self.id_emb = nn.Embedding(150, id_emb_size, padding_idx=122)
        self.act_emb = nn.Embedding(10, act_emb_size, padding_idx=8)
        self.speed_emb = nn.Embedding(25, speed_emb_size, padding_idx=24)
        self.x_emb = nn.Embedding(20, x_emb_size, padding_idx=19)
        self.y_emb = nn.Embedding(13, y_emb_size, padding_idx=12)
        self.stack_emb = nn.Sequential(nn.Linear(21 + id_emb_size, stack_emb_size), #stack feature + id_emb
                                       nn.ReLU(inplace=True)
                                       )
        self.stack_fc = nn.Sequential(
                                      my_reshape([-1,stack_emb_size * 14]),
                                      nn.Linear(stack_emb_size * 14, stack_fc_size),
                                      nn.ReLU(inplace=True))
                                      # nn.Linear(128, stack_fc_size),
                                      # nn.ReLU(inplace=True))
        self.stack_plane_conv  = nn.Sequential(my_reshape([-1, 3, 11, 17]),
                                               nn.Conv2d(3, out_channels=8, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 14 * 8, 11, 17]),
                                               nn.Conv2d(14 * 8, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(32, out_channels=3, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 3 * 11 * 17]))
        self.global_plane_conv = nn.Sequential(my_reshape([-1, 3, 11, 17]),
                                               nn.Conv2d(3, out_channels=8, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(8, out_channels=8, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(8, out_channels=3, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 3 * 11 * 17]))
        self.stack_plane_flat = nn.Sequential(nn.Linear(stack_fc_size + 3 * 11 * 17 * 2, all_fc_size),nn.ReLU(inplace=True),nn.Linear(all_fc_size, 256),nn.ReLU(inplace=True))
        self.actor = nn.Linear(256, Battle.act_size_flat)
        self.critic = nn.Linear(256, 1)
        #
        self.id_emb.to(self.device)
        self.stack_fc.to(self.device)
        self.stack_emb.to(self.device)
        self.stack_plane_conv.to(self.device)
        self.global_plane_conv.to(self.device)
        self.stack_plane_flat.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
    #@profile
    def forward2(self,ind,attri_stack,attri_stack_orig,planes_stack,plane_glb,action_mask=None):
        orig_shape = ind.shape
        batch_shape = orig_shape[:-1]
        ind = to_tensor(ind, dtype=torch.int32)
        id_emb = self.id_emb(ind)  ##self.id_emb(attri_stack[...,[2]].squeeze(-1).long())
        stack_emb = self.stack_emb(torch.cat([id_emb,attri_stack],dim=-1))
        stack_fc = self.stack_fc(stack_emb)
        planes_conv = self.stack_plane_conv(planes_stack)
        glb_conv = self.global_plane_conv(plane_glb)
        all_fc = self.stack_plane_flat(torch.cat([stack_fc,planes_conv,glb_conv],dim=-1))
        post_shape = all_fc.shape
        rebatch_shape = [*batch_shape,post_shape[-1]]
        all_fc = all_fc.reshape(rebatch_shape)
        return all_fc
    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: Dict) -> Dict:
        all_fc = self.forward2(**x)
        logit = self.actor(all_fc)
        return {'logit': logit,'action_mask':x['action_mask']}

    def compute_critic(self, x: torch.Tensor) -> Dict:
        all_fc = self.forward2(**x)
        v = self.critic(all_fc)
        return {'value': v}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        all_fc = self.forward2(**x)
        logit = self.actor(all_fc)
        v = self.critic(all_fc)
        return {'logit': logit, 'value': v,'action_mask':x['action_mask']}
