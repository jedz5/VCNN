import torch
from torch import nn
import numpy as np
# from H3_battle import logger
dist_fn = torch.distributions.Categorical
import copy

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
        self.id_emb = nn.Embedding(150, 64, padding_idx=122)
        self.act_emb = nn.Embedding(10, 64, padding_idx=8)
        self.stack_emb = nn.Sequential(nn.Linear(21 + 64, 64),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(64, 64)
                                       )
        self.stack_fc = nn.Sequential(
                                      my_reshape([-1,64 * 14]),
                                      nn.Linear(64 * 14, 512),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(512, 512),
                                      nn.ReLU(inplace=True))
        self.stack_plane_conv  = nn.Sequential(my_reshape([-1, 3, 11, 17]),
                                               nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               # nn.MaxPool2d(kernel_size=2),
                                               my_reshape([-1, 14 * 32, 11, 17]),
                                               nn.Conv2d(14 * 32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(32, out_channels=3, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 3 * 11 * 17]))
        self.global_plane_conv = nn.Sequential(my_reshape([-1, 3, 11, 17]),
                                               nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               # nn.MaxPool2d(kernel_size=2),
                                               nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(32, out_channels=3, kernel_size=3, stride=1, padding=1),
                                               nn.ReLU(inplace=True),
                                               my_reshape([-1, 3 * 11 * 17]))
        self.stack_plane_flat = nn.Sequential(nn.Linear(512 + 3 * 11 * 17 * 2, 512),nn.ReLU(inplace=True))
        #
        self.id_emb.to(self.device)
        self.stack_fc.to(self.device)
        self.stack_emb.to(self.device)
        self.stack_plane_conv.to(self.device)
        self.global_plane_conv.to(self.device)
        self.stack_plane_flat.to(self.device)
    #@profile
    def forward(self,ind,attri_stack,planes_stack,plane_glb):
        id_emb = self.id_emb(ind)
        stack_emb = self.stack_emb(torch.cat([id_emb,attri_stack],dim=-1))
        stack_fc = self.stack_fc(stack_emb)
        planes_conv = self.stack_plane_conv(planes_stack)
        glb_conv = self.global_plane_conv(plane_glb)
        all_fc = self.stack_plane_flat(torch.cat([stack_fc,planes_conv,glb_conv],dim=-1))
        return all_fc,stack_emb
class H3_net(nn.Module):
    def __init__(self, device='cpu'):
        super(H3_net,self).__init__()
        self.device = device
        self.inpipe = in_pipe(device=self.device)
        self.act_ids = nn.Linear(512, 5) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),)#,nn.BatchNorm1d(5))
        self.position_h = nn.Linear(512, 64)
        self.position = nn.Linear(192, 11*17) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),)#,nn.BatchNorm1d(11*17))
        self.targets_h = nn.Linear(512, 64) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),)#,nn.BatchNorm1d(7))
        self.targets = nn.Linear(128, 14)
        self.spells = nn.Linear(512, 10) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512,10))#,nn.BatchNorm1d(10))
        self.critic = nn.Linear(512, 1) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512,1))#,nn.BatchNorm1d(1))

        self.act_ids.to(self.device)
        self.position_h.to(self.device)
        self.position.to(self.device)
        self.targets_h.to(self.device)
        self.targets.to(self.device)
        self.spells.to(self.device)
        self.critic.to(self.device)
    #@profile
    def forward(self,ind,attri_stack,planes_stack,plane_glb,critic_only = False):
        ind_t = torch.tensor(ind, device=self.device, dtype=torch.long)
        attri_stack_t = torch.tensor(attri_stack, device=self.device, dtype=torch.float)
        planes_stack_t = torch.tensor(planes_stack, device=self.device, dtype=torch.float)
        plane_glb_t = torch.tensor(plane_glb, device=self.device, dtype=torch.float)
        h,stack_emb = self.inpipe(ind_t,attri_stack_t,planes_stack_t,plane_glb_t)
        value = self.critic(h)
        if critic_only:
            return value
        act_logits = self.act_ids(h)
        targets_logits = self.targets_h(h)
        position_logits = self.position_h(h)
        spell_logits = self.spells(h)
        return act_logits,targets_logits,stack_emb,position_logits,spell_logits,value

    def get_act_emb(self,act_id):
        act_emb = self.inpipe.act_emb(act_id)
        return act_emb
    def get_target_loggits(self,act_id,target_h,single_batch=False):
        if single_batch:
            '''single mode xxx_id is int
              after act emb ->(1,) -> (1,embbeding) 
           '''
            act_emb = self.inpipe.act_emb(torch.tensor([act_id]))
        else:
            '''batch mode xxx_id is pre_processed like shape = (batch,1)
              after act emb -> (batch,1,embbeding) -> (batch,embbeding)
           '''
            act_emb = self.inpipe.act_emb(act_id).squeeze(1)
        target_h = torch.cat([target_h,act_emb],dim=-1)
        target_logits = self.targets(target_h)
        '''target attention, not tested'''
        # act_emb = torch.sigmoid(act_emb)
        # target_h *= act_emb
        # target_logits = torch.mm(target_h.unsqueeze(dim=-2),target_emb.permute(0,2,1))
        return target_logits
    def get_position_loggits(self,act_id,target_id,target_embs,mask_orig,position_h,single_batch=False):
        if single_batch:
            '''single mode xxx_id is int -> (1,) -> act emb -> (1,embbeding)
              target emb=(1,14,embbeding)[:,target_id] -> (1,embbeding)
                        '''
            act_emb = self.inpipe.act_emb(torch.tensor([act_id]))
            '''target_id 0 is current stack'''
            if 0 < target_id < 14:
                target_emb = target_embs[:,target_id]
            else:
                target_emb = torch.zeros((1,target_embs.shape[-1]))
            position_h = torch.cat([position_h, act_emb, target_emb], dim=-1)
            position_logits = self.position(position_h)
        else:
            '''batch mode xxx_id is pre_processed like shape = (batch,1) -> act emb -> (batch,1,embbeding) -> (batch,embbeding)
              target emb=(batch,14,embbeding).gather((batch,1,embbeding))  -> (batch,1,embbeding) -> (batch,embbeding)
            '''
            mask = torch.tensor(mask_orig, dtype=torch.float, device=target_embs.device)
            act_emb = self.inpipe.act_emb(act_id).squeeze(1)
            expande_shape = (target_embs.shape[0],1,target_embs.shape[-1])
            '''target_id 0 needs padding'''
            target_embs = target_embs * mask.unsqueeze(-1)
            '''target_id (batch,1) -> (batch,1,1) -> (batch,1,embbeding)'''
            target_emb = target_embs.gather(dim=1,index=target_id.unsqueeze(-1).expand(expande_shape)).squeeze(1)
            position_h = torch.cat([position_h,act_emb,target_emb],dim=-1)
            position_logits = self.position(position_h)
        return position_logits

class H3_Q_net(nn.Module):
    def __init__(self, device='cpu'):
        super(H3_Q_net,self).__init__()
        self.device = device
        self.inpipe = in_pipe(device=self.device)
        self.act_ids = nn.Linear(512, 5) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),)#,nn.BatchNorm1d(5))
        self.position_h = nn.Linear(512, 64)
        self.position = nn.Linear(192, 11*17) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),)#,nn.BatchNorm1d(11*17))
        self.targets_h = nn.Linear(512, 64) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),)#,nn.BatchNorm1d(7))
        self.targets = nn.Linear(128, 14)
        self.spells = nn.Linear(512, 10) #nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True),nn.Linear(512,10))#,nn.BatchNorm1d(10))

        self.act_ids.to(self.device)
        self.position_h.to(self.device)
        self.position.to(self.device)
        self.targets_h.to(self.device)
        self.targets.to(self.device)
        self.spells.to(self.device)

        self.act_imt = nn.Linear(512, 5)
        self.position_imt = nn.Linear(192, 11 * 17)
        self.targets_imt = nn.Linear(128, 14)
        # self.spells_imt = nn.Linear(512,10)
        self.act_imt.to(self.device)
        self.position_imt.to(self.device)
        self.targets_imt.to(self.device)
        # self.spells_imt.to(self.device)

    #@profile
    def forward(self,ind,attri_stack,planes_stack,plane_glb):
        ind_t = torch.tensor(ind, device=self.device, dtype=torch.long)
        attri_stack_t = torch.tensor(attri_stack, device=self.device, dtype=torch.float32)
        planes_stack_t = torch.tensor(planes_stack, device=self.device, dtype=torch.float32)
        plane_glb_t = torch.tensor(plane_glb, device=self.device, dtype=torch.float32)
        h,stack_emb = self.inpipe(ind_t,attri_stack_t,planes_stack_t,plane_glb_t)
        act_logits = self.act_ids(h)
        act_imt = self.act_imt(h)
        targets_logits = self.targets_h(h)
        position_logits = self.position_h(h)
        spell_logits = self.spells(h)
        return act_logits,act_imt,targets_logits,stack_emb,position_logits,spell_logits

    def get_act_emb(self,act_id):
        act_emb = self.inpipe.act_emb(act_id)
        return act_emb
    def get_target_loggits(self,act_id,target_h,single_batch=False):
        if single_batch:
            '''single mode xxx_id is int
              after act emb ->(1,) -> (1,embbeding) 
           '''
            act_emb = self.inpipe.act_emb(torch.tensor([act_id]))
        else:
            '''batch mode xxx_id is pre_processed like shape = (batch,1)
              after act emb -> (batch,1,embbeding) -> (batch,embbeding)
           '''
            act_emb = self.inpipe.act_emb(act_id).squeeze(1)
        target_h = torch.cat([target_h,act_emb],dim=-1)
        target_logits = self.targets(target_h)
        target_imt =self.targets_imt(target_h)
        '''target attention, not tested'''
        # act_emb = torch.sigmoid(act_emb)
        # target_h *= act_emb
        # target_logits = torch.mm(target_h.unsqueeze(dim=-2),target_emb.permute(0,2,1))
        return target_logits,target_imt
    def get_position_loggits(self,act_id,target_id,target_embs,mask_orig,position_h,single_batch=False):
        if single_batch:
            '''single mode xxx_id is int -> (1,) -> act emb -> (1,embbeding)
              target emb=(1,14,embbeding)[:,target_id] -> (1,embbeding)
                        '''
            act_emb = self.inpipe.act_emb(torch.tensor([act_id]))
            '''target_id 0 is current stack'''
            if 0 < target_id < 14:
                target_emb = target_embs[:,target_id]
            else:
                target_emb = torch.zeros((1,target_embs.shape[-1]))
            position_h = torch.cat([position_h, act_emb, target_emb], dim=-1)
            position_logits = self.position(position_h)
            position_imt = self.position_imt(position_h)
        else:
            '''batch mode xxx_id is pre_processed like shape = (batch,1) -> act emb -> (batch,1,embbeding) -> (batch,embbeding)
              target emb=(batch,14,embbeding).gather((batch,1,embbeding))  -> (batch,1,embbeding) -> (batch,embbeding)
            '''
            mask = torch.tensor(mask_orig, dtype=torch.float, device=target_embs.device)
            act_emb = self.inpipe.act_emb(act_id).squeeze(1)
            '''target_id 0 needs padding'''
            target_embs = target_embs * mask.unsqueeze(-1)
            expande_shape = (target_embs.shape[0],1,target_embs.shape[-1])
            '''target_id (batch,1) -> (batch,1,1) -> (batch,1,embbeding)'''
            target_emb = target_embs.gather(dim=1,index=target_id.unsqueeze(-1).expand(expande_shape)).squeeze(1)
            position_h = torch.cat([position_h,act_emb,target_emb],dim=-1)
            position_logits = self.position(position_h)
            position_imt = self.position_imt(position_h)
        return position_logits,position_imt





