import torch
import time
import numpy as np
from torch import nn
from H3_battle import Battle
from H3_battle import actionType
from H3_battle import logger
import H3_battleInterface
import pygame
dist_fn = torch.distributions.Categorical


def unsparse(mask, size):
    value_size = len(mask)
    I = torch.tensor([[0] * value_size, mask])
    mask_targets = torch.sparse_coo_tensor(I, torch.ones(value_size), (1, size))
    return mask_targets

def softmax(logits, mask,dev):
    mask = torch.tensor(mask, dtype=torch.int,device=dev)
    logits = torch.exp(logits)
    logits = logits * mask
    logits = logits / (torch.sum(logits) + 1E-5)
    return logits
class Net(nn.Module):
    def __init__(self, layer_num, state_shape, device='cuda'):
        super().__init__()
        self.device = device
        self.inpipe = [
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.inpipe += [nn.Linear(1024, 1024), nn.ReLU(inplace=True)]
        self.inpipe = nn.Sequential(*self.inpipe)

        self.act_ids = nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(inplace=True),nn.Linear(1024, 5))
        self.position = nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(inplace=True),nn.Linear(1024, 11*17))
        self.targets = nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(inplace=True),nn.Linear(1024, 7))
        self.spells = nn.Sequential(nn.Linear(1024, 1024),nn.ReLU(inplace=True),nn.Linear(1024,10))

        self.inpipe.to(self.device)
        self.act_ids.to(self.device)
        self.position.to(self.device)
        self.targets.to(self.device)
        self.spells.to(self.device)

    def forward(self, s,env):
        h = self.inpipe(s)
        mask_acts = env.legal_act(level = 0)
        act_logits = self.act_ids(h)
        targets_logits = self.targets(h)
        position_logits = self.position(h)
        spell_logits = self.spells(h)
        act_logits = softmax(act_logits,mask_acts,self.device)
        act_id = dist_fn(act_logits).sample()[0].item()
        if act_id == actionType.move.value:
            mask_position = env.legal_act(level=1, act_id=act_id)
            position_logits = softmax(position_logits,mask_position,self.device)
            position_id = dist_fn(position_logits).sample()[0].item()
            #
            mask_targets = torch.zeros(7,device=self.device)
            target_id = -1
            mask_spell = torch.zeros(10,device=self.device)
            spell_id = -1
        elif act_id == actionType.attack.value:
            mask_targets = env.legal_act(level = 1,act_id=act_id)
            targets_logits = softmax(targets_logits,mask_targets,self.device)
            if torch.sum(targets_logits).item() > 0.5:
                target_id = dist_fn(targets_logits).sample()[0].item()
            else:
                logger.info("no attack target found!!")
                assert 0
            mask_position = env.legal_act(level=2, act_id=act_id,target_id=target_id)
            position_logits = softmax(position_logits,mask_position,self.device)
            if torch.sum(position_logits).item() > 0.5:
                position_id = dist_fn(position_logits).sample()[0].item()
            else:
                position_id = -1
            #
            mask_spell = torch.zeros(10,device=self.device)
            # mask_spell = unsparse(mask_spell, 10)
            spell_id = -1
        else:
            #
            mask_position = torch.zeros(11*17,device=self.device)
            position_id = -1
            mask_targets = torch.zeros(7,device=self.device)
            target_id = -1
            mask_spell = torch.zeros(10,device=self.device)
            spell_id = -1
        return {'act_id':act_id, 'spell_id':spell_id,'target_id':target_id,'position_id':position_id,
                'mask_acts':mask_acts,'mask_spell':mask_spell,'mask_targets':mask_targets,'mask_position':mask_position}




def start_game():
    #初始化 agent
    dev = 'cuda'
    agent = Net(3, 1024, device=dev)
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    battle = Battle(agent=agent)
    battle.loadFile("D:/project/VCNN/ENV/selfplay.json")
    battle.checkNewRound()
    bi = H3_battleInterface.BattleInterface(battle)
    bi.next_act = battle.curStack.active_stack()
    act = bi.next_act
    # 事件循环(main loop)
    i = 0
    last = time.time()
    while True:
        if i % 20 == 0:
            if act:
                cost = time.time() - last
                print(f'cost time {cost}')
            act = bi.handleEvents()
            if act:
                i += 1
                last = time.time()
            bi.handleBattle(act)
            if battle.check_battle_end():
                print("battle end")
                return
            bi.renderFrame()
        else:
            i += 1
            battle.doAction(bi.next_act)
            battle.checkNewRound()
            if battle.check_battle_end():
                print("battle end")
                return
            bi.next_act = battle.curStack.active_stack()


if __name__ == '__main__':
    start_game()