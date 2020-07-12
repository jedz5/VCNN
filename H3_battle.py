import random
import copy
import numpy as np
from enum import Enum
import logging
import torch
import os
import json
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
std_logger = logging.getLogger('train')
handler = logging.FileHandler('train.log','w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
std_logger.addHandler(handler)
import sys
sys.path.extend(['D:\\project\\VCNN\\VCCC\\x64\\Release'])
import VCbattle

class log_with_gui(object):
    def __init__(self,std_logger):
        self.logger = std_logger
        self.log_text = []
    def info(self,text,to_gui = False):
        # pass
        self.logger.info(text)
        if(to_gui):
            self.log_text.append(text)
logger = log_with_gui(std_logger)
diretMap = {'0':3,'1':4,'2':5,'3':0,'4':1,'5':2}


class action_type(Enum):
    wait = 0
    defend = 1
    move = 2
    attack = 3
    shoot = 4
    spell = 5
class action_query_type(Enum):
    can_move = 0
    move_to = 1
    can_attack = 2
    attack_target = 3
    attack_from = 4
    spell = 5


def neb_id(self, nb):
    if self.y == nb.y:
        if self.x < nb.x:
            return 2
        else:
            return 5
    elif self.y > nb.y:
        if self.y % 2 == 0:
            if self.x == nb.x:
                return 0
            else:
                return 1
        else:
            if self.x == nb.x:
                return 1
            else:
                return 0
    else:
        if self.y % 2 == 0:
            if self.x == nb.x:
                return 4
            else:
                return 3
        else:
            if self.x == nb.x:
                return 3
            else:
                return 4
class BHex:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        if(other):
            return self.y == other.y and self.x == other.x
        return False
    def flat(self):
        return self.y * Battle.bFieldHeight + self.x
class BStack(object):
    def __init__(self):
        self.amount = 0
        self.attack = 0
        self.defense = 0
        self.max_damage = 0
        self.min_damage = 0
        self.first_HP_Left = 0
        self.health = 0
        self.side = 0
        self.had_moved = False
        self.had_retaliated = False
        self.had_waited = False
        self.had_defended = False
        self.speed = 0
        self.luck = 0
        self.morale = 0
        self.id = 0
        self.shots = 10
        # self.hex_type = hexType.creature
        #辅助参数

        self.by_AI = 1
        self.name = 'unKnown'
        self.slotId = 0
        self.x = 0
        self.y = 0
        self.is_wide = False
        self.is_fly = False
        self.is_shooter = False
        self.block_retaliate = False
        self.attack_nearby_all = False
        self.wide_breath = False
        self.infinite_retaliate = False
        self.attack_twice = False
        self.amount_base = 0
        self.in_battle = 0  #Battle()
    def __eq__(self, other):
        if(other):
            return self.x == other.x and self.y == other.y
        return False
    def get_global_state(self, query_type = -1, exclude_me = True):
        return VCbattle.get_global_state(self,self.in_battle.stacks,query_type,exclude_me)
        # bf = np.ones((self.in_battle.bFieldHeight, self.in_battle.bFieldWidth))
        # bf.fill(-1)
        # bf[:, 0] = 100
        # bf[:, -1] = 100
        # for sts in self.in_battle.stacks:
        #     if(sts.is_alive()):
        #         bf[sts.y,sts.x] = 400 if sts.side == self.side else 200
        # for obs in self.in_battle.obstacles:
        #     bf[obs.y,obs.x] = 800
        # #init battleField end
        # # accessable  begin
        # travellers = []
        # bf[self.y,self.x] = self.speed
        # travellers.append(self)
        # if(not self.is_fly):
        #     while(len(travellers) > 0):
        #         current = travellers.pop()
        #         speedLeft = bf[current.y,current.x] - 1
        #         for adj in self.get_neighbor(current):
        #             if(bf[adj.y,adj.x] < speedLeft):
        #                 bf[adj.y,adj.x] = speedLeft
        #                 if (speedLeft > 0):
        #                     travellers.append(adj)
        #                     if query_type == action_query_type.can_move:
        #                         return True
        # else: #fly
        #     for ii in range(self.in_battle.bFieldHeight):
        #         for jj in range(1, self.in_battle.bFieldWidth - 1):
        #             if bf[ii,jj] > 50:
        #                 continue
        #             d = self.get_distance(BHex(jj, ii))
        #             if(0 < d <= self.speed):
        #                 bf[ii,jj] = self.speed - d
        #                 if query_type == action_query_type.can_move:
        #                     return True
        # #no space to move to
        # if query_type == action_query_type.can_move:
        #     return False
        # #accessable  end
        # #attackable begin
        # for sts in self.in_battle.stacks:
        #     if(not sts.is_alive()):
        #         continue
        #     if sts.side != self.side:
        #         if (self.can_shoot()):
        #             bf[sts.y,sts.x] = 201  # enemy and attackbale
        #             if query_type == action_query_type.can_attack:
        #                 return True
        #         else:
        #             for neib in self.get_neighbor(sts):
        #                 if (0 <= bf[neib.y,neib.x] < 50):
        #                     bf[sts.y,sts.x] = 201
        #                     if query_type == action_query_type.can_attack:
        #                         return True
        #                     break
        # #no target to reach
        # if query_type == action_query_type.can_attack:
        #     return False
        # if exclude_me:
        #     bf[self.y,self.x] = 401
        # return bf
    def damaged(self,damage):
        hpInAll = self.health * (self.amount - 1) + self.first_HP_Left
        if (damage >= hpInAll):
            damage = hpInAll
            killed = self.amount
            first_HP_Left = 0
        else:
            rest = int((hpInAll - damage - 1) / self.health) + 1
            first_HP_Left = (hpInAll - damage - 1) % self.health + 1
            killed = self.amount - rest
        self.amount -= killed
        self.first_HP_Left = first_HP_Left
        return damage,killed,first_HP_Left
    def computeCasualty(self,opposite,stand,is_reta, estimate=False):
        total_damage = 0
        if(self.attack >= opposite.attack):
            damageMin = int(self.min_damage * (1 + (self.attack - opposite.attack) * 0.05) * self.amount)
            damageMax = int(self.max_damage * (1 + (self.attack - opposite.attack) * 0.05) * self.amount)
        else:
            damageMin = int(self.min_damage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
            damageMax = int(self.max_damage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
        damage = int((damageMin+damageMax)/2) if estimate else random.randint(damageMin,damageMax)
        can_shoot = self.can_shoot()
        if(self.is_shooter):
            if(not can_shoot or self.is_half(opposite)):
                damage = int(damage/2)
        else:
            others = self.get_attacked_stacks(opposite,stand)
            for st_tmp in others:
                if(estimate):
                    st = copy.copy(st_tmp)
                else:
                    st = st_tmp
                real_damage, killed, first_HP_Left = st.damaged(damage)
                if(self.side == st.side):
                    total_damage += -real_damage
                else:
                    total_damage += real_damage
                if (not estimate):
                    head = "reta" if is_reta else "make"
                    tt = "{} {} dmg killed {} {} {} left, HP {}".format(head,real_damage, killed, st.name,st.amount, first_HP_Left)
                    logger.info(tt,True)
                    if(opposite.amount == 0):
                        logger.info("{} perished".format(opposite.name),True)
        real_damage, killed, first_HP_Left = opposite.damaged(damage)
        total_damage += real_damage
        if(not estimate):
            if(can_shoot): # damage is done and enemy may die, so this flag may change
                half = "(half)" if self.is_half(opposite) else "(full)"
                logger.info("shoot{} {}".format(half,opposite.name),True)
            head = "reta" if is_reta else "make"
            tt = "{} {} dmg killed {} {} {} left, HP {}".format(head,real_damage, killed, opposite.name,opposite.amount, first_HP_Left)
            logger.info(tt,True)
            if(opposite.amount == 0):
                logger.info("{} perished".format(opposite.name),True)
        return total_damage,killed,first_HP_Left
    def meeleAttack(self,opposite,dest,is_retaliate):
        self.do_attack(opposite,dest)
    def can_shoot(self, opposite = None):
        if (not self.is_shooter or self.shots <= 0):
            return False
        for enemy in self.in_battle.stacks:
            if(enemy.side != self.side and enemy.is_alive() and self.get_distance(enemy) == 1):
                return False
        return True
    def get_attacked_stacks(self,defender,stand):
        attacked = []
        if(self.attack_nearby_all):
            neibs = self.get_neighbor(stand)
            for st in self.in_battle.stacks:
                if st != defender and st.is_alive() and st.side != self.side:
                    for nb in neibs:
                        if (st == nb):
                            attacked.append(st)
        elif(self.wide_breath):
            df = defender.get_position()
            df.x += (0.5 if df.y % 2 == 0 else 0)
            at = BHex(stand.x,stand.y)
            at.x += (0.5 if at.y % 2 == 0 else 0)
            other = BHex(int(df.x * 2 - at.x),int(df.y * 2 - at.y))
            for st in self.in_battle.stacks:
                if st != defender and st.is_alive():
                    if (st == other):
                        attacked.append(st)
        return attacked
    def get_position(self):
        return BHex(self.x,self.y)
    def get_neighbor(self, src = None):
        adj = []
        if(not src ):
            src = self
        self.checkAndPush(src.x - 1,src.y,  adj)
        self.checkAndPush(src.x + 1,src.y,  adj)
        self.checkAndPush((src.x - 1) if src.y%2 != 0 else (src.x + 1) ,src.y - 1,  adj)
        self.checkAndPush(src.x,src.y - 1,  adj)
        self.checkAndPush((src.x - 1) if src.y%2 != 0 else (src.x + 1),src.y + 1,  adj)
        self.checkAndPush(src.x, src.y + 1, adj)
        return adj
    def checkAndPush(self,x,y,adj):
        if(self.checkPosition(x,y)):
            adj.append(BHex(x,y))
    def checkPosition(self,x,y):
        return 0 <= y < Battle.bFieldHeight and 0 < x < Battle.bFieldWidth - 1
    def is_half(self, dist):
        return self.get_distance(dist) > Battle.bPenaltyDistance
    def get_distance(self, dist):
        return  Battle.bGetDistance(self,dist)
    def shoot(self,opposite):
        logger.info('{} shooting {}'.format(self.name,opposite.name))
        self.do_attack(opposite,self.get_position())
        self.shots -= 1
        return
    def is_alive(self):
        return self.amount > 0
    def wait(self):
        self.had_waited = True
    def defend(self):
        self.defense += 2
        self.had_defended = True
        self.had_moved = True
    def newRound(self):
        if self.had_defended:
            self.defense -= 2
        self.had_moved = False
        self.had_retaliated = False
        self.had_waited = False
        self.had_defended = False
        return
    # def legalMoves(self):
    #     if (self.had_moved):
    #         logger.info("sth wrong happen! {} is moved!!!".format(self.name))
    #         return 0
    #     #ret = {'wait': self.had_waited(), 'defend': True, 'move': [], 'melee': [], 'shoot': []}
    #     legalMoves = []
    #     if(not self.had_waited):
    #         legalMoves.append(0) #waite
    #     legalMoves.append(1) #defend
    #     aa = self.get_global_state()
    #     for i in range(0, self.inBattle.bFieldHeight):
    #         for j in range(1, self.inBattle.bFieldWidth - 1):
    #             if (aa[i,j] >= 0 and aa[i,j] < 50 and aa[i,j] != self.speed):
    #                 #ret['move'].append(BAction(action_type.move, BHex(i,j)))
    #                 legalMoves.append(self.inBattle.actionToIndex(BAction(action_type.move, BHex(j,i))))
    #             if (aa[i,j] == -1):
    #                 if (self.can_shoot()):
    #                     #ret['shoot'].append(BAction(action_type.shoot,0,BHex(i,j)))target
    #                     legalMoves.append(self.inBattle.actionToIndex(BAction(action_type.shoot,target=BHex(j,i))))
    #                 else:
    #                     att = BHex(j,i)
    #                     for nb in self.getNeibours(att):
    #                         if(aa[nb.y,nb.x] >= 0 and aa[nb.y,nb.x] < 50):
    #                             #ret['melee'].append(BAction(action_type.shoot,nb,BHex(i,j)))
    #                             legalMoves.append(self.inBattle.actionToIndex(BAction(action_type.attack,nb,att)))
    #     return legalMoves



    def potential_target(self):
        bf = self.get_global_state(exclude_me=False)  #
        attackable = []
        unreach = []
        for sts in self.in_battle.stacks:
            if bf[sts.y,sts.x] == 201:
                if(self.can_shoot()):
                    attackable.append((sts,0))
                else:
                    for nb in sts.get_neighbor():
                        if(0 <= bf[nb.y,nb.x] < 50):
                            attackable.append((sts, nb))
            elif bf[sts.y,sts.x] == 200:
                unreach.append(sts)
        return attackable,unreach
    def do_attack(self,defender_orig,stand,estimate = False):
        damage_get = 0
        damage_dealt = 0
        if(estimate):
            attacker = copy.copy(self)
            defender  = copy.copy(defender_orig)
        else:
            attacker = self
            defender = defender_orig
        N_attack = 1
        if attacker.attack_twice :
            N_attack = 2
            if attacker.is_shooter and not attacker.can_shoot(defender):
                N_attack = 1
        for i in range(N_attack):
            if(not attacker.is_alive()):
                break
            damage1,killed1,first_HP_Left1 = attacker.computeCasualty(defender,stand,False,estimate)
            damage_dealt += damage1
            attacker.had_moved = True
            if (not defender.is_alive()):
                break
            if(not attacker.can_shoot() and defender.is_alive() and not (defender.had_retaliated or attacker.block_retaliate)):
                damage2, killed2, first_HP_Left2 = defender.computeCasualty(attacker,defender.get_position(),True,estimate)
                damage_get += damage2
                if(not defender.infinite_retaliate):
                    defender.had_retaliated = True
        return damage_dealt,damage_get
    def go_toward(self,target):
        df = self.get_global_state()
        min_dist = 999
        dest = None
        for i in range(Battle.bFieldHeight):
            for j in range(Battle.bFieldWidth):
                if(0 <= df[i,j] < 50):
                    temp = BHex(j, i)
                    distance = Battle.bGetDistance(temp,target)
                    if(distance < min_dist):
                        min_dist = distance
                        dest = temp
        if(Battle.bGetDistance(self,target) <= min_dist):
            return BAction(action_type.defend)
        else:
            return BAction(action_type.move, dest=dest)

    def active_stack(self,ret_obs = False,print_act = False):
        if self.by_AI == 1:
            attackable, unreach = self.potential_target()
            if not self.in_battle.debug:
                if (self.in_battle.round == 0 and not self.had_waited):
                    return BAction(action_type.wait)
            if (len(attackable) > 0):
                att = [self.do_attack(target, stand, estimate=True) for target, stand in attackable]
                dmgs = [delt - get for delt, get in att]
                best = np.argmax(dmgs)
                best_target = attackable[best]
                if (self.can_shoot()):
                    return BAction(action_type.shoot, target=best_target[0])
                else:
                    return BAction(action_type.attack, target=best_target[0], dest=best_target[1])
            else:
                distants = [self.get_distance(x) for x in unreach]
                closest = np.argmin(distants)
                return self.go_toward(unreach[closest])
        elif self.by_AI == 2 and self.in_battle.agent:
            # {'act_id':act_id, 'spell_id':spell_id,'target_id':target_id,'position_id':position_id,
            #             'mask_acts':mask_acts,'mask_spell':mask_spell,'mask_targets':mask_targets,'mask_position':mask_position}
            agent = self.in_battle.agent
            ind, attri_stack, planes_stack, plane_glb = self.in_battle.current_state_feature()
            if agent.in_train:
                result = agent(ind[None], attri_stack[None], planes_stack[None], plane_glb[None], self.in_battle, print_act)
            else:
                with torch.no_grad():
                    result = agent(ind[None], attri_stack[None], planes_stack[None], plane_glb[None], self.in_battle,
                                   print_act)
            act_id = result['act_id']
            position_id = result['position_id']
            target_id = result['target_id']
            spell_id = result['spell_id']
            if act_id == action_type.wait.value:
                next_act = BAction(action_type.wait)
            elif act_id == action_type.defend.value:
                next_act = BAction(action_type.defend)
            elif act_id == action_type.move.value:
                next_act = BAction(action_type.move, dest=BHex(position_id % Battle.bFieldWidth, int(position_id / Battle.bFieldWidth)))
            elif act_id == action_type.attack.value:
                t = self.in_battle.defender_stacks[target_id] if self.side == 0 else self.in_battle.attacker_stacks[target_id]
                if self.can_shoot():
                    next_act = BAction(action_type.shoot, dest=BHex(position_id % Battle.bFieldWidth, int(position_id / Battle.bFieldWidth)), target=t)
                else:
                    next_act = BAction(action_type.attack, dest=BHex(position_id % Battle.bFieldWidth, int(position_id / Battle.bFieldWidth)), target=t)
            else:
                logger.info("not implemented action!!",True)
            if not ret_obs:
                return next_act
            # act_id = torch.tensor([[0]]) if act_id < 0 else torch.tensor([[act_id]])
            # position_id = torch.tensor([[0]]) if position_id < 0 else torch.tensor([[position_id]])
            # target_id = torch.tensor([[0]]) if target_id < 0 else torch.tensor([[target_id]])
            # spell_id = torch.tensor([[0]]) if spell_id < 0 else torch.tensor([[spell_id]])
            act_id = 0 if act_id < 0 else act_id
            position_id = 0 if position_id < 0 else position_id
            target_id = 0 if target_id < 0 else target_id
            spell_id = 0 if spell_id < 0 else spell_id
            obs = {'ind':ind, 'attri_stack':attri_stack, 'planes_stack':planes_stack, 'plane_glb':plane_glb}
            acts = {'act_id':act_id,'position_id':position_id,'target_id':target_id,'spell_id':spell_id}
            mask = {'mask_acts':result['mask_acts'],'mask_spell':result['mask_spell'],'mask_targets':result['mask_targets'],'mask_position':result['mask_position']}
            return next_act,obs,acts,mask
        elif self.by_AI == 0:
            return
        else:
            logger.info("no way to contrl the stack!!")
            exit(-1)


class BObstacle(object):
    def __init__(self,kind = 0):
        self.kind = kind
        self.x = 0
        self.y = 0
        # self.hexType = hexType.obstacle
class BObstacleInfo:
    def __init__(self,pos,w,h,isabs,imname):
        self.y = int(pos / Battle.bFieldWidth)
        self.x = pos % Battle.bFieldWidth
        self.width = w
        self.height = h
        self.isabs = isabs
        self.imname = imname

batId = 0
#[0 fly,1 shooter,2 block_retaliate,3 attack_all,4 wide_breath,5 infinite_retaliate]
creature_ability = {1:[0,0,0,0,0,0,0],3:[0,1,0,0,0,0,1],5:[1,0,0,0,0,1,0],7:[0,0,0,0,0,0,1],19:[0,1,0,0,0,0,1],
                            50:[0,0,0,0,0,0,0],51:[0,0,0,0,0,0,0],52:[1,0,0,0,0,0,0],119:[1,0,1,0,0,0,0],
                            121:[0,0,1,1,0,0,0],125:[0,0,0,0,0,0,0],131:[1,0,0,0,1,0,0],}
class Battle(object):
    bFieldWidth = 17
    bFieldHeight = 11
    bFieldStackProps = 18
    bFieldStackPlanes = 46
    bPenaltyDistance = 10
    bFieldSize = (bFieldWidth - 2)* bFieldHeight
    bTotalFieldSize = 2 + 8*bFieldSize
    def __init__(self,by_AI = [2,1], gui = None , load_file = None,agent = None,debug = False):
        #self.bField = [[0 for col in range(battle.bFieldWidth)] for row in range(battle.bFieldHeight)]
        self.stacks = []
        self.defender_stacks = []
        self.attacker_stacks = []
        self.round = 0
        self.obstacles = []
        self.obsinfo = []
        self.toMove = []
        self.waited = []
        self.moved = []
        self.stackQueue = []
        self.cur_stack = None
        self.last_stack = None
        self.batId = 0
        self.by_AI = by_AI
        self.bat_interface = None
        self.agent = agent
        self.debug = debug
        if(gui):
            self.bat_interface = gui
        if(load_file):
            self.loadFile(load_file)
            self.checkNewRound()
    def getCopy(self):
        global batId
        batId += 1
        cp = Battle()
        cp.batId = batId
        cp.round = self.round
        cp.obstacles = copy.deepcopy(self.obstacles)
        def copyStack(st,newBat):
            newSt = copy.copy(st)
            newSt.in_battle = newBat
            return newSt
        cp.stacks = [copyStack(st,cp) for st in self.stacks]
        cp.defender_stacks = list(filter(lambda x: x.side == 1, cp.stacks))
        cp.attacker_stacks = list(filter(lambda x: x.side == 0, cp.stacks))
        cp.sortStack()
        return cp
    def loadFile(self,file,shuffle_postion = True):
        with open("ENV/creatureData.json") as JsonFile:
            crList = json.load(JsonFile)["creatures"]
        with open(file) as jsonFile:
            root = json.load(jsonFile)
            for i in range(2):
                li = list(range(11))
                ys = np.random.choice(li,size=len(root['army{}'.format(i)]), replace=False)
                j = 0
                for id,num,py,px in root['army{}'.format(i)]:
                    x = crList[id]
                    st = BStack()
                    st.attack = x['attack']
                    st.defense = x['defense']
                    st.amount = num
                    st.amount_base = num
                    st.health = x['health']
                    st.first_HP_Left = x['health']
                    st.id = id
                    st.side = i
                    st.by_AI = self.by_AI[i]
                    #st.isWide = x['isWide']
                    st.luck = x['luck']
                    st.morale = x['morale']
                    st.max_damage = x['max_damage']
                    st.min_damage = x['min_damage']
                    st.name = x['name']
                    st.speed = x['speed']
                    st.shots = 16
                    st.is_fly = creature_ability[id][0]
                    st.is_shooter = creature_ability[id][1]
                    st.block_retaliate = creature_ability[id][2]
                    st.attack_nearby_all = creature_ability[id][3]
                    st.wide_breath = creature_ability[id][4]
                    st.infinite_retaliate = creature_ability[id][5]
                    st.attack_twice = creature_ability[id][6]
                    #st.slotId = x['slot']
                    if shuffle_postion:
                        st.x = 15 if i else 1
                        # st.x = px
                        st.y = int(ys[j])
                    else:
                        st.x = px
                        st.y = py
                    j += 1
                    st.in_battle = self
                    self.stacks.append(st)
                    if st.side:
                        self.defender_stacks.append(st)
                    else:
                        self.attacker_stacks.append(st)
            # for x in root['obstacles']:
            #     obs = BObstacle()
            #     obs.kind = x['type']
            #     obs.x = x['x']
            #     obs.y = x['y']
            #     self.obstacles.append(obs)
            # for x in root['obs']:
            #     oi = BObstacleInfo(x["pos"],x["width"],x["height"],bool(x["isabs"]),x["imname"])
            #     self.obsinfo.append(oi)
    def load_curriculum(self,file):
        with open("ENV/creatureData.json") as JsonFile:
            crList = json.load(JsonFile)["creatures"]
        with open(file) as JsonFile:
            js = json.load(JsonFile)
            curr = js["stacks"]
            self.round = js['round']
        curr = np.array(curr).reshape(14,-1)
        for i in range(14):
            py, px, id,side, amount, amount_base, first_HP_Left, health, attack, defense, max_damage, min_damage, had_moved, had_defended, had_retaliated, had_waited, speed, luck, morale, shots = curr[i]
            if px == 0:
                break
            x = crList[id]
            st = BStack()
            st.attack = attack
            st.defense = defense
            st.amount = amount
            st.amount_base = amount_base
            st.health = health
            st.first_HP_Left = first_HP_Left
            st.id = id
            st.side = side
            st.by_AI = self.by_AI[side]
            st.luck = luck
            st.morale = morale
            st.max_damage = max_damage
            st.min_damage = min_damage
            st.name = x['name']
            st.speed = speed
            st.shots = shots
            st.had_moved = had_moved
            st.had_defended = had_defended
            st.had_retaliated = had_retaliated
            st.had_waited = had_waited
            st.is_fly = creature_ability[id][0]
            st.is_shooter = creature_ability[id][1]
            st.block_retaliate = creature_ability[id][2]
            st.attack_nearby_all = creature_ability[id][3]
            st.wide_breath = creature_ability[id][4]
            st.infinite_retaliate = creature_ability[id][5]
            st.attack_twice = creature_ability[id][6]
            st.x = px
            st.y = py
            st.in_battle = self
            self.stacks.append(st)
            if st.side:
                self.defender_stacks.append(st)
            else:
                self.attacker_stacks.append(st)
    def dump_curriculum(self,dir):
        files = []
        for f in os.listdir(dir):
            tmp_f = os.path.join(dir,f)
            if os.path.isdir(tmp_f):
                continue
            else:
                files.append(int(os.path.splitext(f)[0]))
        if len(files):
            nmax = max(files) + 1
        else:
            nmax = 0
        dump_in = os.path.join(dir, f'{nmax}.json')
        attri_stack = self.current_state_feature(curriculum=True)
        with open(dump_in,'w', encoding='utf-8') as JsonFile:
            json.dump({'stacks':attri_stack.reshape(-1).tolist(),'round':self.round},JsonFile)
        print(f"states dumped in {dump_in}")
    def canReach(self,bFrom,bTo,bAtt = None):
        curSt = bFrom
        if(not curSt.checkPosition(bTo.x,bTo.y)):
            logger.info('dist {},{} not valid'.format(bTo.y,bTo.x))
            return False
        bf = curSt.get_global_state(exclude_me=False)
        if(bAtt):
            return self.bGetDistance(bTo,bAtt) == 1 and 50 > bf[bTo.y,bTo.x] >= 0 and bf[bAtt.y,bAtt.x] == 201
        else:
            return 50 > bf[bTo.y,bTo.x] >= 0
    @staticmethod
    def bGetDistance(src,dist):
        y1,x1 = src.x,src.y #历史原因交换了.x 和 .y -_ -
        y2,x2 = dist.x,dist.y
        y1 = int(y1+x1*0.5)
        y2 = int(y2+x2*0.5)
        yDst = y2 - y1
        xDst = x2 - x1
        if((yDst >= 0 and xDst >= 0) or (yDst < 0 and xDst < 0)):
            return max(abs(yDst),abs(xDst))
        return abs(yDst) + abs(xDst)
    def move(self,bFrom,bTo):
        srcs = self.findStack(bFrom)
        if(len(srcs) == 0):
            logger.info("sth wrong move from {},not exist".format(bFrom))
            return
        src = srcs[0]
        src.x = bTo.x
        src.y = bTo.y
        src.had_moved = True

    def sortStack(self):
        self.last_stack = self.cur_stack
        self.toMove = list(filter(lambda elem: elem.is_alive() and not elem.had_moved and not elem.had_waited, self.stacks))
        self.waited = list(filter(lambda elem: elem.is_alive() and not elem.had_moved and elem.had_waited, self.stacks))
        self.moved = list(filter(lambda elem: elem.is_alive() and elem.had_moved, self.stacks))

        self.toMove.sort(key=lambda elem:(-elem.speed,elem.y,elem.x))
        self.waited.sort(key=lambda elem: (elem.speed, elem.y, elem.x))
        self.moved.sort(key=lambda elem: (-elem.speed, elem.y, elem.x))
        self.stackQueue = self.toMove + self.waited + self.moved
        self.cur_stack = self.stackQueue[0]
    def currentPlayer(self):
        return 1 if self.stackQueue[0].side else 0

    def currentState(self):
        pass
    def current_state_feature(self,curriculum = False):
        planes_stack  = np.zeros((14,3,self.bFieldHeight,self.bFieldWidth),bool)
        attri_stack = np.zeros((14,14),dtype=int) if not curriculum else np.zeros((14,20),dtype=int)
        ind = np.array([122] * 14,dtype=int)
        for i,st in enumerate(self.stackQueue):
            bf = st.get_global_state()
            planes_stack[i, 0] = (bf >= 0) & (bf < 50)
            planes_stack[i, 1] = bf == 401
            planes_stack[i, 2] = bf == 201
            #
            ind[i] = st.id
            if curriculum:
                attri_stack[i] = np.array(
                    [st.y, st.x, st.id, st.side, st.amount, st.amount_base, st.first_HP_Left, st.health, st.attack, st.defense, st.max_damage, st.min_damage,
                     int(st.had_moved), int(st.had_defended), int(st.had_retaliated), int(st.had_waited), st.speed, st.luck, st.morale,
                     st.shots])
            else:
                attri_stack[i] = np.array(
                    [st.side, st.amount, st.first_HP_Left, st.attack, st.defense, st.max_damage, st.min_damage,
                     int(st.had_moved), int(st.had_retaliated), int(st.had_waited), st.speed, st.luck, st.morale,
                     st.shots])
        if curriculum:
            return attri_stack
        plane_glb = np.zeros([3,self.bFieldHeight,self.bFieldWidth],bool)
        for st in self.attacker_stacks:
            plane_glb[0,st.y, st.x] = 1
        for st in self.defender_stacks:
            plane_glb[1,st.y, st.x] = 1
        for st in self.obstacles:
            plane_glb[2,st.y, st.x] = 1
        return ind, attri_stack, planes_stack, plane_glb



    def getStackHPBySlots(self):
        pass

    def findStack(self,dist,alive=True):
        ret = list(filter(lambda elem: elem.x == dist.x and elem.y == dist.y and elem.is_alive() == alive, self.stacks))
        return ret
    def direction_to_hex(self, mySelf, dirct):
        zigzagCorrection =0 if (mySelf.y % 2) else 1
        if(dirct < 0 or dirct > 5):
            logger.info('wrong direction {}'.format(dirct))
            return None
        if(dirct == 0):
            return BHex(mySelf.x - 1,mySelf.y)
        if(dirct == 1):
            return BHex(mySelf.x - 1 + zigzagCorrection,mySelf.y - 1)
        if(dirct == 2):
            return BHex(mySelf.x + zigzagCorrection,mySelf.y - 1)
        if(dirct == 3):
            return BHex(mySelf.x + 1,mySelf.y)
        if(dirct == 4):
            return BHex(mySelf.x + zigzagCorrection,mySelf.y + 1)
        if(dirct == 5):
            return BHex(mySelf.x - 1 + zigzagCorrection,mySelf.y + 1)
    def hexToDirection(self,mySelf,hex):
        pass
    def actionToIndex(self,action):
        pass
    def indexToAction(self,move):
        pass
    def action2Str(self,act):
        #act = self.indexToAction(act)
        if(act.type == action_type.wait):
            return "wait"
        if (act.type == action_type.defend):
            return "defend"
        if (act.type == action_type.move):
            return "move to ({},{})".format(act.dest.y,act.dest.x)
        if (act.type == action_type.attack):
            return "melee ({},{}),({},{})".format(act.dest.y,act.dest.x,act.target.y,act.target.x)
        if (act.type == action_type.shoot):
            return "shoot ({},{})".format(act.target.y,act.target.x)
    def end(self):
        live = {0:False,1:False}
        for st in self.stacks:
            live[st.side] = live[st.side] or st.is_alive()
        if self.round > 20:
            return True,-1
        return not (live[0] and live[1]),self.currentPlayer()

    def getHash(self):
        pass
    def check_battle_end(self):
        att_alive = False
        def_alive = False
        for x in self.attacker_stacks:
            if x.is_alive():
                att_alive = True
                break
        for x in self.defender_stacks:
            if x.is_alive():
                def_alive = True
                break
        return not att_alive or not def_alive
    def get_winner(self):
        def_alive = False
        for x in self.defender_stacks:
            if x.is_alive():
                def_alive = True
                break
        return int(def_alive)
    def checkNewRound(self,is_self_play = 0):
        self.sortStack()
        if self.check_battle_end():
            logger.info("battle end~")
            return
        if(self.stackQueue[0].had_moved):
            self.newRound()
            self.sortStack()
        if(not is_self_play):
            side = "" if self.cur_stack.side else "me "
            logger.info("now it's {}{} turn".format(side, self.cur_stack.name), True)

    def newRound(self):
        self.round += 1
        logger.info("now it's round {}".format(self.round))
        for st in self.stacks:
            if(st.is_alive()):
                st.newRound()
    def doAction(self,action):
        logger.log_text.clear()
        logger.info(self.action2Str(action),True)
        if(self.cur_stack.had_moved):
            logger.info("{} is already moved".format(self.cur_stack))
            return
        if(action.type == action_type.wait):
            if (self.cur_stack.had_waited):
                logger.info("{} is already waited".format(self.cur_stack))
                return
            self.cur_stack.wait()
        elif(action.type == action_type.defend):
            self.cur_stack.defend()
        elif(action.type == action_type.move):
            if (self.cur_stack.x == action.dest.x and self.cur_stack.y == action.dest.y):
                logger.info("can't move to where you already are!!")
                return
            if (self.canReach(self.cur_stack, action.dest)):
                self.move(self.cur_stack, action.dest)
            else:
                logger.info("you can't reach ({},{})".format(action.dest.y,action.dest.x))
        elif(action.type == action_type.attack):
            dests = self.findStack(action.target,True)
            if(len(dests) == 0):
                logger.info("wrong attack dist ({},{})".format(action.target.y,action.target.x))
                return
            dest = dests[0]
            if (self.canReach(self.cur_stack, action.dest, action.target)):
                self.move(self.cur_stack, action.dest)
                self.cur_stack.meeleAttack(dest, action.dest, False)
            else:
                logger.info("you can't reach ({},{}) and attack {}".format(action.dest.y,action.dest.x,action.target.name))
                exit(-1)
        elif(action.type == action_type.shoot):
            dists = self.findStack(action.target, True)
            if (len(dists) == 0):
                logger.info("wrong shoot dist ({},{})".format(action.target.y,action.target.x))
                return
            dist = dists[0]
            if (self.cur_stack.can_shoot(action.target)):
                self.cur_stack.shoot(dist)
            else:
                logger.info("{} can't shoot {}".format(self.cur_stack.name, dist.name))
        elif (action.type == action_type.spell):
            logger.info("spell not implemented yet")
    def legal_act(self,level=0,act_id=0,spell_id=0,target_id=0):
        cur_stack = self.cur_stack
        if cur_stack.had_moved:
            return None

        if level ==0:
            legals = np.zeros((5,))
            if not cur_stack.had_waited:
                legals[action_type.wait.value] = 1
            if not cur_stack.had_defended:
                legals[action_type.defend.value] = 1
            if cur_stack.get_global_state(query_type=action_query_type.can_move.value):
                legals[action_type.move.value] = 1
            if cur_stack.get_global_state(query_type=action_query_type.can_attack.value):
                legals[action_type.attack.value] = 1
            return legals
        elif level == 1:
            if act_id == action_type.move.value:
                bf = cur_stack.get_global_state().flatten()
                mask = (bf >= 0) & (bf < 50)
                return mask.flatten()
            elif act_id == action_type.attack.value:
                bf = cur_stack.get_global_state()
                mask = np.zeros((7,))
                targets = self.defender_stacks if cur_stack.side == 0 else self.attacker_stacks
                for i in range(len(targets)):
                    if bf[targets[i].y, targets[i].x] == 201:
                        mask[i] = 1
                return mask
            else:
                return np.zeros((self.bFieldHeight,self.bFieldWidth))
        elif level == 2:
            mask = np.zeros((self.bFieldHeight,self.bFieldWidth))
            if act_id == action_type.attack.value:
                bf = cur_stack.get_global_state(exclude_me=False)
                target = self.defender_stacks[target_id] if cur_stack.side == 0 else self.attacker_stacks[target_id]
                nb = target.get_neighbor()
                for t in nb:
                    if 0 <= bf[t.y, t.x] < 50:
                        mask[t.y, t.x] = 1
                return mask.flatten()
            else:
                return mask.flatten()
class  BPlayer(object):
    def getAction(self,battle):
        return  battle.cur_stack.active_stack()
class BAction:
    def __init__(self,type = 0,dest = None,target = None,spell=None):
        self.type = type
        self.dest = dest
        self.target = target
        self.spell = spell
def main():
    pass
if __name__ == '__main__':
    main()


