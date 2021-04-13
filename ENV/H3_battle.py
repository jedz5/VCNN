import random
import copy
from typing import List, Tuple
from operator import itemgetter
import numpy as np
from enum import Enum
import logging
import os
import json
# from ENV.python_json_numpy_print import save_formatted
np.set_printoptions(precision=2,suppress=True,sign=' ',linewidth=400,formatter={'float': '{: 0.2f}'.format})
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
std_logger = logging.getLogger('train')
handler = logging.FileHandler('train.log','w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
std_logger.addHandler(handler)
import platform
Linux = "Linux" == platform.system()
import sys
if Linux:
    sys.path.extend(['/home/enigma/work/project/VCNN/','/home/enigma/work/project/VCNN/VCCC/VCbattle/build'])
else:
    sys.path.extend(['D:\\project\\VCNN', 'D:\\project\\VCNN\\VCCC\\x64\\Release'])
import VCbattle  as vb
from VCbattle import BHex
def set_logger(lg_on,lg):
    global log_gui_on
    log_gui_on = lg_on
    global logger
    logger = lg
def get_logger():
    return log_gui_on,logger
set_logger(False,std_logger)
diretMap = {'0':3,'1':4,'2':5,'3':0,'4':1,'5':2}
battle_start_pos_att = \
[[ 86 ],
 [ 35, 137 ],
 [ 35, 86, 137 ],
 [ 1, 69, 103, 171 ],
 [ 1, 35, 86, 137, 171 ],
 [ 1, 35, 69, 103, 137, 171 ],
 [ 1, 35, 69, 86, 103, 137, 171 ]]
battle_start_pos_def = \
[[ 100 ],
 [ 49, 151 ],
 [ 49, 100, 151 ],
 [ 15, 83, 117, 185 ],
 [ 15, 49, 100, 151, 185 ],
 [ 15, 49, 83, 117, 151, 185 ],
 [ 15, 49, 83, 100, 117, 151, 185 ]]
class action_type(Enum):
    wait = 0
    defend = 1
    move = 2
    attack = 3
    spell = 4
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
# class BHex:
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#     def __eq__(self, other):
#         if(other):
#             return self.y == other.y and self.x == other.x
#         return False
#     def flat(self):
#         return self.y * Battle.bFieldHeight + self.x
class BStack(BHex):
    def __init__(self):
        super().__init__()
        self.amount = 0
        self.amount_base = 0
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
        self.slotId = 0
        # self.hex_type = hexType.creature
        #辅助参数
        self.by_AI = 1
        self.name = 'unKnown'
        self.is_wide = False
        self.is_fly = False
        self.is_shooter = False
        self.block_retaliate = False
        self.attack_nearby_all = False
        self.wide_breath = False
        self.infinite_retaliate = False
        self.attack_twice = False
        self.in_battle = None #type:Battle
    def __copy__(self):
        cp = BStack()
        cp.amount = self.amount
        cp.amount_base = self.amount_base
        cp.attack = self.attack
        cp.defense = self.defense
        cp.max_damage = self.max_damage
        cp.min_damage = self.min_damage
        cp.first_HP_Left = self.first_HP_Left
        cp.health = self.health
        cp.side = self.side
        cp.had_moved = self.had_moved
        cp.had_retaliated = self.had_retaliated
        cp.had_waited = self.had_waited
        cp.had_defended = self.had_defended
        cp.speed = self.speed
        cp.luck = self.luck
        cp.morale = self.morale
        cp.id = self.id
        cp.shots = self.shots
        cp.slotId = self.slotId
        # self.hex_type = hexType.creature
        # 辅助参数
        cp.x = self.x
        cp.y = self.y
        cp.by_AI = self.by_AI
        cp.name = self.name
        cp.is_wide = self.is_wide
        cp.is_fly = self.is_fly
        cp.is_shooter = self.is_shooter
        cp.block_retaliate = self.block_retaliate
        cp.attack_nearby_all = self.attack_nearby_all
        cp.wide_breath = self.wide_breath
        cp.infinite_retaliate = self.infinite_retaliate
        cp.attack_twice = self.attack_twice
        cp.in_battle = self.in_battle
        cp.ai_value = self.ai_value
        cp.fight_value = self.fight_value
        return cp
    def __repr__(self):
        w = 'w'if self.had_waited else ''
        m = 'm'if self.had_moved else ''
        return f'{self.name}_{self.amount}_{w}{m} at ({self.y},{self.x})'
    def __eq__(self, other):
        if isinstance(other,BStack):
            return other.id == self.id and other.amount == other.amount and super(BStack, self).__eq__(other)
        return super(BStack, self).__eq__(other)
    def get_global_state(self, query_type = -1, exclude_me = True):
        return vb.get_global_state(self,self.in_battle.stacks,query_type,exclude_me)
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
        #             d = self.vb.get_distance(BHex(jj, ii))
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
        if(self.attack >= opposite.defense):
            power = 1 + min((self.attack - opposite.defense),60) * 0.05
            damageMin = int(self.min_damage * power * self.amount)
            damageMax = int(self.max_damage * power * self.amount)
        else:
            power = 1 + max((self.attack - opposite.defense),-30) * 0.025
            damageMin = max(int(self.min_damage * power * self.amount),1)
            damageMax = max(int(self.max_damage * power * self.amount),1)
        damage = int((damageMin+damageMax)/2) if estimate else random.randint(damageMin,damageMax)
        can_shoot = self.can_shoot()
        if(self.is_shooter):
            if(not can_shoot or self.is_half(opposite)):
                damage = int(damage/2)
                damage = max(damage, 1)
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
                    logger.debug(tt,True)
                    if(opposite.amount == 0):
                        logger.debug("{} perished".format(opposite.name),True)
        real_damage, killed, first_HP_Left = opposite.damaged(damage)
        total_damage += real_damage
        if(not estimate):
            if(can_shoot): # damage is done and enemy may die, so this flag may change
                half = "(half)" if self.is_half(opposite) else "(full)"
                logger.debug("shoot{} {}".format(half,opposite.name),True)
            head = "reta" if is_reta else "make"
            tt = "{} {} dmg killed {} {} {} left, HP {}".format(head,real_damage, killed, opposite.name,opposite.amount, first_HP_Left)
            logger.debug(tt,True)
            if(opposite.amount == 0):
                logger.debug("{} perished".format(opposite.name),True)
        return total_damage,killed,first_HP_Left
    def meeleAttack(self,opposite,dest,is_retaliate):
        return self.do_attack(opposite,dest)
    def can_shoot(self, opposite = None):
        if (not self.is_shooter or self.shots <= 0):
            return False
        for enemy in self.in_battle.stacks:
            if(enemy.side != self.side and enemy.is_alive() and vb.get_distance(self,enemy) == 1):
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
        if(vb.check_position(x,y)):
            adj.append(BHex(x,y))
    def is_half(self, dist):
        return vb.get_distance(self,dist) > Battle.bPenaltyDistance
    # def vb.get_distance(self, dist):
    #     return  vb.get_distance(self,dist)
    def shoot(self,opposite):
        logger.debug('{} shooting {}'.format(self.name,opposite.name))
        damage_dealt, damage_get, killed_dealt, killed_get = self.do_attack(opposite,self.get_position())
        self.shots -= 1
        return damage_dealt,damage_get,killed_dealt,killed_get
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

    def potential_target(self): #->  type: Tuple[List[Tuple[BStack,int]],List]
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
        if self.had_moved:
            logger.error(f"{self.name}({self.y},{self.x}) had moved already")
            exit(-1)
        damage_get = 0
        damage_dealt = 0
        killed_dealt = 0
        killed_get = 0
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
            killed_dealt += killed1
            attacker.had_moved = True
            if (not defender.is_alive()):
                break
            if(not attacker.can_shoot() and defender.is_alive() and not (defender.had_retaliated or attacker.block_retaliate)):
                damage2, killed2, first_HP_Left2 = defender.computeCasualty(attacker,defender.get_position(),True,estimate)
                damage_get += damage2
                killed_get += killed2
                if(not defender.infinite_retaliate):
                    defender.had_retaliated = True
        return damage_dealt,damage_get,killed_dealt,killed_get
    def go_toward(self,target):
        df = self.get_global_state()
        min_dist = 999
        dest = None
        for i in range(Battle.bFieldHeight):
            for j in range(Battle.bFieldWidth):
                if(0 <= df[i,j] < 50):
                    temp = BHex(j, i)
                    distance = vb.get_distance(temp,target)
                    if(distance < min_dist):
                        min_dist = distance
                        dest = temp
        if(vb.get_distance(self,target) <= min_dist):
            return BAction(action_type.defend)
        else:
            return BAction(action_type.move, dest=dest)

    def active_stack(self,ret_obs = False,print_act = False,action_known:tuple=None):
        if self.by_AI == 1:
            attackable, unreach = self.potential_target() #type: Tuple[List[Tuple[BStack,int]],List]
            if not self.had_waited and len(attackable) == 0:
                return BAction(action_type.wait)
            if (len(attackable) > 0):
                att = [(self.do_attack(target, stand, estimate=True) + (-target.is_shooter,-target.had_retaliated,idx)) for idx,(target, stand) in enumerate(attackable)]
                dmgs = [(is_shooter,had_retaliated,-delt,idx) for delt, get,kd,kg,is_shooter,had_retaliated,idx in att]
                # best = np.argmax(dmgs)
                best = sorted(dmgs,key=itemgetter(0,1,2))
                best_target = attackable[best[0][3]]
                if (self.can_shoot()):
                    return BAction(action_type.attack, target=best_target[0])
                else:
                    return BAction(action_type.attack, target=best_target[0], dest=best_target[1])
            else:
                distants = [vb.get_distance(self,x) for x in unreach]
                closest = np.argmin(distants)
                return self.go_toward(unreach[closest])
        elif self.by_AI == 2 and self.in_battle.agent:
            # {'act_id':act_id, 'spell_id':spell_id,'target_id':target_id,'position_id':position_id,
            #             'mask_acts':mask_acts,'mask_spell':mask_spell,'mask_targets':mask_targets,'mask_position':mask_position}
            return self.in_battle.agent.choose_action(self.in_battle,ret_obs,print_act,action_known=action_known)
        elif self.by_AI == 0:
            return
        else:
            logger.error("no way to contrl the stack!!")
            sys.exit(-1)

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
creature_ability = {0:[0,0,0,0,0,0,0],1:[0,0,0,0,0,0,0],3:[0,1,0,0,0,0,1],5:[1,0,0,0,0,1,0],7:[0,0,0,0,0,0,1],19:[0,1,0,0,0,0,1],
                    41: [0, 0, 0, 0, 0, 0, 0],50:[0,0,0,0,0,0,0],51:[0,0,0,0,0,0,0],52:[1,0,0,0,0,0,0],85:[0,0,0,0,0,0,0],99:[0,0,0,0,0,0,0],112:[0,0,0,0,0,0,0],119:[1,0,1,0,0,0,0],
                            121:[0,0,1,1,0,0,0],125:[0,0,0,0,0,0,0],131:[1,0,0,0,1,0,0],}
class Battle(object):
    bFieldWidth = 17
    bFieldHeight = 11
    bFieldStackProps = 18
    bFieldStackPlanes = 46
    bPenaltyDistance = 10
    bFieldSize = (bFieldWidth - 2)* bFieldHeight
    bTotalFieldSize = 2 + 8*bFieldSize
    def __init__(self,by_AI = [2,1], gui = None , load_file = None,agent = None,debug = False,cur_stack:BStack=None,last_stack:BStack=None):
        #self.bField = [[0 for col in range(battle.bFieldWidth)] for row in range(battle.bFieldHeight)]
        self.stacks = [] #type:List[BStack]
        self.defender_stacks = [] #type:List[BStack]
        self.attacker_stacks = [] #type:List[BStack]
        self.round = 0
        self.obstacles = []
        self.obsinfo = []
        self.toMove = []
        self.waited = []
        self.moved = []
        self.stackQueue = [] #type:List[BStack]
        self.cur_stack = cur_stack
        self.last_stack = last_stack
        self.batId = 0
        self.by_AI = by_AI
        self.ai_value = [0,0]
        self.bat_interface = None
        self.agent = agent
        self.debug = debug
        if(gui):
            self.bat_interface = gui
        if(load_file):
            self.loadFile(load_file)
            self.checkNewRound()

    def reset(self):
        for att in self.attacker_stacks:
            att.amount_base = att.amount
            att.first_HP_Left = att.health
            att.y, att.x, att.slotId = 0, 0, 0
            att.had_waited = False
            att.had_moved = False
            att.had_retaliated = False
            att.had_defended = False
            att.shots = 16
        for deff in self.defender_stacks:
            deff.amount = deff.amount_base
            deff.first_HP_Left = deff.health
            deff.y, deff.x, deff.slotId = 0, 0, 0
            deff.had_waited = False
            deff.had_moved = False
            deff.had_retaliated = False
            deff.had_defended = False
            deff.shots = 16
        self.round = 0
        self.toMove.clear()
        self.waited.clear()
        self.moved.clear()
        self.stackQueue.clear()
        self.cur_stack = None
        self.last_stack = None
    def split_army(self,side=0):
        cmap = {}
        if side == 0:
            i = 0
            stacks = list(filter(lambda elem: elem.is_alive(),self.attacker_stacks)) #list(filter(lambda elem: elem.is_alive() and elem.had_moved, self.stacks))
            assert len(stacks) > 0,"your army is gone..."
            '''merge stacks'''
            while i != len(stacks):
                st = stacks[i]
                assert st.amount == st.amount_base,f"{st.amount} != {st.amount_base}"
                if st.id in cmap:
                    st_0 = stacks[cmap[st.id]]
                    st_0.amount += st.amount
                    st_0.amount_base = st_0.amount
                    stacks.pop(i)
                else:
                    cmap[st.id] = i
                    i += 1
            '''shooter first'''
            stacks.sort(key=lambda elem:(-elem.is_shooter))
            st_to_split = stacks[-1]
            if st_to_split.amount_base >= 7 - len(stacks):
                num_st = 7 - len(stacks)
                st_to_split.amount_base -= num_st
                st_to_split.amount = st_to_split.amount_base
                for i in range(num_st):
                    new_st = copy.copy(st_to_split)
                    new_st.amount_base, new_st.amount = 1, 1
                    stacks.append(new_st)
            else:
                num_st = st_to_split.amount_base - 1
                st_to_split.amount_base -= num_st
                st_to_split.amount = st_to_split.amount_base
                for i in range(num_st):
                    new_st = copy.copy(st_to_split)
                    new_st.amount_base, new_st.amount = 1, 1
                    stacks.append(new_st)
            sl = len(stacks)
            '''shooter and pikeman'''
            if not st_to_split.is_shooter:
                stacks.remove(st_to_split)
                stacks.insert(sl//2,st_to_split)
            self.attacker_stacks = stacks
            self.stacks = self.attacker_stacks + self.defender_stacks
            self.format_postions()


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
    def loadFile(self,file,shuffle_postion = True,load_ai_side = True):
        bf = np.zeros([self.bFieldHeight,self.bFieldWidth])
        with open("ENV/creatureData.json") as JsonFile:
            crList = json.load(JsonFile)["creatures"]
        with open(file) as jsonFile:
            root = json.load(jsonFile)
            if load_ai_side:
                self.by_AI = [1,1]
                agent_s = random.choice(root["agent_side"])
                self.by_AI[agent_s] = 2
            slot0, slot1 = 0, 0
            for i in range(2):
                li = list(range(11))
                j = 0
                for id,num,py,px in root['army{}'.format(i)]:
                    if not shuffle_postion:
                        assert bf[py, px] != 1
                    bf[py, px] = 1
                    x = crList[id]
                    st = BStack()
                    st.attack = x['attack']
                    st.defense = x['defense']
                    st.amount = num
                    st.amount_base = num
                    st.health = x['health']
                    st.first_HP_Left = x['health']
                    st.id = id
                    if i == 0:
                        st.slotId = slot0
                        slot0 += 1
                    else:
                        st.slotId = slot1
                        slot1 += 1
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
                    # if shuffle_postion:
                    #     st.x = 15 if i else 1
                    #     # st.x = px
                    #     st.y = int(ys[j])
                    # else:
                    #     st.x = px
                    #     st.y = py
                    st.ai_value = x['ai_value']
                    st.fight_value = x['fight_value']
                    j += 1
                    st.in_battle = self
                    self.stacks.append(st)
                    if st.side:
                        self.defender_stacks.append(st)
                    else:
                        self.attacker_stacks.append(st)
        if shuffle_postion:
            self.init_stack_position()
            # for x in root['obstacles']:
            #     obs = BObstacle()
            #     obs.kind = x['type']
            #     obs.x = x['x']
            #     obs.y = x['y']
            #     self.obstacles.append(obs)
            # for x in root['obs']:
            #     oi = BObstacleInfo(x["pos"],x["width"],x["height"],bool(x["isabs"]),x["imname"])
            #     self.obsinfo.append(oi)

    def load_battle(self,file,load_ai_side = True, shuffle_postion=False,format_postion = False):
        bf = np.zeros([self.bFieldHeight,self.bFieldWidth])
        with open("ENV/creatureData.json") as JsonFile:
            crList = json.load(JsonFile)["creatures"]
        if isinstance(file,str):
            with open(file) as JsonFile:
                js = json.load(JsonFile)
                curr = js["stacks"]
                curr = np.array(curr)
                self.round = js['round']
                if load_ai_side:
                    agent_side = random.choice(js['agent_side'])
                    self.by_AI = [1,1]
                    self.by_AI[agent_side] = 2
        else:
            curr,self.round = file
        for i in range(14):
            side, slotId, id, amount, amount_base, first_HP_Left, health, luck, attack, defense, max_damage, min_damage,speed, morale,shots,py, px, had_moved, had_waited, had_retaliated, had_defended= curr[i]
            if px == 0:
                break
            if not (shuffle_postion or format_postion):
                assert bf[py,px] != 1
                bf[py, px] = 1
            x = crList[id]
            st = BStack()
            st.attack = attack
            st.defense = defense
            st.amount = amount
            st.amount_base = amount_base
            st.health = health
            st.first_HP_Left = first_HP_Left
            st.id = id
            st.slotId = slotId
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
            st.ai_value = x['ai_value']
            st.fight_value = x['fight_value']
            # st.ai_R1 = speed * int((max_damage + min_damage) / 2) * attack * (1 + int(st.is_shooter))
            # st.ai_R2 = health * defense
            st.in_battle = self
            self.stacks.append(st)
            if st.side:
                self.defender_stacks.append(st)
            else:
                self.attacker_stacks.append(st)
        if shuffle_postion:
            self.init_stack_position()
        if format_postion:
            self.format_postions()
    def format_postions(self):
        for stacks, pos in [(self.attacker_stacks, battle_start_pos_att), (self.defender_stacks, battle_start_pos_def)]:
            sl = len(stacks)
            sp = pos[sl - 1]
            for i in range(sl):
                stacks[i].y = sp[i] // Battle.bFieldWidth
                stacks[i].x = sp[i] % Battle.bFieldWidth
                stacks[i].slotId = i
    def dump_battle(self,dir):
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
        # with open(dump_in,'w', encoding='utf-8') as JsonFile:
            # JsonFile.write(pprint.pformat({'stacks':attri_stack.tolist(),'round':self.round},indent=1,width=256).replace('\'','"'))
        '''formated'''
        save_formatted(dump_in,{'stacks':attri_stack,'round':self.round})
        print(f"states dumped in {dump_in}")
    def canReach(self,bFrom,bTo,bAtt = None):
        curSt = bFrom
        if(not vb.check_position(bTo.x,bTo.y)):
            logger.error('dist {},{} not valid'.format(bTo.y,bTo.x))
            sys.exit(-1)
        bf = curSt.get_global_state(exclude_me=False)
        if(bAtt):
            return vb.get_distance(bTo,bAtt) == 1 and 50 > bf[bTo.y,bTo.x] >= 0 and bf[bAtt.y,bAtt.x] == 201
        else:
            return 50 > bf[bTo.y,bTo.x] >= 0
    def move(self,bFrom,bTo,and_attack = False):
        if bFrom.had_moved:
            logger.error(f"{bFrom.name}({bFrom.y},{bFrom.x}) had moved already")
            exit(-1)
        bFrom.x = bTo.x
        bFrom.y = bTo.y
        if and_attack:
            return
        bFrom.had_moved = True

    def init_stack_position(self):
        mask = np.zeros([11, 17])
        mask[:, 0] = 1
        mask[:, 16] = 1
        # self.attacker_stacks[0].amount = random.randint(25,35)
        # self.defender_stacks[0].amount = random.randint(3, 5)
        for st in self.stacks:
            # base1 = random.random() * 2 + 0.1
            # st.amount_base = int(st.amount_base * base1)
            # st.amount = st.amount_base
            pos = random.randint(1, 11 * 17 - 1)
            while True:
                if mask[int(pos / 17), pos % 17]:
                    pos = random.randint(1, 11 * 17 - 1)
                else:
                    break
            st.x = pos % 17
            st.y = int(pos / 17)
            mask[st.y, st.x] = 1
    #TODO get_act_masks
    def get_act_masks(self,act):
        mask_targets = np.zeros((14,))
        mask_position = np.zeros((self.bFieldHeight * self.bFieldWidth,))
        mask_spell = np.zeros((10,))
        mask_acts = self.legal_act(level=0)
        if act.type == action_type.move:
            mask_position = self.legal_act(level=1,act_id=act.type.value)
        if act.type == action_type.attack:
            mask_targets = self.legal_act(level=1, act_id=act.type.value)
            if not self.cur_stack.can_shoot():
                mask_position = self.legal_act(level=2, act_id=act.type.value,target=act.target)
        return mask_acts,mask_spell,mask_targets,mask_position
    def sortStack(self):
        self.last_stack = self.cur_stack
        self.toMove = list(filter(lambda elem: elem.is_alive() and not elem.had_moved and not elem.had_waited, self.stacks))
        self.waited = list(filter(lambda elem: elem.is_alive() and not elem.had_moved and elem.had_waited, self.stacks))
        self.moved = list(filter(lambda elem: elem.is_alive() and elem.had_moved, self.stacks))

        self.toMove.sort(key=lambda elem:(-elem.speed,elem.slotId,elem.side))
        self.waited.sort(key=lambda elem: (elem.speed, -elem.slotId,-elem.side))
        self.moved.sort(key=lambda elem: (-elem.speed))
        self.stackQueue = self.toMove + self.waited + self.moved
        self.cur_stack = self.stackQueue[0]
    def currentPlayer(self):
        return 1 if self.stackQueue[0].side else 0

    def currentState(self):
        pass
    def current_state_feature(self,curriculum = True):
        planes_stack  = np.zeros((14,3,self.bFieldHeight,self.bFieldWidth),bool)
        attri_stack = np.zeros((14,21),dtype=int)
        ind = np.array([122] * 14,dtype=int)
        for i,st in enumerate(self.stackQueue):
            bf = st.get_global_state()
            planes_stack[i, 0] = ((bf >= 0) & (bf < 50))
            planes_stack[i, 1] = (bf == 401)
            planes_stack[i, 2] = (bf == 201)
            #
            ind[i] = st.id
            # if curriculum:
            attri_stack[i] = np.array(
                [st.side,st.slotId, st.id , st.amount, st.amount_base, st.first_HP_Left, st.health, st.luck,st.attack, st.defense, st.max_damage, st.min_damage,
                  st.speed,  st.morale,
                 st.shots,st.y, st.x,int(st.had_moved), int(st.had_waited), int(st.had_retaliated), int(st.had_defended)])
            # else:
            #     attri_stack[i] = np.array(
            #         [st.side, st.amount, st.first_HP_Left, st.attack, st.defense, st.max_damage, st.min_damage,
            #          int(st.had_moved), int(st.had_retaliated), int(st.had_waited), st.speed, st.luck, st.morale,
            #          st.shots])
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


    def state_represent(self):
        attri_stack = []
        for i, st in enumerate(self.stackQueue):

            attri_stack.append((st.side,st.y, st.x, st.id,2 if st.amount > 1 else 1,int(st.had_moved),int(st.had_waited)))
        return tuple(attri_stack)
    def getStackHPBySlots(self):
        pass

    def findStack(self,dest,alive=True):
        ret = list(filter(lambda elem: elem.x == dest.x and elem.y == dest.y and elem.is_alive() == alive, self.stacks))
        return ret
    def direction_to_hex(self, mySelf, dirct):
        zigzagCorrection =0 if (mySelf.y % 2) else 1
        if(dirct < 0 or dirct > 5):
            logger.error('wrong direction {}'.format(dirct))
            sys.exit()
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
            if not self.cur_stack.can_shoot():
                return "melee ({},{}),({},{})".format(act.dest.y,act.dest.x,act.target.y,act.target.x)
            else:
                return "shoot ({},{})".format(act.target.y,act.target.x)
    def end(self):
        live = {0:False,1:False}
        for st in self.stacks:
            live[st.side] = live[st.side] or st.is_alive()
        if self.round > 20:
            return True,0
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
            logger.debug("battle end~")
            return
        if(self.stackQueue[0].had_moved):
            self.newRound()
            self.sortStack()
        if(not is_self_play):
            side = "" if self.cur_stack.side else "me "
            logger.debug("now it's {}{} turn".format(side, self.cur_stack.name), True)

    def newRound(self):
        self.round += 1
        logger.debug("round {}".format(self.round),True)
        for st in self.stacks:
            if(st.is_alive()):
                st.newRound()
    def doAction(self,action):
        damage_dealt, damage_get, killed_dealt, killed_get = 0,0,0,0
        if log_gui_on:
            logger.log_text.clear()
        logger.debug(self.action2Str(action),True)
        if(self.cur_stack.had_moved):
            logger.error("{} is already moved".format(self.cur_stack))
            sys.exit()
        if(action.type == action_type.wait):
            if (self.cur_stack.had_waited):
                logger.error("{} is already waited".format(self.cur_stack))
                sys.exit()
            self.cur_stack.wait()
        elif(action.type == action_type.defend):
            self.cur_stack.defend()
        elif(action.type == action_type.move):
            if (self.cur_stack.x == action.dest.x and self.cur_stack.y == action.dest.y):
                logger.error("can't move to where you already are!!")
                sys.exit()
            if (self.canReach(self.cur_stack, action.dest)):
                self.move(self.cur_stack, action.dest)
            else:
                logger.error("you can't reach ({},{})".format(action.dest.y,action.dest.x))
                sys.exit()
        elif(action.type == action_type.attack):
            targets = self.findStack(action.target,True)
            if(len(targets) == 0):
                logger.error("wrong attack dist ({},{})".format(action.target.y,action.target.x))
                sys.exit()
            target = targets[0]
            if target.side == self.cur_stack.side:
                logging.error(f"target {dest.name}is in your side!")
                sys.exit()
            if self.cur_stack.can_shoot():
                damage_dealt, damage_get, killed_dealt, killed_get = self.cur_stack.shoot(target)
            elif self.canReach(self.cur_stack, action.dest, action.target):
                self.move(self.cur_stack, action.dest,and_attack=True)
                damage_dealt, damage_get, killed_dealt, killed_get = self.cur_stack.meeleAttack(target, action.dest, False)
            else:
                logger.error("you can't reach ({},{}) and attack {}".format(action.dest.y,action.dest.x,action.target.name))
                sys.exit(-1)
        else:
            logger.error("spell not implemented yet")
            sys.exit()
        return damage_dealt, damage_get, killed_dealt, killed_get
    def legal_act(self,level=0,act_id=0,spell_id=0,target_id=-1,target = None):
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
                mask = np.zeros((14,))
                # targets = self.defender_stacks if cur_stack.side == 0 else self.attacker_stacks
                targets = self.stackQueue
                for i in range(len(targets)):
                    if bf[targets[i].y, targets[i].x] == 201:
                        mask[i] = 1
                return mask
            else:
                logger.error(f"wrong act_id = {act_id}")
                sys.exit()
        elif level == 2:
            mask = np.zeros((self.bFieldHeight,self.bFieldWidth))
            if act_id == action_type.attack.value:
                bf = cur_stack.get_global_state(exclude_me=False)
                if not target:
                    assert target_id >= 0,"need target_id!!"
                    # target = self.defender_stacks[target_id] if cur_stack.side == 0 else self.attacker_stacks[target_id]
                    target = self.stackQueue[target_id]
                nb = target.get_neighbor()
                for t in nb:
                    if 0 <= bf[t.y, t.x] < 50:
                        mask[t.y, t.x] = 1
                return mask.flatten()
            else:
                logger.error(f"{act_id} of level2 is not implemented yet!")
                sys.exit()
class  BPlayer(object):
    def getAction(self,battle):
        return  battle.cur_stack.active_stack()
class BAction:
    def __init__(self,type:action_type = None,dest:BHex = None,target:BStack = None,spell=None):
        self.type = type
        self.dest = dest
        self.target = target
        self.spell = spell
    def __repr__(self):
        if self.type == action_type.wait:
            return f'action wait'
        if self.type == action_type.defend:
            return f'action defend'
        if self.type == action_type.move:
            return f'action move to ({self.dest.y},{self.dest.x})'
        if self.type == action_type.attack:
            if self.dest:
                return f'action melee attack {self.target.name}_{self.target.amount}_({self.target.y},{self.target.x}) at ({self.dest.y},{self.dest.x})'
            else:
                return f'action shoot {self.target.name}_{self.target.amount}_({self.target.y},{self.target.x})'
        assert False,f'wrong action {self.type.value}!!'


