import random
import json
import copy
import numpy as np
from enum import Enum
import logging
import sys
import torch

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
std_logger = logging.getLogger('train')
handler = logging.FileHandler('train.log','w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
std_logger.addHandler(handler)


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

def printF(bh,stacks,curSt):
    def f(n):
        if n >= 0 and n < 50:
            if(n == 0 or n == curSt.speed - 1 or n%3 == 0):
                return n
            else:
                return BHex.mapp[51]
        else:
            return BHex.mapp[n]
    bf = [[f(n) for n in line] for line in bh]
    def ff(l):
        for n in l:
            if(not isinstance(n,str)):
                print('%3d'%n, end=".")
            else:
                print(n, end=".")
        print()

    bf[curSt.y,curSt.x] = 'M'
    for st in stacks:
        if(st.isAlive()):
            bf[st.y,st.x] = bf[st.y,st.x]+str(st.amount)
    print("      ",end="")
    [print("%3d "%i,end="") for i in range(1,len(bf[0]) - 1)]
    print()
    i = 0
    for line in bf:
        if(i % 2 == 0):
            print("%3d  "%i,end="")
        else:
            print("%3d"%i,end="")
        i += 1
        ff(line)


class actionType(Enum):
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
class hexType(Enum):
    creature = 0
    obstacle = 1
mapp = {100: '|', -9: '   ', -8: ' * ', -4: 'M', -2: 'E', -1: 'A', 51: '   '}
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
        self.maxDamage = 0
        self.minDamage = 0
        self.firstHPLeft = 0
        self.health = 0
        self.isMoved = False
        self.side = 0
        self.had_retaliated = False
        self.isWaited = False
        self.speed = 0
        self.luck = 0
        self.morale = 0
        self.id = 0
        self.shots = 10
        self.hexType = hexType.creature
        #辅助参数
        self.isDenfenced = False
        self.name = 'unKnown'
        self.slotId = 0
        self.x = 0
        self.y = 0
        self.isWide = False
        self.isFly = False
        self.isShooter = False
        self.blockRetaliate = False
        self.attack_nearby_all = False
        self.wide_breath = False
        self.infinite_retaliate = False
        self.attack_twice = False
        self.amountBase = 0
        self.inBattle = 0  #Battle()
    def __eq__(self, other):
        if(other):
            return self.x == other.x and self.y == other.y
        return False
    def acssessableAndAttackable(self,query_type = None,exclude_me = True):
        bf = np.ones((self.inBattle.bFieldHeight,self.inBattle.bFieldWidth))
        bf.fill(-1)
        bf[:, 0] = 100
        bf[:, -1] = 100
        for sts in self.inBattle.stacks:
            if(sts.isAlive()):
                bf[sts.y,sts.x] = 400 if sts.side == self.side else 200
        for obs in self.inBattle.obstacles:
            bf[obs.y,obs.x] = 800
        #init battleField end
        # accessable  begin
        travellers = []
        bf[self.y,self.x] = self.speed
        travellers.append(self)
        if(not self.isFly):
            while(len(travellers) > 0):
                current = travellers.pop()
                speedLeft = bf[current.y,current.x] - 1
                for adj in self.getNeibours(current):
                    if(bf[adj.y,adj.x] < speedLeft):
                        bf[adj.y,adj.x] = speedLeft
                        if (speedLeft > 0):
                            travellers.append(adj)
                            if query_type == action_query_type.can_move:
                                return True
        else: #fly
            for ii in range(self.inBattle.bFieldHeight):
                for jj in range(1,self.inBattle.bFieldWidth-1):
                    if bf[ii,jj] > 50:
                        continue
                    d = self.getDistance(BHex(jj,ii))
                    if(0 < d <= self.speed):
                        bf[ii,jj] = self.speed - d
                        if query_type == action_query_type.can_move:
                            return True
        #no space to move to
        if query_type == action_query_type.can_move:
            return False
        #accessable  end
        #attackable begin
        for sts in self.inBattle.stacks:
            if(not sts.isAlive()):
                continue
            if sts.side != self.side:
                if (self.canShoot()):
                    bf[sts.y,sts.x] = 201  # enemy and attackbale
                    if query_type == action_query_type.can_attack:
                        return True
                else:
                    for neib in self.getNeibours(sts):
                        if (0 <= bf[neib.y,neib.x] < 50):
                            bf[sts.y,sts.x] = 201
                            if query_type == action_query_type.can_attack:
                                return True
                            break
        #no target to reach
        if query_type == action_query_type.can_attack:
            return False
        if exclude_me:
            bf[self.y,self.x] = 401
        return bf
    def damaged(self,damage):
        hpInAll = self.health * (self.amount - 1) + self.firstHPLeft
        if (damage >= hpInAll):
            damage = hpInAll
            killed = self.amount
            firstHPLeft = 0
        else:
            rest = int((hpInAll - damage - 1) / self.health) + 1
            firstHPLeft = (hpInAll - damage - 1) % self.health + 1
            killed = self.amount - rest
        self.amount -= killed
        self.firstHPLeft = firstHPLeft
        return damage,killed,firstHPLeft
    def computeCasualty(self,opposite,stand,is_reta, estimate=False):
        total_damage = 0
        if(self.attack >= opposite.attack):
            damageMin = int(self.minDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount)
            damageMax = int(self.maxDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount)
        else:
            damageMin = int(self.minDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
            damageMax = int(self.maxDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
        damage = int((damageMin+damageMax)/2) if estimate else random.randint(damageMin,damageMax)
        if(self.isShooter):
            if(not self.canShoot() or self.isHalf(opposite) ):
                damage = int(damage/2)
        else:
            others = self.get_attacked_stacks(opposite,stand)
            for st_tmp in others:
                if(estimate):
                    st = copy.copy(st_tmp)
                else:
                    st = st_tmp
                real_damage, killed, firstHPLeft = st.damaged(damage)
                if(self.side == st.side):
                    total_damage += -real_damage
                else:
                    total_damage += real_damage
                if (not estimate):
                    head = "reta" if is_reta else "make"
                    tt = "{} {} dmg killed {} {} {} left, HP {}".format(head,real_damage, killed, st.name,st.amount, firstHPLeft)
                    logger.info(tt,True)
                    if(opposite.amount == 0):
                        logger.info("{} perished".format(opposite.name),True)
        real_damage, killed, firstHPLeft = opposite.damaged(damage)
        total_damage += real_damage
        if(not estimate):
            if(self.canShoot()):
                half = "(half)" if self.isHalf(opposite) else "(full)"
                logger.info("shoot{} {}".format(half,opposite.name),True)
            head = "reta" if is_reta else "make"
            tt = "{} {} dmg killed {} {} {} left, HP {}".format(head,real_damage, killed, opposite.name,opposite.amount, firstHPLeft)
            logger.info(tt,True)
            if(opposite.amount == 0):
                logger.info("{} perished".format(opposite.name),True)
        return total_damage,killed,firstHPLeft
    def meeleAttack(self,opposite,dest,is_retaliate):
        self.do_attack(opposite,dest)
    def canShoot(self,opposite = None):
        if (not self.isShooter or self.shots <= 0):
            return False
        for enemy in self.inBattle.stacks:
            if(enemy.side != self.side and enemy.isAlive() and self.getDistance(enemy) == 1):
                return False
        return True
    def get_attacked_stacks(self,defender,stand):
        attacked = []
        if(self.attack_nearby_all):
            neibs = self.getNeibours(stand)
            for st in self.inBattle.stacks:
                if st != defender and st.isAlive() and st.side != self.side:
                    for nb in neibs:
                        if (st == nb):
                            attacked.append(st)
        elif(self.wide_breath):
            df = defender.get_position()
            df.x += (0.5 if df.y % 2 == 0 else 0)
            at = BHex(stand.x,stand.y)
            at.x += (0.5 if at.y % 2 == 0 else 0)
            other = BHex(int(df.x * 2 - at.x),int(df.y * 2 - at.y))
            for st in self.inBattle.stacks:
                if st != defender and st.isAlive():
                    if (st == other):
                        attacked.append(st)
        return attacked
    def get_position(self):
        return BHex(self.x,self.y)
    def getNeibours(self,src = None):
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
        if (0 <= y < Battle.bFieldHeight and 0 < x < Battle.bFieldWidth - 1):
            return True
        return False
    def isHalf(self,dist):
        if(self.getDistance(dist) <= Battle.bPenaltyDistance):
            return False
        return True
    def getDistance(self,dist):
        return  Battle.bGetDistance(self,dist)
    def shoot(self,opposite):
        logger.info('{} shooting {}'.format(self.name,opposite.name))
        self.do_attack(opposite,self.get_position())
        self.shots -= 1
        return
    def isAlive(self):
        return self.amount > 0
    def beShoot(self,attacker):
        return
    def wait(self):
        self.isWaited = True
    def defend(self):
        self.defense += 2
        self.isDenfenced = True
        self.isMoved = True
    def newRound(self):
        if self.isDenfenced:
            self.defense -= 2
        self.isMoved = False
        self.had_retaliated = False
        self.isWaited = False
        self.isDenfenced = False
        return
    # def legalMoves(self):
    #     if (self.isMoved):
    #         logger.info("sth wrong happen! {} is moved!!!".format(self.name))
    #         return 0
    #     #ret = {'wait': self.isWaited(), 'defend': True, 'move': [], 'melee': [], 'shoot': []}
    #     legalMoves = []
    #     if(not self.isWaited):
    #         legalMoves.append(0) #waite
    #     legalMoves.append(1) #defend
    #     aa = self.acssessableAndAttackable()
    #     for i in range(0, self.inBattle.bFieldHeight):
    #         for j in range(1, self.inBattle.bFieldWidth - 1):
    #             if (aa[i,j] >= 0 and aa[i,j] < 50 and aa[i,j] != self.speed):
    #                 #ret['move'].append(BAction(actionType.move, BHex(i,j)))
    #                 legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.move, BHex(j,i))))
    #             if (aa[i,j] == -1):
    #                 if (self.canShoot()):
    #                     #ret['shoot'].append(BAction(actionType.shoot,0,BHex(i,j)))target
    #                     legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.shoot,target=BHex(j,i))))
    #                 else:
    #                     att = BHex(j,i)
    #                     for nb in self.getNeibours(att):
    #                         if(aa[nb.y,nb.x] >= 0 and aa[nb.y,nb.x] < 50):
    #                             #ret['melee'].append(BAction(actionType.shoot,nb,BHex(i,j)))
    #                             legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.attack,nb,att)))
    #     return legalMoves



    def potential_target(self):
        bf = self.acssessableAndAttackable(exclude_me=False) #
        attackable = []
        unreach = []
        for sts in self.inBattle.stacks:
            if bf[sts.y,sts.x] == 201:
                if(self.canShoot()):
                    attackable.append((sts,0))
                else:
                    for nb in sts.getNeibours():
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
            if attacker.isShooter and not attacker.canShoot(defender):
                N_attack = 1
        for i in range(N_attack):
            if(not attacker.isAlive()):
                break
            damage1,killed1,firstHPLeft1 = attacker.computeCasualty(defender,stand,False,estimate)
            damage_dealt += damage1
            attacker.isMoved = True
            if (not defender.isAlive()):
                break
            if(not attacker.canShoot() and defender.isAlive() and not (defender.had_retaliated or attacker.blockRetaliate)):
                damage2, killed2, firstHPLeft2 = defender.computeCasualty(attacker,defender.get_position(),True,estimate)
                damage_get += damage2
                if(not defender.infinite_retaliate):
                    defender.had_retaliated = True
        return damage_dealt,damage_get
    def go_toward(self,target):
        df = self.acssessableAndAttackable()
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
            return BAction(actionType.defend)
        else:
            return BAction(actionType.move,dest=dest)

    def active_stack(self):
        if self.inBattle.agent:
            agent = self.inBattle.agent
            # st = time.time()
            ins = torch.rand((1, 1024), device='cpu')
            ins = ins.to(agent.device)
            # ins.to(agent.device)
            # ins = ins.to(agent.device)
            result = agent(ins, self.inBattle)
            # print(time.time() - st)
            act_id = result['act_id']
            position_id = result['position_id']
            target_id = result['target_id']
            if act_id == actionType.wait.value:
                next_act = BAction(actionType.wait)
            elif act_id == actionType.defend.value:
                next_act = BAction(actionType.defend)
            elif act_id == actionType.move.value:
                next_act = BAction(actionType.move,dest=BHex(int(position_id / Battle.bFieldWidth),position_id % Battle.bFieldWidth))
            elif act_id == actionType.attack.value:
                t = self.inBattle.defender_stacks[target_id] if self.side == 0 else self.inBattle.attacker_stacks[target_id]
                if self.canShoot():
                    next_act = BAction(actionType.shoot, dest=BHex(int(position_id / Battle.bFieldWidth),position_id % Battle.bFieldWidth), target=t)
                else:
                    next_act = BAction(actionType.attack,dest=BHex(int(position_id / Battle.bFieldWidth),position_id % Battle.bFieldWidth),target=t)
            else:
                logger.info("not implemented action!!",True)
            return next_act
        else:
            attackable, unreach = self.potential_target()
            if not self.inBattle.debug:
                if(self.inBattle.round == 0 and not self.isWaited):
                    return BAction(actionType.wait)
            if(len(attackable) > 0):
                att = [self.do_attack(target, stand,estimate=True) for target, stand in attackable]
                dmgs = [delt - get for delt, get in att]
                best = np.argmax(dmgs)
                best_target = attackable[best]
                if(self.canShoot()):
                    return BAction(actionType.shoot,target=best_target[0])
                else:
                    return BAction(actionType.attack,target=best_target[0],dest=best_target[1])
            else:
                distants = [self.getDistance(x) for x in unreach]
                closest = np.argmin(distants)
                return self.go_toward(unreach[closest])


class BObstacle(object):
    def __init__(self,kind = 0):
        self.kind = kind
        self.x = 0
        self.y = 0
        self.hexType = hexType.obstacle
class BObstacleInfo:
    def __init__(self,pos,w,h,isabs,imname):
        self.y = int(pos / Battle.bFieldWidth)
        self.x = pos % Battle.bFieldWidth
        self.width = w
        self.height = h
        self.isabs = isabs
        self.imname = imname

batId = 0

class Battle(object):
    bFieldWidth = 17
    bFieldHeight = 11
    bFieldStackProps = 18
    bFieldStackPlanes = 46
    bPenaltyDistance = 10
    bFieldSize = (bFieldWidth - 2)* bFieldHeight
    bTotalFieldSize = 2 + 8*bFieldSize
    def __init__(self,gui = None , load_file = None,agent = None,debug = False):
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
        self.curStack = 0
        self.last_stack = None
        self.batId = 0
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
            newSt.inBattle = newBat
            return newSt
        cp.stacks = [copyStack(st,cp) for st in self.stacks]
        cp.defender_stacks = list(filter(lambda x: x.side == 1, cp.stacks))
        cp.attacker_stacks = list(filter(lambda x: x.side == 0, cp.stacks))
        cp.sortStack()
        return cp
    def loadFile(self,file,shuffle_postion = True):
        #[0 fly,1 shooter,2 block_retaliate,3 attack_all,4 wide_breath,5 infinite_retaliate]
        creature_ability = {1:[0,0,0,0,0,0,0],3:[0,1,0,0,0,0,1],5:[1,0,0,0,0,1,0],7:[0,0,0,0,0,0,1],19:[0,1,0,0,0,0,1],
                            50:[0,0,0,0,0,0,0],51:[0,0,0,0,0,0,0],52:[1,0,0,0,0,0,0],119:[1,0,1,0,0,0,0],
                            121:[0,0,1,1,0,0,0],125:[0,0,0,0,0,0,0],131:[1,0,0,0,1,0,0],}
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
                    st.amountBase = num
                    st.health = x['health']
                    st.firstHPLeft = x['health']
                    st.id = id
                    st.side = i
                    #st.isWide = x['isWide']
                    st.luck = x['luck']
                    st.morale = x['morale']
                    st.maxDamage = x['maxDamage']
                    st.minDamage = x['minDamage']
                    st.name = x['name']
                    st.isFly = creature_ability[id][0]
                    st.isShooter = creature_ability[id][1]
                    st.blockRetaliate = creature_ability[id][2]
                    st.attack_nearby_all = creature_ability[id][3]
                    st.wide_breath = creature_ability[id][4]
                    st.infinite_retaliate = creature_ability[id][5]
                    st.attack_twice = creature_ability[id][6]
                    st.speed = x['speed']
                    #st.slotId = x['slot']
                    if shuffle_postion:
                        st.x = 15 if i else 1
                        st.y = int(ys[j])
                    else:
                        st.x = px
                        st.y = py
                    j += 1
                    st.shots = 16
                    st.inBattle = self
                    self.stacks.append(st)
                    if st.side:
                        self.defender_stacks.append(st)
                    else:
                        self.attacker_stacks.append(st)
            for x in root['obstacles']:
                obs = BObstacle()
                obs.kind = x['type']
                obs.x = x['x']
                obs.y = x['y']
                self.obstacles.append(obs)
            for x in root['obs']:
                oi = BObstacleInfo(x["pos"],x["width"],x["height"],bool(x["isabs"]),x["imname"])
                self.obsinfo.append(oi)

    def canReach(self,bFrom,bTo,bAtt = None):
        curSt = bFrom
        if(not curSt.checkPosition(bTo.x,bTo.y)):
            logger.info('dist {},{} not valid'.format(bTo.x,bTo.y))
            return False
        bf = curSt.acssessableAndAttackable(exclude_me=False)
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
        src.isMoved = True

    def sortStack(self):
        self.last_stack = self.curStack
        self.toMove = list(filter(lambda elem:elem.isAlive() and not elem.isMoved and not elem.isWaited,self.stacks))
        self.waited = list(filter(lambda elem:elem.isAlive() and not elem.isMoved and elem.isWaited,self.stacks))
        self.moved = list(filter(lambda elem:elem.isAlive() and elem.isMoved,self.stacks))

        self.toMove.sort(key=lambda elem:(-elem.speed,elem.y,elem.x))
        self.waited.sort(key=lambda elem: (elem.speed, elem.y, elem.x))
        self.moved.sort(key=lambda elem: (-elem.speed, elem.y, elem.x))
        self.stackQueue = self.toMove + self.waited + self.moved
        self.curStack = self.stackQueue[0]
    def currentPlayer(self):
        return 1 if self.stackQueue[0].side else 0

    def currentState(self):
        pass
    def currentStateFeature(self):
        # for st in self.stackQueue:
        #     bf = st.acssessableAndAttackable()
        #     reachable = (bf >= 0) & (bf < 50)
        #     me = bf ==
        pass
    def getStackHPBySlots(self):
        leftBase = [0]*7
        rightBase = [0]*7
        left = [0]*7
        right = [0]*7
        for st in self.stacks:
            if(st.side == 0):
                leftBase[st.slotId] = st.amountBase * st.health
                left[st.slotId] = ((st.amount - 1) * st.health + st.firstHPLeft) if st.amount > 0 else 0
            else:
                rightBase[st.slotId] = st.amountBase * st.health
                right[st.slotId] = ((st.amount - 1) * st.health + st.firstHPLeft) if st.amount > 0 else 0
        return left,leftBase,right,rightBase

    def findStack(self,dist,alive=True):
        ret = list(filter(lambda elem:elem.x == dist.x and elem.y == dist.y and elem.isAlive() == alive,self.stacks))
        return ret
    def directionToHex(self,mySelf,dirct):
        if(dirct < 0 or dirct > 5):
            logger.info('wrong direction {}'.format(dirct))
            return 0
        if(dirct == 0):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.x,mySelf.y - 1)
            else:
                return BHex(mySelf.x - 1,mySelf.y - 1)
        if(dirct == 1):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.x + 1,mySelf.y - 1)
            else:
                return BHex(mySelf.x,mySelf.y - 1)
        if(dirct == 2):
            return BHex(mySelf.x + 1,mySelf.y)
        if(dirct == 5):
            return BHex(mySelf.x - 1,mySelf.y)
        if(dirct == 4):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.x,mySelf.y + 1)
            else:
                return BHex(mySelf.x - 1,mySelf.y + 1)
        if (dirct == 3):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.x + 1,mySelf.y + 1)
            else:
                return BHex(mySelf.x,mySelf.y + 1)
    def hexToDirection(self,mySelf,hex):
        if(hex.y == mySelf.y - 1):
            if(mySelf.y % 2 == 0): #top left right
                if(hex.x == mySelf.x):
                    return 0
                elif(hex.x == mySelf.x + 1):
                    return 1
                else:
                    return -1
            else:
                if (hex.x == mySelf.x - 1):
                    return 0
                elif (hex.x == mySelf.x):
                    return 1
                else:
                    return -1
        elif(hex.y == mySelf.y): #left right
            if (hex.x == mySelf.x - 1):
                return 5
            elif(hex.x == mySelf.x + 1):
                return 2
            else:
                return -1
        elif(hex.y == mySelf.y + 1): #bottom left right
            if (mySelf.y % 2 == 0):
                if (hex.x == mySelf.x):
                    return 4
                elif (hex.x == mySelf.x + 1):
                    return 3
                else:
                    return -1
            else:
                if (hex.x == mySelf.x - 1):
                    return 4
                elif (hex.x == mySelf.x):
                    return 3
                else:
                    return -1
        else:
            return -1
    def actionToIndex(self,action):
        pass
    def indexToAction(self,move):
        pass
    def action2Str(self,act):
        #act = self.indexToAction(act)
        if(act.type == actionType.wait):
            return "wait"
        if (act.type == actionType.defend):
            return "defend"
        if (act.type == actionType.move):
            return "move to ({},{})".format(act.dest.x,act.dest.y)
        if (act.type == actionType.attack):
            return "melee ({},{}),({},{})".format(act.dest.x,act.dest.y,act.target.x,act.target.y)
        if (act.type == actionType.shoot):
            return "shoot ({},{})".format(act.target.x,act.target.y)
    def end(self):
        live = {0:False,1:False}
        for st in self.stacks:
            live[st.side] = live[st.side] or st.isAlive()
        if self.round > 20:
            return True,-1
        return not (live[0] and live[1]),self.currentPlayer()

    def getHash(self):
        pass
    def check_battle_end(self):
        att_alive = False
        def_alive = False
        for x in self.attacker_stacks:
            if x.isAlive():
                att_alive = True
                break
        for x in self.defender_stacks:
            if x.isAlive():
                def_alive = True
                break
        if not att_alive or not def_alive:
            return True
        return False
    def checkNewRound(self,is_self_play = 0):
        if self.check_battle_end():
            logger.info("battle end~")
            return
        self.sortStack()
        if(self.stackQueue[0].isMoved):
            self.newRound()
            self.sortStack()
        if(not is_self_play):
            side = "" if self.curStack.side else "me "
            logger.info("now it's {}{} turn".format(side,self.curStack.name),True)

    def newRound(self):
        self.round += 1
        logger.info("now it's round {}".format(self.round))
        for st in self.stacks:
            if(st.isAlive()):
                st.newRound()
    def doAction(self,action):
        logger.log_text.clear()
        logger.info(self.action2Str(action),True)
        if(self.curStack.isMoved):
            logger.info("{} is already moved".format(self.curStack))
            return
        if(action.type == actionType.wait):
            if (self.curStack.isWaited):
                logger.info("{} is already waited".format(self.curStack))
                return
            self.curStack.wait()
        elif(action.type == actionType.defend):
            self.curStack.defend()
        elif(action.type == actionType.move):
            if (self.curStack.x == action.dest.x and self.curStack.y == action.dest.y):
                logger.info("can't move to where you already are!!")
                return
            if (self.canReach(self.curStack, action.dest)):
                self.move(self.curStack, action.dest)
            else:
                logger.info("you can't reach ({},{})".format(action.dest.y,action.dest.x))
        elif(action.type == actionType.attack):
            dests = self.findStack(action.target,True)
            if(len(dests) == 0):
                logger.info("wrong attack dist ({},{})".format(action.target.y,action.target.x))
                return
            dest = dests[0]
            if (self.canReach(self.curStack, action.dest,action.target)):
                self.move(self.curStack, action.dest)
                self.curStack.meeleAttack(dest,action.dest,False)
            else:
                logger.info("you can't reach ({},{}) and attack {}".format(action.dest.y,action.dest.x,action.target.name))
        elif(action.type == actionType.shoot):
            dists = self.findStack(action.target, True)
            if (len(dists) == 0):
                logger.info("wrong shoot dist ({},{})".format(action.target.y,action.target.x))
                return
            dist = dists[0]
            if (self.curStack.canShoot(action.target)):
                self.curStack.shoot(dist)
            else:
                logger.info("{} can't shoot {}".format(self.curStack.name,dist.name))
        elif (action.type == actionType.spell):
            logger.info("spell not implemented yet")
    def legal_act(self,level=0,act_id=0,spell_id=0,target_id=0):
        cur_stack = self.curStack
        if cur_stack.isMoved:
            return None

        if level ==0:
            legals = np.zeros((5,))
            if not cur_stack.isWaited:
                legals[actionType.wait.value] = 1
            if not cur_stack.isDenfenced:
                legals[actionType.defend.value] = 1
            if cur_stack.acssessableAndAttackable(query_type=action_query_type.can_move):
                legals[actionType.move.value] = 1
            if cur_stack.acssessableAndAttackable(query_type=action_query_type.can_attack):
                legals[actionType.attack.value] = 1
            return legals
        elif level == 1:
            if act_id == actionType.move.value:
                bf = cur_stack.acssessableAndAttackable().flatten()
                mask = (bf >= 0) & (bf < 50)
                return mask.flatten()
            elif act_id == actionType.attack.value:
                bf = cur_stack.acssessableAndAttackable()
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
            if act_id == actionType.attack.value:
                bf = cur_stack.acssessableAndAttackable(exclude_me=False)
                target = self.defender_stacks[target_id] if cur_stack.side == 0 else self.attacker_stacks[target_id]
                nb = target.getNeibours()
                for t in nb:
                    if 0 <= bf[t.y, t.x] < 50:
                        mask[t.y, t.x] = 1
                return mask.flatten()
            else:
                return mask.flatten()
    def start_self_play(self,player,take_control=0,is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        #best_policy = policy_value_net_tensorflow.PolicyValueNet(self.bFieldWidth - 2,self.bFieldHeight,self.bFieldStackPlanes,self.bTotalFieldSize)
        # if take_control:
        #     trainer = BPlayer()
        # else:
        #     trainer = MCTSPlayer(best_policy.policy_value_fn,c_puct=5,n_playout=100,is_selfplay=1,side=self.currentPlayer())

        states, mcts_probs, current_players,left_bases,right_bases,lefts,rights = [], [], [],[],[],[],[]
        while (not self.end()[0]):
            if take_control:
                printF(self.curStack.acssessableAndAttackable(), self.stacks, self.curStack)
            actInd,move_probs = player.getAction(self)  #temp=temp,return_prob=1
            if not take_control:
                printF(self.curStack.acssessableAndAttackable(), self.stacks, self.curStack)
            logger.info("-------final action: {}, move_p={} by {}".format(self.action2Str(actInd),move_probs[actInd],self.curStack.name))
            legals = self.curStack.legalMoves()
            if (actInd not in legals):
                logger.info('...sth  wrong.....actInd not in legals')
            else:
                # store the data
                states.append(self.currentStateFeature())
                mcts_probs.append(move_probs)
                current_players.append([-1.0] if self.currentPlayer() else [1.0])
                left, leftBase, right, rightBase = self.getStackHPBySlots()
                left_bases.append(left)
                right_bases.append(right)
            act = self.indexToAction(actInd)
            self.doAction(act)
            self.checkNewRound()
            # update the root node and reuse the search tree
            if not take_control:
                player.mcts.update_with_move(actInd, self)
        winner = self.currentPlayer()
        if winner != -1:
            left, leftBase, right, rightBase = self.getStackHPBySlots()
            [lefts.append(left) for x in range(len(left_bases))]
            [rights.append(right) for x in range(len(right_bases))]
        # reset MCTS root node
        if not take_control:
            player.reset_player(self)
        logger.info("final Game end. Winner is player: {}".format(winner))
        return zip(states, mcts_probs, current_players,lefts,left_bases,rights,right_bases)



class  BPlayer(object):
    def getAction(self,battle):
        return  battle.curStack.active_stack()
class BAction:
    def __init__(self,type = 0,dest = 0,target = 0,spell=0):
        self.type = type
        self.dest = dest
        self.target = target
        self.spell = spell
def main():
    pass
if __name__ == '__main__':
    main()

