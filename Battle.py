import random
import json
import copy
import numpy as np
from enum import Enum
import logging
import sys
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
std_logger = logging.getLogger('train')
handler = logging.FileHandler('train.log','w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
std_logger.addHandler(handler)
from BattleInterface import log_with_gui
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

    bf[curSt.y][curSt.x] = 'M'
    for st in stacks:
        if(st.isAlive()):
            bf[st.y][st.x] = bf[st.y][st.x]+str(st.amount)
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
class hexType(Enum):
    creature = 0
    obstacle = 1
class BHex:
    mapp = {100:'|',-9:'   ',-8:' * ',-4:'M',-2:'E',-1:'A',51:'   '}
    def __init__(self,x = 0,y = 0,bhex = None):
        assert isinstance(x,int) and isinstance(y,int)
        if(bhex):
            self.x = bhex.x
            self.y = bhex.y
        else:
            self.x = x
            self.y = y
    def __eq__(self, other):
        if(other):
            return self.y == other.y and self.x == other.x
        return False
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
    def acssessableAndAttackable(self):
        ini = lambda x: -9 if x > 0 and x < self.inBattle.bFieldWidth - 1 else 100 # side colunm
        bf = [[ini(col) for col in range(self.inBattle.bFieldWidth)] for row in range(self.inBattle.bFieldHeight)]
        for sts in self.inBattle.stacks:
            if(sts.isAlive()):
                bf[sts.y][sts.x] = 400 if sts.side == self.side else 200
        for obs in self.inBattle.obstacles:
            bf[obs.y][obs.x] = 800
        #init battleField end
        # accessable  begin
        travellers = []
        bf[self.y][self.x] = self.speed
        travellers.append(self)
        if(not self.isFly):
            while(len(travellers) > 0):
                current = travellers.pop()
                speedLeft = bf[current.y][current.x] - 1
                for adj in self.getNeibours(current):
                    if(bf[adj.y][adj.x] < speedLeft):
                        bf[adj.y][adj.x] = speedLeft
                        if (speedLeft > 0):
                            travellers.append(adj)
        else: #fly
            for ii in range(self.inBattle.bFieldHeight):
                for jj in range(1,self.inBattle.bFieldWidth-1):
                    d = self.getDistance(BHex(jj,ii))
                    if(d <= self.speed):
                        bf[ii][jj] = self.speed - d
        #accessable  end
        #attackable begin
        for obs in self.inBattle.obstacles:  #obstacle,enemy and attackable,left
            bf[obs.y][obs.x] = -8
        for sts in self.inBattle.stacks:
            if(not sts.isAlive()):
                continue
            if sts.side == self.side:
                bf[sts.y][sts.x] = -4
            else:
                if (self.canShoot()):
                    bf[sts.y][sts.x] = -1  # enemy and attackbale
                else:
                    bf[sts.y][sts.x] = -2  # enemy
                    for neib in self.getNeibours(sts):
                        if (bf[neib.y][neib.x] >= 0 and bf[neib.y][neib.x] < 50):
                            bf[sts.y][sts.x] = -1  # attackbale
                            break
        bf[self.y][self.x] = self.speed
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
                if (not estimate):
                    head = "reta" if is_reta else "make"
                    tt = "{} {} dmg killed {} {} {} left, HP {}".format(head,real_damage, killed, st.name,st.amount, firstHPLeft)
                    logger.info(tt,True)
                    if(opposite.amount == 0):
                        logger.info("{} perished".format(opposite.name),True)
                if(self.side == st.side):
                    total_damage += -real_damage
                else:
                    total_damage += real_damage
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
    # def attacked(self,attacker,is_retaliate):
    #     if(not self.isAlive()):
    #         return
    #     if(not is_retaliate):
    #         if(attacker.blockRetaliate):
    #             logger.info("blockRetaliate",True)
    #             return
    #         if(not self.had_retaliated):
    #             logger.info("{} prepare retaliate".format(self.name),True)
    #             self.meeleAttack(attacker,self.get_position(),True)
    #             if(not self.infinite_retaliate):
    #                 self.had_retaliated = True
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
                        if (st.y == nb.y and st.x == nb.x):
                            attacked.append(st)
        elif(self.wide_breath):
            df = defender.get_position()
            df.x += (0.5 if df.y % 2 == 0 else 0)
            at = BHex(bhex=stand)
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
        if (y >= 0 and y < Battle.bFieldHeight and x > 0 and x < Battle.bFieldWidth - 1):
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
    def legalMoves(self):
        if (self.isMoved):
            logger.info("sth wrong happen! {} is moved!!!".format(self.name))
            return 0
        #ret = {'wait': self.isWaited(), 'defend': True, 'move': [], 'melee': [], 'shoot': []}
        legalMoves = []
        if(not self.isWaited):
            legalMoves.append(0) #waite
        legalMoves.append(1) #defend
        aa = self.acssessableAndAttackable()
        for i in range(0, self.inBattle.bFieldHeight):
            for j in range(1, self.inBattle.bFieldWidth - 1):
                if (aa[i][j] >= 0 and aa[i][j] < 50 and aa[i][j] != self.speed):
                    #ret['move'].append(BAction(actionType.move, BHex(i,j)))
                    legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.move, BHex(j,i))))
                if (aa[i][j] == -1):
                    if (self.canShoot()):
                        #ret['shoot'].append(BAction(actionType.shoot,0,BHex(i,j)))
                        legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.shoot,attack=BHex(j,i))))
                    else:
                        att = BHex(j,i)
                        for nb in self.getNeibours(att):
                            if(aa[nb.y][nb.x] >= 0 and aa[nb.y][nb.x] < 50):
                                #ret['melee'].append(BAction(actionType.shoot,nb,BHex(i,j)))
                                legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.attack,nb,att)))
        return legalMoves
    def potential_target(self):
        bf = self.acssessableAndAttackable()
        attackable = []
        unreach = []
        for sts in self.inBattle.stacks:
            if bf[sts.y][sts.x] == -1:
                if(self.canShoot()):
                    attackable.append((sts,0))
                else:
                    for nb in sts.getNeibours():
                        if(bf[nb.y][nb.x] >=0 and bf[nb.y][nb.x] < 50):
                            attackable.append((sts, nb))
            elif bf[sts.y][sts.x] == -2:
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
        N_attack = 2 if attacker.attack_twice else 1
        for i in range(N_attack):
            damage1,killed1,firstHPLeft1 = attacker.computeCasualty(defender,stand,False,estimate)
            attacker.isMoved = True
            if(not attacker.canShoot() and defender.amount > 0 and not (defender.had_retaliated or attacker.blockRetaliate)):
                damage2, killed2, firstHPLeft2 = defender.computeCasualty(attacker,stand,True,estimate)
                if(not defender.infinite_retaliate):
                    defender.had_retaliated = True
                damage_get += damage2
            damage_dealt += damage1
        return damage_dealt,damage_get
    def go_toward(self,target):
        df = self.acssessableAndAttackable()
        min_dist = 999
        dest = 0
        for i in range(Battle.bFieldHeight):
            for j in range(Battle.bFieldWidth):
                if(df[i][j] >= 0 and df[i][j] < 50):
                    distance = Battle.bGetDistance(BHex(j,i),target)
                    if(distance < min_dist):
                        min_dist = distance
                        dest = BHex(j,i)
        if(dest == self.get_position()):
            return BAction(actionType.defend)
        else:
            return BAction(actionType.move,dest=dest)

    def active_stack(self):
        attackable, unreach = self.potential_target()
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
    def __init__(self,gui = None , load_file = 0):
        #self.bField = [[0 for col in range(battle.bFieldWidth)] for row in range(battle.bFieldHeight)]
        self.stacks = []
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
        cp.sortStack()
        return cp
    def loadFile(self,file):
        #[0 fly,1 shooter,2 block_retaliate,3 attack_all,4 wide_breath,5 infinite_retaliate]
        creature_ability = {1:[0,0,0,0,0,0,0],3:[0,1,0,0,0,0,1],5:[1,0,0,0,0,1,0],7:[0,0,0,0,0,0,1],19:[0,1,0,0,0,0,1],
                            50:[0,0,0,0,0,0,0],52:[1,0,0,0,0,0,0],119:[1,0,1,0,0,0,0],
                            121:[0,0,1,1,0,0,0],125:[0,0,0,0,0,0,0],131:[1,0,0,0,1,0,0],}
        with open(r"D:\project\VCNN\ENV\creatureData.json") as JsonFile:
            crList = json.load(JsonFile)["creatures"]
        with open(file) as jsonFile:
            root = json.load(jsonFile)
            for i in range(2):
                li = list(range(11))
                ys = np.random.choice(li,size=len(root['army{}'.format(i)]), replace=False)
                j = 0
                for id,num in root['army{}'.format(i)]:
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
                    st.x = 15 if i else 1
                    st.y = int(ys[j])
                    j += 1
                    st.shots = 16
                    st.inBattle = self
                    self.stacks.append(st)

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
        if (curSt):
            if(not curSt.checkPosition(bTo.x,bTo.y)):
                logger.info('dist {},{} not valid'.format(bTo.x,bTo.y))
                return False
            bf = curSt.acssessableAndAttackable()
            if(bAtt):
                return self.bGetDistance(bTo,bAtt) == 1 and bf[bTo.y][bTo.x] >= 0 and bf[bAtt.y][bAtt.x] == -1
            else:
                return bf[bTo.y][bTo.x] >= 0
        logger.info('src {},{} not valid'.format(bFrom.x,bFrom.y))
        return False
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
    def checkNewRound(self,is_self_play = 0):
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
                logger.info("you can't reach {},{}".format(action.dest.x,action.dest.y))
        elif(action.type == actionType.attack):
            dests = self.findStack(action.target,True)
            if(len(dests) == 0):
                logger.info("wrong attack dist {} {}".format(action.target.x,action.target.y))
                return
            dest = dests[0]
            if (self.canReach(self.curStack, action.dest,action.target)):
                self.move(self.curStack, action.dest)
                self.curStack.meeleAttack(dest,action.dest,False)
            else:
                logger.info("you can't reach {} and attack {}".format(action.dest,action.target))
        elif(action.type == actionType.shoot):
            dists = self.findStack(action.target, True)
            if (len(dists) == 0):
                logger.info("wrong shoot dist {} {}".format(action.target.x,action.target.y))
                return
            dist = dists[0]
            if (self.curStack.canShoot(action.target)):
                self.curStack.shoot(dist)
            else:
                logger.info("{} can't shoot {}".format(self.curStack.name,dist.name))
        elif (action.type == actionType.spell):
            logger.info("spell not implemented yet")

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
    pl1 = BPlayer()
    pl2 = BPlayer()
    players = [pl1, pl2]
    battle = Battle()
    battle.loadFile("D:/project/VCNN/ENV/selfplay.json")
    battle.checkNewRound()
    while(not battle.end()[0]):
        battle.checkNewRound()
        cplayer = battle.currentPlayer()
        printF(battle.curStack.acssessableAndAttackable(),battle.stacks,battle.curStack)
        act = players[cplayer].getAction(battle)
        if(act == 0):
            continue
        # legals = battle.curStack.legalMoves()
        # myMove = battle.actionToIndex(act)
        # if(myMove not in legals):
        #     logger.info('...sth  wrong.....')
        battle.doAction(act)
if __name__ == '__main__':
    main()


