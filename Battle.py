import random
import json
import copy
import numpy as np
from enum import Enum
import logging
import sys
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train')
handler = logging.FileHandler('train.log','w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
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

    bf[curSt.x][curSt.y] = 'M'
    for st in stacks:
        if(st.isAlive()):
            bf[st.x][st.y] = bf[st.x][st.y]+str(st.amount)
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
    def __init__(self,x = 0,y = 0):
        self.y = y
        self.x = x

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
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
        self.isRetaliated = False
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
        self.y = 0
        self.x = 0
        self.isWide = False
        self.isFly = False
        self.isShooter = False
        self.blockRetaliate = False
        self.attack_nearby_all = False
        self.wide_breath = False
        self.amountBase = 0
        self.inBattle = 0  #Battle()
    def __eq__(self, other):
        return self.id == other.id
    def acssessableAndAttackable(self):
        ini = lambda x: -9 if x > 0 and x < self.inBattle.bFieldWidth - 1 else 100 # side colunm
        bf = [[ini(col) for col in range(self.inBattle.bFieldWidth)] for row in range(self.inBattle.bFieldHeight)]
        for sts in self.inBattle.stacks:
            if(sts.isAlive()):
                bf[sts.x][sts.y] = 400 if sts.side == self.side else 200
        for obs in self.inBattle.obstacles:
            bf[obs.x][obs.y] = 800
        #init battleField end
        # accessable  begin
        travellers = []
        bf[self.x][self.y] = self.speed
        travellers.append(self)
        if(not self.isFly):
            while(len(travellers) > 0):
                current = travellers.pop()
                speedLeft = bf[current.x][current.y] - 1
                for adj in self.getNeibours(current):
                    if(bf[adj.x][adj.y] < speedLeft):
                        bf[adj.x][adj.y] = speedLeft
                        if (speedLeft > 0):
                            travellers.append(adj)
        else: #fly
            for ii in range(self.inBattle.bFieldHeight):
                for jj in range(1,self.inBattle.bFieldWidth-1):
                    d = self.getDistance(BHex(ii,jj))
                    if(d <= self.speed):
                        bf[ii][jj] = self.speed - d
        #accessable  end
        #attackable begin
        for obs in self.inBattle.obstacles:  #obstacle,enemy and attackable,left
            bf[obs.x][obs.y] = -8
        for sts in self.inBattle.stacks:
            if(not sts.isAlive()):
                continue
            if sts.side == self.side:
                bf[sts.x][sts.y] = -4
            else:
                if (self.canShoot()):
                    bf[sts.x][sts.y] = -1  # enemy and attackbale
                else:
                    bf[sts.x][sts.y] = -2  # enemy
                    for neib in self.getNeibours(sts):
                        if (bf[neib.x][neib.y] >= 0 and bf[neib.x][neib.y] < 50):
                            bf[sts.x][sts.y] = -1  # attackbale
                            break
        bf[self.x][self.y] = self.speed
        return bf
    def damaged(self,damage,estimate = False):
        hpInAll = self.health * (self.amount - 1) + self.firstHPLeft
        if (damage >= hpInAll):
            damage = hpInAll
            killed = self.amount
            firstHPLeft = 0
        else:
            rest = int((hpInAll - damage - 1) / self.health) + 1
            firstHPLeft = (hpInAll - damage - 1) % self.health + 1
            killed = self.amount - rest
        if(not estimate):
            self.amount -= killed
            self.firstHPLeft = firstHPLeft
        return damage,killed,firstHPLeft
    def computeCasualty(self,opposite,stand,estimate=False):
        total_damage = 0
        if(self.attack >= opposite.attack):
            damageMin = int(self.minDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount)
            damageMax = int(self.maxDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount)
        else:
            damageMin = int(self.minDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
            damageMax = int(self.maxDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
        damage = int((damageMin+damageMax)/2) if estimate else random.randint(damageMin,damageMax)
        if(self.isShooter):
            if(not self.canShoot() or self.isHalf() ):
                damage = int(damage/2)
        else:
            others = self.get_attacked_stacks(opposite,stand)
            for st in others:
                real_damage, killed, firstHPLeft = st.damaged(damage, estimate)
                if (not estimate):
                    logger.info("make {} damage killed {} {},firstHPLeft {}".format(real_damage, killed, opposite.name, firstHPLeft))
                if(self.side == st.side):
                    total_damage += -real_damage
                else:
                    total_damage += real_damage
        real_damage, killed, firstHPLeft = opposite.damaged(damage,estimate)
        total_damage += real_damage
        if(not estimate):
            logger.info("make {} damage killed {} {},firstHPLeft {}".format(real_damage,killed,opposite.name,firstHPLeft))
        return total_damage,killed,firstHPLeft
    def meeleAttack(self,opposite,retaliate):
        logger.info('{} meele attacking {}'.format(self.name, opposite.name))
        damage,killed,firstHPLeft = self.computeCasualty(opposite)
        self.isMoved = True
        if(opposite.amount == 0):
            logger.info("{} perished".format(opposite.name))
        else:
            opposite.attacked(self,retaliate)
    def attacked(self,attacker,retaliate):
        if(not retaliate):
            if(attacker.blockRetaliate):
                logger.info("blockRetaliate")
                return
            if(not self.isRetaliated):
                logger.info("prepare retaliate")
                self.meeleAttack(attacker,True)
                self.isRetaliated = True
    def canShoot(self,opposite = 0):
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
                if st != defender:
                    for nb in neibs:
                        if (st.x == nb.x and st.y == nb.y):
                            attacked.append(st)
        elif(self.wide_breath):
            df = defender.get_position()
            df.y += (0.5 if df.x % 2 == 0 else 0)
            at = stand
            at.y += (0.5 if at.x % 2 == 0 else 0)
            other = BHex(int(df.x * 2 - at.x),int(df.y * 2 - at.y))
            for st in self.inBattle.stacks:
                if st != defender:
                    if (st.x == other.x and st.y == other.y):
                        attacked.append(st)
        return attacked
    def get_position(self):
        return BHex(self.x,self.y)
    def getNeibours(self,src = None):
        adj = []
        if(not src ):
            src = self
        self.checkAndPush(src.x,src.y - 1,  adj)
        self.checkAndPush(src.x,src.y + 1,  adj)
        self.checkAndPush(src.x - 1,(src.y - 1) if src.x%2 != 0 else (src.y + 1) ,  adj)
        self.checkAndPush(src.x - 1,src.y,  adj)
        self.checkAndPush(src.x + 1,(src.y - 1) if src.x%2 != 0 else (src.y + 1),  adj)
        self.checkAndPush(src.x + 1,src.y,  adj)
        return adj
    def checkAndPush(self,x,y,adj):
        if(self.checkPosition(x,y)):
            adj.append(BHex(x,y))
    def checkPosition(self,x,y):
        if (y > 0 and y < Battle.bFieldWidth - 1 and x >= 0 and x < Battle.bFieldHeight):
            return True
        return False
    def isHalf(self,dist):
        if(self.getDistance(dist) <= Battle.bPenaltyDistance):
            return False
        return True
    def getDistance(self,dist):
        y1,x1 = self.y,self.x
        y2,x2 = dist.y,dist.x
        y1 = int(y1+x1*0.5)
        y2 = int(y2+x2*0.5)
        yDst = y2 - y1
        xDst = x2 - x1
        if((yDst >= 0 and xDst >= 0) or (yDst < 0 and xDst < 0)):
            return max(abs(yDst),abs(xDst))
        return abs(yDst) + abs(xDst)
    def shoot(self,opposite):
        logger.info('{} shooting {}'.format(self.name,opposite.name))
        damage,killed,firstHPLeft = self.computeCasualty(opposite)
        if (opposite.amount == 0):
            logger.info("{} perished".format(opposite.name))
        self.isMoved = True
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
        self.isRetaliated = False
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
                    legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.move, BHex(i,j))))
                if (aa[i][j] == -1):
                    if (self.canShoot()):
                        #ret['shoot'].append(BAction(actionType.shoot,0,BHex(i,j)))
                        legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.shoot,attack=BHex(i,j))))
                    else:
                        att = BHex(i,j)
                        for nb in self.getNeibours(att):
                            if(aa[nb.x][nb.y] >= 0 and aa[nb.x][nb.y] < 50):
                                #ret['melee'].append(BAction(actionType.shoot,nb,BHex(i,j)))
                                legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.attack,nb,att)))
        return legalMoves
    def potential_target(self):
        bf = self.acssessableAndAttackable()
        attackable = []
        unreach = []
        for sts in self.inBattle.stacks:
            if bf[sts.x][sts.y] == -1:
                attackable.append(sts)
            elif bf[sts.x][sts.y] == -2:
                unreach.append(sts)
        return attackable,unreach

    def activeStack(self):
        attackable, unreach = self.potential_target()
        max_damage = -9999
        best_target = 0
        for target in attackable:
            delt,get = Battle.estimateDamage(target)
            damage = delt - get
            if damage > max_damage:
                max_damage = damage
                best_target = BHex(target.x,target.y)
        # if best_target:
        #     if

class BObstacle(object):
    def __init__(self,kind = 0):
        self.kind = kind
        self.y = 0
        self.x = 0
        self.hexType = hexType.obstacle
class BObstacleInfo:
    def __init__(self,pos,w,h,isabs,imname):
        self.x = int(pos / Battle.bFieldWidth)
        self.y = pos % Battle.bFieldWidth
        self.width = w
        self.height = h
        self.isabs = isabs
        self.imname = imname
def isFly(x):
    return x['ability'][0]
def isShoot(x):
    return x['ability'][1]
def blockRetaliate(x):
    return x['ability'][2]
def attack_nearby_all(x):
    return x['ability'][3]
def wide_breath(x):
    return x['ability'][4]
batId = 0
class Battle(object):
    bFieldWidth = 17
    bFieldHeight = 11
    bFieldStackProps = 18
    bFieldStackPlanes = 46
    bPenaltyDistance = 10
    bFieldSize = (bFieldWidth - 2)* bFieldHeight
    bTotalFieldSize = 2 + 8*bFieldSize
    def __init__(self,load_file = 0):
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
        self.batId = 0
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
        with open(file) as jsonFile:
            root = json.load(jsonFile)
            for x in root['stacks']:
                st = BStack()
                st.attack = x['attack']
                st.defense = x['defense']
                st.amount = x['baseAmount']
                st.amountBase = x['baseAmount']
                st.health = x['health']
                st.firstHPLeft = x['health']
                st.id = x['id']
                st.side = 0 if x['isHuman'] else 1
                st.isWide = x['isWide']
                st.luck = x['luck']
                st.morale = x['morale']
                st.maxDamage = x['maxDamage']
                st.minDamage = x['minDamage']
                st.name = x['name']
                st.isFly = isFly(x)
                st.isShooter = isShoot(x)
                st.blockRetaliate = blockRetaliate(x)
                st.attack_nearby_all = attack_nearby_all(x)
                st.wide_breath = wide_breath(x)
                st.speed = x['speed']
                st.slotId = x['slot']
                st.y = x['x']
                st.x = x['y']
                if('shots' in x):
                    st.shots = x['shots']
                st.inBattle = self
                self.stacks.append(st)

            for x in root['obstacles']:
                obs = BObstacle()
                obs.kind = x['type']
                obs.y = x['x']
                obs.x = x['y']
                self.obstacles.append(obs)
            for x in root['obs']:
                oi = BObstacleInfo(x["pos"],x["width"],x["height"],bool(x["isabs"]),x["imname"])
                self.obsinfo.append(oi)

    def canReach(self,bFrom,bTo,bAtt = 0):
        curSt = bFrom
        if (curSt != 0):
            if(not curSt.checkPosition(bTo.y,bTo.x)):
                logger.info('dist {},{} not valid'.format(bTo.x,bTo.y))
                return False
            bf = curSt.acssessableAndAttackable()
            if(bAtt != 0):
                return self.bGetDistance(bTo,bAtt) == 1 and bf[bTo.x][bTo.y] >= 0 and bf[bAtt.x][bAtt.y] == -1
            else:
                return bf[bTo.x][bTo.y] >= 0
        logger.info('src {},{} not valid'.format(bFrom.x, bFrom.y))
        return False
    def bGetDistance(self,src,dist):
        y1,x1 = src.y,src.x
        y2,x2 = dist.y,dist.x
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
        src.y = bTo.y
        src.x = bTo.x
        src.isMoved = True

    def sortStack(self):
        self.toMove = list(filter(lambda elem:elem.isAlive() and not elem.isMoved and not elem.isWaited,self.stacks))
        self.waited = list(filter(lambda elem:elem.isAlive() and not elem.isMoved and elem.isWaited,self.stacks))
        self.moved = list(filter(lambda elem:elem.isAlive() and elem.isMoved,self.stacks))

        self.toMove.sort(key=lambda elem:(-elem.speed,elem.x,elem.y))
        self.waited.sort(key=lambda elem: (elem.speed, elem.x, elem.y))
        self.moved.sort(key=lambda elem: (-elem.speed, elem.x, elem.y))
        self.stackQueue = self.toMove + self.waited + self.moved
        self.curStack = self.stackQueue[0]
    def currentPlayer(self):
        return 1 if self.stackQueue[0].side else 0

    def currentState(self):
        #state = [[[0 for k in range(self.bFieldStackProps)] for j in range(self.bFieldWidth - 2)] for i in range(self.bFieldHeight)]
        state = np.zeros((self.bFieldHeight, self.bFieldWidth - 2, self.bFieldStackProps), dtype=int)
        planeStruct = [[x for x in range(1,self.bFieldStackProps)],0]
        def fillStack(stack):
            #stack = BStack()
            state[stack.x][stack.y - 1][planeStruct[0]] = [stack.side,stack.id,stack.amount,stack.attack,stack.defense,stack.maxDamage,stack.minDamage,stack.health,int(stack.isMoved),int(stack.isRetaliated),int(stack.isWaited),stack.speed,stack.luck,stack.morale,stack.shots,stack.isFly,stack.isShooter] #,stack.firstHPLeft
        def fillObs(ob):
            state[ob.x][ob.y - 1][planeStruct[1]] = 1
        [fillStack(st) for st in self.stacks]
        [fillObs(ob) for ob in self.obstacles]
        return state
    def currentStateFeature(self):
        state = np.zeros((self.bFieldHeight, self.bFieldWidth - 2, self.bFieldStackPlanes), dtype=int)
        planeStruct = [[x for x in range(21)], [x for x in range(21,42)],42,43,44,45]
        def fillStack(stack):
            #stack = BStack()
            idList = [0, 0, 0, 0, 0]
            idList[stack.id] = 1
            state[stack.x][stack.y - 1][planeStruct[stack.side]] = [stack.amount,stack.attack,stack.defense,stack.maxDamage,stack.minDamage,stack.firstHPLeft,stack.health,stack.isMoved,stack.isRetaliated,stack.isWaited,stack.speed,stack.luck,stack.morale,stack.shots,stack.isFly,stack.isShooter] +idList
        def fillObs(ob):
            state[ob.x][ob.y - 1][planeStruct[2]] = 1
        def fillMe(me):
            state[me.x][me.y - 1][planeStruct[3]] = 1
        def fillReachable(r):
            state[r.x][r.y - 1][planeStruct[4]] = 1
        def fillAttackable(r):
            state[r.x][r.y - 1][planeStruct[5]] = 1
        [fillStack(st) for st in self.stacks]
        [fillObs(st) for st in self.obstacles]
        fillMe(self.curStack)
        aa = self.curStack.acssessableAndAttackable()
        for i in range(self.bFieldHeight):
            for j in range(self.bFieldWidth):
                if(aa[i][j] == -1):
                    fillAttackable(BHex(i,j))
                elif(aa[i][j]>=0 and aa[i][j] < 50):
                    fillReachable(BHex(i,j))
        return state
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
        ret = list(filter(lambda elem:elem.y == dist.y and elem.x == dist.x and elem.isAlive() == alive,self.stacks))
        return ret
    def directionToHex(self,mySelf,dirct):
        if(dirct < 0 or dirct > 5):
            logger.info('wrong direction {}'.format(dirct))
            return 0
        if(dirct == 0):
            if (mySelf.x % 2 == 0):
                return BHex(mySelf.x - 1,mySelf.y)
            else:
                return BHex(mySelf.x - 1, mySelf.y - 1)
        if(dirct == 1):
            if (mySelf.x % 2 == 0):
                return BHex(mySelf.x - 1,mySelf.y + 1)
            else:
                return BHex(mySelf.x - 1, mySelf.y)
        if(dirct == 2):
            return BHex(mySelf.x, mySelf.y + 1)
        if(dirct == 5):
            return BHex(mySelf.x, mySelf.y - 1)
        if(dirct == 4):
            if (mySelf.x % 2 == 0):
                return BHex(mySelf.x + 1,mySelf.y)
            else:
                return BHex(mySelf.x + 1, mySelf.y - 1)
        if (dirct == 3):
            if (mySelf.x % 2 == 0):
                return BHex(mySelf.x + 1, mySelf.y + 1)
            else:
                return BHex(mySelf.x + 1, mySelf.y)
    def hexToDirection(self,mySelf,hex):
        if(hex.x == mySelf.x - 1):
            if(mySelf.x % 2 == 0): #top left right
                if(hex.y == mySelf.y):
                    return 0
                elif(hex.y == mySelf.y + 1):
                    return 1
                else:
                    return -1
            else:
                if (hex.y == mySelf.y - 1):
                    return 0
                elif (hex.y == mySelf.y):
                    return 1
                else:
                    return -1
        elif(hex.x == mySelf.x): #left right
            if (hex.y == mySelf.y - 1):
                return 5
            elif(hex.y == mySelf.y + 1):
                return 2
            else:
                return -1
        elif(hex.x == mySelf.x + 1): #bottom left right
            if (mySelf.x % 2 == 0):
                if (hex.y == mySelf.y):
                    return 4
                elif (hex.y == mySelf.y + 1):
                    return 3
                else:
                    return -1
            else:
                if (hex.y == mySelf.y - 1):
                    return 4
                elif (hex.y == mySelf.y):
                    return 3
                else:
                    return -1
        else:
            return -1
    def actionToIndex(self,action):
        if(actionType.wait == action.type):
            return 0
        if (actionType.defend == action.type):
            return 1
        if (actionType.move == action.type):
            i = action.move.x
            j = action.move.y
            return 2 + i* (self.bFieldWidth - 2) + j - 1
        if (actionType.attack == action.type):
            enemy = action.attack
            direct = self.hexToDirection(action.attack,action.move)
            i = enemy.x
            j = enemy.y
            return 2 + self.bFieldSize +(i * (self.bFieldWidth - 2) + (j - 1))*6+ direct
        if (actionType.shoot == action.type):
            enemy = action.attack
            i = enemy.x
            j = enemy.y
            return 2 + (7)*self.bFieldSize+(i * (self.bFieldWidth - 2) + j - 1)
        logger.info('actionToIndex wrong action {}'.format(action))
    def indexToAction(self,move):
        if(move < 0):
            logger.info('wrong move {}'.format(move))
            return 0
        if(move == 0):
            return BAction(actionType.wait)
        elif(move == 1):
            return BAction(actionType.defend)
        elif((move - 2) >= 0 and (move - 2) < self.bFieldSize):
            y = int((move - 2)/(self.bFieldWidth - 2))
            x = (move - 2)%(self.bFieldWidth - 2)
            return BAction(actionType.move,BHex(y,x + 1))
        elif((move - 2 -self.bFieldSize) >= 0 and (move - 2 -self.bFieldSize) < 6 * self.bFieldSize):
            direction = (move - 2 -self.bFieldSize) % 6
            enemy = int((move - 2 -self.bFieldSize) / 6)
            y = int(enemy /(self.bFieldWidth - 2))
            x = enemy % (self.bFieldWidth - 2)
            enemy = BHex(y,x + 1)
            stand = self.directionToHex(enemy,direction)
            return BAction(actionType.attack, stand,enemy)
        elif((move - 2 - 7*self.bFieldSize) >= 0 and (move - 2 - 7*self.bFieldSize) < self.bFieldSize):
            enemy = (move - 2 - 7*self.bFieldSize)
            y = int(enemy / (self.bFieldWidth - 2))
            x = enemy % (self.bFieldWidth - 2)
            return BAction(actionType.shoot,attack=BHex(y,x+1))
        else:
            logger.info("wrong move {}".format(move))
    def action2Str(self,act):
        act = self.indexToAction(act)
        if(act.type == actionType.wait):
            return "wait"
        if (act.type == actionType.defend):
            return "defend"
        if (act.type == actionType.move):
            return "move to ({},{})".format(act.move.x,act.move.y)
        if (act.type == actionType.attack):
            return "melee ({},{}),({},{})".format(act.move.x,act.move.y,act.attack.x,act.attack.y)
        if (act.type == actionType.shoot):
            return "shoot ({},{})".format(act.attack.x,act.attack.y)
    def end(self):
        live = {0:False,1:False}
        for st in self.stacks:
            live[st.side] = live[st.side] or st.isAlive()
        if self.round > 20:
            return True,-1
        return not (live[0] and live[1]),self.currentPlayer()

    def getHash(self):
        state = self.currentState()
        a = ''
        for i in range(self.bFieldHeight):
            for j in range(self.bFieldWidth - 2):
                for k in range(self.bFieldStackProps):
                    a += str(state[i][j][k])
        return hash(a)
    def checkNewRound(self,is_self_play = 0):
        self.sortStack()
        if(self.stackQueue[0].isMoved):
            self.newRound()
            self.sortStack()
        if(not is_self_play):
            logger.info("now it's {} turn".format(self.curStack.name))

    def newRound(self):
        self.round += 1
        logger.info("now it's round {}".format(self.round))
        for st in self.stacks:
            if(st.isAlive()):
                st.newRound()
    def doAction(self,action):
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
            if (self.curStack.y == action.move.y and self.curStack.x == action.move.x):
                logger.info("can't move to where you already are!!")
                return
            if (self.canReach(self.curStack, action.move)):
                self.move(self.curStack, action.move)
            else:
                logger.info("you can't reach {},{}".format(action.move.x, action.move.y))
        elif(action.type == actionType.attack):
            dists = self.findStack(action.attack,True)
            if(len(dists) == 0):
                logger.info("wrong attack dist {} {}".format(action.attack.x,action.attack.y))
                return
            dist = dists[0]
            if (self.canReach(self.curStack, action.move,action.attack)):
                self.move(self.curStack, action.move)
                self.curStack.meeleAttack(dist,False)
            else:
                logger.info("you can't reach {} and attack {}".format(action.move,action.attack))
        elif(action.type == actionType.shoot):
            dists = self.findStack(action.attack, True)
            if (len(dists) == 0):
                logger.info("wrong shoot dist {} {}".format(action.attack.x, action.attack.y))
                return
            dist = dists[0]
            if (self.curStack.canShoot(action.attack)):
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

    def estimateDamage(self,attacker_orig,defender_orig):
        damage_get = 0
        defender  = copy.copy(defender_orig)
        damage1,killed1,firstHPLeft1 = attacker_orig.computeCasualty(defender,estimate=True)
        defender.amount -= killed1
        defender.firstHPLeft = firstHPLeft1
        if(defender.amount > 0 and not (defender.isRetaliated or attacker_orig.blockRetaliate)):
            #attacker = copy.copy(attacker_orig)
            damage2, killed2, firstHPLeft2 = defender.computeCasualty(attacker_orig,estimate=True)
            damage_get = damage2
        damage_dealt = damage1
        return damage_dealt,damage_get

class  BPlayer(object):
    def getAction(self,battle):
        action = BAction()
        try:
            act = input("input: ")
            if isinstance(act,str):
                ii = act.split(',')
                acts = [int(a) for a in ii]
                action.type = actionType(acts[0])
                if(action.type == actionType.wait or action.type == actionType.defend):
                    pass
                elif(action.type == actionType.move):
                    action.move = BHex(acts[1],acts[2])
                elif(action.type == actionType.attack):
                    action.move = BHex(acts[1],acts[2])
                    action.attack = BHex(acts[3],acts[4])
                elif((action.type == actionType.shoot)):
                    action.attack = BHex(acts[1],acts[2])
                else:
                    logger.info("action not implementedd yet")
        except Exception as e:
            action = 0
            logger.info("Exception:  ",e)
        actIndex = battle.actionToIndex(action)
        legals = battle.curStack.acssessableAndAttackable()
        all_moves = np.zeros((battle.bTotalFieldSize),dtype=float)
        for i in range(battle.bTotalFieldSize):
            if i == actIndex:
                all_moves[i] = 0.6
            elif i in legals:
                all_moves[i] = 0.4/len(legals) - 1
        return actIndex,all_moves
class SimpleAI(BPlayer):
    def getAction(self,battle):
        pass
class BAction:
    def __init__(self,type = 0,move = 0,attack = 0,spell=0):
        self.type = type
        self.move = move
        self.attack = attack
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
        legals = battle.curStack.legalMoves()
        myMove = battle.actionToIndex(act)
        if(myMove not in legals):
            logger.info('...sth  wrong.....')
        battle.doAction(act)
if __name__ == '__main__':
    main()


