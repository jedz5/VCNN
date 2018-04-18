import random
import json
import copy
import numpy as np
from enum import Enum
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train')
handler = logging.FileHandler('train.log')
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
    def __init__(self,y = 0,x = 0):
        self.x = x
        self.y = y
class BStack(object):
    def  __init__(self):
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
        self.x = 0
        self.y = 0
        self.isWide = False
        self.isFly = False
        self.isShooter = False
        self.blockRetaliate = False
        self.amountBase = 0
        self.inBattle = 0  #Battle()

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
                    d = self.getDistance(BHex(ii,jj))
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
    def computeCasualty(self,opposite,shoot=False,half=True):
        if(self.attack >= opposite.attack):
            damageMin = int(self.minDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount)
            damageMax = int(self.maxDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount)
        else:
            damageMin = int(self.minDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
            damageMax = int(self.maxDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount)
        damage = random.randint(damageMin,damageMax)
        if(self.isShooter):
            if(not shoot or half ):
                damage = int(damage/2)
        logger.info('make {} damage'.format(damage))
        hpInAll = opposite.health*(opposite.amount-1) + opposite.firstHPLeft
        if(damage >= hpInAll):
            killed = opposite.amount
            firstHPLeft = 0
        else:
            rest = int((hpInAll - damage - 1)/opposite.health) + 1
            firstHPLeft = (hpInAll - damage - 1)%opposite.health + 1
            killed = opposite.amount - rest
        logger.info("killed {} {},firstHPLeft {}".format(killed,opposite.name,firstHPLeft))
        return killed,firstHPLeft
    def meeleAttack(self,opposite,retaliate):
        logger.info('{} meele attacking {}'.format(self.name, opposite.name))
        killed,firstHPLeft = self.computeCasualty(opposite)
        opposite.amount -= killed
        opposite.firstHPLeft = firstHPLeft
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
    def getNeibours(self,src = 0):
        adj = []
        if(src == 0):
            src = self
        self.checkAndPush(src.x - 1, src.y, adj)
        self.checkAndPush(src.x + 1, src.y, adj)
        self.checkAndPush((src.x - 1) if src.y%2 != 0 else (src.x + 1) , src.y - 1, adj)
        self.checkAndPush(src.x, src.y - 1, adj)
        self.checkAndPush((src.x - 1) if src.y%2 != 0 else (src.x + 1), src.y + 1, adj)
        self.checkAndPush(src.x, src.y + 1, adj)
        return adj
    def checkAndPush(self,x,y,adj):
        if(self.checkPosition(x,y)):
            adj.append(BHex(y,x))
    def checkPosition(self,x,y):
        if (x > 0 and x < Battle.bFieldWidth - 1 and y >= 0 and y < Battle.bFieldHeight):
            return True
        return False
    def isHalf(self,dist):
        if(self.getDistance(dist) <= Battle.bPenaltyDistance):
            return False
        return True
    def getDistance(self,dist):
        x1,y1 = self.x,self.y
        x2,y2 = dist.x,dist.y
        x1 = int(x1+y1*0.5)
        x2 = int(x2+y2*0.5)
        xDst = x2 - x1
        yDst = y2 - y1
        if((xDst >= 0 and yDst >= 0) or (xDst < 0 and yDst < 0)):
            return max(abs(xDst),abs(yDst))
        return abs(xDst) + abs(yDst)
    def shoot(self,opposite):
        logger.info('{} shooting {}'.format(self.name,opposite.name))
        killed,firstHPLeft = self.computeCasualty(opposite,True,self.isHalf(opposite))
        opposite.amount -= killed
        opposite.firstHPLeft = firstHPLeft
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
                            if(aa[nb.y][nb.x] >= 0 and aa[nb.y][nb.x] < 50):
                                #ret['melee'].append(BAction(actionType.shoot,nb,BHex(i,j)))
                                legalMoves.append(self.inBattle.actionToIndex(BAction(actionType.attack,nb,att)))
        return legalMoves


class BObstacle(object):
    def __init__(self,kind = 0):
        self.kind = kind
        self.x = 0
        self.y = 0
        self.hexType = hexType.obstacle

def isFly(x):
    if "ability" in x:
        for y in x['ability']:
            if y['type'] == 43:
                return 1
    return 0

def isShoot(x):
        if "ability" in x:
            for y in x['ability']:
                if y['type'] == 44:
                    return 1
        return 0
def blockRetaliate(x):
    if "ability" in x:
        for y in x['ability']:
            if y['type'] == 68:
                return 1
    return 0
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
                st.speed = x['speed']
                st.slotId = x['slot']
                st.x = x['x']
                st.y = x['y']
                if('shots' in x):
                    st.shots = x['shots']
                st.inBattle = self
                self.stacks.append(st)

            for x in root['obstacles']:
                obs = BObstacle()
                obs.kind = x['type']
                obs.x = x['x']
                obs.y = x['y']
                self.obstacles.append(obs)
    def canReach(self,bFrom,bTo,bAtt = 0):
        curSt = bFrom
        if (curSt != 0):
            if(not curSt.checkPosition(bTo.x,bTo.y)):
                logger.info('dist {},{} not valid'.format(bTo.y,bTo.x))
                return False
            bf = curSt.acssessableAndAttackable()
            if(bAtt != 0):
                return self.bGetDistance(bTo,bAtt) == 1 and bf[bTo.y][bTo.x] >= 0 and bf[bAtt.y][bAtt.x] == -1
            else:
                return bf[bTo.y][bTo.x] >= 0
        logger.info('src {},{} not valid'.format(bFrom.y, bFrom.x))
        return False
    def bGetDistance(self,src,dist):
        x1,y1 = src.x,src.y
        x2,y2 = dist.x,dist.y
        x1 = int(x1+y1*0.5)
        x2 = int(x2+y2*0.5)
        xDst = x2 - x1
        yDst = y2 - y1
        if((xDst >= 0 and yDst >= 0) or (xDst < 0 and yDst < 0)):
            return max(abs(xDst),abs(yDst))
        return abs(xDst) + abs(yDst)
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
        #state = [[[0 for k in range(self.bFieldStackProps)] for j in range(self.bFieldWidth - 2)] for i in range(self.bFieldHeight)]
        state = np.zeros((self.bFieldHeight, self.bFieldWidth - 2, self.bFieldStackProps), dtype=int)
        planeStruct = [[x for x in range(1,self.bFieldStackProps)],0]
        def fillStack(stack):
            #stack = BStack()
            state[stack.y][stack.x - 1][planeStruct[0]] = [stack.side,stack.id,stack.amount,stack.attack,stack.defense,stack.maxDamage,stack.minDamage,stack.health,int(stack.isMoved),int(stack.isRetaliated),int(stack.isWaited),stack.speed,stack.luck,stack.morale,stack.shots,stack.isFly,stack.isShooter] #,stack.firstHPLeft
        def fillObs(ob):
            state[ob.y][ob.x - 1][planeStruct[1]] = 1
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
            state[stack.y][stack.x - 1][planeStruct[stack.side]] = [stack.amount,stack.attack,stack.defense,stack.maxDamage,stack.minDamage,stack.firstHPLeft,stack.health,stack.isMoved,stack.isRetaliated,stack.isWaited,stack.speed,stack.luck,stack.morale,stack.shots,stack.isFly,stack.isShooter] +idList
        def fillObs(ob):
            state[ob.y][ob.x - 1][planeStruct[2]] = 1
        def fillMe(me):
            state[me.y][me.x - 1][planeStruct[3]] = 1
        def fillReachable(r):
            state[r.y][r.x - 1][planeStruct[4]] = 1
        def fillAttackable(r):
            state[r.y][r.x - 1][planeStruct[5]] = 1
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
        ret = list(filter(lambda elem:elem.x == dist.x and elem.y == dist.y and elem.isAlive() == alive,self.stacks))
        return ret
    def directionToHex(self,mySelf,dirct):
        if(dirct < 0 or dirct > 5):
            logger.info('wrong direction {}'.format(dirct))
            return 0
        if(dirct == 0):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.y - 1,mySelf.x)
            else:
                return BHex(mySelf.y - 1, mySelf.x - 1)
        if(dirct == 1):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.y - 1,mySelf.x + 1)
            else:
                return BHex(mySelf.y - 1, mySelf.x)
        if(dirct == 2):
            return BHex(mySelf.y, mySelf.x + 1)
        if(dirct == 5):
            return BHex(mySelf.y, mySelf.x - 1)
        if(dirct == 4):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.y + 1,mySelf.x)
            else:
                return BHex(mySelf.y + 1, mySelf.x - 1)
        if (dirct == 3):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.y + 1, mySelf.x + 1)
            else:
                return BHex(mySelf.y + 1, mySelf.x)
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
        if(actionType.wait == action.type):
            return 0
        if (actionType.defend == action.type):
            return 1
        if (actionType.move == action.type):
            i = action.move.y
            j = action.move.x
            return 2 + i* (self.bFieldWidth - 2) + j - 1
        if (actionType.attack == action.type):
            enemy = action.attack
            direct = self.hexToDirection(action.attack,action.move)
            i = enemy.y
            j = enemy.x
            return 2 + self.bFieldSize +(i * (self.bFieldWidth - 2) + (j - 1))*6+ direct
        if (actionType.shoot == action.type):
            enemy = action.attack
            i = enemy.y
            j = enemy.x
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
            return "move to ({},{})".format(act.move.y,act.move.x)
        if (act.type == actionType.attack):
            return "melee ({},{}),({},{})".format(act.move.y,act.move.x,act.attack.y,act.attack.x)
        if (act.type == actionType.shoot):
            return "shoot ({},{})".format(act.attack.y,act.attack.x)
    def end(self):
        live = {0:False,1:True}
        for st in self.stacks:
            live[st.side] = live[st.side] or st.isAlive()
        if self.round > 50:
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
            if (self.curStack.x == action.move.x and self.curStack.y == action.move.y):
                logger.info("can't move to where you already are!!")
                return
            if (self.canReach(self.curStack, action.move)):
                self.move(self.curStack, action.move)
            else:
                logger.info("you can't reach {},{}".format(action.move.y, action.move.x))
        elif(action.type == actionType.attack):
            dists = self.findStack(action.attack,True)
            if(len(dists) == 0):
                logger.info("wrong attack dist {} {}".format(action.attack.y,action.attack.x))
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
                logger.info("wrong shoot dist {} {}".format(action.attack.y, action.attack.x))
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
            logger.info("-------final action: {}".format(self.action2Str(actInd)))
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
        if winner != -1:
            logger.info("Game end. Winner is player: {}".format(winner))
        else:
            logger.info("Game end. Tie")
        return zip(states, mcts_probs, current_players,lefts,left_bases,rights,right_bases)




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

class BAction:
    def __init__(self,type = 0,move = 0,attack = 0,spell=0):
        self.type = type
        self.move = move
        self.attack = attack
        self.spell = spell

    # pl1 = BPlayer(0)
    # pl2 = BPlayer(1)
    # players = [pl1, pl2]
    # battle = Battle()
    # pl1.setBattle(battle)
    # pl2.setBattle(battle)
    # battle.loadFile("D:/project/VCNN/train/selfplay.json")
    # battle.newRound()
    # while(not battle.end()):
    #     battle.checkGameEndOrNewRound()
    #     cplayer = battle.currentPlayer()
    #     printF(battle.curStack.acssessableAndAttackable(),battle.stacks,battle.curStack)
    #     act = players[cplayer].getAction()
    #     if(act == 0):
    #         continue
    #     legals = battle.curStack.legalMoves()
    #     myMove = battle.actionToIndex(act)
    #     if(myMove not in legals):
    #         logger.info('...sth  wrong.....')
    #     battle.doAction(act)




