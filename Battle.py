import random
import json
import numpy as np
import mcts_alpha
from mcts_alpha import MCTSPlayer
from enum import Enum
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
        self.firstHPLeft = 0
        self.health = 0
        self.isMoved = False
        self.side = 0
        self.isRetaliated = False
        self.isWaited = False
        self.maxDamage = 0
        self.minDamage = 0
        self.speed = 0
        self.luck = 0
        self.morale = 0
        self.id = 0
        self.shots = 10
        self.hexType = hexType.creature
        #辅助参数
        self.isDenfenced = False
        self.name = 'unKnown'
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
        for obs in self.inBattle.obstacles:  #obstacle,enemy and attackable,mine
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
        hpInAll = opposite.health*(opposite.amount-1) + opposite.firstHPLeft
        if(damage >= hpInAll):
            killed = opposite.amount
            firstHPLeft = 0
        else:
            rest = int((hpInAll - damage)/opposite.health)+1
            firstHPLeft = (hpInAll - damage)%opposite.health
            if(firstHPLeft == 0):
                firstHPLeft = opposite.health
            killed = opposite.amount - rest
        print("killed {} {},firstHPLeft {}".format(killed,opposite.name,firstHPLeft))
        return killed,firstHPLeft
    def meeleAttack(self,opposite,retaliate):
        print('{} meele attacking {}'.format(self.name, opposite.name))
        killed,firstHPLeft = self.computeCasualty(opposite)
        opposite.amount -= killed
        opposite.firstHPLeft = firstHPLeft
        self.isMoved = True
        if(opposite.amount == 0):
            print("{} perished".format(opposite.name))
        else:
            opposite.attacked(self,retaliate)
    def attacked(self,attacker,retaliate):
        if(not retaliate):
            if(attacker.blockRetaliate):
                print("blockRetaliate")
                return
            if(not self.isRetaliated):
                print("prepare retaliate")
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
        print('{} shooting {}'.format(self.name,opposite.name))
        killed,firstHPLeft = self.computeCasualty(opposite,True,self.isHalf(opposite))
        opposite.amount -= killed
        opposite.firstHPLeft = firstHPLeft
        if (opposite.amount == 0):
            print("{} perished".format(opposite.name))
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
            print("sth wrong happen! {} is moved!!!".format(self.name))
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
class Battle(object):
    bFieldWidth = 17
    bFieldHeight = 11
    bPenaltyDistance = 10
    bFieldSize = (bFieldWidth - 2)* bFieldHeight
    def __init__(self):
        #self.bField = [[0 for col in range(battle.bFieldWidth)] for row in range(battle.bFieldHeight)]
        self.stacks = []
        self.round = 0
        self.obstacles = []
        self.toMove = []
        self.waited = []
        self.moved = []
        self.stackQueue = []
        self.curStack = 0

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
                st.x = x['x']
                st.y = x['y']
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
                print('dist {},{} not valid'.format(bTo.y,bTo.x))
                return False
            bf = curSt.acssessableAndAttackable()
            if(bAtt != 0):
                return self.bGetDistance(bTo,bAtt) == 1 and bf[bTo.y][bTo.x] >= 0 and bf[bAtt.y][bAtt.x] == -1
            else:
                return bf[bTo.y][bTo.x] >= 0
        print('src {},{} not valid'.format(bFrom.y, bFrom.x))
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
            print("sth wrong move from {},not exist".format(bFrom))
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
    def currentPlayer(self):
        return 1 if self.stackQueue[0].side else 0
    def findStack(self,dist,alive=True):
        ret = list(filter(lambda elem:elem.x == dist.x and elem.y == dist.y and elem.isAlive() == alive,self.stacks))
        return ret
    def directionToHex(self,mySelf,dirct):
        if(dirct < 0 or dirct > 5):
            print('wrong direction {}'.format(dirct))
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
                return BHex(mySelf.y + 1,mySelf.x + 1)
            else:
                return BHex(mySelf.y + 1, mySelf.x)
        if (dirct == 3):
            if (mySelf.y % 2 == 0):
                return BHex(mySelf.y + 1, mySelf.x)
            else:
                return BHex(mySelf.y + 1, mySelf.x - 1)
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
        print('actionToIndex wrong action {}'.format(action))
    def indexToAction(self,move):
        if(move < 0):
            print('wrong move {}'.format(move))
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
            print("wrong move {}".format(move))
    def end(self):
        live = {0:False,1:False}
        for st in self.stacks:
            live[st.side] = live[st.side] or st.isAlive()
        return not (live[0] and live[1])
    def checkGameEndOrNewRound(self):
        if(self.end()):
            print("game over")
            exit(0)
        self.sortStack()
        if(self.stackQueue[0].isMoved):
            self.newRound()
            self.sortStack()
        self.curStack = self.stackQueue[0]
        print("now it's {} turn".format(self.curStack.name))

    def newRound(self):
        self.round += 1
        print("now it's round ",self.round)
        for st in self.stacks:
            if(st.isAlive()):
                st.newRound()
    def doAction(self,action):
        if(self.curStack.isMoved):
            print("{} is already moved".format(self.curStack))
            return
        if(action.type == actionType.wait):
            if (self.curStack.isWaited):
                print("{} is already waited".format(self.curStack))
                return
            self.curStack.wait()
        elif(action.type == actionType.defend):
            self.curStack.defend()
        elif(action.type == actionType.move):
            if (self.curStack.x == action.move.x and self.curStack.y == action.move.y):
                print("can't move to where you already are!!")
                return
            if (self.canReach(self.curStack, action.move)):
                self.move(self.curStack, action.move)
            else:
                print("you can't reach {},{}".format(action.move.y, action.move.x))
        elif(action.type == actionType.attack):
            dists = self.findStack(action.attack,True)
            if(len(dists) == 0):
                print("wrong attack dist {} {}".format(action.attack.y,action.attack.x))
                return
            dist = dists[0]
            if (self.canReach(self.curStack, action.move,action.attack)):
                self.move(self.curStack, action.move)
                self.curStack.meeleAttack(dist,False)
            else:
                print("you can't reach {} and attack {}".format(action.move,action.attack))
        elif(action.type == actionType.shoot):
            dists = self.findStack(action.attack, True)
            if (len(dists) == 0):
                print("wrong shoot dist {} {}".format(action.attack.y, action.attack.x))
                return
            dist = dists[0]
            if (self.curStack.canShoot(action.attack)):
                self.curStack.shoot(dist)
            else:
                print("{} can't shoot {}".format(self.curStack.name,dist.name))
        elif (action.type == actionType.spell):
            print("spell not implemented yet")





class  BPlayer(object):
    def __init__(self,side):
        self.side = side
    def setBattle(self,battle):
        self.battle = battle
    def getAction(self):
        action = BAction()
        try:
            act = input("action: ")
            if isinstance(act,str):
                ii = act.split(',')
                acts = [int(a) for a in ii]
                action.type = actionType(acts[0])
                if(action.type == actionType.wait or action.type == actionType.defend):
                    return action
                elif(action.type == actionType.move):
                    action.move = BHex(acts[1],acts[2])
                elif(action.type == actionType.attack):
                    action.move = BHex(acts[1],acts[2])
                    action.attack = BHex(acts[3],acts[4])
                elif((action.type == actionType.shoot)):
                    action.attack = BHex(acts[1],acts[2])
                else:
                    print("action not implementedd yet")
        except Exception as e:
            action = 0
            print("Exception:  ",e)
        return action
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.start_self_play(temp=1.0)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)
    def start_self_play(self,is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        battle = Battle()
        trainer = MCTSPlayer(mcts_alpha.policy_value_fn,c_puct=5,n_playout=5,is_selfplay=1)
        battle.loadFile("D:/project/VCNN/train/selfplay.json")
        battle.newRound()
        states, mcts_probs, current_players = [], [], []
        while (not battle.end()):
            battle.checkGameEndOrNewRound()
            if(is_shown):
                printF(battle.curStack.acssessableAndAttackable(), battle.stacks, battle.curStack)
            act,move_probs = trainer.getAction(self,temp=temp,return_prob=1)
            if (act == 0):
                continue
            legals = battle.curStack.legalMoves()
            myMove = battle.actionToIndex(act)
            if (myMove not in legals):
                print('...sth  wrong.....')
            else:
                # store the data
                states.append(self.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.currentPlayer())
            battle.doAction(act)
        winner = self.currentPlayer()
        # winner from the perspective of the current player of each state
        winners_z = np.zeros(len(current_players))
        if winner != -1:
            winners_z[np.array(current_players) == winner] = 1.0
            winners_z[np.array(current_players) != winner] = -1.0
        # reset MCTS root node
        trainer.reset_player()
        if is_shown:
            if winner != -1:
                print("Game end. Winner is player:", winner)
            else:
                print("Game end. Tie")
        return winner, zip(states, mcts_probs, winners_z)
class BAction:
    def __init__(self,type = 0,move = 0,attack = 0,spell=0):
        self.type = type
        self.move = move
        self.attack = attack
        self.spell = spell


if __name__ == '__main__':
    pl1 = BPlayer(0)
    pl2 = BPlayer(1)
    players = [pl1, pl2]
    battle = Battle()
    pl1.setBattle(battle)
    pl2.setBattle(battle)
    battle.loadFile("D:/project/VCNN/train/selfplay.json")
    battle.newRound()
    while(not battle.end()):
        battle.checkGameEndOrNewRound()
        cplayer = battle.currentPlayer()
        printF(battle.curStack.acssessableAndAttackable(),battle.stacks,battle.curStack)
        act = players[cplayer].getAction()
        if(act == 0):
            continue
        legals = battle.curStack.legalMoves()
        myMove = battle.actionToIndex(act)
        if(myMove not in legals):
            print('...sth  wrong.....')
        battle.doAction(act)




