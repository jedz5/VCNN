import numpy as np
import operator
import random
from enum import Enum
class BStack(object):
    def  __init__(self):
        self.amount = 0
        self.attack = 0
        self.defense = 0
        self.firstHPLeft = 0
        self.health = 0
        self.isMoved = False
        self.side = 0
        self.isRetaliate = True
        self.isWaited = False
        self.maxDamage = 0
        self.minDamage = 0
        self.speed = 0
        self.luck = 0
        self.morale = 0
        self.id = 0
        self.shots = 10
        #辅助参数
        self.isDenfenced = False
        self.name = 'unKnown'
        self.x = 0
        self.y = 0
        self.isWide = False
        self.isFly = False
        self.isShooter = False
        self.amountBase = 0
        self.inBattle = Battle()

    def availableDists(self):

        dists = []
        if(self.isFly):
            for adj in self.getNeibours():

    def computeCasualty(self,opposite,shoot=False,half=True):
        if(self.attack >= opposite.attack):
            damageMin = self.minDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount
            damageMax = self.maxDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount
        else:
            damageMin = self.minDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount
            damageMax = self.maxDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount
        damage = random.randint(damageMin,damageMax)
        if(shoot and half):
            damage /= 2
        hpInAll = opposite.health*(opposite.amount-1) + opposite.firstHPLeft
        if(damage >= hpInAll):
            killed = opposite.amount
            firstHPLeft = 0
        else:
            rest = int((hpInAll - damage)/opposite.health)+1
            firstHPLeft = (hpInAll - damage)%opposite.health
            killed = opposite.amount - rest
        print("killed {} {},firstHPLeft {}".format(killed,opposite.name,firstHPLeft))
        return killed,firstHPLeft
    def attack(self,opposite,retaliate):
        killed,firstHPLeft = self.computeCasualty(opposite)
        opposite.amount -= killed
        opposite.firstHPLeft = firstHPLeft
        self.isMoved = True
        if(opposite.amount == 0):
            print("{} perished".format(opposite.name))
        else:
            opposite.attacked(self,opposite,retaliate)
    def attacked(self,attacker,retaliate):
        if(not retaliate):
            if(self.isRetaliate):
                self.attack(attacker,True)
                self.isRetaliate = True
    def canShoot(self,opposite):
        adj = self.getNeibours()
        for xy in adj:
            if(self.inBattle[xy.y][xy.x] != 0 and self.inBattle[xy.y][xy.x].side != self.side):
                return False
        return True
    def getNeibours(self):
        adj = []
        self.checkAndPush(self.x - 1, self.y, adj)
        self.checkAndPush(self.x + 1, self.y, adj)
        self.checkAndPush(self.x - 1, self.y - 1, adj)
        self.checkAndPush(self.x, self.y - 1, adj)
        self.checkAndPush(self.x - 1, self.y + 1, adj)
        self.checkAndPush(self.x, self.y + 1, adj)
        return adj
    def checkAndPush(self,x,y,adj):
        if(x>0 and x<Battle.bFieldWidth - 1 and y >= 0 and y < Battle.bFieldHeight):
            adj.append((x,y))
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
        killed,firstHPLeft = self.computeCasualty(opposite,True,self.isHalf(opposite))
        opposite.amount -= killed
        opposite.firstHPLeft = firstHPLeft
        if (opposite.amount == 0):
            print("{} perished".format(opposite.name))
        self.isMoved = True
        return
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
        self.isRetaliate = True
        self.isWaited = False
        self.isDenfenced = False
        return

class BObstacle(object):
    def __init__(self,kind = 0):
        self.kind = kind
        self.x = 0
        self.y = 0

class actionType(Enum):
    wait = 0
    defend = 1
    move = 2
    attack = 3
    spell = 4

class Battle(object):
    bFieldWidth = 17
    bFieldHeight = 11
    bPenaltyDistance = 10
    def __init__(self,player1,player2):
        self.bField = np.array((17,11),dtype=object)
        self.stackQueue = []
        self.players = [player1,player2]
        player1.setBattle(self)
        player2.setBattle(self)
        self.curStack = BStack()

    def canReach(self,bFrom,bTo):
        return True
    def move(self,bFrom,bTo):
        if(self.bField[bFrom.x][bFrom.y] != 0 and self.bField[bFrom.x][bFrom.y].id == bFrom.id):
            if(self.canReach(bFrom,bTo)):
                self.bField[bTo.x][bTo.y] = self.bField[bFrom.x][bFrom.y]
                self.bField[bFrom.x][bFrom.y] = 0
                return
        print("sth wrong from {} to {}".format(bFrom,bTo))
    def sortStack(self):
        cmpfun = operator.attrgetter('isWaited','speed','x')
        self.stackQueue.sort(key=cmpfun)
    def getCurrentPlayer(self):
        return self.players[0] if self.stackQueue[0].isHuman else self.players[1]
    def end(self):
        return False
    def checkNewRound(self):
        self.sortStack()
        if(self.stackQueue[0].isMoved):
            self.newRound()

    def newRound(self):
        for st in self.stackQueue:
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
            if(self.canReach(self.curStack,actionType.move)):
                self.move(self.curStack,actionType.move)
            else:
                print("you can't reach {}".format(actionType.move))
        elif(action.type == actionType.attack):
            if(self.curStack.canShoot(action.attack)):
                self.curStack.shoot(self.bField[action.attack.x][action.attack.y])
            else:
                if (self.canReach(self.curStack, actionType.move)):
                    self.move(self.curStack, actionType.move)
                    self.curStack.attack(self.bField[action.attack.x][action.attack.y])
                else:
                    print("you can't reach {}".format(actionType.move))
        elif(action.type == actionType.spell):
            print("spell not implemented yet")



class  BPlayer(object):
    def __init__(self,side):
        self.side = side
    def setBattle(self,battle):
        self.battle = battle
    def getAction(self):
        return
class BAction:
    def __init__(self,type,move = 0,attack = 0,spell=0):
        self.type = type
        self.move = move
        self.attack = attack
        self.spell = spell
if __name__ == '__main__':
    pl1 = BPlayer(0)
    pl2 = BPlayer(1)
    battle = Battle(pl1,pl2)
    while(not battle.end()):
        battle.checkNewRound()
        cplayer = battle.getCurrentPlayer()
        act = cplayer.getAction()
        battle.doAction(act)




