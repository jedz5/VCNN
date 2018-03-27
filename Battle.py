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
        self.isHuman = True
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
        self.name = 'unKnown'
        self.x = 0
        self.y = 0
        self.isWide = False
        self.isFly = False
        self.isShooter = False
        self.amountBase = 0
    def computeCasualty(self,opposite):
        if(self.attack >= opposite.attack):
            damageMin = self.minDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount
            damageMax = self.maxDamage*(1+(self.attack - opposite.attack)*0.05)*self.amount
        else:
            damageMin = self.minDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount
            damageMax = self.maxDamage * (1 + (self.attack - opposite.attack) * 0.025) * self.amount
        damage = random.randint(damageMin,damageMax)
        hpInAll = opposite.health*(opposite.amount-1) + opposite.firstHPLeft
        if(damage >= hpInAll):
            killed = opposite.amount
            opposite.amount = 0
        else:
            rest = int((hpInAll - damage)/opposite.health)+1
            opposite.firstHPLeft = (hpInAll - damage)%opposite.health
            killed = opposite.amount - rest
            opposite.amount = rest
        print("killed {} {}".format(killed,opposite.name))
        return killed
    def attack(self,opposite,retaliate):
        killed = self.computeCasualty(opposite)
        return
    def attacked(self,attacker,retaliate):
        return
    def canShoot(self,opposite):
        return
    def shoot(self,opposite):
        return
    def beShoot(self,attacker):
        return
    def wait(self):
        self.isWaited = True
    def defend(self):
        self.defense += 2
        self.isMoved = True
    def newRound(self):
        if self.isMoved:
            self.defense -= 2
        self.isMoved = False
        self.isRetaliate = True
        self.isWaited = False
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
    def __init__(self,player1,player2):
        self.bField = np.array((17,11),dtype=object)
        self.stackQueue = []
        self.players = [player1,player2]
        player1.setBattle(self)
        player2.setBattle(self)
        self.curStack = BStack()
    def canReach(self,bFrom,bTo):
        return True
    def nextTo(self,bFrom,bTo):
        return
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




