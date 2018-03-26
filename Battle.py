import numpy as np
import operator
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
        #辅助参数
        self.name = 'unKnown'
        self.x = 0
        self.y = 0
        self.isWide = False
        self.isFly = False
        self.isShooter = False
        self.amountBase = 0
    def move(self):
        return
    def canMove(self):
        return not self.isMoved
    def attack(self,opposite):
        return
    def attacked(self,attacker):
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
    def newTurn(self):


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
    def canReach(self,bFrom,bTo):
        return True
    def move(self,bFrom,bTo):
        if(self.bField[bFrom.x][bFrom.y] != 0 and self.bField[bFrom.x][bFrom.y].id == bFrom.id):
            if(self.canReach(bFrom,bTo)):
                self.bField[bTo.x][bTo.y] = self.bField[bFrom.x][bFrom.y]
                self.bField[bFrom.x][bFrom.y] = 0
                return True
        return False
    def sortStack(self):
        cmpfun = operator.attrgetter('isWaited','speed','x')
        self.stackQueue.sort(key=cmpfun)
    def getCurrentPlayer(self):
        return self.players[0] if self.stackQueue[0].isHuman else self.players[1]
    def newTurn(self):
        return
    def doAction(self,action):
        if(action.type == actionType.wait)

class  BPlayer(object):
    def __init__(self,battle,side):
        self.battle = battle
        self.side = side
    def getAction(self):
        return
class BAction:
    def __init__(self,type,move = 0,spell=0):
        self.type = type
        self.move = move
        self.spell = spell



