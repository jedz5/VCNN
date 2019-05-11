import pygame  # 导入pygame库
from pygame.locals import *  # 导入pygame库中的一些常量
from sys import exit  # 导入sys库中的exit函数
from Battle import *
# Enum EBattleCursors { COMBAT_BLOCKED, COMBAT_MOVE, COMBAT_FLY, COMBAT_SHOOT,
# 						COMBAT_HERO, COMBAT_QUERY, COMBAT_POINTER,
# 						//various attack frames
# 						COMBAT_SHOOT_PENALTY = 15, COMBAT_SHOOT_CATAPULT, COMBAT_HEAL,
# 						COMBAT_SACRIFICE, COMBAT_TELEPORT}
from enum import Enum
COMBAT_BLOCKED, COMBAT_MOVE, COMBAT_FLY, COMBAT_SHOOT,COMBAT_HERO, COMBAT_QUERY, COMBAT_POINTER = range(7)
COMBAT_SHOOT_PENALTY,COMBAT_SHOOT_CATAPULT, COMBAT_HEAL,COMBAT_SACRIFICE, COMBAT_TELEPORT = range(15,20)


class BPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class CClickableHex:
    def __init__(self, hex_i=0, hex_j=0, pixels_x=0, pixels_y=0):
        self.hex_i = hex_i
        self.hex_j = hex_j
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.hovered = False

    def mouseMoved(self):
        self.hovered = True
    @classmethod
    def XYtoIJ(self,mouse_x,mouse_y):
        i = (int)((mouse_y - 86) / 42)
        j = (int)((mouse_x - (14 + (22 if i % 2 == 0 else 0))) / 44)
        return i,j
    @classmethod
    def getXYUnitAnim(self,hex,stack):
        ret = BPoint(-500,-500)
        basePos = BPoint(-190,-139) # position of creature in topleft corner
        imageShiftX = 30 # X offset to base pos for facing right stacks, negative for facing left
        ret.x = basePos.x + 22 * ((hex.x + 1) % 2) + 44 * hex.y;
        ret.y = basePos.y + 42 * hex.x;
        if not stack.side:
            ret.x += imageShiftX
        else:
            ret.x -= imageShiftX
        return ret
        # if stack.isWide:
    def getHexXY(self):
        if not self.pixels_x :
            self.pixels_x = 14 + (22 if self.hex_i % 2 == 0 else 0) + 44 * self.hex_j
            self.pixels_y = 86 + 42 * self.hex_i
        return self.pixels_x,self.pixels_y
class BattleInterface:
    def __init__(self, battleEngine):
        # 定义窗口的分辨率
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.BFIELD_WIDTH = 17;
        self.BFIELD_HEIGHT = 11;
        self.battleEngine = battleEngine
        self.current_hex = CClickableHex()
        self.transColor = pygame.Color(255, 0, 255)
        self.stimgs = {}
        self.cursor = [0] * 20
        self.clock = pygame.time.Clock()
        self.loadIMGs()
        self.hex_shader = self.stimgs["hex_shader"]
        self.background = self.stimgs["background"]
    def loadIMGs(self):
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])  # 初始化一个用于显示的窗口
        stacks = self.battleEngine.stacks
        background = pygame.image.load("D:/project/vcnn/imgs/bgrd.bmp")
        self.stimgs["background"] = background
        shader = pygame.image.load("D:/project/vcnn/imgs/CCellShd.bmp").convert_alpha()
        hex_shader = pygame.Surface(shader.get_size())
        hex_shader.blit(shader, (0, 0))
        hex_shader.set_colorkey(self.transColor)
        hex_shader.set_alpha(100)
        self.stimgs["hex_shader"] = hex_shader
        for st in stacks:
            if st.name not in self.stimgs:
                img = pygame.image.load("imgs/creatures/"+st.name+".bmp").convert_alpha()
                imgback = pygame.Surface(img.get_size())
                imgback.blit(img, (0, 0))
                imgback.set_colorkey((16,16,16))
                self.stimgs[st.name] = imgback
        for oi in self.battleEngine.obsinfo:
            if oi.imname not in self.stimgs:
                img = pygame.image.load("imgs/obstacles/" + oi.imname + ".bmp").convert_alpha()
                imgback = pygame.Surface(img.get_size())
                imgback.blit(img, (0, 0))
                if oi.isabs:
                    imgback.set_colorkey((0, 255, 255))
                else:
                    imgback.set_colorkey((0, 0, 0))
                self.stimgs[oi.imname] = imgback
        for idx in range(20):
            img = pygame.image.load("imgs/cursor/" + str(idx) + ".bmp").convert_alpha()
            imgback = pygame.Surface(img.get_size())
            imgback.blit(img, (0, 0))
            imgback.set_colorkey((0, 255, 255))
            self.cursor[idx] = imgback
    def renderFrame(self):
        self.clock.tick(60)
        self.update()
        pygame.display.update()

    def showBackground(self):
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.hex_shader, (self.current_hex.pixels_x, self.current_hex.pixels_y))

    def showBattlefieldObjects(self):
        ##########
        # BStack.acssessableAndAttackable()
        # Battle.curStack
        ##########
        stacks = self.battleEngine.stacks
        #show active stack
        curStackRange = self.battleEngine.curStack.acssessableAndAttackable()
        for i in range(self.battleEngine.bFieldHeight):
            for j in range(self.battleEngine.bFieldWidth):
                if curStackRange[i][j] >=0 and curStackRange[i][j] < 50:
                    self.screen.blit(self.hex_shader, CClickableHex(i,j).getHexXY())
        # hover on stack
        hoveredStack = 0
        for stack in stacks:
            if self.current_hex.hex_i == stack.x and self.current_hex.hex_j == stack.y:
                hoveredStack = stack
        if hoveredStack:
            hoveredRange = hoveredStack.acssessableAndAttackable()
            for i in range(self.battleEngine.bFieldHeight):
                for j in range(self.battleEngine.bFieldWidth):
                    if hoveredRange[i][j] >= 0 and hoveredRange[i][j] < 50:
                        self.screen.blit(self.hex_shader, CClickableHex(i,j).getHexXY())

        # show stacks
        for stack in stacks:
            coord = CClickableHex.getXYUnitAnim(BHex(stack.x,stack.y),stack)
            img = self.stimgs[stack.name]
            img = pygame.transform.flip(img, True, False) if stack.side else img
            self.screen.blit(img,(coord.x,coord.y))

        #show obstacles
        for oi in self.battleEngine.obsinfo:
            img = self.stimgs[oi.imname]
            x,y = self.getObstaclePosition(img,oi)
            self.screen.blit(img, (x, y))

        #show cursor
        # cursorMap = curStackRange.deepcopy()
        # if cursorMap[self.current_hex.hex_i,self.current_hex.hex_j] < 0
    def update(self):
        self.showBackground()
        self.showBattlefieldObjects()
        # self.showProjectiles(to)
        # updateBattleAnimations()
        # showInterface(to)

    def handleEvents(self):
        # 处理游戏退出
        # 从消息队列中循环取
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                move_x, move_y = event.rel
                self.handleMouseMotion(mouse_x, mouse_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_x, mouse_y = event.pos
                i,j = CClickableHex.XYtoIJ(mouse_x, mouse_y)
                #if


    def handleMouseMotion(self, mouse_x, mouse_y):
        #hover on hex
        if mouse_x < 58 or mouse_y <86:
            return
        i,j = CClickableHex.XYtoIJ(mouse_x,mouse_y)
        self.current_hex.hex_i = i
        self.current_hex.hex_j = j
        self.current_hex.pixels_x = mouse_x - (mouse_x - (14 + (22 if i % 2 == 0 else 0))) % 44
        self.current_hex.pixels_y = mouse_y - (mouse_y - 86) % 42
        # print("hovered on pixels{},{} location{},{} repixels{},{}".format(mouse_x,mouse_y,i,j,self.current_hex.pixels_x,self.current_hex.pixels_y))

    def handleBattle(self,act):
        if not act:
            return
        battle = self.battleEngine
        battle.checkNewRound()
        cplayer = battle.currentPlayer()
        printF(battle.curStack.acssessableAndAttackable(), battle.stacks, battle.curStack)
        # act = players[cplayer].getAction(battle)
        legals = battle.curStack.legalMoves()
        myMove = battle.actionToIndex(act)
        if (myMove not in legals):
            logger.info('...sth  wrong.....')
        battle.doAction(act)

    def getObstaclePosition(self,img, obinfo):
        if obinfo.isabs:
            return obinfo.width,obinfo.height
        offset = img.get_height() % 42
        if offset > 37:
            offset -=42
        bh = CClickableHex(obinfo.x,obinfo.y)
        x,y = bh.getHexXY()
        y += 42 - img.get_height() + offset
        return x,y
def start_game():
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    pl1 = BPlayer()
    pl2 = BPlayer()
    players = [pl1, pl2]
    battle = Battle()
    battle.loadFile("D:/project/VCNN/train/selfplay.json")
    battle.checkNewRound()
    bi = BattleInterface(battle)
    # 事件循环(main loop)
    while True:
        act = bi.handleEvents()
        bi.handleBattle(act)
        bi.renderFrame()
if __name__ == '__main__':
    start_game()

