import pygame  # 导入pygame库
from pygame.locals import *  # 导入pygame库中的一些常量
from sys import exit  # 导入sys库中的exit函数
from Battle import *

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
    def IJtoXY(self,hex_i,hex_j):
        x = hex_j*44 + (14 + (22 if hex_i % 2 == 0 else 0))
        y = hex_i*42 + 86
        return x,y
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
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])  # 初始化一个用于显示的窗口

        self.background = pygame.image.load("D:/project/vcmi/imgs/bgrd.bmp")
        self.creIMG = {1:pygame.transform.scale(pygame.image.load("D:/project/vcmi/imgs/CBONG1A2.bmp"),(40,40))}
        shader = pygame.image.load("D:/project/vcmi/imgs/CCellShd.bmp").convert_alpha()
        self.hex_shader = pygame.Surface(shader.get_size())
        self.hex_shader.blit(shader, (0, 0))
        self.hex_shader.set_colorkey(self.transColor)
        self.hex_shader.set_alpha(100)
        self.clock = pygame.time.Clock()

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
                    self.screen.blit(self.hex_shader, CClickableHex.IJtoXY(i,j))
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
                        self.screen.blit(self.hex_shader, CClickableHex.IJtoXY(i, j))
        # show stacks
        # cre = self.creIMG[1]
        # self.screen.blit(cre, CClickableHex.IJtoXY(3, 5))
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

    def handleMouseMotion(self, mouse_x, mouse_y):
        #hover on hex
        if mouse_x < 58 or mouse_y <86:
            return
        i = (int)((mouse_y - 86) / 42)
        j = (int)((mouse_x - (14 + (22 if i % 2 == 0 else 0))) / 44)
        self.current_hex.hex_i = i
        self.current_hex.hex_j = j
        self.current_hex.pixels_x = mouse_x - (mouse_x - (14 + (22 if i % 2 == 0 else 0))) % 44
        self.current_hex.pixels_y = mouse_y - (mouse_y - 86) % 42
        print("hovered on pixels{},{} location{},{} repixels{},{}".format(mouse_x,mouse_y,i,j,self.current_hex.pixels_x,self.current_hex.pixels_y))

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

