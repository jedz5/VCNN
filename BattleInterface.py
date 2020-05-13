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

class log_with_gui(object):
    def __init__(self,std_logger):
        self.logger = std_logger
        self.log_text = []
    def info(self,text,to_gui = False):
        self.logger.info(text)
        if(to_gui):
            self.log_text.append(text)
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
        ret.x = basePos.x + 22 * ((hex.y + 1) % 2) + 44 * hex.x;
        ret.y = basePos.y + 42 * hex.y;
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
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 600
        self.BFIELD_WIDTH = 17
        self.BFIELD_HEIGHT = 11
        battleEngine.bat_interface = self
        self.battleEngine = battleEngine
        self.current_hex = CClickableHex()
        self.transColor = pygame.Color(255, 0, 255)
        self.stimgs = {}
        self.stimgs_dead = {}
        self.cursor = [0] * 20
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 11)
        self.font.set_bold(True)
        self.loadIMGs()
        self.next_act = None
    def loadIMGs(self):
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])  # 初始化一个用于显示的窗口
        stacks = self.battleEngine.stacks
        background = pygame.image.load("D:/project/vcnn/imgs/bgrd.bmp")
        self.background = background
        self.hex_shader = pygame.image.load("D:/project/vcnn/imgs/CCellShd_gray.bmp")
        self.hex_shader.set_colorkey(self.transColor)
        self.hex_shader.set_alpha(100)
        self.shader_cur_stack = pygame.image.load("D:/project/vcnn/imgs/CCellShd_green.bmp")
        self.shader_cur_stack.set_colorkey(self.transColor)
        self.shader_cur_target = pygame.image.load("D:/project/vcnn/imgs/CCellShd_red.bmp")
        self.shader_cur_target.set_colorkey(self.transColor)
        self.amout_backgrd = pygame.image.load("D:/project/vcnn/imgs/CmNumWin_purple.bmp").convert_alpha()
        self.amout_backgrd_enemy = pygame.image.load("D:/project/vcnn/imgs/CmNumWin_blue.bmp").convert_alpha()
        for st in stacks:
            if st.name not in self.stimgs:
                img = pygame.image.load("imgs/creatures/"+st.name+".bmp").convert_alpha()
                imgback = pygame.Surface(img.get_size())
                imgback.blit(img, (0, 0))
                imgback.set_colorkey((16,16,16))
                self.stimgs[st.name] = imgback
                #dead
                img = pygame.image.load("imgs/creatures/dead/" + st.name + ".bmp").convert_alpha()
                imgback = pygame.Surface(img.get_size())
                imgback.blit(img, (0, 0))
                imgback.set_colorkey((16, 16, 16))
                self.stimgs_dead[st.name] = imgback
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
        self.screen.fill((0,0,0))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.hex_shader, (self.current_hex.pixels_x, self.current_hex.pixels_y))

    def showBattlefieldObjects(self):
        ##########
        ##########
        stacks = self.battleEngine.stacks
        #show active stack
        curStackRange = self.battleEngine.curStack.acssessableAndAttackable()
        for i in range(self.battleEngine.bFieldHeight):
            for j in range(self.battleEngine.bFieldWidth):
                if curStackRange[i][j] >=0 and curStackRange[i][j] < 50:
                    self.screen.blit(self.hex_shader, CClickableHex(i,j).getHexXY())
        self.screen.blit(self.shader_cur_stack, CClickableHex(self.battleEngine.curStack.y, self.battleEngine.curStack.x).getHexXY())

        #stack target
        if (self.next_act.type == actionType.wait):
            pass
        elif (self.next_act.type == actionType.defend):
            pass
        elif (self.next_act.type == actionType.move):
            self.screen.blit(self.shader_cur_target,
                             CClickableHex(self.next_act.dest.y, self.next_act.dest.x).getHexXY())
        elif (self.next_act.type == actionType.attack):
            self.screen.blit(self.shader_cur_target,
                             CClickableHex(self.next_act.dest.y, self.next_act.dest.x).getHexXY())
        if (self.next_act.type == actionType.shoot):
            self.screen.blit(self.shader_cur_target,
                             CClickableHex(self.next_act.target.y, self.next_act.target.x).getHexXY())
        # show obstacles
        for oi in self.battleEngine.obsinfo:
            img = self.stimgs[oi.imname]
            x, y = self.getObstaclePosition(img, oi)
            self.screen.blit(img, (x, y))

        # hover on stack
        hoveredStack = 0
        for stack in stacks:
            if self.current_hex.hex_i == stack.y and self.current_hex.hex_j == stack.x:
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
                if (stack.isAlive()):
                    img = self.stimgs[stack.name]
                    img = pygame.transform.flip(img, True, False) if stack.side else img
                else:
                    img = self.stimgs_dead[stack.name]
                    img = pygame.transform.flip(img, True, False) if stack.side else img
                self.screen.blit(img,(coord.x,coord.y))

        # show stack amounts
        for stack in stacks:
            coord = CClickableHex.getXYUnitAnim(BHex(stack.x, stack.y), stack)
            if (stack.isAlive()):
                if(stack.side == 0):
                    amount_bgrd = copy.copy(self.amout_backgrd)
                else:
                    amount_bgrd = copy.copy(self.amout_backgrd_enemy)
                moved = "m" if stack.isMoved else ""
                retaliate = "" if stack.had_retaliated else "r"
                # text_surface = self.font.render(u"123            #", True,(255, 255, 255))
                # amount_bgrd.blit(text_surface, (0, -2))
                text_surface = self.font.render(u"{}{}{}              #".format(stack.amount,retaliate,moved), True, (255, 255, 255))
                amount_bgrd.blit(text_surface, (2, -2))
                xadd = 220 - (44 if stack.side else -22)
                yadd = 260 - 42 * 3
                self.screen.blit(amount_bgrd, (coord.x + 450 - xadd - amount_bgrd.get_width(), coord.y + 400 - yadd - amount_bgrd.get_height()))

        #info
        if(not self.battleEngine.last_stack):
            return
        root = pygame.Surface((300, 200))
        #root.fill((int(back_color[0]), int(back_color[1]), int(back_color[2])))
        root.set_colorkey((0,0,0))
        last_stack_coord = CClickableHex.getXYUnitAnim(BHex(self.battleEngine.last_stack.x, self.battleEngine.last_stack.y), self.battleEngine.last_stack)
        start_height = 0
        for text in logger.log_text:
            ftext = self.font.render(text, True, (255, 255, 255))
            root.blit(ftext, (0,start_height))
            start_height = start_height + ftext.get_height() + 5
        xadd = 220 - (44 if self.battleEngine.last_stack.side else -22)
        yadd = 260 - 42 * 3
        self.screen.blit(root, (last_stack_coord.x + 450 - xadd, last_stack_coord.y + 400 - yadd))
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
                return self.next_act


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
        bf = battle.curStack.acssessableAndAttackable()
        battle.doAction(act)
        battle.checkNewRound()
        self.next_act = battle.curStack.active_stack()

    def getObstaclePosition(self,img, obinfo):
        if obinfo.isabs:
            return obinfo.width,obinfo.height
        offset = img.get_height() % 42
        if offset > 37:
            offset -=42
        bh = CClickableHex(obinfo.y,obinfo.x)
        x,y = bh.getHexXY()
        y += 42 - img.get_height() + offset
        return x,y
def start_game():
    # 初始化游戏
    pygame.init()  # 初始化pygame
    pygame.display.set_caption('This is my first pyVCMI')  # 设置窗口标题
    battle = Battle()
    battle.loadFile("D:/project/VCNN/ENV/selfplay.json")
    battle.checkNewRound()
    bi = BattleInterface(battle)
    bi.next_act = battle.curStack.active_stack()
    # 事件循环(main loop)
    while True:
        act = bi.handleEvents()
        bi.handleBattle(act)
        bi.renderFrame()
if __name__ == '__main__':
    start_game()

