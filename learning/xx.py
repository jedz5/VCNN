import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
import timeit
import pdb
from enum import IntEnum
import random
import matplotlib.pyplot as plt
from collections import  Counter
import glob
from math import log2
import numpy as np, matplotlib.pyplot as plt
a = [1,2,4,20]
b = [1,2,4,200]
X = np.array([1/16,1/8,1/4,1/3,1/2])
def hh():
    plt.figure(figsize=(8, 5))
    sz = 100
    x1 = np.sort(np.random.normal(1.7,0.1,sz))
    x2 = np.sort(np.random.normal(1.6,0.1,sz))
    # normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
    def normfun(x, mu, sigma):
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        return pdf

    # x的范围为60-150，以1为单位,需x根据范围调试
    u1 = np.arange(1.3, 2.0, 0.02)
    # x数对应的概率密度
    y1 = normfun(u1, 1.7,0.1)
    y2 = normfun(u1, 1.55, 0.1)
    # 参数,颜色，线宽
    plt.plot(u1, y1, color='g', linewidth=3)
    plt.plot(u1, y2, color='r', linewidth=3)
    plt.scatter(x1,[1]*sz,marker='x')
    plt.scatter(x2, [1.2]*sz,marker='x')
    print(x1)
    print(x2)
    plt.show()
def hh2():
    # plt.figure(figsize=(8,5))
    b = np.linspace(1, 6, 100)
    b_true=2
    b_start = 3.5
    for i in range(0,100):
        x0 = np.random.randint(-100,100)
        y0 = (x0 - b_true)**2#((x0-b)**2-y0)**2
        r1 = 2*x0 - b_true
        z = np.poly1d([-1,x0])
        z = z*z - [y0]
        z = z * z
        dz = z.deriv()
        dzb = dz(b_start)
        if b_start > x0:
            print("2--{}-{}-{}".format(x0,b_start,r1))
        else:
            print("2-{}--{}-{}".format(b_start,x0, r1))
        b_start -= 0.001 * dzb
        print("b = {}, dzb = {}".format(b_start,dzb))
        print("")
def entrop(p):
    p = np.array(p) / sum(p)
    x = -np.log2(p) #[log2(1 / i) for i in p]
    h = - p * np.log2(p) #[-i * log2(i) for i in p]
    return x,h
# -*- coding: utf-8 -*-
# 记住上面这行是必须的，而且保存文件的编码要一致！
import pygame
from pygame.locals import *
from sys import exit

pygame.init()
screen = pygame.display.set_mode((640, 480), 0, 32)

#font = pygame.font.SysFont("宋体", 40)
#上句在Linux可行，在我的Windows 7 64bit上不行，XP不知道行不行
#font = pygame.font.SysFont("simsunnsimsun", 40)
#用get_fonts()查看后看到了这个字体名，在我的机器上可以正常显示了
font = pygame.font.SysFont("Arial", 11)
CCellShd = pygame.image.load("D:/project/vcnn/imgs/CCellShd.bmp") #
#CCellShd.set_colorkey(pygame.Color(255, 0, 255))
CCellShd = CCellShd.convert_alpha()
font.set_bold(True)
background = pygame.image.load("D:/project/vcnn/imgs/bgrd.bmp")
amout_backgrd = pygame.image.load("D:/project/vcnn/imgs/CmNumWin.bmp").convert_alpha()
#amout_backgrd.fill((0,0,255))
screen.blit(background, (0,0))
def blend_img(img,color):
    """Inverts the colors of a pygame Screen"""

    img.lock()

    for x in range(img.get_width()):
        for y in range(img.get_height()):
            RGBA = img.get_at((x,y))
            if(RGBA[0] == 255):
                continue
            for i in range(3):
                #new_RGBA = blend_color(RGBA,pygame.Color("purple"),1)
                img.set_at((x,y),color )

    img.unlock()
def blend_color(color1, color2, blend_factor):
    r1, g1, b1,a1 = color1
    r2, g2, b2,a2 = color2
    r = r1 + (r2 - r1) * blend_factor
    g = g1 + (g2 - g1) * blend_factor
    b = b1 + (b2 - b1) * blend_factor
    return int(r), int(g), int(b),255
text = "today is a good day to play "
# blend_img(CCellShd,(0,255,0))

screen.blit(amout_backgrd, (240, 240))
#CCellShd.set_colorkey(pygame.Color(255, 0, 255))

#text = font.render(u"{}     #".format(123), True, (255, 255, 255))
#amout_backgrd.blit(text,(0,-2))
blend_img(amout_backgrd,pygame.Color("blueviolet"))
screen.blit(amout_backgrd, (300, 240))
pygame.image.save(amout_backgrd, "D:/project/vcnn/imgs/CmNumWin_purple.bmp")
# black_scale_surface = pygame.surface.Surface((40, 40))
# black_scale_surface.fill((0,0,0,100))

screen.blit(CCellShd, (360, 240))
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()

    # screen.blit(text_surface,(10,10))

    pygame.display.update()
