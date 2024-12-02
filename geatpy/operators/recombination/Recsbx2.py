# -*- coding: utf-8 -*-
import random

import numpy as np
from geatpy.core.recsbx import recsbx

from geatpy.operators.recombination.Recombination import Recombination


class Recsbx2(Recombination):
    """
    Recsbx - class : 一个用于调用内核中的函数recsbx(模拟二进制交叉)的类，
                     该类的各成员属性与内核中的对应函数的同名参数含义一致，
                     可利用help(recsbx)查看各参数的详细含义及用法。
    """

    def __init__(self, XOVR=0.7, Half_N=False, n=20, Parallel=False, rate=0.5):
        self.XOVR = XOVR  # 发生交叉的概率
        self.Half_N = Half_N  # 该参数是旧版的输入参数Half的升级版，用于控制交配方式
        self.n = n  # 分布指数，必须不小于0
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省时默认为False
        self.rate = rate  # 表示采用交叉2的概率

    def do(self, OldChrom, gen, max_gen):  # 执行内核函数
        if random.random()>self.rate:
            return recsbx(OldChrom, self.XOVR, self.Half_N, self.n, self.Parallel)
        else:
            rows, columns = OldChrom.shape
            index = int(gen/max_gen * columns)
            changed = recsbx(np.array(OldChrom[:,index:]),self.XOVR, self.Half_N, self.n, self.Parallel)
            nn = OldChrom[random.randint(0, 1)][:index]
            outChrom = np.hstack((np.array(nn).reshape(1,-1),changed))
            return outChrom

    def getHelp(self):  # 查看内核中的重组算子的API文档
        help(recsbx)
