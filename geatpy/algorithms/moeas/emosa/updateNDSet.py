# -*- coding: utf-8 -*-
import numpy as np

import geatpy as ea


def updateNDSet(population, maxormins, MAXSIZE, NDSet=None):
    """

    """

    [levels, criLevel] = ea.ndsortDED(population.ObjV, None, 1, population.CV, maxormins)  # 只对个体划分出第一层
    CombinObjV = ea.awGA(population.ObjV, population.CV, maxormins)  # 计算适应性权重以及多目标的加权单目标
    population.FitnV = (np.max(CombinObjV) - CombinObjV + 0.5) / (
            np.max(CombinObjV) - np.min(CombinObjV) + 0.5)  # 计算种群适应度
    # 更新NDSet
    if NDSet is None:
        return population[np.where(levels == 1)[0]]
    else:
        tempPop = population[np.where(levels == 1)[0]] + NDSet  # 将种群可行个体与NDSet合并
        [levels, criLevel] = ea.ndsortDED(tempPop.ObjV, None, 1, tempPop.CV, maxormins)  # 只对个体划分出第一层
        liveIdx = np.where(levels == 1)[0]  # 选择非支配个体
        NDSet = tempPop[liveIdx]
        # 对种群中被NDSet支配的个体进行惩罚
        punishFlag = np.zeros(population.sizes)
        punishFlag[np.where(liveIdx < population.sizes)[0]] = 1
        population.FitnV[np.where(punishFlag == 0)[0]] *= 0.5
        if len(liveIdx) > MAXSIZE:  # 若要保留下来的NDSet个体数大于MAXSIZE，则根据拥挤距离进行筛选
            dis = ea.crowdis(NDSet.ObjV, levels[liveIdx])  # 计算拥挤距离
            NDSet = NDSet[ea.selecting('dup', np.array([dis]).T, MAXSIZE)]  # 进行筛选
        return NDSet
