# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
import geatpy as ea


def fitness_calculate(fitness_fun, population, **kwargs) -> ndarray:
    """
    适应度计算
    :param population: Population
    :param fitness_fun: 适应度函数
    :return: 被选择的概率列表
    """
    n, gene_len = population.shape
    fitness_list = np.arange(n)
    for i, x in enumerate(population):
        fitness_list[i] = fitness_fun(x, **kwargs)
    possibility_list = fitness_list / fitness_list.sum()
    return possibility_list


def fitness_MMT(sequence: ndarray, index: int, compare_graph: ndarray) -> float:
    """
    :param sequence: 选择的个体（这里是测试用的一个排序）
    :param index: 前面的几个执行完后，全部代码已经被覆盖
    :param compare_graph: 时间向量
    :return:
    """
    ans = 0
    for i in sequence[:index]:
        ans += compare_graph[i - 1]
    return ans


def fitness_APBC(sequence: ndarray, compare_graph: ndarray) -> float:
    """
    代码的块覆盖率适应度函数
    :param sequence:选择的个体（这里是测试用的一个排序）
    :param compare_graph: 测试用例默认编号代码块的关系矩阵（n*m，m为代码块，n为测试用例的数量)
    :return:输出评价指标的值
    """
    ans = 0
    n, m = compare_graph.shape
    sorted = []
    for i in sequence:
        sorted.append(compare_graph[i - 1])
    snp = np.array(sorted)
    min_index = 1
    for col_id in range(m):
        list = snp[:, col_id]
        flag = False
        for i, dom in enumerate(list):
            if dom:
                flag = True
                ans += i + 1
                min_index = max(min_index, i)
                break
        if not flag:
            ans += n

    return 1 - ans / (n * m) + 1 / (2 * n), min_index


def fitness_APFDc(sequence: ndarray, compare_graph: ndarray, time_graph: ndarray) -> float:
    """
    代码的块覆盖率适应度函数
    :param sequence:选择的个体（这里是测试用的一个排序）
    :param compare_graph: bug覆盖矩阵（n*m，m为bug数量，n为测试用例的数量)
    :param time_graph: 时间向量（n*1)
    :return:输出评价指标的值
    """
    ans = 0
    n, m = compare_graph.shape
    sorted = []
    for i in sequence:
        sorted.append(compare_graph[i - 1])
    snp = np.array(sorted)
    min_index = 1
    for col_id in range(m):
        list = snp[:, col_id]
        flag = False
        temp = 0
        for i, dom in enumerate(list):
            if not dom:
                temp += time_graph[i]
                min_index = max(min_index, i)
            else:
                flag = True
                temp = 1 - temp
                break
        if flag:
            ans = temp - time_graph[i] / 2
    return ans


class TSP1(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2, Dim=None, code_graph1=None, code_graph2=None, code_graph3=None, timeMMS=None):  # M : 目标维数；Dim : 决策变量维数
        name = 'TSP1'  # 初始化name（函数名称，可以随意设置）
        maxormins = [-1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = np.array([1] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1] * Dim  # 决策变量下界 P 编码，上界与DIM相同
        ub = [Dim] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self._code_graph1 = code_graph1
        self._code_graph2 = code_graph2
        self._code_graph3 = code_graph3
        self._timeMMS = timeMMS
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        if self.M == 4:
            f1_list = []
            f2_list = []
            f3_list = []
            f4_list = []
            for var in Vars:
                f1, min_index = fitness_APBC(var, compare_graph=self._code_graph1)
                f2, min_index = fitness_APBC(var, compare_graph=self._code_graph2)
                f3, min_index = fitness_APBC(var, compare_graph=self._code_graph3)
                f4 = fitness_MMT(var, min_index, self._timeMMS)
                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                f4_list.append(f4)
            f1 = np.array(f1_list).reshape(-1, 1)
            f2 = np.array(f2_list).reshape(-1, 1)
            f3 = np.array(f3_list).reshape(-1, 1)
            f4 = np.array(f4_list).reshape(-1, 1)
            return np.hstack([f1, f2, f3, f4])
        else:
            f1_list = []
            f4_list = []
            for var in Vars:
                f1, min_index = fitness_APBC(var, compare_graph=self._code_graph1)
                f4 = fitness_MMT(var, min_index, self._timeMMS)
                f1_list.append(f1)
                f4_list.append(f4)
            f1 = np.array(f1_list).reshape(-1, 1)
            f4 = np.array(f4_list).reshape(-1, 1)
            return np.hstack([f1, f4])

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        uniformPoint, ans = ea.crtup(self.M, 10000)  # 生成10000个在各目标的单位维度上均匀分布的参考点
        referenceObjV = uniformPoint / 2
        return referenceObjV
