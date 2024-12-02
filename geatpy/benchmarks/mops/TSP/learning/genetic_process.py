import logging
import random
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import pandas as pd
import os



def calculate_metric(metric_type: str, sequence: ndarray, compare_graph: ndarray) -> float:
    """
    评价指标的计算，需要将测试套件在新构建的项目中运行之后才能得到这个值，执行后会有一个测试用例序号与故障之间的矩阵图
    :param metric_type:评价指标的类型
    :param sequence:选择的个体（这里是测试用的一个排序）
    :param compare_graph: 测试用例默认编号与故障的关系矩阵（n*m，m为故障数，n为测试用例的数量)
    :return:输出评价指标的值
    """
    if metric_type != "APFD":
        raise "该参数指标不可用"
    ans = 0
    n, m = compare_graph.shape
    leave_m = m
    for order, index in enumerate(sequence):
        value = order + 1
        bug_info = compare_graph[index]
        for i, v in enumerate(bug_info):
            if leave_m == 0:
                break
            if v:
                ans += value
                leave_m -= 1
                compare_graph[i, :] = False
    ans = 1 - ans / (n * m) + 1 / (2 * n)
    return ans


def get_graph(name: str, version: str, _type: str) -> ndarray:
    """
    获取graph图并转为ndarray
    :param name: 项目名字
    :param version: 版本
    :param _type: 获取的覆盖矩阵类型
    :return: 返回的覆盖矩阵
    """
    _type = _type + "Graph"
    df = pd.read_csv(r"/home/ldf/MOEA/database/open_{}_cover/{}_{}_{}.csv".format(name, name, _type, version))
    return np.array(df.values[:, 1:-1])

def get_MMS(name: str, version: str):
    df = pd.read_csv(r"/home/ldf/MOEA/database/open_{}_cover/{}_{}_{}.csv".format(name, name, "timeMMS", version))
    na =  np.array(df.values[:, :])
    na =  np.array(na[:,1])
    na =  na / na.sum()
    return na/na.sum()



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


def selection(population: ndarray, select_type: str, fitness_fun, **kwargs) -> ndarray:
    """
    选择
    :param population: Population
    :param select_type: 选择的方法
    :param fitness_fun: 适应度函数
    :return: 输出两个被选中的对象
    """
    if select_type != "Wheel":
        raise "不支持该选择算法"
    possibility_list = fitness_calculate(fitness_fun, population, **kwargs)
    possibility_cumsum = np.cumsum(possibility_list)
    # 选第一个
    alpha = random.random()
    choose_index = population.shape[0]
    for i, v in enumerate(possibility_cumsum):
        if alpha < v:
            choose_index = i
            break
    # 选第二个，直到选择一个跟第一个不一样的
    choose_index2 = choose_index
    while choose_index2 == choose_index:
        alpha = random.random()
        choose_index2 = population.shape[0]
        for i, v in enumerate(possibility_cumsum):
            if alpha < v:
                choose_index2 = i
                break
    return np.array([population[choose_index], population[choose_index2]])


def mutation(selected_parents: ndarray, rate: float = 0.1):
    """
    突变
    :param selected_parents: 选择的父类，每一行为一个排序
    :param rate: 突变的比例
    :return:
    """
    _num, _coding_num = selected_parents.shape
    for selected_one in selected_parents:
        if random.random() < rate:
            candidate_index_list = random.sample([_ for _ in range(_coding_num)], int(_coding_num / 3))
            candidate_index_list_shuffle = candidate_index_list.copy()
            random.shuffle(candidate_index_list_shuffle)
            first_id = candidate_index_list[0]  # 第一个id需要被保存
            for aid, bid in zip(candidate_index_list, candidate_index_list_shuffle):
                selected_one[aid] = selected_one[bid]
            selected_one[candidate_index_list_shuffle[-1]] = selected_one[first_id]  # 最后一个需要替换


def cross_over(selected_parents: ndarray, is_elite: int, choose_number: int, fitness_fun, **kwargs) -> ndarray:
    """
     交叉过程
     :param selected_parents: 被选择的两个父代
     :param is_elite: 是否经营策略 0 不使用， 1 随机选择父子， 2 绝对精英
     :param choose_number: 固定不变的因子数量
     :param fitness_fun: 适应度函数
     :return: 返回两个子代
     """
    parents_num, gene_len = selected_parents.shape
    if choose_number >= gene_len:
        raise "选择的数量过多"
    if parents_num != 2:
        raise "父代数量有误"
    temp = [i for i in range(gene_len)]
    selected_list = random.sample(temp, choose_number)
    a = np.array(selected_parents[0])
    b = np.array(selected_parents[1])
    temp_a = a.copy()
    temp_b = b.copy()
    new_a = np.zeros((gene_len,), int)
    new_b = np.zeros((gene_len,), int)
    crossover_sub(a, gene_len, new_a, selected_list, temp_b)
    crossover_sub(b, gene_len, new_b, selected_list, temp_a)
    children = np.vstack((new_a.reshape(1, gene_len), new_b.reshape(1, gene_len)))
    candidates = np.vstack((children, a, b))  # 把父代与子代一起放入候选集
    # print(candidates)
    if is_elite == 0:
        return children
    elif is_elite == 1:
        temp = random.sample((0, 1, 2, 3), 2)
        return candidates[[temp[0], temp[1]], :]
    else:
        temp = np.array([[_, fitness_fun(candidate, **kwargs)] for (_, candidate) in enumerate(candidates)])
        choose_list = temp[temp[:, 1].argsort()]  # 获取candidates的增序排序
        return candidates[[int(choose_list[-1][0]), int(choose_list[-2][0])], :]


def crossover_sub(a, gene_len, new_a, selected_list, temp_b):
    """
    交叉过程生成一个子代的子函数
    :param a:
    :param gene_len:
    :param new_a:
    :param selected_list:
    :param temp_b:
    :return:
    """
    for i in selected_list:
        value = a[i]
        new_a[i] = value
        for j, v in enumerate(temp_b):
            if v == value:
                temp_b[j] = -1
    new_index = 0
    compare_index = 0
    while new_index < gene_len and compare_index < gene_len:
        if new_a[new_index] == 0 and temp_b[compare_index] != -1:
            new_a[new_index] = temp_b[compare_index]
            new_index += 1
            compare_index += 1
        elif new_a[new_index] == 0:
            compare_index += 1
        elif temp_b[compare_index] != -1:
            new_index += 1
        else:
            new_index += 1
            compare_index += 1


def fitness_f1(x: ndarray):
    """
    测试用的目标函数，实际的目标函数需要调用代码覆盖矩阵表进行计算
    :param x:
    :return:
    """
    return x.sum()


def fitness_APBC(sequence: ndarray, compare_graph: ndarray) -> float:
    """
    代码的块覆盖率适应度函数
    :param sequence:选择的个体（这里是测试用的一个排序）
    :param compare_graph: 测试用例默认编号代码块的关系矩阵（n*m，m为代码块，n为测试用例的数量)
    :return:输出评价指标的值
    """
    ans = 0
    n, m = compare_graph.shape
    leave_m = m
    for gene_order, index in enumerate(sequence):
        value = gene_order + 1
        code_info = compare_graph[index]
        for i, v in enumerate(code_info):
            if leave_m == 0:
                break
            if v:
                ans += value
                leave_m -= 1
                compare_graph[:, i] = False
    ans = 1 - ans / (n * m) + 1 / (2 * n)
    return ans


def init(project_name, version):
    """
    初始化比较数据，bug图和代码覆盖图
    :return: bug图和代码覆盖图
    """
    _bug_graph = get_graph(project_name, version, "bug")
    _classes_graph = get_graph(project_name, version, "classes")
    _blocks_graph = get_graph(project_name, version, "blocks")
    _timeMMS = get_MMS(project_name, version)
    return _bug_graph, _blocks_graph, _classes_graph, _timeMMS


def init_population(p_size, test_num):
    order = [_ for _ in range(test_num)]
    p_pre = []
    for _ in range(p_size):
        p_pre.append(random.sample(order, test_num))
    return np.array(p_pre)


if __name__ == '__main__':
    """利用代码覆盖矩阵实现的测试用例排序"""
    P_SIZE = 20  # 种群数量
    ITERATION = 20  # 迭代次数
    PROJECT_NAME = "ant"  # 项目名称
    VERSION = "v7"  # 版本号
    GRAPH_TYPE = "classes"  # 图类型
    bug_graph, code_graph = init(PROJECT_NAME, VERSION, GRAPH_TYPE)
    TEST_NUM, _ = bug_graph.shape
    P = init_population(P_SIZE, TEST_NUM)
    metric_list = []
    for i in range(ITERATION):
        metric_list.append(max(calculate_metric("APFD", p, bug_graph) for p in P))
        logging.warning("第{}次迭代".format(i))
        children_list = []
        for _ in range(int(P_SIZE / 2)):
            parents = selection(P, "Wheel", fitness_APBC, compare_graph=code_graph)
            children_list.append(cross_over(parents, 2, int(TEST_NUM / 2), fitness_APBC, compare_graph=code_graph))
        mutation(P)
        P = np.vstack(children_list)

    plt.plot([i for i in range(ITERATION)], metric_list)
    plt.show()
