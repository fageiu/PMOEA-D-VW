from numpy import ndarray
import numpy as np


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


if __name__ == '__main__':
    graph = np.zeros((3, 4), bool)
    graph[0, 0] = True
    graph[0, 1] = True
    graph[1, 1] = True
    graph[1, 2] = True
    graph[2, 2] = True
    graph[2, 3] = True
    se = np.array([[0, 2, 1], [0, 1, 2]])
    print(max([calculate_metric("APFD", se[0], graph) for i in se]))
