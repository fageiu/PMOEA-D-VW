import numpy as np
from numpy import ndarray
import pandas as pd


def get_graph(name: str, version: str, _type: str) -> ndarray:
    """
    获取graph图并转为ndarray
    :param name: 项目名字
    :param version: 版本
    :param _type: 获取的覆盖矩阵类型
    :return: 返回的覆盖矩阵
    """
    if _type != "timeMMS":
        _type = _type + "Graph"
    df = pd.read_csv(r"/home/ldf/MOEA/database/open_{}_cover/{}_{}_{}.csv".format(name, name, _type, version))
    return np.array(df.values[:, 1:-1])


if __name__ == '__main__':
    print(get_graph("ant", "v7", "bug").shape)
    print(get_graph("ant", "v7", "classes").shape)
