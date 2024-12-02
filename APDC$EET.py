# 实验重做
import os
import pickle

import geatpy as ea  # import geatpy
import pandas as pd
from geatpy.benchmarks.mops.TSP.learning.genetic_process import init

project_name_list = ['ant', 'Derby', 'Jboss', 'jmeter', 'jtopas', 'xml-security']
version_list = ['v7', 'v7', 'v10', 'v7', 'v4', 'v4']

P_SIZE = 100  # 种群数量
ITERATION = 500  # 迭代次数
REDO = True
MAXGEN = 500  # 最大迭代次数
NIND = 100  # 基因长度

# 2个目标的TSP问题
if __name__ == '__main__':
    for project_name, version in zip(project_name_list, version_list):
        _bug_graph, _blocks_graph, _classes_graph, _timeMMS = init(project_name, version)
        print("loading {} down".format(project_name))
        TEST_NUM, _ = _classes_graph.shape
        problem = ea.benchmarks.TSP1(M=2, Dim=TEST_NUM, code_graph1=_bug_graph, timeMMS=_timeMMS)
        algorithm = ea.moea_MOEAD_VW_templet(problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1)
        dirname = "../resultRQ1/{}_{}M{}$APDC$EET{}_result".format(algorithm.name, problem.name, problem.M, project_name)
        # 判断该方法有没有执行过,没有执行过才去执行
        if REDO or not os.path.exists(dirname + "/optPop/algorithm_log.csv"):
            res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
                              dirName=dirname)
            df = pd.DataFrame(algorithm.log)
            # 将历史数据写入csv
            df.to_csv(dirname + "/optPop/algorithm_log.csv")
            # 将 res, algorithm数据写入pickle
            with open(dirname + "/res.pickle", 'wb') as f1, open(dirname + "/algorithm.pickle", 'wb') as f2:
                pickle.dump(res, f1)
                pickle.dump(algorithm, f2)
        else:
            print(dirname + "已经执行")
