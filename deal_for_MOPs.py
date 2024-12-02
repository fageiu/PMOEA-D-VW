import numpy as np
import geatpy as ea
import pandas as pd
import os
import logging
import pickle

m_dim = 5
m_dim2 = 2
d_dim = None

MAXGEN = 500  # 最大迭代次数
NIND = 100  # 基因长度


def _get_algorithms(_problem):
    _algorithms = []
    _algorithms.append(ea.moea_MOEAD_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_DE_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_archive_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_NSGA3_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_NSGA3_DE_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_NSGA2_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_NSGA2_DE_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_RVEA_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_PPS_MOEAD_DE_archive_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_VW_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_VW1_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_VW2_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_VW3_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_URAW_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_ARMOEA_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_EMOSA_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_M2M_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_AM2M_templet(_problem, ea.Population(Encoding='RI', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    return _algorithms


def _get_algorithms_for_TSP(_problem):
    _algorithms = []
    _algorithms.append(ea.moea_MOEAD_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_URAW_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOEAD_VW_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_NSGA2_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_NSGA3_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOPSO_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(ea.moea_MOBBO_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    _algorithms.append(
        ea.moea_MOCO_templet(_problem, ea.Population(Encoding='P', NIND=NIND), MAXGEN=MAXGEN, logTras=1))
    return _algorithms


if __name__ == '__main__':
    logging.basicConfig(filename="log/log.info")
    # UF1-2, DTLZ1-7, ZDT1-6
    problems = [ea.benchmarks.DTLZ1(M=m_dim, Dim=d_dim), ea.benchmarks.DTLZ2(M=m_dim, Dim=d_dim), ea.benchmarks.DTLZ3(M=m_dim, Dim=d_dim), ea.benchmarks.DTLZ4(M=m_dim, Dim=d_dim), ea.benchmarks.DTLZ5(M=m_dim, Dim=d_dim), ea.benchmarks.DTLZ6(M=m_dim, Dim=d_dim),
                ea.benchmarks.DTLZ7(M=m_dim, Dim=d_dim), ea.benchmarks.ZDT1(), ea.benchmarks.ZDT2(), ea.benchmarks.ZDT3(), ea.benchmarks.ZDT4(), ea.benchmarks.ZDT5(), ea.benchmarks.ZDT6(), ea.benchmarks.UF1(), ea.benchmarks.UF2()]

    # DTLZ系列，5、10、15个目标
    DTLZ_problems = [ea.benchmarks.DTLZ1(M=5), ea.benchmarks.DTLZ1(M=10), ea.benchmarks.DTLZ1(M=15), ea.benchmarks.DTLZ2(M=5), ea.benchmarks.DTLZ2(M=10), ea.benchmarks.DTLZ2(M=15), ea.benchmarks.DTLZ3(M=5), ea.benchmarks.DTLZ3(M=10), ea.benchmarks.DTLZ3(M=15), ea.benchmarks.DTLZ4(M=5),
                     ea.benchmarks.DTLZ4(M=10), ea.benchmarks.DTLZ4(M=15), ea.benchmarks.DTLZ5(M=5), ea.benchmarks.DTLZ5(M=10), ea.benchmarks.DTLZ5(M=15)]
    for problem in DTLZ_problems:
        # 构建算法
        algorithms = _get_algorithms(problem)
        for algorithm in algorithms[-2:]:
            # 求解
            dirname = "结果/{}_{}M{:0>2d}D{}_result".format(algorithm.name, problem.name, problem.M, problem.Dim)
            # 判断该方法有没有执行过,没有执行过才去执行
            if not os.path.exists(dirname + "/optPop/algorithm_log.csv"):
                logging.warning("【START】正在用算法{}执行{}问题".format(algorithm.name, problem.name))
                res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
                                  dirName=dirname)
                df = pd.DataFrame(algorithm.log)
                # 将历史数据写入csv
                df.to_csv(dirname + "/optPop/algorithm_log.csv")
                # 将 res, algorithm数据写入pickle
                with open(dirname + "/res.pickle", 'wb') as f1, open(dirname + "/algorithm.pickle", 'wb') as f2:
                    pickle.dump(res, f1)
                    pickle.dump(algorithm, f2)
                logging.warning("【END】算法{}执行{}问题结束,执行时间:{}".format(algorithm.name, problem.name, res['executeTime']))
            else:
                print(dirname + "已经执行")
