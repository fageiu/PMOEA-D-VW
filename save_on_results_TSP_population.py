import os
import pickle
import re

import numpy as np
import pandas as pd
from geatpy.benchmarks.mops.TSP.TSP1 import fitness_APFDc
from geatpy.benchmarks.mops.TSP.learning.genetic_process import init

choose_methods = ['RVEA', 'ARMOEA', 'NSGA2', 'NSGA3', 'MOEAD-VW', 'MOEAD-URAW']
project_name_list = ['ant', 'Derby', 'Jboss', 'jmeter', 'jtopas', 'xml-security']
version_list = ['v7', 'v7', 'v10', 'v7', 'v4', 'v4']

aim_dir = r"../dealed_result/RQ1"


# 转储data信息
def resave_data_info():
    for project_name, version in zip(project_name_list, version_list):
        _bug_graph, _blocks_graph, _classes_graph, _timeMMS = init(project_name, version)
        for dir_name in os.listdir("../resultRQ1"):
            if dir_name == ".DS_Store":
                continue
            method_name, problem_name, _ = dir_name.split("_")
            if re.match(".*{}.*".format(project_name), problem_name) and (re.match(".*MOEAD-VW.*", method_name) or method_name == "MOEAD"):  # 将匹配的方法列出来
                dirname = r'../结果/{}'.format(dir_name)
                if os.path.getsize(dirname + "/res.pickle") > 0:
                    with open(dirname + "/res.pickle", 'rb') as f1, open(dirname + "/algorithm.pickle", 'rb') as f2:
                        res = pickle.load(f1)
                        algorithm = pickle.load(f2)
                        phen = res['optPop'].Phen
                        ObV = res['optPop'].ObjV
                        APFDc = []
                        print(len(phen))
                        for p in phen:
                            APFDc.append(fitness_APFDc(p, _bug_graph, _timeMMS))  # 获取APDFc值
                        ObV = np.hstack([ObV, np.array(APFDc).reshape(-1, 1)])
                        pd.DataFrame(phen).to_csv(aim_dir + "/Phen/{}.csv".format(dir_name))
                        frame = pd.DataFrame(ObV)
                        frame.to_csv(aim_dir + "/ObjV/{}.csv".format(dir_name))


if __name__ == '__main__':
    resave_data_info()
