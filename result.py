import os
import pickle
import re

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

# 数据提取并保存到结果中
if __name__ == '__main__':
    ans_map = {}
    for dir_name in os.listdir("../结果"):
        if dir_name == ".DS_Store" or dir_name == "dealed_result":
            continue
        method_name, problem_name, _ = dir_name.split("_")
        if re.match(".*TSP.*", problem_name) and method_name == "MOEAD-VW":  # 将匹配的方法列出来
            dirname = r'../resultRQ1/{}'.format(dir_name)
            print(dirname)
            if os.path.getsize(dirname + "/res.pickle") > 0:
                with open(dirname + "/res.pickle", 'rb') as f1, open(dirname + "/algorithm.pickle", 'rb') as f2:
                    res = pickle.load(f1)
                    algorithm = pickle.load(f2)
                    dealed_problem_name = re.sub("TSP1", "", problem_name)
                    ls = dealed_problem_name.split("EET")
                    if len(ls) > 1:
                        l1, l2 = ls
                    else:
                        l1, l2 = dealed_problem_name.split("ALL")
                        l2 = l2 + "ALL"
                    key = l2 + l1
                    if not ans_map.get(key):
                        ans_map[key] = {}
                    df = pd.read_csv("../dealed_result/RQ1/ObjV/{}.csv".format(dir_name))
                    APFDc = df[df.columns[-1]].mean()
                    APBC = df[df.columns[1]].mean() if "APBC" in key else 0
                    APBC = df[df.columns[1]].mean() if "ALL" in key else APBC
                    APSC = df[df.columns[1]].mean() if "APSC" in key else 0
                    APSC = df[df.columns[2]].mean() if "ALL" in key else APSC
                    APDC = df[df.columns[1]].mean() if "APDC" in key else 0
                    APDC = df[df.columns[2]].mean() if "ALL" in key else APDC
                    EET = df[df.columns[2]].mean() if "ALL" not in key else df[df.columns[-2]].mean()
                    ans_map[key]['HV'] = res['hv']
                    ans_map[key]['Time'] = res['executeTime']
                    ans_map[key]['APFDc'] = APFDc
                    ans_map[key]['APBC'] = APBC
                    ans_map[key]['APSC'] = APSC
                    ans_map[key]['APDC'] = APDC
                    ans_map[key]["EET"] = EET
    df = pd.DataFrame(ans_map)
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv("res/ans.csv")
