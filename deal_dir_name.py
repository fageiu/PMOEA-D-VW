import os
import re
import shutil

# 默认的算法名MOEA/D会导致名字错误，这里进行了文件夹处理
def rename():
    # 设置MOEA文件夹路径
    moea_folder_path = "resultRQ3/MOEA"  # 假设 MOEA 文件夹在当前工作目录中
    parent_folder = os.path.dirname(moea_folder_path)

    # 遍历 MOEA 文件夹下的子文件夹
    for folder_name in os.listdir(moea_folder_path):
        old_path = os.path.join(moea_folder_path, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(old_path):
            # 加上 "MOEA" 前缀，并构造新路径
            new_folder_name = f"MOEA{folder_name}"
            new_path = os.path.join(parent_folder, new_folder_name)

            # 移动并重命名文件夹
            shutil.move(old_path, new_path)
            print(f"Moved and renamed '{folder_name}' to '{new_folder_name}'")

    # 删除原 MOEA 文件夹
    if os.path.isdir(moea_folder_path):
        shutil.rmtree(moea_folder_path)
# M5改成M05这样图片里展示的顺序才是对的

# TCP写错了，全部改成TSP

# RQ3里面名字修改
if __name__ == '__main__':
    rename()
    # for i in os.listdir(r"/Users/jiachengyin/论文/resultRQ3"):
    #     a, b, c, d = i.split("_")
    #     y = "Num{:0>2}_{}_{}_{}".format(c, a, b, d)
    #     os.rename("/Users/jiachengyin/论文/resultRQ3/" + i, "/Users/jiachengyin/论文/resultRQ3/" + y)
