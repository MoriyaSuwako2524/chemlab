'''
该文件处理Gaussian等计算化学软件输出的理论ECD光谱文件
inp输入对应的文件名
输出ECDout.csv为输出
'''



import numpy as np
import matplotlib.pyplot as plt
inp = open("COM24-THF.txt","r").read().split("\n")
out = open("spectra.csv","w")




count = 0
for i in range(len(inp)):
    data = inp[i].split(" ")
    if len(data) == 1:
        data = inp[i].split("\t")

    if data == [""]:
        continue
    for k in data:
        out.write(k+",")
    out.write("\n")








