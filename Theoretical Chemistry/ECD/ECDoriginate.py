'''
处理CD色谱(通常是ECD，电子激子光谱)文件源，提取并绘图。
通过solv 作为溶剂文件，消除基线
通过filename输入ECD光谱输出文件
输出ECDout.csv
'''
import matplotlib.pyplot as plt
solv = "CH3OH.txt"
filename = "LT-1453-332.txt"#"LT-1453-332.txt","LT-1453-349.txt","LT-1458-587.txt","CH3OH.txt"
inp = open(filename,"r").read().split("\n")
solvfile = open(solv,"r").read().split("\n")
X_list = []
Y1_list = []
X_solv_list = []
Y_solv_list = []
K = 0
J= 0
'''
YUNITS	CD[mdeg]
Y2UNITS	HT[V]
Y3UNITS	ABSORBANCE
'''
for i in range(len(inp)):


    if "XYDATA" in inp[i]:
        K = 1
        continue
    if K == 0:
        continue
    elif K==1:
        data = inp[i].split("\t")
        X_list.append(float(data[0]))
        Y1_list.append(float(data[1])/3.2982)
for i in range(len(solvfile)):


    if "XYDATA" in solvfile[i]:
        J= 1
        continue
    if J == 0:
        continue
    elif J==1:
        data = solvfile[i].split("\t")
        X_solv_list.append(float(data[0]))
        Y_solv_list.append(float(data[1])/3.2982)
'''
print(X_list)
plt.plot(X_list, Y1_list, "b--",)
plt.xlabel("wavelength(nm)")
plt.ylabel("CD(mdeg)")
plt.title(filename)
plt.show()
'''
output = open("ECDout.csv","w")
for i in range(len(X_list)):
    if X_list[i] == X_solv_list[i]:
        output.write(str(X_list[i])+","+str(Y1_list[i])+"\n")