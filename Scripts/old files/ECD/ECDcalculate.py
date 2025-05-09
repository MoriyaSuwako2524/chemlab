'''
该文件处理Gaussian等计算化学软件输出的理论ECD光谱文件
inp输入对应的文件名
输出ECDout.csv为输出
'''




import matplotlib.pyplot as plt
inp = open("gau00001_tdsp_ecd.txt","r").read().split("\n")

ecd_exp = []
ecd_13 = []
ecd_17 = []
ecd_22 = []
ecd_31 = []
out = open("2.csv","w")
for i in range(len(inp)):
    if i == 0:
        continue
    j = inp[i].split(",")
    if j == [""]:
        continue
    if j[0] != "":
        ecd_exp.append([j[0],j[1]])
    if j[2] != "":
        ecd_22.append([j[2], j[3]])
    if j[4] != "":
        ecd_13.append([j[4], j[5]])
    if j[6] != "":
        ecd_17.append([j[6], j[7]])
    if j[10] != "":
        ecd_31.append([j[10], j[11]])
new_ecd = []
def linear_insert(ecd):
    ecd.sort()
    ecd_x = []
    ecd_y = []
    new_ecd = []
    for i in range(len(ecd)):
        ecd_x.append(float(ecd[i][0]))
        ecd_y.append(float(ecd[i][1]))

    k = 0

    for i in range(1001):
        j = 200.0+ 150*i/1000

        if j > ecd_x[-1]:
            new_ecd.append([j,0])
            continue
        if j < ecd_x[k]:
            new_ecd.append([j,0])
            continue
        if ecd_x[k] <= j <= ecd_x[k+1]:
            y = ecd_y[k]*(j-ecd_x[k])+ecd_y[k+1]*(ecd_x[k+1]-j)
        elif j:
            k+=1
            y = ecd_y[k] * (j - ecd_x[k]) + ecd_y[k + 1] * (ecd_x[k + 1] - j)
        new_ecd.append([j,y])
    return new_ecd
newecd_exp = linear_insert(ecd_exp)
newecd_13 = linear_insert(ecd_13)
newecd_17 = linear_insert(ecd_17)
newecd_22 = linear_insert(ecd_22)
newecd_31 = linear_insert(ecd_31)
def format_trans(ecd):
    ecd_x = []
    ecd_y = []
    for i in range(len(ecd)):
        ecd_x.append(float(ecd[i][0]))
        ecd_y.append(float(ecd[i][1]))
    return [ecd_x,ecd_y]
ecd_13 = format_trans(newecd_13)
ecd_31 = format_trans(newecd_31)
ecd_17 = format_trans(newecd_17)
ecd_22 = format_trans(newecd_22)
ecd_exp = format_trans(newecd_exp)


plt.plot(ecd_13[0],ecd_13[1],label="332-13")
plt.plot(ecd_17[0],ecd_17[1],label="332-17")
plt.plot(ecd_22[0],ecd_22[1],label="332-22")
plt.plot(ecd_31[0],ecd_31[1],label="332-31")
plt.plot(ecd_exp[0],ecd_exp[1],label="332-exp")
plt.legend()
plt.show()

#print(format_trans(linear_insert(ecd_exp)))
def output(ecd):
    for i in range(len(ecd)):
        out.write(str(ecd[i][0])+","+str(ecd[i][1])+"\n")



newecdout = []
'''
for i in range(len(newecd_31)):
    k = 1-i/len(newecd_31)
    print(newecd_17[i][0],newecd_31[i][0])
    newy = newecd_17[i][1]*k +newecd_31[i][1]*(1-k)
    out.write(str(newecd_17[i][0])+","+str(newy)+"\n")
'''






