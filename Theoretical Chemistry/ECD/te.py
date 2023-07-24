filename = "tran00004_tdsp_ecd.txt"
inp = open(filename,"r",encoding='utf-8').read().split("\n")
X_list = []
Y_list = []
K = 0
for i in range(len(inp)):

    if "DY/DX" in inp[i]:
        K = 1
        continue
    if K == 0:
        continue
    elif K==1:
        data = inp[i].split(" ")
        while "" in data:
            data.remove("")
        if data == []:
            continue
        X_list.append(float(data[0]))
        Y_list.append(float(data[1])/3.2982)

output = open("ECDout.csv","w")
for i in range(len(X_list)):
    if 400>= X_list[i] >=200:
        output.write(str(X_list[i])+","+str(Y_list[i])+"\n")