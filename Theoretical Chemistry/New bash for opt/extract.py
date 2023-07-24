import sys
opt_out = open("tem_opt_out.txt","r").read().split("opt.log")
ts_out = open("tem_opt_out.txt","r").read().split("ts.log")
ene_out = open("tem_ene_out.txt","r").read().split("ene.log")
output_file = open("summary.csv","w")
tem_ene_out = []
tem_opt_out = []
tem_name_ene = []
tem_name_opt = []
if len(opt_out) == 2:
    ts_out[0] = ts_out[0].split("\n")[1]
    opt_out = ts_out
for i in range(len(opt_out)):
    if opt_out[i] == "*_":
        opt_out.remove("*_")
        break



for i in range(len(opt_out)):
    if i == 0 :
        tem_name_opt.append(opt_out[i].split(" ")[-1])
        continue
    if opt_out[i].split(" ") == ["\n"]:
        continue
    tem_float = float(opt_out[i].split(" ")[-2])
    tem_opt_out.append(tem_float)
    tem_name_opt.append(opt_out[i].split(" ")[-1])


for i in range(len(ene_out)):
    if i == 0:
        tem_name_ene.append(ene_out[i].split(" ")[-1])
        continue
    if opt_out[i].split(" ") == ["\n"]:
        continue
    tem_float = float(ene_out[i].split(" ")[7])
    tem_ene_out.append(tem_float)
    tem_name_ene.append(ene_out[i].split(" ")[-1])
print(tem_name_opt)
print(tem_name_ene)
output_file.write("name,correct_to_G,exac_ene,total_G\n")
for i in range(len(tem_ene_out)):
    if tem_name_opt[i] == tem_name_ene[i]:
        output_file.write(str(tem_name_opt[i][0:-1])+","+str(tem_opt_out[i])+","+str(tem_ene_out[i])+","+str(tem_opt_out[i]+tem_ene_out[i])+"\n")