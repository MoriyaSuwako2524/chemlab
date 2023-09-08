


input_file = open("ori.txt", "r", encoding="utf-8").read().split("\n")
out = open("out.txt", "w", encoding="utf-8")


j = 4
def form(test1):
    for i in range(len(test1)):
        if test1[i].isdigit():
            continue
        else:
            test1 = test1[i:-1]
            break
    return test1

for i in input_file:
    new_form = ""
    list1 = form(i).split(";")

    name_list = list1[0]
    name_list = name_list.split(",")
    while "" in name_list:
        name_list.remove("")
    while " " in name_list:
        name_list.remove(" ")
    if name_list == []:
        continue
    if len(name_list) <=6:
        for l in name_list:
            new_form += l+". "
    else:
        for l in range(len(name_list)):
            if l <= 2:
                new_form += name_list[l]
        new_form += ",et al."
    title = list1[1].split("https://doi")[0]
    new_form += title
    DOI =(list1[1].split("https://doi"))[1]
    for p in range(len(DOI)):
        if "1" == DOI[p]:
            if "0" == DOI[p+1]:
                DOI = DOI[p:-1]+DOI[-1]
                break
    new_form += "[PMID: DOI:{} ]".format(DOI)
    out.write(str(j)+" "+new_form+"\n")
    j+=1



