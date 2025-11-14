import os

import sys
def main(row,col,path):
    row = int(row)
    col = int(col)
    folder_path = path

    all_entries = os.listdir(folder_path)

    file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(folder_path, entry))]
    file_name_dict = {}

    for filename in file_names:
        if ".out" in filename:
            continue
        filename = filename[:-4]
        file = filename.split("_")
        rows = int(file[-2][3:])
        cols = int(file[-1][3:])
        file_name_dict[(rows,cols)] = filename
    print("filename={}".format(file_name_dict[(row,col)]))
#main(sys.argv[1],sys.argv[2],sys.argv[3])

