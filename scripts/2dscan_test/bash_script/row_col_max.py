import sys
import os
def main(path):
    folder_path = path

    all_entries = os.listdir(folder_path)

    file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(folder_path, entry))]
    row_list = []
    col_list = []
    for filename in file_names:
        if ".out" in filename:
            continue
        filename = filename[:-4]
        file = filename.split("_")

        row = file[-2][3:]
        col = file[-1][3:]
        row_list.append(int(row))
        col_list.append(int(col))

    max_row = max(row_list)
    max_col = max(col_list)
    print("max_row={}".format(max_row))
    print("max_col={}".format(max_col))
main(sys.argv[1])
