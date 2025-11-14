from qchemfile import qchem_out_file,qchem_file,molecule
import matplotlib.pyplot as plt
import numpy as np
import os
import math as ma
import heapq

Hartree_to_kcal = 627.51
def locate_row_col(file_name_dict,row,col):
    row = int(row)
    col = int(col)
    return file_name_dict[(row,col)]
def generate_addtional_suffix(i, j, p, q):
    return "_" + str(i) + "_" + str(j) + "_row" + str(p) + "_col" + str(q)




def read_all_xyz(path,suffix):
    all_entries = os.listdir(path)
    file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(path, entry))]
    mol_dict = {}
    for filename in file_names:
        if ".out" in filename or ".ref" in filename:
            continue

        mol = molecule()
        mol.read_xyz(path+filename)

        mol_dict[mol.filename.split("/")[-1][:-len(suffix)]] = mol
    return mol_dict

class scan_2d(object):
    def __init__(self,row_max,col_max):
        self.row_max = row_max
        self.col_max = col_max
        self.process_array = np.zeros((row_max,col_max))
        self.check_array = np.zeros((row_max,col_max))
        self.ref = qchem_file()
        self.new_inp_dict = {}
        self.job_dict = {}
        self.new_inp_path = ""
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.prefix_name = ""
        self.path = ""
        self.row_start = 0
        self.col_start = 0
        self.row_distance = 0
        self.col_distance = 0
        self.row_mix_list = []
        self.col_mix_list =[]
        self.row_type = "r12"
        self.col_type = "r12mr34"
        self.out_path = "./new_input/"
    def init_ref(self,ref_path,ref_filename,default=True):
        if default:
            self.ref.molecule.check = True
            self.ref.opt2.check = True
        self.ref_path = ref_path
        self.ref_filename = ref_filename
        self.ref.read_from_file(ref_path + ref_filename)
    def read_row_tot_ene_jobs(self):
        r = 1
        for row in range(1, self.row_max + 1):
            c = 1
            row_filename = self.path + "{}_row{}.inp.out".format(self.prefix_name, row)
            row_file = qchem_out_file()
            row_file.read_multiple_jobs_out(row_filename)
            for jobs in row_file.out_texts:
                opt_job = qchem_out_file()
                opt_job.read_opt_from_file("", out_text=jobs)
                if opt_job.opt_converged:
                    self.process_array[r - 1, c - 2] = opt_job.return_final_molecule_energy() * Hartree_to_kcal
                    self.check_array[r - 1, c - 2] = True
                    self.job_dict[(r,c-1)] = opt_job
                c += 1
            r += 1

        min_energy = np.min(self.process_array)
        self.process_array -= min_energy
        for row in range(self.row_max):
            for col in range(self.col_max):
                if self.process_array[row, col] == -min_energy:
                    self.process_array[row, col] = 0
        return self.process_array

    def plot_2d_scan(self,array,path=None,max_ene=None):
        plt.imshow(array, cmap="GnBu")
        for row in range(self.row_max):
            for col in range(self.col_max):
                plt.text(col - 0.25, row, "{}".format(array[row, col].round(1)))
        plt.yticks(np.arange(0, self.row_max, 2), np.around(np.arange(1.5, 3.9, 0.4), 2), fontsize=15)
        plt.xticks(np.arange(0, self.col_max, 2), np.around(np.arange(-2.0, 2.0, 0.4), 2), fontsize=15)
        plt.xlabel(self.xlabel, fontsize=15)
        plt.ylabel(self.ylabel, fontsize=15)
        plt.title(self.title, fontsize=15)
        plt.colorbar()
        # === MEP ===
        if path:
            y_coords, x_coords = zip(*path)
            print(y_coords,x_coords)
            plt.plot(x_coords, y_coords, color='red', linewidth=2.5, label='Minimax Path')
            plt.scatter(x_coords, y_coords, color='red', s=20)

            # find the max energy point
            max_idx = np.argmax([array[i, j] for (i, j) in path])
            max_point = path[max_idx]

            plt.text(max_point[1] + 0.4, max_point[0]+0.5, fr"Max $\Delta E$ = {(max_ene).round(2)}",
                     color='black', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))


        plt.show()
    def generate_inp_dict_from_job_dict(self):
        row_max = self.row_max
        col_max = self.col_max
        for row in range(row_max):
            for col in range(col_max):
                row_dis = round(self.row_start + self.row_distance * row, 2)
                col_dis = round(self.col_start + self.col_distance * col, 2)
                if (row+1,col+1) in self.job_dict.keys():
                    job = self.job_dict[(row+1,col+1)]
                    inp = qchem_file()
                    inp.molecule.check = True
                    inp.opt2.check = True
                    inp.read_from_file(self.ref_path + self.ref_filename)
                    inp.molecule.carti = job.return_final_molecule_carti()
                    inp.opt2.modify_r12(0, row_dis)
                    inp.opt2.modify_r12mr34(0, col_dis)
                    self.new_inp_dict[(row+1,col+1)] = inp


    def write_new_scan_2d_inp(self,self_generate=True): #should have finished reading output
        row_max = self.row_max
        col_max = self.col_max
        if self_generate:
            self.generate_inp_dict_from_job_dict()
        for row in range(row_max):
            for col in range(col_max):
                if self.check_array[row,col] == 0:
                    continue
                row_dis = round(self.row_start + self.row_distance * row, 2)
                col_dis = round(self.col_start + self.col_distance * col, 2)
                print()
                out_file = open(self.out_path+"{}_{}_{}_row{}_col{}.inp".format(self.prefix_name, row_dis,col_dis,row + 1,col+1), "w")
                inp = self.new_inp_dict[(row+1,col+1)]
                text = inp.molecule.return_output_format()+inp.remain_texts+inp.opt2.return_output_format()
                out_file.write(text)

    def return_2d_scan_ene_seperate_map(self,converge_only=True, optimizer="default",process_array=np.zeros((1,1))):
        path = self.path
        row_max = self.row_max
        col_max = self.col_max
        row_col_list, file_name_dict = self.return_row_cow_list()
        check_array = np.zeros((row_max, col_max))

        if process_array.all() == 0 :
            process_array = np.zeros((row_max, col_max))
        filename = locate_row_col(file_name_dict, 1, 1) + ".inp.out"
        ref_qof = qchem_out_file()
        ref_qof.read_opt_from_file(path + filename)
        for row_col in row_col_list:
            filename = locate_row_col(file_name_dict, row_col[0], row_col[1]) + ".inp.out"
            qof = qchem_out_file()
            qof.optimizer = optimizer
            qof.read_opt_from_file(path + filename)
            if qof.opt_converged:
                ene = (qof.return_final_molecule_energy()) * 627.51
                process_array[row_col[0] - 1, row_col[1] - 1] = ene
                check_array[row_col[0] - 1, row_col[1] - 1] = True
            if converge_only==False and qof.geom_have_energy==1 :
                ene = (qof.return_final_molecule_energy()) * 627.51
                process_array[row_col[0] - 1, row_col[1] - 1] = ene
                check_array[row_col[0] - 1, row_col[1] - 1] = True
        min_energy = np.min(process_array)
        process_array -= min_energy
        for row in range(row_max):
            for col in range(col_max):
                if process_array[row, col] == -min_energy:
                    process_array[row, col] = 1e-10
        self.check_array = check_array
        self.process_array = process_array
        return process_array
    def fix_process_array(self,process_array):
        check_array=self.check_array
        row_max=self.row_max
        col_max=self.col_max
        for row in range(row_max):
            for col in range(col_max):
                if process_array[row, col] != 1e-10:
                    check_array[row, col] = True
                    continue
                else:
                    tem_value = 0
                    tem_n = 0
                    if check_array[row - 1, col] == True:
                        tem_n += 1
                        tem_value += process_array[row - 1, col]
                    if check_array[row, col - 1] == True:
                        tem_n += 1
                        tem_value += process_array[row, col - 1]
                    if row + 1 < row_max and check_array[row + 1, col] == True:
                        tem_n += 1
                        tem_value += process_array[row + 1, col]
                    if col + 1 < col and check_array[row, col + 1] == True:
                        tem_n += 1
                        tem_value += process_array[row, col + 1]

                    process_array[row, col] = tem_value / tem_n
                    check_array[row, col] = True
        self.process_array = process_array
        self.check_array = check_array
        return process_array
    def minimax_shortest_path(self, start, end):
        process_array = self.process_array
        rows, cols = process_array.shape
        visited = np.full((rows, cols), False)
        max_energy = np.full((rows, cols), np.inf)
        path_len = np.full((rows, cols), np.inf)
        parent = np.full((rows, cols, 2), -1, dtype=int)

        heap = [(process_array[start], 0, start)]  # (max_energy_so_far, path_length, (i,j))
        max_energy[start] = process_array[start]
        path_len[start] = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while heap:
            cur_max, cur_len, (i, j) = heapq.heappop(heap)
            if visited[i, j]:
                continue
            visited[i, j] = True

            if (i, j) == end:
                path = []
                while (i, j) != (-1, -1):
                    path.append((i, j))
                    i, j = parent[i, j]
                return path[::-1], cur_max-process_array[start]

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    new_max = max(cur_max, process_array[ni, nj])
                    new_len = cur_len + 1

                    # update condition: smaller max energy, or same max but shorter path
                    if (new_max < max_energy[ni, nj]) or (
                            np.isclose(new_max, max_energy[ni, nj]) and new_len < path_len[ni, nj]
                    ):
                        max_energy[ni, nj] = new_max
                        path_len[ni, nj] = new_len
                        parent[ni, nj] = [i, j]
                        heapq.heappush(heap, (new_max, new_len, (ni, nj)))

        return None, np.inf
    def modify_opt2(self,qchem_input_file,type,type_order,num):
        if type == "r12" :
            qchem_input_file.opt2.modify_r12(type_order,num)
        elif type == "r12mr34":
            qchem_input_file.opt2.modify_r12mr34(type_order,num)
    def generate_inp(self):
        # initial variables
        ref_qchem_file = qchem_file()
        ref_qchem_file.opt2.check = True
        ref_qchem_file.read_from_file(self.ref_path + self.ref_filename)
        qchem_file_reference_name_without_suffix = self.ref_filename.split(".")[0]
        row_start = self.row_start
        row_distance = self.row_distance
        row_max = self.row_max
        col_start = self.col_start
        col_distance = self.col_distance
        col_max = self.col_max

        for row in range(row_max):
            row_dis = round(row_start + row_distance*row,2)
            self.modify_opt2(ref_qchem_file,self.row_type,0,row_dis)
            for col in range(col_max):
                col_dis = round(col_start + col_distance*col,2)
                additional_suffix = "_{}_{}_row{}_col{}".format(row_dis,col_dis,row+1,col+1)
                if self.row_type == self.col_type:
                    self.modify_opt2(ref_qchem_file, self.col_type, 1, row_dis)
                else:
                    self.modify_opt2(ref_qchem_file, self.col_type, 0, col_dis)
                new_file = open(self.new_inp_path + qchem_file_reference_name_without_suffix + additional_suffix + ".inp", "w")
                new_file.write(ref_qchem_file.remain_texts + ref_qchem_file.opt2.return_output_format())


    def return_row_cow_list(self):
        output_path = self.out_path
        row_max = self.row_max
        col_max = self.col_max
        prefix = self.prefix_name
        row_cow_list = []
        all_entries = os.listdir(output_path)

        file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(output_path, entry))]
        file_name_dict = {}
        for filename in file_names:
            if prefix not in filename :
                continue
            filename = filename.split(".")[0]
            file = filename.split("_")
            row = int(file[-2][3:])
            col = int(file[-1][3:])
            if row > row_max or col > col_max:
                continue
            row_cow_list.append([row, col])
            file_name_dict[(row, col)] = filename
        return row_cow_list, file_name_dict
    def read_geom_from_converged_jobs_and_generate_new_input(self, write_method="single"):
        out_path = self.out_path
        row_col_list, file_name_dict = self.return_row_cow_list()


def run_3119():
    ID3119 = scan_2d(12,20)

    ID3119.ref_filename = "ID3119.ref"
    ID3119.ref_path = "ref/"
    ID3119.prefix_name = "ID3119"
    ID3119.new_inp_path = "tem_input/"
    ID3119.row_type = "r12"
    ID3119.col_type = "r12mr34"
    ID3119.row_start = 1.5
    ID3119.col_start = -2.0
    ID3119.row_distance = 0.2
    ID3119.col_distance = 0.2
    ID3119.generate_inp()
#run_3119()



