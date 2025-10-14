from qchemfile import qchem_out_file, qchem_file, molecule
import matplotlib.pyplot as plt
import numpy as np
import os
import math as ma
import heapq

Hartree_to_kcal = 627.51  # Conversion factor from Hartree to kcal/mol

# Helper function: get the file name for a specific (row, col) coordinate
def locate_row_col(file_name_dict, row, col):
    return file_name_dict[(int(row), int(col))]

# Helper function: create a suffix string based on coordinate information
def generate_addtional_suffix(i, j, p, q):
    return f"_{i}_{j}_row{p}_col{q}"

# Read all XYZ files from a folder and return them as a dictionary of molecule objects
def read_all_xyz(path, suffix):
    all_entries = os.listdir(path)
    file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(path, entry))]
    mol_dict = {}
    for filename in file_names:
        if ".out" in filename or ".ref" in filename:
            continue
        mol = molecule()
        mol.read_xyz(path + filename)
        key = mol.filename.split("/")[-1][:-len(suffix)]
        mol_dict[key] = mol
    return mol_dict

# Class for managing a 2D potential energy surface scan
class scan_2d(object):
    def __init__(self, row_max, col_max):
        self.row_max = row_max
        self.col_max = col_max
        self.process_array = np.zeros((row_max, col_max))  # stores energy values
        self.check_array = np.zeros((row_max, col_max))    # marks valid points
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
        self.col_mix_list = []
        self.row_type = "r12"
        self.col_type = "r12mr34"
        self.out_path = "./new_input/"

    # Initialize the reference input file for generating other inputs
    def init_ref(self, ref_path, ref_filename, default=True):
        if default:
            self.ref.molecule.check = True
            self.ref.opt2.check = True
        self.ref_path = ref_path
        self.ref_filename = ref_filename
        self.ref.read_from_file(ref_path + ref_filename)

    # Read all jobs from multiple .inp.out files and store energies into array
    def read_row_tot_ene_jobs(self):
        r = 1
        for row in range(1, self.row_max + 1):
            c = 1
            row_filename = f"{self.path}{self.prefix_name}_row{row}.inp.out"
            row_file = qchem_out_file()
            row_file.read_multiple_jobs_out(row_filename)
            for jobs in row_file.out_texts:
                opt_job = qchem_out_file()
                opt_job.read_opt_from_file("", out_text=jobs)
                if opt_job.opt_converged:
                    self.process_array[r - 1, c - 2] = opt_job.return_final_molecule_energy() * Hartree_to_kcal
                    self.check_array[r - 1, c - 2] = True
                    self.job_dict[(r, c - 1)] = opt_job
                c += 1
            r += 1

        # Normalize by subtracting the minimum energy
        min_energy = np.min(self.process_array)
        self.process_array -= min_energy
        self.process_array[self.process_array == -min_energy] = 0
        return self.process_array

    # Plot the 2D energy scan with optional minimax path overlay
    def plot_2d_scan(self, array, path=None, max_ene=None):
        row_start = self.row_start
        col_start = self.col_start
        plt.imshow(array, cmap="GnBu")
        for row in range(self.row_max):
            for col in range(self.col_max):
                plt.text(col - 0.25, row, f"{array[row, col].round(1)}")
        plt.yticks(np.arange(0, self.row_max, 2), np.around(np.arange(self.row_start, self.row_start+self.row_distance*self.row_max, self.row_distance*2), 2), fontsize=15)
        plt.xticks(np.arange(0, self.col_max, 2), np.around(np.arange(self.col_start, self.col_start+self.col_distance*self.col_max,self.col_distance*2), 2), fontsize=15)
        plt.xlabel(self.xlabel, fontsize=15)
        plt.ylabel(self.ylabel, fontsize=15)
        plt.title(self.title, fontsize=15)
        plt.colorbar()

        if path:
            y_coords, x_coords = zip(*path)
            plt.plot(x_coords, y_coords, color='red', linewidth=2.5, label='Minimax Path')
            plt.scatter(x_coords, y_coords, color='red', s=20)
            max_idx = np.argmax([array[i, j] for (i, j) in path])
            max_point = path[max_idx]
            plt.text(max_point[1] + 0.4, max_point[0] + 0.5, fr"Max $\Delta E$ = {max_ene.round(2)}",
                     color='black', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
        plt.show()

    # Generate a dictionary of Q-Chem inputs from optimized jobs
    def generate_inp_dict_from_job_dict(self):
        for row in range(self.row_max):
            for col in range(self.col_max):
                if (row+1, col+1) in self.job_dict:
                    job = self.job_dict[(row+1, col+1)]
                    inp = qchem_file()
                    inp.molecule.check = True
                    inp.opt2.check = True
                    inp.read_from_file(self.ref_path + self.ref_filename)
                    inp.molecule.carti = job.return_final_molecule_carti()
                    inp.opt2.modify_r12(0, round(self.row_start + self.row_distance * row, 2))
                    inp.opt2.modify_r12mr34(0, round(self.col_start + self.col_distance * col, 2))
                    self.new_inp_dict[(row+1, col+1)] = inp

    # Write new .inp files for 2D scan
    def write_new_scan_2d_inp(self, self_generate=True):
        if self_generate:
            self.generate_inp_dict_from_job_dict()
        for row in range(self.row_max):
            for col in range(self.col_max):
                if not self.check_array[row, col]:
                    continue
                row_dis = round(self.row_start + self.row_distance * row, 2)
                col_dis = round(self.col_start + self.col_distance * col, 2)
                filename = f"{self.prefix_name}_{row_dis}_{col_dis}_row{row+1}_col{col+1}.inp"
                with open(self.out_path + filename, "w") as out_file:
                    inp = self.new_inp_dict[(row+1, col+1)]
                    text = inp.molecule.return_output_format() + inp.remain_texts + inp.opt2.return_output_format()
                    out_file.write(text)

    # Parse energy surface from .inp.out files and return normalized energy map
    def return_2d_scan_ene_seperate_map(self, converge_only=True, optimizer="default", process_array=np.zeros((1,1))):
        if process_array.all() == 0:
            process_array = np.zeros((self.row_max, self.col_max))
        check_array = np.zeros_like(process_array)
        row_col_list, file_name_dict = self.return_row_cow_list()
        ref_qof = qchem_out_file()
        ref_qof.read_opt_from_file(self.path + locate_row_col(file_name_dict, 1, 1) + ".inp.out")

        for row, col in row_col_list:
            filename = self.path + locate_row_col(file_name_dict, row, col) + ".inp.out"
            qof = qchem_out_file()
            qof.optimizer = optimizer
            qof.read_opt_from_file(filename)
            if qof.opt_converged or (not converge_only and qof.geom_have_energy):
                ene = qof.return_final_molecule_energy() * Hartree_to_kcal
                process_array[row - 1, col - 1] = ene
                check_array[row - 1, col - 1] = True

        min_energy = np.min(process_array)
        process_array -= min_energy
        process_array[process_array == -min_energy] = 1e-10

        self.check_array = check_array
        self.process_array = process_array
        return process_array

    # Interpolate missing points using neighbor averaging
    def fix_process_array(self, process_array):
        for row in range(self.row_max):
            for col in range(self.col_max):
                if process_array[row, col] != 1e-10:
                    self.check_array[row, col] = True
                    continue
                # Neighbor averaging
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = row+dr, col+dc
                    if 0 <= nr < self.row_max and 0 <= nc < self.col_max and self.check_array[nr, nc]:
                        neighbors.append(process_array[nr, nc])
                if neighbors:
                    process_array[row, col] = sum(neighbors)/len(neighbors)
                    self.check_array[row, col] = True
        self.process_array = process_array
        return process_array

    # Compute the shortest minimax path (least maximum energy) on 2D energy grid
    def minimax_shortest_path(self, start, end):
        process_array = self.process_array
        rows, cols = process_array.shape
        visited = np.full((rows, cols), False)
        max_energy = np.full((rows, cols), np.inf)
        path_len = np.full((rows, cols), np.inf)
        parent = np.full((rows, cols, 2), -1, dtype=int)
        heap = [(process_array[start], 0, start)]
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
                return path[::-1], cur_max - process_array[start]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    new_max = max(cur_max, process_array[ni, nj])
                    new_len = cur_len + 1
                    if new_max < max_energy[ni, nj] or (
                        np.isclose(new_max, max_energy[ni, nj]) and new_len < path_len[ni, nj]
                    ):
                        max_energy[ni, nj] = new_max
                        path_len[ni, nj] = new_len
                        parent[ni, nj] = [i, j]
                        heapq.heappush(heap, (new_max, new_len, (ni, nj)))
        return None, np.inf

    # Modify r12 or r12mr34 constraints in input
    def modify_opt2(self, qchem_input_file, type, type_order, num):
        if type == "r12":
            qchem_input_file.opt2.modify_r12(type_order, num)
        elif type == "r12mr34":
            qchem_input_file.opt2.modify_r12mr34(type_order, num)

    # Generate .inp files over a full 2D scan grid
    def generate_inp(self):
        ref = qchem_file()
        ref.opt2.check = True
        ref.read_from_file(self.ref_path + self.ref_filename)
        base_name = self.ref_filename.split(".")[0]
        for row in range(self.row_max):
            row_dis = round(self.row_start + self.row_distance * row, 2)
            self.modify_opt2(ref, self.row_type, 0, row_dis)
            for col in range(self.col_max):
                col_dis = round(self.col_start + self.col_distance * col, 2)
                if self.row_type == self.col_type:
                    self.modify_opt2(ref, self.col_type, 1, row_dis)
                else:
                    self.modify_opt2(ref, self.col_type, 0, col_dis)
                suffix = f"_{row_dis}_{col_dis}_row{row+1}_col{col+1}"
                with open(self.new_inp_path + base_name + suffix + ".inp", "w") as f:
                    f.write(ref.remain_texts + ref.opt2.return_output_format())

    # Return list of row/col indices and file name mapping
    def return_row_cow_list(self):
        file_names = os.listdir(self.out_path)
        row_col_list = []
        file_name_dict = {}
        for name in file_names:
            if self.prefix_name not in name:
                continue
            parts = name.split(".")[0].split("_")
            row = int(parts[-2][3:])
            col = int(parts[-1][3:])
            if row <= self.row_max and col <= self.col_max:
                row_col_list.append([row, col])
                file_name_dict[(row, col)] = "_".join(parts)
        return row_col_list, file_name_dict

# Example function to run a 2D scan setup
def run_3119():
    ID3119 = scan_2d(12, 20)
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

run_3119()
