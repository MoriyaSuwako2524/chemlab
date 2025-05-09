import sys
import numpy as np

# Function to classify input files by their name (currently unused)
def judge(filename):
    if "tem1" in filename:
        return 1
    elif "tem2" in filename:
        return 2
    elif "tem3" in filename:
        return 3
    elif "Q.inp" in filename:
        return 0
    elif "G.inp" in filename:
        return 10
    elif "O.inp" in filename:
        return 20
    else:
        return -1

# Main class representing a Q-Chem input file
class qchem_file(object):
    def __init__(self):
        self.molecule = molecule()  # molecular geometry
        self.rem = rem()            # Q-Chem $rem section
        self.pcm = pcm()            # PCM solvent model
        self.opt = opt()            # optimization constraints
        self.opt2 = opt2()          # extended constraints
        self.remain_texts = ""      # stores any unclassified text

    def generate_inp(self, new_file_name):  # placeholder for file writing
        new_file = open(new_file_name,"w")

    # Read Q-Chem input from a file and parse relevant sections
    def read_from_file(self, filename):
        self.filename = filename
        file = open(filename, "r").read().split("\n")
        file_length = len(file)
        module_start = 0
        specifix_const = 0

        for i in range(file_length):
            line = file[i]
            if "$end" in line:
                if module_start == -1:
                    self.remain_texts += line + "\n\n"
                module_start = 0
                continue

            # Identify start of different sections
            elif "$molecule" in line:
                module_start = 1 if self.molecule.check else -1
                if module_start == -1:
                    self.remain_texts += "\n" + line + "\n"
                continue
            elif "$rem" in line:
                module_start = 2 if self.rem.check else -1
                if module_start == -1:
                    self.remain_texts += "\n" + line + "\n"
                continue
            elif "$opt" in line:
                if "$opt2" in line and self.opt2.check:
                    module_start = 3.5
                elif self.opt.check:
                    module_start = 3
                else:
                    module_start = -1
                    self.remain_texts += "\n" + line + "\n"
                continue
            elif "$" in line:
                module_start = -1
                self.remain_texts += "\n" + line + "\n"
                continue
            elif "@" in line:
                self.remain_texts += "\n" + line + "\n"
                continue

            # Parse each module according to context
            if module_start == -1:
                self.remain_texts += line + "\n"

            elif module_start == 1:  # $molecule
                content = line.split()
                if not content:
                    continue
                if "read" in content:
                    self.remain_texts += "\n$molecule\nread\n$end\n\n"
                elif len(content) == 2:
                    self.molecule.charge, self.molecule.multistate = content
                else:
                    self.molecule.carti.append(content)

            elif module_start == 2:  # $rem
                content = line.split()
                if not content:
                    continue
                self.rem.texts += line + "\n"
                if len(content) >= 3:
                    setattr(self.rem, content[0].lower(), content[2])

            elif module_start == 3:  # $opt
                content = line.split()
                if not content:
                    continue
                if "CONSTRAINT" in content:
                    specifix_const = 1
                    continue
                elif "ENDCONSTRAINT" in content:
                    specifix_const = 0
                    continue
                elif "FIXED" in content:
                    specifix_const = 2
                    continue
                elif "ENDFIXED" in content:
                    specifix_const = 0
                    continue
                if specifix_const == 1:
                    self.opt.constraint.stre.append(content)
                elif specifix_const == 2:
                    self.opt.fix_atom.input_fixed_atoms(content[0], content[-1])

            elif module_start == 3.5:  # $opt2
                content = line.split()
                if not content:
                    continue
                if content[0] == "r12":
                    self.opt2.r12.append(content[1:])
                elif content[0] == "r12mr34":
                    self.opt2.r12mr34.append(content[1:])

# Class to represent molecular structure in $molecule section
class molecule(object):
    def __init__(self):
        self.charge = 0
        self.multistate = 1
        self.carti = []
        self.check = False

    def return_atom_xyz(self, atom):
        return np.array(self.carti[atom][1:4]).astype(float)

    def return_output_format(self):
        out = "\n$molecule\n" + f"{self.charge} {self.multistate}\n"
        for cors in self.carti:
            out += " ".join(cors) + "\n"
        out += "$end\n\n"
        return out

    def read_xyz(self, filename):
        file = open(filename,"r").read().split("\n")
        carti = []
        for texts in file:
            text = texts.split()
            if len(text) < 3:
                continue
            carti.append(text)
        self.carti = carti

# Class for $rem section parameters
class rem(object):
    def __init__(self):
        self.texts = ""
        self.check = False

    def modify(self, stuff, new_stuff):
        lines = self.texts.split("\n")
        lines = [line for line in lines if line.strip()]
        result = ""
        for line in lines:
            if stuff in line:
                if stuff == "METHOD" and "SOLVENT" in line:
                    result += line + "\n"
                    continue
                if new_stuff == "":
                    continue
                key_val = line.split("=")
                line = f"{key_val[0]}= {new_stuff}"
            result += line + "\n"
        self.texts = result

    def return_output_format(self):
        return f"\n$rem\n{self.texts}$end\n\n"

# PCM solvent model (not fully implemented)
class pcm(object):
    def __init__(self):
        self.name = "pcm"
        self.theory = "IEFPCM"
        self.solvent = ""
        self.solvent_die = 4.0
        self.check = False

# Constraint-handling class for $opt section
class opt(object):
    def __init__(self):
        self.constraint = constraint()
        self.fix_atom = fix_atom()
        self.check = False

    def return_output_format(self):
        out = "\n$opt\nCONSTRAINT\n"
        text = " ".join(self.constraint.stre[0]) + "\n"
        out += text + "ENDCONSTRAINT\n$end\n\n"
        return out

class constraint(object):
    def __init__(self):
        self.stre = []

    def modify_stre(self, which_line, j):
        self.stre[which_line][3] = str(j)

# Extended optimization parameters class ($opt2)
class opt2(object):
    def __init__(self):
        self.check = False
        self.r12 = []
        self.r12mr34 = []
        self.r12pr34 = []

    def modify_r12(self, which_line, j):
        self.r12[which_line][2] = j

    def modify_r12mr34(self, which_line, j):
        self.r12mr34[which_line][4] = j

    def return_output_format(self):
        out = "\n$opt2\n"
        for block, label in [(self.r12, "r12"), (self.r12mr34, "r12mr34"), (self.r12pr34, "r12pr34")]:
            for entry in block:
                out += label + " " + " ".join(entry) + "\n"
        out += "$end\n\n"
        return out

# Class for managing fixed atoms in optimization
class fix_atom(object):
    def __init__(self):
        self.check = False
        self.fixed_atoms = []

    def input_fixed_atoms(self, atom, car="xyz"):
        self.fixed_atoms.append([atom, car])

# Class to parse and analyze Q-Chem output files
class qchem_out_file(object):
    def __init__(self):
        self.molecule = molecule()
        self.molecule.check = True
        self.opt_converged = False
        self.read_scf_time = False
        self.read_final_cpu_time = False
        self.geoms = []
        self.filename = ""
        self.optimizer = "default"

    # Utility functions to access geometry data
    def index_of_geom_i(self, index): return self.geoms[index].index
    def energy_of_geom_i(self, index): return self.geoms[index].energy
    def gradient_of_geom_i(self, index): return self.geoms[index].gradient
    def displacement_of_geom_i(self, index): return self.geoms[index].displacement

    def return_final_molecule_carti(self):
        if self.optimizer == "default":
            return self.geoms[-1].molecule.carti
        elif self.optimizer == "1996":
            return self.geoms[-2].molecule.carti

    def return_final_molecule_energy(self):
        if not self.opt_converged:
            print(f"{self.filename} doesn't have energy")
        for geom in reversed(self.geoms):
            if geom.have_energy:
                return geom.energy

    def return_average_scf_time(self):
        return sum(g.scf_time for g in self.geoms) / len(self.geoms)

    def return_opt_soc_ene(self):
        return self.opt_ene_list[-1]

    def read_ene_job(self, filename):
        for line in open(filename).readlines():
            if " Total energy" in line:
                self.ene = float(line.split("=")[1])

    def read_multiple_jobs_out(self, filename):
        self.filename = filename
        with open(filename, "r") as f:
            self.out_texts = f.read().split(" Welcome to Q-Chem")
        self.total_jobs = len(self.out_texts) - 1

    def read_pes_scan_from_file(self, filename, return_type="list"):
        scan_value_list, scan_ene_list = [], []
        for line in open(filename):
            if "PES scan" in line:
                parts = line.split(":")
                scan_value_list.append(float(parts[1].split()[-2]))
                scan_ene_list.append(float(parts[2]))
        if return_type in ("numpy", "np"):
            scan_value_list = np.array(scan_value_list)
            scan_ene_list = np.array(scan_ene_list)
        self.scan_value_list, self.scan_ene_list = scan_value_list, scan_ene_list
        return scan_value_list, scan_ene_list

    def read_soc_opt(self, filename, return_type):  # custom for Peizheng's SOC jobs
        opt_time_list, opt_ene_list = [], []
        with open(filename) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "E_adiab: " in line:
                opt_time_list.append(i + 1)
                opt_ene_list.append(float(line.split(":")[-1]))
        if return_type in ("numpy", "np"):
            opt_time_list = np.array(opt_time_list)
            opt_ene_list = np.array(opt_ene_list)
        self.opt_time_list, self.opt_ene_list = opt_time_list, opt_ene_list
        return opt_time_list, opt_ene_list

    def read_opt_from_file(self, filename, out_text=""):
        # ... skipped for brevity (can annotate if needed)
        pass

# Geometry class for storing parsed optimization information
class geoms(object):
    def __init__(self):
        self.index = 0
        self.energy = 0
        self.gradient = 0
        self.displacement = 0
        self.molecule = molecule()
