import sys
import numpy as np
Hartree_to_kcal = 627.51
atom_charge_dict = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,
"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,
"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"I":53,}
SPIN_REF = {1:"s",2:"d",3:"t",4:"q",5:"q"}



class qchem_file(object): #standard qchem inp file class
    def __init__(self):
        self.molecule = molecule()
        self.rem = rem()
        self.pcm = pcm()
        self.opt = opt()
        self.opt2 = opt2()
        self.remain_texts = ""
    @property
    def output(self):
        """
        Concatenate all available input sections to form full Q-Chem input text.
        """
        sections = []
        if self.molecule.check:
            sections.append(self.molecule.output)
        if self.remain_texts.strip() != "":
            sections.append(self.remain_texts)
        if self.rem.check:
            sections.append(self.rem.output)
        if self.opt.check:
            sections.append(self.opt.output)
        if self.opt2.check:
            sections.append(self.opt2.output)
        return "".join(sections)
    def show_molecule(self):
        return self.molecule.show()
    def generate_inp(self, new_file_name):
        with open(new_file_name, "w") as f:
            f.write(self.output)
            
    def read_from_file(self,filename="",out_text=""):# read standard inp file, require filename that consist of path and suffix. use class.check to determine which part you want to modify
        if out_text!= "":
            file = out_text.split("\n")
        else:
            self.filename = filename
            file = open(filename, "r").read().split("\n")
        file_length = len(file)
        module_start = 0
        specifix_const= 0
        for i in range(file_length):
            if "$end" in file[i]:
                if module_start == -1:
                    self.remain_texts += file[i]+"\n\n"
                module_start = 0

                continue
            elif "$molecule" in file[i]:
                if self.molecule.check == True:
                    module_start = 1
                else:
                    module_start = -1
                    self.remain_texts += "\n" + file[i] + "\n"
                continue
            elif "$rem" in file[i]:
                if self.rem.check == True:
                    module_start = 2
                else:
                    module_start = -1
                    self.remain_texts += "\n" + file[i] + "\n"
                continue
            elif "$opt" in file[i]:
                if "$opt2" in file[i] and self.opt2.check == True:
                    module_start = 3.5
                elif self.opt.check == True:
                    module_start = 3
                else:
                    module_start = -1
                    self.remain_texts += "\n" + file[i] + "\n"

                continue
            elif "$" in file[i] :
                module_start = -1
                self.remain_texts += "\n"+file[i]+"\n"
                continue
            elif "@" in file[i]:
                self.remain_texts += "\n"+file[i]+"\n"
                continue

            if module_start == -1:
                self.remain_texts += file[i] +"\n"


            elif module_start == 1:
                content = file[i].split(" ")
                while "" in content:
                    content.remove("")
                if file[i] == "":
                    continue
                if "read" in content:
                    self.molecule.read = True
                elif len(content) == 2 :
                    self.molecule.charge = content[0]
                    self.molecule.multistate = content[1]
                else:
                    self.molecule.carti.append(content)

            elif module_start == 2:
                text = file[i]
                content = file[i].split(" ")
                while "" in content:
                    content.remove("")
                if file[i] == "":
                    continue
                self.rem.texts += text+"\n"
                setattr(self.rem,content[0].lower(),content[2])

            elif module_start == 3:

                content = file[i].split(" ")
                while "" in content:
                    content.remove("")
                if file[i] == "":
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
                elif  "ENDFIXED" in content:
                    specifix_const = 0
                    continue
                if specifix_const == 1:
                    self.opt.constraint.stre.append(content)
                if specifix_const == 2:
                    self.opt.fix_atom.input_fixed_atoms(content[0],content[-1])
            elif module_start == 3.5:
                content = file[i].split(" ")
                while "" in content:
                    content.remove("")
                if file[i] == "" or content == []:
                    continue
                if "r12" == content[0]:
                    self.opt2.r12.append(content[1:])
                elif "r12mr34" == content[0]:
                    self.opt2.r12mr34.append(content[1:])







class qchem_inp_block:
    def __init__(self):
        self.check = False  # Common attribute

    @property
    def output(self):
        return self.return_output_format()

class molecule(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.charge = 0
        self.multistate = 1
        self.carti = []
        self.read = False
    @property
    def natom(self):
        if self.carti is not None and len(self.carti) > 0:
            return len(self.return_atom_type_list())
        else:
            return None
    @property
    def xyz(self):
        return self.return_xyz_list()
    @property
    def atoms(self):
        return self.return_atom_type_list()
    def show(self, show_index=False):
        from ase import Atoms
        atoms = Atoms(symbols=self.atoms, positions=self.xyz)
        view = nv.show_ase(atoms)
    
        if show_index:
            for i, pos in enumerate(self.xyz):
                # Add label using NGL JavaScript interface
                js = (
                    f"stage.compList[0].addLabel("
                    f"{{text: '{i}', position: [{pos[0]}, {pos[1]}, {pos[2]}]}})"
                )
                view._js(js)
    
        return view
    def return_xyz_list(self):
        carti = np.array(self.carti)
        return carti[:,1:]
    def return_atom_type_list(self):
        carti = np.array(self.carti)
        return carti[:,0]
    def replace_new_xyz(self,xyz):
        atom_type = np.array(self.carti)[:, 0].reshape(1,-1)
        combined = np.vstack([atom_type, xyz])
        self.carti = combined.T
        return combined.T
    def return_output_format(self):
        if self.read:
            return "\n$molecule\nread\n$end\n\n"
        else:
            out = "\n$molecule\n"
            out += f"{self.charge} {self.multistate}\n"
            for cors in self.carti:
                text = ""
                for j in cors:
                    if isinstance(j, float):
                        text += f" {j:.10f} "
                    else:
                        text += f" {j} "
                text += "\n"
                out += text
            out += "$end\n\n"
            return out
    def read_xyz(self,filename):
        file = open(filename,"r").read().split("\n")
        self.filename = filename
        carti = []
        for texts in file:
            text = texts.split(" ")
            while "" in text:
                text.remove("")
            if len(text) <3:
                continue
            else:
                carti.append(text)
        self.carti = carti
    def calc_distance_of_2_atoms(self,i,j):
        R_atom_1_atom_2 = self.calc_array_from_atom_1_to_atom_2(i,j)
        return np.dot(R_atom_1_atom_2,R_atom_1_atom_2) **0.5
    def calc_array_from_atom_1_to_atom_2(self,i,j):
        atom_1_xyz = np.array(self.carti[i][1:]).astype(float)
        atom_2_xyz = np.array(self.carti[j][1:]).astype(float)
        R_atom_1_atom_2 = atom_1_xyz-atom_2_xyz
        return R_atom_1_atom_2
    def transform_atom_type_into_charge(self):
        for i in range(len(self.carti)):
            self.carti[i][0] = atom_charge_dict[self.carti[i][0]]
    def calc_angle(self, atom_i, atom_j, atom_k):
        """
        Calculate the bond angle (in degrees) between three atoms: atom_i–atom_j–atom_k.
        The angle is measured at atom_j and lies within [0, 180] degrees.
        """
        coordinates = self.xyz.astype(float)

        vector_ij = coordinates[atom_i] - coordinates[atom_j]
        vector_kj = coordinates[atom_k] - coordinates[atom_j]

        unit_vector_ij = vector_ij / np.linalg.norm(vector_ij)
        unit_vector_kj = vector_kj / np.linalg.norm(vector_kj)

        cosine_angle = np.clip(np.dot(unit_vector_ij, unit_vector_kj), -1.0, 1.0)
        angle_radians = np.arccos(cosine_angle)

        return np.degrees(angle_radians)

    def calc_dihedral(self, atom_i, atom_j, atom_k, atom_l):
        """
        Calculate the dihedral angle (in degrees) defined by four atoms: i–j–k–l.
        The angle is in the range [-180, 180] and describes the torsion around the j–k bond.
        """
        coordinates = self.xyz.astype(float)

        point_i = coordinates[atom_i]
        point_j = coordinates[atom_j]
        point_k = coordinates[atom_k]
        point_l = coordinates[atom_l]

        bond_vector_1 = point_j - point_i
        bond_vector_2 = point_k - point_j
        bond_vector_3 = point_l - point_k

        unit_bond_vector_2 = bond_vector_2 / np.linalg.norm(bond_vector_2)

        # Plane normals
        normal_plane_1 = np.cross(bond_vector_1, unit_bond_vector_2)
        normal_plane_2 = np.cross(unit_bond_vector_2, bond_vector_3)

        unit_normal_1 = normal_plane_1 / np.linalg.norm(normal_plane_1)
        unit_normal_2 = normal_plane_2 / np.linalg.norm(normal_plane_2)

        orthogonal_vector = np.cross(unit_normal_1, unit_bond_vector_2)

        x_component = np.dot(unit_normal_1, unit_normal_2)
        y_component = np.dot(orthogonal_vector, unit_normal_2)

        dihedral_angle_radians = np.arctan2(y_component, x_component)
        return np.degrees(dihedral_angle_radians)

class rem(qchem_inp_block):
    def __init__(self):
        super().__init__()  # Set self.check = False
        self.texts = ""
    def modify(self,stuff,new_stuff):
        texts = self.texts.split("\n")
        while "" in texts:
            texts.remove("")
        text = ""
        for i in range(len(texts)):

            if stuff in texts[i]:
                if stuff == "METHOD" and "SOLVENT" in texts[i]:
                    text += texts[i] + "\n"
                    continue
                if new_stuff == "":
                    continue
                things = texts[i].split("=")
                texts[i] = "{}= {}".format(things[0],new_stuff)

            text += texts[i] +"\n"

        self.texts = text


    def return_output_format(self):
        out = "\n$rem\n"
        out += self.texts
        out += "$end\n\n"
        return out

class pcm(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.name = "pcm"
        self.theory = "IEFPCM"
        self.solvent = ""
        self.solvent_die = 4.0
class opt(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.constraint = constraint()
        self.fix_atom = fix_atom()

    def return_output_format(self):
        out = "\n$opt\nCONSTRAINT\n"
        text = ""
        for i in self.constraint.stre[0]:
            text += i + " "
        text += "\n"
        out += text
        out += "ENDCONSTRAINT\n$end\n\n"
        return out

class constraint(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.stre =[]
    def modify_stre(self,which_line,j):
        self.stre[which_line][3] = str(j)

class opt2(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.r12 = []
        self.r12mr34 = []
        self.r12pr34 = []
        self.opt2_info = {} 
    def modify_r12(self,which_line,j):
        self.r12[which_line][2] = j
    def modify_r12mr34(self,which_line,j):
        self.r12mr34[which_line][4] = j
    def return_output_format(self):
        lines = ["\n$opt2"]
        
        for r12_entry in self.r12:
            lines.append("r12 " + " ".join(str(x) for x in r12_entry))
        
        for r12mr34_entry in self.r12mr34:
            lines.append("r12mr34 " + " ".join(str(x) for x in r12mr34_entry))
        
        for r12pr34_entry in self.r12pr34:
            lines.append("r12pr34 " + " ".join(str(x) for x in r12pr34_entry))
        
        lines.append("$end\n")
        return "\n".join(lines)



class fix_atom(object):
    def __init__(self):
        self.check = False
        self.fixed_atoms = []
    def input_fixed_atoms(self,atom,car="xyz"):
        self.fixed_atoms.append([atom,car])

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
        self.soc_restrain = False
        self.cpu_time = 0
        self.wall_time = 0
        
    @property
    def final_geom(self):
        for geom in reversed(self.geoms):
            if hasattr(geom, "carti") and hasattr(geom, "energy") and geom.carti and geom.energy is not None:
                return geom.carti
        return None
    @property
    def final_ene(self):
        return self.geoms[-1].energy
    @property
    def final_geom_class(self):
        for geom in reversed(self.geoms):
            if hasattr(geom, "carti") and hasattr(geom, "energy") and geom.carti and geom.energy is not None:
                return geom
        return None
    @property
    def final_adiabatic_ene(self):
        return self.opt_ene_list[-1]
    @property
    def final_soc_ene(self):
        return self.opt_esoc_list[-1]
    @property
    def final_soc_e1(self):
        return self.opt_e1_list[-1]
    @property
    def final_soc_e2(self):
        return self.opt_e2_list[-1]
    @property
    def final_soc_er(self):
        return self.opt_er_list[-1]
    def index_of_geom_i(self, index):
        return self.geoms[index].index

    def energy_of_geom_i(self, index):
        return self.geoms[index].energy

    def gradient_of_geom_i(self, index):
        return self.geoms[index].gradient

    def displacement_of_geom_i(self, index):
        return self.geoms[index].displacement

    def return_final_molecule_carti(self):
        return self.final_geom

    def return_final_molecule_energy(self):
        return self.final_ene

    def return_average_scf_time(self):
        total_scf_time = sum(getattr(g, 'scf_time', 0.0) for g in self.geoms)
        return total_scf_time / len(self.geoms) if self.geoms else 0.0
    def return_opt_soc_ene(self):
        return self.opt_ene_list[-1]
    def read_time_result(self,line):
        data = line.split(":")[1].split(",")
        cpu_time = data[1][:-6]
        wall_time = data[1][:-7]
        print(cpu_time,wall_time)
        self.cpu_time += cpu_time
        self.wall_time += wall_time
    def read_ene_job(self, filename="", out_text=""):
        if filename:
            self.filename = filename
        content = out_text if out_text else open(self.filename, "r").read()
        lines = content.split("\n")

        # Read total energy
        for line in lines:
            if " Total energy" in line:
                self.ene = float(line.split("=")[1])
                break

        # Read only the final "Standard Nuclear Orientation" structure block
        structure_started = False
        carti = []
        structure_blocks = []

        for i, line in enumerate(lines):
            if "Standard Nuclear Orientation" in line:
                structure_started = True
                carti = []
                continue
            if " Total job time:" in line:
                read_time_result(line)
            if structure_started:
                if "-----" in line or "Atom" in line or line.strip() == "":
                    continue
                tokens = line.strip().split()
                if len(tokens) == 5:
                    try:
                        atom = tokens[1]
                        x, y, z = map(float, tokens[2:5])
                        carti.append([atom, x, y, z])
                    except ValueError:
                        continue
                elif len(tokens) == 0 and carti:
                    structure_blocks.append(carti.copy())
                    structure_started = False

        # Use only the last complete orientation block
        if structure_blocks:
            mol = molecule()
            mol.carti = structure_blocks[-1]
            mol.check = True
            self.molecule = mol
    def read_pes_scan_from_file(self, filename="", return_type="list", out_text=""):
        if out_text == "":
            file = open(filename, "r").read().split("\n")
        else:
            file = out_text.split("\n")
        
        scan_value_list = []
        scan_ene_list = []
        scan_structure_list = []
    
        i = 0
        converged = False
        while i < len(file):
            line = file[i]
            if "PES scan" in line:
                data = line.split(":")
                scan_value = float(data[1].split(" ")[-2])
                energy = float(data[2])
                scan_value_list.append(scan_value)
                scan_ene_list.append(energy)
                # 向后找结构
                converged = False
            if "OPTIMIZATION CONVERGED" in file[i]:
                structure = []
                converged = True
            if converged and "Coordinates (Angstroms)" in file[i]:
                k = i + 2
                while k < i + 1000:
                    tokens = file[k].split()
                    if len(tokens) != 5:
                        break
                    structure.append(tokens[1:])
                    k += 1
                scan_structure_list.append(structure)
            i += 1
    
        if return_type == "numpy" or return_type == "np":
            scan_value_list = np.array(scan_value_list)
            scan_ene_list = np.array(scan_ene_list)
            # structure list 不转 numpy，因为不同步数原子数不一定一致
    
        self.scan_value_list = scan_value_list
        self.scan_ene_list = scan_ene_list
        self.scan_structure_list = scan_structure_list
    
        return scan_value_list, scan_ene_list, scan_structure_list


    def read_soc_opt(self,filename,return_type="np"): #这个函数是给peizheng的soc单独写的！ this function can only use in soc opt jobs written by peizheng!
        file = open(filename, "r").read().split("\n")
        num = 1
        opt_time_list = []
        opt_ene_list = []
        opt_e1_list = []
        opt_e2_list = []
        opt_er_list = []
        opt_esoc_list = []
        for i in range(len(file)):
            if "E_adiab: " in file[i]:
                data = file[i].split(":")
                e1 = float(data[1].split()[0])
                e2 = float(data[2].split()[0])
                esoc = float(data[3].split()[0])
                opt_time_list.append(num)
                opt_ene_list.append(float(data[-1]))
                opt_e1_list.append(e1)
                opt_e2_list.append(e2)
                er = float(data[3].split()[0])
                opt_er_list.append(er)
                esoc = float(data[4].split()[0])
                opt_esoc_list.append(esoc)
                num += 1
            
        if return_type == "list":
            a = 1
        elif return_type == "numpy" or return_type == "np":
            opt_time_list = np.array(opt_time_list)
            opt_ene_list = np.array(opt_ene_list)
        self.opt_time_list = opt_time_list
        self.opt_ene_list = opt_ene_list
        self.opt_e1_list = opt_e1_list
        self.opt_e2_list = opt_e2_list
        self.opt_er_list = opt_er_list
        self.opt_esoc_list = opt_esoc_list
        self.ene = opt_ene_list[-1]
        return opt_time_list, opt_ene_list

    def read_opt_from_file(self, filename="", out_text=""):
        if filename:
            self.filename = filename
        out_files = out_text if out_text else open(self.filename, "r").read()

        if self.optimizer == "default":
            out_file = out_files.split("OPTIMIZATION CYCLE")
            for text in out_file[0].split("\n"):
                if "geom_opt_driver = 1996" in text:
                    self.optimizer = "1996"
                    break
        if self.optimizer == "1996" or self.optimizer == "constrain":
            out_file = out_files.split("Optimization Cycle:")

        for i in range(1, len(out_file)):
            geom_texts = out_file[i].split("\n")
            geom = geoms()
            geom.index = i - 1
            internal_index = int(geom_texts[0][2:])

            detected_data = 0
            molecule_check = -1
            detected_molecule = 0
            datas = []
            molecule_cart = []
            self.geom_have_energy = 0

            for text_line in geom_texts:
                if f"Step {internal_index}" in text_line or "Step Taken" in text_line:
                    detected_data = 1
                    continue

                if self.read_scf_time and " SCF time" in text_line:
                    scf_time = [x for x in text_line.split(" ") if x]
                    geom.scf_time = float(scf_time[3][:-1])

                if "I:" in text_line and "EI=" in text_line and "E=" in text_line and self.optimizer == "default":
                    tokens = text_line.replace("=", "").split()
                    opt2_info = {}
                    for i, token in enumerate(tokens):
                        if token.startswith("I:"):
                            opt2_info["I"] = int(token[2:])
                        elif token.startswith("Type:"):
                            opt2_info["Type"] = int(token[5:])
                        elif token == "Value":
                            opt2_info["Value"] = float(tokens[i+1])
                        elif token == "K":
                            opt2_info["K"] = float(tokens[i+1])
                        elif token == "R12":
                            opt2_info["R12"] = float(tokens[i+1])
                        elif token == "R34":
                            opt2_info["R34"] = float(tokens[i+1])
                        elif token == "R12-R34":
                            opt2_info["R12-R34"] = float(tokens[i+1])
                        elif token == "EI":
                            opt2_info["EI"] = float(tokens[i+1])
                        elif token == "E":
                            opt2_info["E"] = float(tokens[i+1])
                    geom.opt2_info = opt2_info

                if "Energy is" in text_line:
                    if self.optimizer == "1996" or self.optimizer == "constrain":
                        geom.energy = float(text_line.split(" ")[-1])
                        geom.have_energy = 1
                        self.geom_have_energy = 1

                if detected_data:
                    datas.append(text_line)

                if "Standard Nuclear Orientation (Angstroms)" in text_line:
                    detected_molecule = 1
                    continue
                if detected_molecule == 1 and " ------------------------------" in text_line:
                    molecule_check += 1
                    continue
                if molecule_check == 0:
                    molecule_cart.append(text_line)
                if "**  OPTIMIZATION CONVERGED  **" in text_line:
                    self.opt_converged = True

            for cart in molecule_cart:
                cart = cart.split()
                if len(cart) > 1:
                    geom.carti.append(cart[1:])

            for data in datas:
                if self.optimizer == "1996" or self.optimizer == "constrain":
                    break
                if "Energy is " in data:
                    geom.energy = float(data.split(" ")[-1])
                    geom.have_energy = 1
                    self.geom_have_energy = 1
                if "Gradient" in data:
                    parts = data.split()
                    geom.gradient = float(parts[1].split("\t")[0])
                if "Displacement" in data:
                    if self.optimizer == "1996":
                        continue
                    parts = data.split()
                    geom.displacement = float(parts[1].split("\t")[0])

                if i == len(out_file) - 1:
                    if self.read_final_cpu_time and " Total job time" in data:
                        jobtime = [x for x in data.split(" ") if x]
                        self.final_cpu_time = float(jobtime[-1][:-6])
                    if "OPTIMIZATION DID NOT CONVERGE" in data:
                        self.opt_converged = False
                    if "END OF GEOMETRY OPTIMIZER USING LIBOPT3" in data:
                        self.opt_converged = True

            if self.geom_have_energy:
                self.geoms.append(geom)

        self.geom_number = len(self.geoms)
    def read_multiple_jobs_out(self,filename):
        self.filename = filename
        file = open(filename,"r").read().split(" Welcome to Q-Chem")
        self.out_texts = file
        self.total_jobs = len(file)-1
    def read_freq_jobs_out(self,filename="",out_text=""):
        if filename != "":
            self.filename = filename
        if out_text == "":
            out_files = open(filename,"r").read()
        else:
            out_files = out_text
        file = out_files.split("\n")
        for text in file:
            if "Total Enthalpy:" in text:
                if "QRRHO" not in text:
                    data = text.split(" ")
                    while "" in data:
                        data.remove("")
                    self.enthalpy = float(data[2])
            if "Total Entropy:" in text:
                if "QRRHO" not in text:
                    data = text.split(" ")
                    while "" in data:
                        data.remove("")
                    self.entropy = float(data[2])
            if " Total energy" in text:
                data = float(text.split("=")[1])
                self.ene = data
    
    def read_force_from_file(self, filename, self_check=False, different_type="analytical"):
        self.filename = filename
        self.molecule.check = True
        out_file = open(filename, "r").read().split("\n")
        force_key_word = "fhiuadjskdnlashalfwwawldjalksfna"
        force_key_word_checked = 0
        force_check = 0
        force = []
        mol_structure = qchem_file()
        mol_structure.molecule.check = True
        mol_structure.read_from_file(filename)
        force_e1 = [] 
        force_e2 = [] 
        if different_type == "soc":
            self.read_soc_opt(filename)
        else:
            self.read_ene_job(filename)
        
        self.molecule = mol_structure.molecule
        for text in out_file:
            if " Total energy" in text:
                data = float(text.split("=")[1])
                self.ene = data

            if force_key_word_checked != 1:
                if self_check:
                    if "ideriv" in text:
                        ideriv_type = text.split()
                        while "" in ideriv_type:
                            ideriv_type.remove("")
                        if int(ideriv_type[-1]) == 1:
                            force_key_word = "zheng adiabatic surface gradient"
                        elif int(ideriv_type[-1]) == 0:
                            force_key_word = "FINAL TENSOR RESULT"
                        force_key_word_checked = 1
                else:
                    if different_type == "soc":
                        force_key_word = "zheng adiabatic surface gradient"
                        force_key_word_e1 = "zheng ASG first state"
                        force_key_word_e2 = "zheng ASG second state"
                    elif different_type == "numercial":
                        force_key_word = "FINAL TENSOR RESULT"
                    elif different_type == "analytical":
                        force_key_word = "Gradient of SCF Energy"
                    elif different_type == "smd":
                        force_key_word = "-- total gradient after adding PCM contribution --"
                    else:
                        force_key_word = different_type
                    force_key_word_checked = 1

            # Start reading force block
            if force_key_word in text:
                force_check = 1
                force = []  # clear buffer
                continue
            if different_type == "soc":
                if force_key_word_e1 in text:
                    force_check = 2
                    force_e1 = []  # clear buffer
                    continue
                elif force_key_word_e2 in text:
                    force_check = 3
                    force_e2 = []  # clear buffer
                    continue

            if force_key_word == "-- total gradient after adding PCM contribution --":
                if "Atom" in text or "----" in text:
                    continue
                if force_check:
                    if "----" in text or text.strip() == "":
                        force_check = 0
                        continue
                    data = text.split()
                    if len(data) == 4:
                        force.append(data[1:])  # skip atom index
            else:
                if "Gradient time" in text or "#" in text or "gradient" in text:
                    force_check = 0
                    continue
                if force_check == 1:
                    data = text.split()
                    while "" in data:
                        data.remove("")
                    force.append(data)
                elif force_check == 2:
                    data = text.split()
                    while "" in data:
                        data.remove("")
                    force_e1.append(data)
                elif force_check == 3:
                    data = text.split()
                    while "" in data:
                        data.remove("")
                    force_e2.append(data)

        
        # Postprocess into numpy array
        if force_key_word == "zheng adiabatic surface gradient":
            force = reshape_force(force)
            force_e1 = reshape_force(force_e1)
            force_e2 = reshape_force(force_e2)
            self.force = force.astype(float)
            self.force_e1 = force_e1.astype(float)
            self.force_e2 = force_e2.astype(float)
        elif force_key_word == "FINAL TENSOR RESULT":
            while [] in force:
                force.remove([])
            force = np.array(force[2:])
            force = force[:, 1:]
        elif force_key_word == "Gradient of SCF Energy":
            while [] in force:
                force.remove([])
            # flatten and reshape
            flat_force = []
            check = 0
            x_force = []
            y_force = []
            z_force = []
            for row in force:
                if check % 4==1:
                    x_force.extend(row[1:])
                    check+=1
                elif check % 4==2:
                    y_force.extend(row[1:])
                    check+=1
                elif check % 4==3:
                    z_force.extend(row[1:])
                    check+=1
                else:
                    check+=1
            x_force = np.array(x_force)
            y_force = np.array(y_force)
            z_force = np.array(z_force)
            force = np.column_stack((x_force, y_force, z_force)).T #Shape should be (3,Natom)
        elif force_key_word == "-- total gradient after adding PCM contribution --":
            force = np.array(force)

        self.force = force.astype(float)

    @property
    def final_EI(self):
        return self.geoms[-1].opt2_info["EI"]
    def calc_gibbs(self):
        global Hartree_to_kcal
        gibbs_energy = self.ene*Hartree_to_kcal  + self.enthalpy - (0.273+0.025)*self.entropy
        self.gibbs_energy = gibbs_energy
        return gibbs_energy
    def plot_optimization(self, show=True, save_as=None):
        import matplotlib.pyplot as plt
        """
        Plot the energy, gradient, and displacement during geometry optimization.
        
        Parameters:
            show (bool): If True, display the plot.
            save_as (str or None): If a string is given, save the plot to this filename.
        """
        if not self.geoms:
            print("No optimization data available. Please run read_opt_from_file() first.")
            return
    
        steps = list(range(len(self.geoms)))
        energies = [g.energy for g in self.geoms]
        gradients = [g.gradient for g in self.geoms if hasattr(g, "gradient")]
        displacements = [g.displacement for g in self.geoms if hasattr(g, "displacement")]
    
        # Normalize energies to initial value
        energies = [e - energies[0] for e in energies]
    
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle("Geometry Optimization Progress", fontsize=14)
    
        axs[0].plot(steps, energies, marker='o', color='tab:blue')
        axs[0].set_ylabel("ΔE (Hartree)")
        axs[0].grid(True)
    
        if gradients:
            axs[1].plot(steps[:len(gradients)], gradients, marker='o', color='tab:green')
            axs[1].set_ylabel("Gradient Norm")
            axs[1].grid(True)
    
        if displacements:
            axs[2].plot(steps[:len(displacements)], displacements, marker='o', color='tab:red')
            axs[2].set_ylabel("Displacement")
            axs[2].set_xlabel("Optimization Step")
            axs[2].grid(True)
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
        if save_as:
            plt.savefig(save_as, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
    def animate_optimization(self): #this function needs to be developed
        """
        Show an interactive animation of the geometry optimization using nglview.
        Requires: nglview, ase
        """
        import nglview as nv
        from ase import Atoms
    
        structures = []
        for geom in self.geoms:
            try:
                from ase import Atoms
                atoms = Atoms(symbols=geom.atoms, positions=geom.xyz.astype(float))
                structures.append(atoms)
            except Exception as e:
                print(f"Skipped step {geom.index}: {e}")
    
        if not structures:
            raise ValueError("No valid geometries to animate.")
    
        return nv.show_ase(structures)
    def read_job_time(self, filename="",out_text=""):
        if filename:
            self.filename = filename
        lines = open(self.filename, "r").read().splitlines()
    
        wall_time = None
        cpu_time = None
    
        for line in lines[::-1]:  # 从文件结尾往前找，能加速匹配
            if "Total job time" in line:
                parts = line.strip().split(',')
                try:
                    wall_time = float(parts[0].split()[3][:-1])  # 去掉"s"
                    cpu_time = float(parts[1].split()[0][:-1])
                except Exception as e:
                    print(f"Failed to parse job time from line: {line}\nError: {e}")
                break
    
        self.wall_time = wall_time
        self.cpu_time = cpu_time
        return wall_time, cpu_time
class MECP_report(object):
    def __init__(self):
        self.geoms = []
        self.filename = "ReportFile"
        self.path = ""
    def read_from_report(self,path="",filename="ReportFile"):
        if path:
            self.path = path
        else:
            path = self.path
        if filename != "ReportFile":
            self.filename = filename
        else:
            filename = self.filename
        
        file = open(path+filename,"r").read().split("Geometry at Step")
        for texts in file:
            geom = geoms()
            texts = texts.split("\n")
            start_read_geom = True
            for text in texts:
                text = text.split(" ")
                while "" in text:
                    text.remove("")
                if start_read_geom == True:
                    if len(text) == 4 and "Gradient:" not in text and 'version' not in text:
                        geom.molecule.carti.append(text)
                    if text == []:
                        start_read_geom = False
                if "Energy" in text:
                    if "First" in text:
                        geom.state_1_energy = float(text[-1])
                    if "Second" in text:
                        geom.state_2_energy = float(text[-1])
            self.geoms.append(geom)
    def return_final_geomery(self): #need read reportfile first
        return self.geoms[-1].molecule.carti
    def return_final_states_energy(self):
        return self.geoms[-1].state_1_energy,self.geoms[-1].state_2_energy
    def return_final_state_energy_difference(self):
        return (self.geoms[-1].state_1_energy - self.geoms[-1].state_2_energy)*Hartree_to_kcal


class geoms(molecule):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.energy = 0
        self.gradient = 0
        self.displacement = 0

def reshape_force(force):
    while [] in force:
        force.remove([])
    # flatten and reshape
    flat_force = []
    check = 0
    x_force = []
    y_force = []
    z_force = []
    for row in force:
        if check % 4==1:
            x_force.extend(row[1:])
            check+=1
        elif check % 4==2:
            y_force.extend(row[1:])
            check+=1
        elif check % 4==3:
            z_force.extend(row[1:])
            check+=1
        else:
            check+=1
    x_force = np.array(x_force)
    y_force = np.array(y_force)
    z_force = np.array(z_force)
    force = np.column_stack((x_force, y_force, z_force)).T #Shape should be (3,Natom)
    return force


class multiple_qchem_jobs(object):
    def __init__(self):
        self.jobs=[]
        self.filename = ""
        self.molecule_check = True
        self.rem_check = True
        self.opt_check = False
        self.pcm_check = False

    def match_check(self, job):
        if self.molecule_check:
            job.molecule.check = True
        else:
            job.molecule.check = False
        if self.rem_check:
            job.rem.check = True
        else:
            job.rem.check = False
        if self.opt_check:
            job.opt.check = True
        else:
            job.opt.check = False
        if self.pcm_check:
            job.pcm.check = True
        else:
            job.pcm.check = False
    @property
    def job_nums(self):
        return len(self.jobs)
    def read_from_file(self,filename):
        self.filename = filename
        jobs = open(filename, "r").read().split("@@@")
        for job in jobs:
            job_file = qchem_file()
            self.match_check(job_file)
            job_file.read_from_file(out_text=job)
            self.jobs.append(job_file)
    @property
    def output(self):
        out = ""
        for i in range(self.job_nums):
            job = self.jobs[i]
            if i == 0:
                out += f"{job.output}\n"
            else:
                out += f"\n@@@\n\n{job.output}\n"
        return out
    def generate_inp(self, new_file_name):
        with open(new_file_name, "w") as f:
            f.write(self.output)
