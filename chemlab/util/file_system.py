import sys
import numpy as np
from chemlab.util.unit import unit_type,complex_unit_type
Hartree_to_kcal = 627.51
ELEMENT_DICT = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,
    "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54
}
SPIN_REF = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartlet", 5: "quintet", 6: "sextuplet"}

NUM2ELEMENT = {v: k for k, v in ELEMENT_DICT.items()}
atom_charge_dict = ELEMENT_DICT.copy() #历史遗留问题这一块
class qchem_file(object):  # standard qchem inp file class
    def __init__(self):
        self.molecule = molecule()
        self.rem = rem()
        self.pcm = pcm()
        self.opt = opt()
        self.opt2 = opt2()
        self.external_charges = external_charges()
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
        if self.external_charges.check:
            sections.append(self.external_charges.output)
        return "".join(sections)

    def show_molecule(self):
        return self.molecule.show()

    def generate_inp(self, new_file_name):
        with open(new_file_name, "w") as f:
            f.write(self.output)

    def read_from_file(self, filename="",
                       out_text=""):  # read standard inp file, require filename that consist of path and suffix. use class.check to determine which part you want to modify
        if out_text != "":
            file = out_text.split("\n")
        else:
            self.filename = filename
            file = open(filename, "r").read().split("\n")
        file_length = len(file)
        module_start = 0
        specifix_const = 0
        for i in range(file_length):
            if "$end" in file[i]:
                if module_start == -1:
                    self.remain_texts += file[i] + "\n\n"
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
            elif "$" in file[i]:
                module_start = -1
                self.remain_texts += "\n" + file[i] + "\n"
                continue
            elif "@" in file[i]:
                self.remain_texts += "\n" + file[i] + "\n"
                continue

            if module_start == -1:
                self.remain_texts += file[i] + "\n"


            elif module_start == 1:
                content = file[i].split(" ")
                while "" in content:
                    content.remove("")
                if file[i] == "":
                    continue
                if "read" in content:
                    self.molecule.read = True
                elif len(content) == 2:
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
                self.rem.texts += text + "\n"
                setattr(self.rem, content[0].lower(), content[2])

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
                elif "ENDFIXED" in content:
                    specifix_const = 0
                    continue
                if specifix_const == 1:
                    self.opt.constraint.stre.append(content)
                if specifix_const == 2:
                    self.opt.fix_atom.input_fixed_atoms(content[0], content[-1])
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
        return carti[:, 1:]

    def return_atom_type_list(self):
        carti = np.array(self.carti)
        return carti[:, 0]

    def replace_new_xyz(self, xyz):
        atom_type = np.array(self.carti)[:, 0].T.reshape(-1, 1)
        combined = np.hstack([atom_type, xyz])
        self.carti = combined
        return combined
    def replace_new_carti(self,carti):
        self.carti = carti
        return carti

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

    def read_xyz(self, filename):
        with open(filename, "r") as f:
            lines = f.read().splitlines()
        self.filename = filename
        carti = []
        if not lines:
            self.carti = []
            self.comment = ""
            return
        try:
            self.natoms = int(lines[0].strip())
        except ValueError:
            self.natoms = None
        self.comment = lines[1].strip() if len(lines) > 1 else ""
        for texts in lines[2:]:
            parts = texts.split()
            if len(parts) < 4:
                continue
            carti.append(parts)

        self.carti = carti

    def calc_distance_of_2_atoms(self, i, j):
        R_atom_1_atom_2 = self.calc_array_from_atom_1_to_atom_2(i, j)
        return np.dot(R_atom_1_atom_2, R_atom_1_atom_2) ** 0.5

    def calc_array_from_atom_1_to_atom_2(self, i, j):
        atom_1_xyz = np.array(self.carti[i][1:]).astype(float)
        atom_2_xyz = np.array(self.carti[j][1:]).astype(float)
        R_atom_1_atom_2 = atom_1_xyz - atom_2_xyz
        return R_atom_1_atom_2

    def modify_bond_length(self, i, j, target_distance, fix=None):
        """
        调整原子i和j之间的键长到指定值

        参数:
            i: 原子1的索引
            j: 原子2的索引
            target_distance: 目标键长
            fix: 固定哪个原子 ('i' 或 'j' 或 None)
                 - 'i': 固定原子i，移动原子j
                 - 'j': 固定原子j，移动原子i
                 - None: 两个原子都移动（各移动一半）
        """
        # 计算当前的距离向量和距离
        R_vec = self.calc_array_from_atom_1_to_atom_2(i, j)
        current_distance = self.calc_distance_of_2_atoms(i, j)

        # 如果当前距离为0，无法调整
        if current_distance == 0:
            raise ValueError("当前两原子距离为0，无法调整")

        # 计算单位向量
        unit_vec = R_vec / current_distance

        # 计算需要移动的总距离
        delta_distance = target_distance - current_distance

        # 根据fix参数决定如何移动
        if fix == 'i':
            # 固定i，移动j
            # j沿着从i到j的方向移动
            displacement = -unit_vec * delta_distance
            self.carti[j][1] = float(self.carti[j][1]) + displacement[0]
            self.carti[j][2] = float(self.carti[j][2]) + displacement[1]
            self.carti[j][3] = float(self.carti[j][3]) + displacement[2]

        elif fix == 'j':
            # 固定j，移动i
            # i沿着从j到i的方向移动
            displacement = unit_vec * delta_distance
            self.carti[i][1] = float(self.carti[i][1]) + displacement[0]
            self.carti[i][2] = float(self.carti[i][2]) + displacement[1]
            self.carti[i][3] = float(self.carti[i][3]) + displacement[2]

        else:
            # 两个原子都移动，各移动一半距离
            displacement = unit_vec * delta_distance / 2

            # 移动原子i
            self.carti[i][1] = float(self.carti[i][1]) + displacement[0]
            self.carti[i][2] = float(self.carti[i][2]) + displacement[1]
            self.carti[i][3] = float(self.carti[i][3]) + displacement[2]

            # 移动原子j
            self.carti[j][1] = float(self.carti[j][1]) - displacement[0]
            self.carti[j][2] = float(self.carti[j][2]) - displacement[1]
            self.carti[j][3] = float(self.carti[j][3]) - displacement[2]

    def transform_atom_type_into_charge(self):
        for i in range(len(self.carti)):
            self.carti[i][0] = ELEMENT_DICT[self.carti[i][0]]

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

    def modify(self, stuff, new_stuff):
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
                texts[i] = "{}= {}".format(things[0], new_stuff)

            text += texts[i] + "\n"

        self.texts = text

    def return_output_format(self):
        out = "\n$rem\n"
        out += self.texts
        out += "$end\n\n"
        return out

class external_charges(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.name = "external_charges"
        self.mm_pos = []
        self.mm_charge = []
    def return_output_format(self):
        mm_pos = self.mm_pos.copy()
        mm_charge = self.mm_charge.copy()
        out = "\n$external_charges\n"
        for i in range(len(mm_pos)):
            out +="".join(["%21.16f" % mm_pos[i, 0],
                             "%21.16f" % mm_pos[i, 1],
                             "%21.16f" % mm_pos[i, 2],
                             "%21.16f" % mm_charge[i], "\n"])
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
        self.stre = []

    def modify_stre(self, which_line, j):
        self.stre[which_line][3] = str(j)


class opt2(qchem_inp_block):
    def __init__(self):
        super().__init__()
        self.r12 = []
        self.r12mr34 = []
        self.r12pr34 = []
        self.opt2_info = {}

    def modify_r12(self, which_line, j):
        self.r12[which_line][2] = j

    def modify_r12mr34(self, which_line, j):
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

    def input_fixed_atoms(self, atom, car="xyz"):
        self.fixed_atoms.append([atom, car])


class geoms(molecule):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.energy = 0
        self.gradient = 0
        self.displacement = 0


class qchem_out:
    def __init__(self, filename=""):
        self.filename = filename
        self.text = ""
        self.molecule = molecule()
        self.molecule.check = True
        self.geoms = []
        self.ene = None
        self.wall_time = None
        self.cpu_time = None
        self.spin = 1
        self.charge = 0


    def read_file(self, filename=None, text=None, read_charge_and_spin=True, read_esp_charge=False):
        if filename:
            self.filename = filename
        if text:
            self.text = text
        else:
            self.text = open(self.filename, "r").read()
        if read_charge_and_spin:
            self._parse_molecule()
        if read_esp_charge:
            self._parse_esp_charge()
        self.parse()

    def parse(self):
        raise NotImplementedError

    def read_job_time(self):
        lines = self.text.splitlines()
        for line in lines[::-1]:
            if "Total job time" in line:
                parts = line.strip().split(',')
                try:
                    self.wall_time = float(parts[0].split()[3][:-1])
                    self.cpu_time = float(parts[1].split()[0][:-1])
                except Exception:
                    pass
                break
        return self.wall_time, self.cpu_time

    @property
    def final_geom(self):
        for geom in reversed(self.geoms):
            if hasattr(geom, "carti") and hasattr(geom, "energy") and geom.carti and geom.energy is not None:
                return geom.carti
        return None

    @property
    def final_geom_class(self):
        for geom in reversed(self.geoms):
            if hasattr(geom, "carti") and hasattr(geom, "energy") and geom.carti and geom.energy is not None:
                return geom
        return None

    @property
    def final_ene(self):
        return self.geoms[-1].energy if self.geoms else None

    import numpy as np

    def _parse_molecule(self):
        lines = self.text.splitlines()
        self.charge = None
        self.spin = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("$molecule"):
                # --- Read charge and spin ---
                next_line = lines[i + 1].strip()
                charge, spin = map(int, next_line.split()[:2])
                self.charge = charge
                self.spin = spin
                self.molecule.charge = charge
                self.molecule.spin = spin

                # --- Read molecular geometry until $end ---
                j = i + 2
                while j < len(lines):
                    l = lines[j].strip()
                    if not l or l.lower().startswith("$end"):
                        break
                    parts = l.split()
                    if len(parts) >= 4:
                        atom_line = parts[0:4]
                        self.molecule.carti.append(atom_line)

                    j += 1
                break  # only the first $molecule section is parsed

        if self.charge is None or self.spin is None:
            raise ValueError("No '$molecule' section found or missing charge/spin line.")

        # Convert to NumPy array for convenience
        self.molecule.carti = np.array(self.molecule.carti)
        return self.charge, self.spin, self.molecule.carti

    def _parse_esp_charge(self, lines):
        """
        Parse the ground-state ESP charge block from Q-Chem output.

        Example block:
            Merz-Kollman ESP Net Atomic Charges

             Atom                 Charge (a.u.)
          ----------------------------------------
              1 C                     0.310046
              2 C                    -0.125879
              3 C                     0.396755
              ...

        Returns:
            np.ndarray of shape (n_atom,)
        """
        text = "\n".join(lines)
        m = re.search(
            r"Merz-Kollman ESP Net Atomic Charges\s+Atom\s+Charge.+?-{5,}\s+(.+?)(?:\n\s*\n|\Z)",
            text,
            re.S,
        )

        if not m:
            return None

        block = m.group(1).strip()
        charges = []
        for ln in block.splitlines():
            parts = re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", ln)
            if len(parts) == 1:  # only charge found
                charges.append(float(parts[0]))
            elif len(parts) >= 2:  # atom index and charge
                charges.append(float(parts[-1]))

        esp_charges = np.array(charges[:-1])
        # assign to ground state (state[0] typically ground)
        if len(self.states) > 0:
            self.states[0].esp_charges = esp_charges
        return esp_charges


class qchem_out_opt(qchem_out):
    def __init__(self, filename=""):
        super().__init__(filename)
        self.opt_converged = False

    def parse(self):
        out_file = self.text.split("Optimization Cycle:".upper())

        for i in range(1, len(out_file)):

            geom_texts = out_file[i].split("\n")
            geom = geoms()
            geom.index = i - 1

            for text_line in geom_texts:
                if "Energy is" in text_line:
                    geom.energy = float(text_line.split()[-1])
                if "Gradient" in text_line:
                    try:
                        geom.gradient = float(text_line.split()[1])
                    except:
                        continue
                if "Displacement" in text_line:
                    try:
                        geom.displacement = float(text_line.split()[1])
                    except:
                        continue
                if "**  OPTIMIZATION CONVERGED  **" in text_line:
                    self.opt_converged = True
                if "Standard Nuclear Orientation" in text_line:
                    geom.carti = self._parse_geom(geom_texts, text_line)

            self.geoms.append(geom)

        if self.geoms:
            self.ene = self.geoms[-1].energy

    def _parse_geom(self, lines, marker):
        carti = []
        start = lines.index(marker) + 3
        for line in lines[start:]:
            parts = line.split()
            if len(parts) == 5:
                carti.append(parts[1:])
            else:
                break
        return carti


class qchem_out_scan(qchem_out):
    def parse(self):
        lines = self.text.splitlines()
        self.scan_value_list = []
        self.scan_ene_list = []
        self.scan_structure_list = []
        converged = False

        for i, line in enumerate(lines):
            if "PES scan" in line:
                data = line.split(":")
                self.scan_value_list.append(float(data[1].split()[-2]))
                self.scan_ene_list.append(float(data[2]))
                converged = False
            if "OPTIMIZATION CONVERGED" in line:
                structure = []
                converged = True
            if converged and "Coordinates (Angstroms)" in line:
                k = i + 2
                while k < len(lines):
                    tokens = lines[k].split()
                    if len(tokens) != 5:
                        break
                    structure.append(tokens[1:])
                    k += 1
                self.scan_structure_list.append(structure)

        if self.scan_ene_list:
            self.ene = self.scan_ene_list[-1]


class qchem_out_soc(qchem_out):
    def parse(self):
        lines = self.text.splitlines()
        self.opt_time_list = []
        self.opt_ene_list = []
        self.opt_e1_list = []
        self.opt_e2_list = []
        self.opt_esoc_list = []
        self.opt_er_list = []

        num = 1
        for line in lines:
            if "E_adiab:" in line:
                data = line.split(":")
                e1 = float(data[1].split()[0])
                e2 = float(data[2].split()[0])
                esoc = float(data[3].split()[0])
                self.opt_time_list.append(num)
                self.opt_e1_list.append(e1)
                self.opt_e2_list.append(e2)
                self.opt_er_list.append(esoc)
                self.opt_esoc_list.append(float(data[4].split()[0]))
                self.opt_ene_list.append(float(data[-1]))
                num += 1

        if self.opt_ene_list:
            self.ene = self.opt_ene_list[-1]

    @property
    def final_soc_ene(self):
        return self.opt_esoc_list[-1] if self.opt_esoc_list else None
    @property
    def final_vsoc_ene(self):
        esoc = self.final_soc_ene
        e1 = self.opt_e1_list[-1]
        e2 = self.opt_e2_list[-1]
        return ((esoc **2 - (e1-e2)**2)/4)**0.5
    @property
    def final_adiabatic_ene(self):
        return self.opt_ene_list[-1]

class qchem_out_force(qchem_out):
    def __init__(self, filename=""):
        super().__init__(filename)
        self.force = None
        self.force_e1 = None
        self.force_e2 = None
        self.ene = None

    def read_file(self, filename=None, text=None, different_type="analytical", self_check=False):

        if filename:
            self.filename = filename
        if text:
            self.text = text
        else:
            self.text = open(self.filename, "r").read()
        self._parse_molecule()

        self.parse(different_type=different_type, self_check=self_check)

    def parse(self, different_type="analytical", self_check=False):
        import numpy as np
        # --- Prepare text lines ---
        if not getattr(self, "text", None):
            # If text is empty but filename is known, populate text
            if getattr(self, "filename", ""):
                try:
                    with open(self.filename, "r") as f:
                        self.text = f.read()
                except Exception:
                    self.text = ""
            else:
                self.text = ""
        lines = self.text.splitlines()

        # --- Try to sync molecule like read_force_from_file() does ---
        if getattr(self, "filename", ""):
            try:
                mol_structure = qchem_file()
                mol_structure.molecule.check = True
                mol_structure.read_from_file(self.filename)
                self.molecule = mol_structure.molecule
                self.molecule.check = True
            except Exception:
                # keep existing molecule if reading fails
                pass

        # --- Capture energy if present (use the last occurrence) ---
        for ln in lines:
            if "Total energy =" in ln:
                try:
                    
                    self.ene = float(ln.split("=")[-1])
                    
                except Exception:
                    pass
        if different_type == "analytical":
            self.force = self._parse_force_analytical(lines)
            return self.force
        # --- Decide gradient block keyword(s) ---
        force_key_word_checked = 0
        force_key_word = "fhiuadjskdnlashalfwwawldjalksfna"  # sentinel
        force_key_word_e1 = None
        force_key_word_e2 = None

        if not force_key_word_checked:
            if self_check:
                # infer from 'ideriv' line: ... ideriv <0/1>
                for ln in lines:
                    if "ideriv" in ln:
                        parts = [p for p in ln.split() if p]
                        try:
                            idv = int(parts[-1])
                        except Exception:
                            idv = None
                        if idv == 1:
                            force_key_word = "zheng adiabatic surface gradient"
                        elif idv == 0:
                            force_key_word = "FINAL TENSOR RESULT"
                        force_key_word_checked = 1
                        break
            if not force_key_word_checked:
                if different_type == "soc":
                    force_key_word = "zheng adiabatic surface gradient"
                    force_key_word_e1 = "zheng ASG first state"
                    force_key_word_e2 = "zheng ASG second state"
                elif different_type == "numerical":
                    force_key_word = "FINAL TENSOR RESULT"
                elif different_type == "analytical":
                    force_key_word = "Gradient of SCF Energy"
                elif different_type == "smd":
                    force_key_word = "-- total gradient after adding PCM contribution --"
                else:
                    force_key_word = different_type
                force_key_word_checked = 1


        force_check = 0  # 0: none, 1: main, 2: e1, 3: e2
        force, force_e1, force_e2 = [], [], []
        for ln in lines:
            # SOC state switches
            if different_type == "soc" and force_key_word_e1 and force_key_word_e2:
                if force_key_word_e1 in ln:
                    force_check = 2
                    force_e1 = []
                    continue
                if force_key_word_e2 in ln:
                    force_check = 3
                    force_e2 = []
                    continue

            # Start of main block
            if force_key_word in ln:
                force_check = 1
                force = []
                continue

            # SMD (PCM total gradient) table handling
            if force_key_word == "-- total gradient after adding PCM contribution --":
                if "Atom" in ln or "----" in ln:
                    continue
                if force_check:
                    if "----" in ln or ln.strip() == "":
                        force_check = 0
                        continue
                    parts = ln.split()
                    if len(parts) == 4:
                        # skip atom index, keep x y z
                        force.append(parts[1:])
                continue

            # Generic stop conditions for non-SMD
            if any(stop in ln for stop in ("Gradient time", "#")) or \
                    (force_check and "gradient" in ln.lower() and force_key_word != "Gradient of SCF Energy"):
                force_check = 0
                continue

            if force_check in (1, 2, 3):
                data = [p for p in ln.split() if p]
                if not data:
                    continue
                if force_check == 1:
                    force.append(data)
                elif force_check == 2:
                    force_e1.append(data)
                elif force_check == 3:
                    force_e2.append(data)

        # --- Post-process by type (match read_force_from_file) ---
        def _reshape_scf_block(block_2d):
            # For "Gradient of SCF Energy"
            # lines come in 4-line groups: header + X + Y + Z (possibly across columns)
            while [] in block_2d:
                block_2d.remove([])
            check = 0
            x_force, y_force, z_force = [], [], []
            for row in block_2d:
                if check % 4 == 1:
                    x_force.extend(row[1:])
                elif check % 4 == 2:
                    y_force.extend(row[1:])
                elif check % 4 == 3:
                    z_force.extend(row[1:])
                check += 1
            x = np.array(x_force, dtype=float)
            y = np.array(y_force, dtype=float)
            z = np.array(z_force, dtype=float)
            # shape (3, Natoms)
            return np.column_stack((x, y, z)).T

        if force_key_word == "zheng adiabatic surface gradient":
            # SOC: three blocks
            self.force = reshape_force(force).astype(float)
            self.force_e1 = reshape_force(force_e1).astype(float)
            self.force_e2 = reshape_force(force_e2).astype(float)

        elif force_key_word == "FINAL TENSOR RESULT":
            # Numerical tensor result; drop empty rows and the first two header lines,
            # then drop the first column (atom index)

            while [] in force:
                force.remove([])
            arr = np.array(force[2:], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                arr = arr[:, 1:]  # remove atom index
            self.force = arr

        elif force_key_word == "Gradient of SCF Energy":
            self.force = _reshape_scf_block(force)

        elif force_key_word == "-- total gradient after adding PCM contribution --":
            self.force = np.array(force, dtype=float)

        else:
            # Custom keyword: try best-effort numeric conversion
            try:
                self.force = np.array(force, dtype=float)
            except Exception:
                self.force = None
        return self
    def _parse_force_analytical(self,lines):
        i = lines.index(' Gradient of SCF Energy')
        total = self.molecule.natom
        j = 1

        x_grad = np.array([])
        y_grad = np.array([])
        z_grad = np.array([])
        while j < ((total-1)//6+1)*4 +1 :
            ln = lines[i+j]
            if j%4 ==1:
                j+=1
                continue
            elif j % 4 == 2:
                x_grad = np.append(x_grad,np.array(ln.split()[1:],dtype=float))
            elif j % 4 == 3:
                y_grad = np.append(y_grad,np.array(ln.split()[1:],dtype=float))
            elif j % 4 == 0:
                z_grad = np.append(z_grad,np.array(ln.split()[1:],dtype=float))
            j+=1
        force = np.column_stack((x_grad, y_grad, z_grad))
        return force


class qchem_out_freq(qchem_out):
    def __init__(self, filename=""):
        super().__init__(filename)
        self.enthalpy = None
        self.entropy = None

    def parse(self):
        for line in self.text.splitlines():
            if "Total Enthalpy:" in line and "QRRHO" not in line:
                self.enthalpy = float(line.split()[2])
            if "Total Entropy:" in line and "QRRHO" not in line:
                self.entropy = float(line.split()[2])
            if " Total energy" in line:
                self.ene = float(line.split("=")[1])

    def calc_gibbs(self):
        global Hartree_to_kcal
        if self.ene is None or self.enthalpy is None or self.entropy is None:
            raise ValueError("Missing ene/enthalpy/entropy for Gibbs calculation.")
        gibbs_energy = self.ene * Hartree_to_kcal + self.enthalpy - (0.273 + 0.025) * self.entropy
        self.gibbs_energy = gibbs_energy
        return gibbs_energy


def reshape_force(force_block):
    x_force, y_force, z_force = [], [], []
    check = 0
    for row in force_block:
        if all(p.isdigit() for p in row):
            check = 0
            continue

        if check % 3 == 0:  # x 分量行
            x_force.extend([float(v) for v in row[1:]])
        elif check % 3 == 1:  # y 分量行
            y_force.extend([float(v) for v in row[1:]])
        elif check % 3 == 2:  # z 分量行
            z_force.extend([float(v) for v in row[1:]])
        check += 1

    x_force = np.array(x_force)
    y_force = np.array(y_force)
    z_force = np.array(z_force)

    if not (len(x_force) == len(y_force) == len(z_force)):
        raise ValueError(f"梯度分量长度不一致: X={len(x_force)}, Y={len(y_force)}, Z={len(z_force)}")

    return np.vstack([x_force, y_force, z_force])


class qchem_out_geomene(qchem_out):

    def __init__(self, filename=""):
        super().__init__(filename)

    def parse(self):
        lines = self.text.splitlines()
        self.ene = None
        carti, blocks = [], []

        for i, line in enumerate(lines):
            if " Total energy" in line:
                try:
                    self.ene = float(line.split("=")[1])
                except Exception:
                    pass

            if "Standard Nuclear Orientation" in line:
                carti = []
                continue

            if carti is not None and len(line.strip().split()) == 5:
                tokens = line.split()
                try:
                    atom, x, y, z = tokens[1], float(tokens[2]), float(tokens[3]), float(tokens[4])
                    carti.append([atom, x, y, z])
                except Exception:
                    continue

            elif carti:
                blocks.append(carti.copy())
                carti = None

        if blocks:
            mol = molecule()
            mol.carti = blocks[-1]  # 最后一帧几何
            mol.check = True
            self.molecule = mol


import re


class qchem_out_aimd(qchem_out):

    def __init__(self, filename=""):
        super().__init__(filename)
        self.aimd_geoms = []
        self.aimd_steps = 0

    def parse(self):
        step_re = re.compile(
            r"^TIME STEP #\s*(\d+)\s*\(t\s*=\s*([+\-0-9.eEdD]+)\s*a\.u\.\s*=\s*([+\-0-9.eEdD]+)\s*fs\)",
            re.M,
        )
        matches = list(step_re.finditer(self.text))
        if not matches:
            return

        def _parse_gradient_from_block(lines):
            key = "Gradient of SCF Energy"
            collecting, rows = False, []
            for ln in lines:
                if key in ln:
                    collecting = True
                    continue
                if not collecting:
                    continue
                low = ln.strip().lower()
                if any(w in low for w in ["gradient time", "maximum gradient", "rms gradient",
                                          "cartesian coordinates", "standard nuclear orientation"]) or ln.strip() == "":
                    if rows:
                        break
                    else:
                        continue
                parts = ln.split()
                if parts:
                    rows.append(parts)
            if not rows:
                return None

            x_vals, y_vals, z_vals, chk = [], [], [], 0
            for r in rows:
                if chk % 4 == 1:
                    x_vals.extend(r[1:] if len(r) >= 2 else r)
                elif chk % 4 == 2:
                    y_vals.extend(r[1:] if len(r) >= 2 else r)
                elif chk % 4 == 3:
                    z_vals.extend(r[1:] if len(r) >= 2 else r)
                chk += 1
            if not (x_vals and y_vals and z_vals):
                return None

            def to_float(a):
                return np.array([float(t.replace("D", "E").replace("d", "e")) for t in a], dtype=float)

            try:
                return np.column_stack((to_float(x_vals), to_float(y_vals), to_float(z_vals))).T
            except Exception:
                return None

        for k, m in enumerate(matches):
            b_start, b_end = m.start(), (matches[k + 1].start() if (k + 1 < len(matches)) else len(self.text))
            block, lines = self.text[b_start:b_end], self.text[b_start:b_end].splitlines()

            step_idx = int(m.group(1))
            t_au = float(m.group(2).replace("D", "E")) if m.group(2) else None
            t_fs = float(m.group(3).replace("D", "E")) if m.group(3) else None

            temp = None
            mtemp = re.search(r"Instantaneous\s+Temperature\s*=\s*([+\-0-9.eEdD]+)\s*K", block)
            if mtemp:
                try:
                    temp = float(mtemp.group(1).replace("D", "E"))
                except Exception:
                    pass

            tmp = qchem_out_geomene()
            tmp.read_file(text=block, read_charge_and_spin=False)

            g = geoms()
            g.index = step_idx
            g.energy = getattr(tmp, "ene", None)
            g.carti = getattr(tmp.molecule, "carti", []) if hasattr(tmp, "molecule") else []

            charge, mult = None, None
            m1 = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", block)
            if m1:
                charge, mult = int(m1.group(1)), int(m1.group(2))
            else:
                m2 = re.search(r"Net\s+Charge\s*[:=]\s*(-?\d+)", block)
                if m2:
                    charge = int(m2.group(1))
                m3 = (re.search(r"Multiplicity\s*[:=]\s*(\d+)", block)
                      or re.search(r"Spin\s+multiplicity\s*[:=]\s*(\d+)", block))
                if m3:
                    mult = int(m3.group(1))
            if charge is not None:
                g.charge = charge
            if mult is not None:
                g.multistate = mult

            grad = _parse_gradient_from_block(lines)
            if grad is not None:
                g.grad = grad

            g.time_au, g.time_fs, g.temperature_K = t_au, t_fs, temp
            self.aimd_geoms.append(g)

        self.aimd_steps = len(self.aimd_geoms)

    def get_trajectory(self):
        return [g.carti for g in self.aimd_geoms]

    def get_energies(self):
        return np.array([g.energy for g in self.aimd_geoms if g.energy is not None])


class multiple_qchem_jobs(object):
    def __init__(self):
        self.jobs = []
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

    def read_from_file(self, filename):
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
        with open(new_file_name, "w") as f: f.write(self.output)


class qchem_out_multi:

    def __init__(self):
        self.tasks = []
        self.filenames = []

    def add_task(self, out_obj):
        if not isinstance(out_obj, qchem_out):
            raise TypeError("只能添加 qchem_out 或其子类对象")
        self.tasks.append(out_obj)

    def read_files(self, filenames, out_cls):
        self.filenames = filenames
        self.tasks = []
        for fn in filenames:
            try:
                out = out_cls(fn)
                out.read_file(fn)
                self.tasks.append(out)
            except:
                print(f"Skipping {fn}")
                continue

    @property
    def ntasks(self):
        return len(self.tasks)

    def get_all_energies(self):
        return [task.final_ene for task in self.tasks]

    def get_all_final_geoms(self):
        return [task.final_geom for task in self.tasks]

    def summary(self):
        print(f"包含 {self.ntasks} 个任务")
        for i, task in enumerate(self.tasks):
            print(f"  [{i}] 文件={task.filename}, 最终能量={task.final_ene}")


class ExcitedState:
    def __init__(self, state_idx, charge=None, multiplicity=None):
        self.state_idx = state_idx
        self.charge = charge
        self.multiplicity = multiplicity
        self.excitation_energy = None  # eV
        self.total_energy = None  # Hartree
        self.osc_strength = None
        self.dipole_mom = None
        self.trans_mom = None  # (Tx, Ty, Tz)
        self.trans_mom_norm = None  # |T|
        self.transitions = []  # list of dict: {"from":..,"to":..,"amplitude":..}
        self.gradient = None  # (Natom, 3)
        self.esp_transition_density = None
        self.esp_charges = None

    def __repr__(self):
        e_ev = f"{self.excitation_energy:.3f}" if self.excitation_energy is not None else "N/A"
        e_tot = f"{self.total_energy:.6f}" if self.total_energy is not None else "N/A"
        f_str = f"{self.osc_strength:.4f}" if self.osc_strength is not None else "N/A"
        mult = self.multiplicity if self.multiplicity is not None else "N/A"
        return f"<ExcitedState {self.state_idx}: E={e_ev} eV, Etot={e_tot} au, f={f_str}, Mult={mult}>"


class qchem_out_excite(qchem_out):
    def __init__(self, filename="", read_esp=True):
        super().__init__(filename)
        self.states = []  # list of ExcitedState
        self.read_esp = read_esp

    def parse(self):
        lines = self.text.splitlines()
        charge, multiplicity = None, None
        carti = []
        for i, line in enumerate(lines):
            if "Standard Nuclear Orientation" in line:
                carti = []
                k = i + 3  # 跳过标题和分隔线
                while k < len(lines):
                    parts = lines[k].split()
                    if len(parts) == 5:  # 格式: index, atom, x, y, z
                        atom, x, y, z = parts[1], float(parts[2]), float(parts[3]), float(parts[4])
                        carti.append([atom, x, y, z])
                    else:
                        break
                    k += 1
        if carti:
            self.molecule.carti = carti
        for ln in lines:
            if "Charge =" in ln and "Multiplicity" in ln:
                parts = ln.split()
                charge = int(parts[2])
                multiplicity = int(parts[-1])
                break

        gs = ExcitedState(0, charge, multiplicity)
        dipole_find = False
        for ln in lines:
            if "SCF   energy" in ln:
                gs.total_energy = float(ln.split("=")[-1])
                break
        for ln in lines:
            if "Dipole Moment (Debye)" in ln:
                dipole_find = True
                continue
            if dipole_find:
                dipoles = ln.split()
                tx, ty, tz = float(dipoles[1]), float(dipoles[3]), float(dipoles[5])
                gs.dipole_mom = (tx, ty, tz)
                dipole_find = False
                break
        self.states.append(gs)

        for i, line in enumerate(lines):
            if line.strip().startswith("Excited state"):
                parts = line.split()
                state_idx = int(parts[2].strip(":"))
                energy_ev = float(parts[-1])
                st = ExcitedState(state_idx, charge, "Singlet")  # 默认Singlet, 后面覆盖
                st.excitation_energy = energy_ev

                j = i + 1
                while j < len(lines) and lines[j].strip() != "":
                    l = lines[j].strip()
                    if l.startswith("Total energy for state"):
                        st.total_energy = float(l.split()[-2])
                    elif l.startswith("Multiplicity"):
                        st.multiplicity = l.split(":")[1].strip()
                    elif l.startswith("Strength"):
                        st.osc_strength = float(l.split(":")[1])
                    elif "->" in l and "amplitude" in l:
                        tparts = l.replace("=", "").split()
                        st.transitions.append({
                            "from": tparts[0],
                            "to": tparts[2],
                            "amplitude": float(tparts[-1])
                        })
                    elif l.startswith("Trans. Mom."):
                        parts = l.replace("Trans. Mom.:", "").split()
                        tx, ty, tz = float(parts[0]), float(parts[2]), float(parts[4])
                        st.trans_mom = (tx, ty, tz)
                        st.trans_mom_norm = (tx ** 2 + ty ** 2 + tz ** 2) ** 0.5

                    elif "amplitude" in l and "-->" in l:
                        segs = l.replace("=", "").split()
                        st.transitions.append({
                            "from": segs[0],
                            "to": segs[2],
                            "amplitude": float(segs[-1])
                        })
                    j += 1

                self.states.append(st)

        if "Gradient of SCF Energy" in self.text:
            start = self.text.index("Gradient of SCF Energy")
            end = self.text.index("Gradient time", start)
            block = self.text[start:end]
            self._parse_gradient_block(block, 0)  # 基态=0
        current_state = None
        for i, line in enumerate(lines):
            if "CIS" in line and "State Energy" in line:
                m = re.search(r"CIS\s+(\d+)\s+State Energy", line)
                if m:
                    current_state = int(m.group(1))

            if "Gradient of the state energy" in line and current_state is not None:
                j = i
                while j < len(lines) and "Gradient time" not in lines[j]:
                    j += 1
                block = "\n".join(lines[i:j + 1])
                self._parse_gradient_block(block, current_state)
                current_state = None  # reset
        if self.read_esp:
            self._parse_excite_esp_blocks(lines)
            self._parse_esp_charge(lines)
        if self.states:
            self.ene = self.states[-1].total_energy

    def _parse_gradient_block(self, block, st_idx):
        force_block = []
        for line in block.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                int(parts[0])  # 原子编号行
                [float(x.replace("D", "E")) for x in parts[1:]]  # 确认其余是数值
                force_block.append(parts)
            except Exception:
                continue  # 非数值行直接跳过

        if force_block:
            grad = reshape_force(force_block)  # 用改进版 reshape
            if grad is not None:
                for st in self.states:
                    if st.state_idx == st_idx:
                        st.gradient = grad.T  # (Natoms, 3)
                        break

    def _parse_excite_esp_blocks(self, lines):
        text = "\n".join(lines)
        # --- ESP charges for excited states ---
        m1 = re.search(r"ESP charges for excited states(.+?)ESP charges for transition densities", text, re.S)
        m2 = re.search(r"ESP charges for transition densities(.+?)(?:\Z|\n\s*\n)", text, re.S)
        if not (m1 and m2):
            return

        block_excited = m1.group(1)
        block_transition = m2.group(1)

        def parse_esp_block(block):
            """
            Parse possibly multi-part ESP block. Concatenate horizontally all sub-blocks.
            Returns array of shape (n_state, n_atom)
            """
            lines = block.strip().splitlines()
            subblocks = []
            i = 0
            while i < len(lines):
                # find header line with state indices, e.g. "1 2 3 4 5 6"
                if "---" in lines[i]:
                    break
                if re.match(r"^\s*\d+(\s+\d+)+\s*$", lines[i]):
                    header_nums = [int(x) for x in re.findall(r"\d+", lines[i])]
                    j = i + 1
                    block_rows = []
                    while j < len(lines):
                        if "---" in lines[j]:
                            break
                        ln = lines[j]
                        # next header or empty line terminates subblock
                        if re.match(r"^\s*\d+(\s+\d+)+\s*$", ln) or not ln.strip():
                            break

                        parts = re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", ln)
                        if len(parts) == len(header_nums):  # includes atom index
                            block_rows.append([float(x) for x in parts[0:]])

                        j += 1
                    if block_rows:
                        subblocks.append(np.array(block_rows))  # shape (n_atom, n_substates)
                    i = j
                else:
                    i += 1
            # print(subblocks)
            if not subblocks:
                return np.zeros((0, 0))
            # concatenate horizontally (same atoms, different states)
            full = np.concatenate(subblocks, axis=1)  # (n_atom, n_total_state)
            return full.T  # (n_state, n_atom)

        esp_excited = parse_esp_block(block_excited)
        esp_trans = parse_esp_block(block_transition)
        # Assign per-state
        n_states = min(len(self.states) - 1, esp_excited.shape[0])
        for i in range(n_states):
            self.states[i + 1].esp_charges = esp_excited[i]
            self.states[i + 1].esp_transition_density = esp_trans[i]

    def summary(self):
        print(f"解析到 {len(self.states)} 个态 (含基态)")
        for st in self.states:
            print(st)
