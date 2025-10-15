import sys
import numpy as np
Hartree_to_kcal = 627.51
atom_charge_dict = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,
    "Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,
    "Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,
    "Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,
    "Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54
}
SPIN_REF = {1:"s",2:"d",3:"t",4:"quartlet",5:"q",6:"sextuplet"}


class unit_type:
    category = None
    def __init__(self, value, unit):
        self.value = np.array(value, dtype=float)
        self.unit = unit
        self.DICT = {}
        self.modify_value = False

    def convert_to(self, target):
        factor = self.DICT[target] / self.DICT[self.unit]
        value = self.value * factor
        if self.modify_value:
            self.value = value
            self.unit = target
        return value

class ENERGY(unit_type):
    category = "energy"
    def __init__(self, value, unit="hartree"):
        super().__init__(value, unit)
        self.DICT = {"hartree": 1, "kcal/mol":627.51,"kcal": 627.51, "ev": 27.2113863, "kj": 2625.5}


class DISTANCE(unit_type):
    category = "distance"
    def __init__(self, value, unit="ang"):
        super().__init__(value, unit)
        self.DICT = {"ang": 1, "bohr": 1/0.529177, "nm": 10}

from itertools import product


class MASS(unit_type):
    category = "mass"
    def __init__(self, value, unit="amu"):
        super().__init__(value, unit)
        self.DICT = {
            "amu": 1,
            "g": 1.66054e-24,
            "kg": 1.66054e-27
        }


class TIME(unit_type):
    category = "time"
    def __init__(self, value, unit="fs"):
        super().__init__(value, unit)
        self.DICT = {
            "fs": 1,
            "ps": 1e-3,
            "ns": 1e-6,
            "s": 1e-15
        }



class complex_unit_type:
    def __init__(self, value, units: dict):
        """
        value:  scalar or numpy array
        units: dict, e.g. {"energy": ("hartree", 1), "distance": ("bohr", -1)}
        """
        self.value = np.array(value, dtype=float)
        self.units = units
        self.modify_value = False

    def convert_to(self, target_units: dict):
        factor = 1.0
        for category, (unit, power) in self.units.items():
            base = UNIT_REGISTRY[category](1.0, unit)
            target_unit = target_units[category][0]
            factor *= (base.convert_to(target_unit)) ** power
        value = self.value * factor
        if self.modify_value:
            self.value = value
            self.units = target_units
        return value

    def generate_all_conversions(self):

        unit_lists = []
        categories = []
        for category, (unit, power) in self.units.items():
            categories.append(category)
            unit_lists.append(list(UNIT_REGISTRY[category](1, unit).DICT.keys()))

        results = {}
        for combo in product(*unit_lists):
            target_units = {cat: (u, self.units[cat][1]) for cat, u in zip(categories, combo)}
            key = " * ".join([f"{u}^{p}" if p != 1 else u
                              for (_, (u, p)) in target_units.items()])
            results[key] = self.convert_to(target_units)
        return results

class CHARGE(unit_type):
    category = "charge"

    def __init__(self, value, unit="e"):
        super().__init__(value, unit)
        self.DICT = {
            "e": 1.0,                 # 1 e = 1 e
            "C": 1.0 / 1.602176634e-19,  # 1 C = 6.2415e18 e
        }

# --- 偶极矩单位 ---
class DIPOLE(complex_unit_type):
    category = "dipole"
    def __init__(self, value, charge_unit="e", distance_unit="bohr"):
        super().__init__(np.array(value, dtype=float), {
            "charge": (charge_unit, 1),
            "distance": (distance_unit, 1)
        })
        # 1 Debye = 0.393430307 e·Å = 0.20819434 e·bohr
        self.DICT = {
            "Debye": 1.0,  # 作为基准
            "e*bohr": 1.0 / 0.20819434,
            "e*ang":  1.0 / 0.393430307,
            "C*m":    1.0 / 3.33564e-30,
        }
class FORCE(complex_unit_type):
    def __init__(self, value, energy_unit="hartree", distance_unit="bohr"):
        super().__init__( -np.array(value, dtype=float), {  # 注意加负号
            "energy": (energy_unit, 1),
            "distance": (distance_unit, -1)
        })


class GRADIENT(complex_unit_type):
    def __init__(self, value, energy_unit="hartree", distance_unit="bohr"):
        super().__init__( np.array(value, dtype=float), {
            "energy": (energy_unit, 1),
            "distance": (distance_unit, -1)
        })
UNIT_REGISTRY = {
    "energy": ENERGY,
    "distance": DISTANCE,
    "mass": MASS,
    "time": TIME,
    "charge": CHARGE,
    "dipole": DIPOLE,
    "force": FORCE,
    "gradient": GRADIENT,
}

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

    def read_file(self, filename=None, text=None):
        if filename:
            self.filename = filename
        if text:
            self.text = text
        else:
            self.text = open(self.filename, "r").read()
        self._parse_charge_and_spin()
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
    def _parse_charge_and_spin(self):
        lines = self.text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("$molecule"):
                next_line = lines[i + 1].strip()
                charge, spin = map(int, next_line.split()[:2])
                self.charge  = charge
                self.spin = spin
                return charge, spin
        raise ValueError("No '$molecule' section found or missing charge/spin line.")

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
class qchem_out_force(qchem_out):
    def __init__(self, filename=""):
        super().__init__(filename)
        self.force = None

    def parse(self):
        lines = self.text.splitlines()
        force_block = []
        reading = False
        for line in lines:
            if "Gradient of SCF Energy" in line:
                reading = True
                continue
            if reading:
                if "Gradient time" in line:
                    break
                parts = line.split()
                if parts:
                    force_block.append(parts)
        if force_block:
            self.force = reshape_force(force_block)
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

        if check % 3 == 0:   # x 分量行
            x_force.extend([float(v) for v in row[1:]])
        elif check % 3 == 1: # y 分量行
            y_force.extend([float(v) for v in row[1:]])
        elif check % 3 == 2: # z 分量行
            z_force.extend([float(v) for v in row[1:]])
        check += 1

    x_force = np.array(x_force)
    y_force = np.array(y_force)
    z_force = np.array(z_force)

    if not (len(x_force) == len(y_force) == len(z_force)):
        raise ValueError(f"梯度分量长度不一致: X={len(x_force)}, Y={len(y_force)}, Z={len(z_force)}")

    return np.vstack([x_force, y_force, z_force])


class qchem_out_geomene(qchem_out):
    """
    Minimal parser: only reads Total energy and last Standard Nuclear Orientation geometry
    """
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
            tmp.read_file(text=block)

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
        self.jobs=[]
        self.filename = ""
        self.molecule_check = True
        self.rem_check = True
        self.opt_check = False
        self.pcm_check = False
    def match_check(self, job):
        if self.molecule_check:
            job.molecule.check = True
        else: job.molecule.check = False
        if self.rem_check:
            job.rem.check = True
        else: job.rem.check = False
        if self.opt_check:
            job.opt.check = True
        else: job.opt.check = False
        if self.pcm_check:
            job.pcm.check = True
        else: job.pcm.check = False
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
            if i == 0: out += f"{job.output}\n"
            else: out += f"\n@@@\n\n{job.output}\n"
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
            out = out_cls(fn)
            out.read_file()
            self.tasks.append(out)

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
        self.excitation_energy = None   # eV
        self.total_energy = None        # Hartree
        self.osc_strength = None
        self.dipole_mom = None
        self.trans_mom = None           # (Tx, Ty, Tz)
        self.trans_mom_norm = None      # |T|
        self.transitions = []           # list of dict: {"from":..,"to":..,"amplitude":..}
        self.gradient = None            # (Natom, 3)

    def __repr__(self):
        e_ev = f"{self.excitation_energy:.3f}" if self.excitation_energy is not None else "N/A"
        e_tot = f"{self.total_energy:.6f}" if self.total_energy is not None else "N/A"
        f_str = f"{self.osc_strength:.4f}" if self.osc_strength is not None else "N/A"
        mult = self.multiplicity if self.multiplicity is not None else "N/A"
        return f"<ExcitedState {self.state_idx}: E={e_ev} eV, Etot={e_tot} au, f={f_str}, Mult={mult}>"


class qchem_out_excite(qchem_out):
    def __init__(self, filename=""):
        super().__init__(filename)
        self.states = []  # list of ExcitedState

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
    def summary(self):
        print(f"解析到 {len(self.states)} 个态 (含基态)")
        for st in self.states:
            print(st)
