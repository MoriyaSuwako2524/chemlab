import sys


def judge(filename): #classification of different input files, return 0 for qchem reference inp
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

class qchem_file(object): #standard qchem inp file class
    def __init__(self):
        self.molecule = molecule()
        self.rem = rem()
        self.pcm = pcm()
        self.opt = opt()
        self.opt2 = opt2()
        self.remain_texts = ""
    def generate_inp(self,new_file_name):
        new_file = open(new_file_name,"w")

    def read_from_file(self,filename):
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
            elif "$" in file[i]:
                module_start = -1
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










class molecule(object):
    def __init__(self):
        self.charge = 0
        self.multistate = 1
        self.carti = []
        self.check = False
    def return_output_format(self):
        out = "\n$molecule\n"
        out += str(self.charge)+" "+str(self.multistate)+"\n"
        for cors in self.carti:
            text = ""
            for j in cors:
                text+= " "+j +" "
            text+="\n"
            out += text
        out += "$end\n\n"
        return out

class rem(object):
    def __init__(self):
        self.texts = ""
        self.check = False
    def return_output_format(self):
        out = ["\n$rem\n"]
        out.append(self.texts)
        out.append("$end\n\n")
        return out

class pcm(object):
    def __init__(self):
        self.name = "pcm"
        self.theory = "IEFPCM"
        self.solvent = ""
        self.solvent_die = 4.0
        self.check = False

class opt(object):
    def __init__(self):
        self.constraint = constraint()
        self.fix_atom = fix_atom()
        self.check = False

class constraint(object):
    def __init__(self):
        self.stre =[]

class opt2(object):
    def __init__(self):
        self.check = False
        self.r12 = []
        self.r12mr34 = []
        self.r12pr34 = []

    def modify_r12(self,which_line,j):
        self.r12[which_line][2] = [j]
    def modify_r12mr34(self,which_line,j):
        self.r12mr34[which_line][4] = [j]
    def return_output_format(self):
        out = "\n$opt2\n"
        text = ""
        for i in self.r12:
            text += "r12 "
            for j in i:
                text += str(j) +" "
            text += "\n"
        for i in self.r12mr34:
            text += "r12mr34 "
            for j in i:
                text += str(j) +" "
            text += "\n"
        for i in self.r12pr34:
            text += "r12pr34 "
            for j in i:
                text += str(j) +" "
            text += "\n"
        out += text
        out += "$end\n\n"
        return out



class fix_atom(object):
    def __init__(self):
        self.check = False
        self.fixed_atoms = []
    def input_fixed_atoms(self,atom,car="xyz"):
        self.fixed_atoms.append([atom,car])

#qchem = qchem_file()
#qchem.molecule.check = True
#qchem.read_from_file("1a.inp")
#print(qchem.remain_texts)
#print(qchem.molecule.carti)

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
    def index_of_geom_i(self,index):
        return self.geoms[index].index
    def energy_of_geom_i(self,index):
        return self.geoms[index].energy
    def gradient_of_geom_i(self,index):
        return self.geoms[index].gradient
    def displacement_of_geom_i(self,index):
        return self.geoms[index].displacement
    def return_final_molecule_carti(self):
        if self.optimizer == "default":
            return self.geoms[-1].molecule.carti
        elif self.optimizer == "1996":
            return self.geoms[-2].molecule.carti
    def return_final_molecule_energy(self):
        return self.geoms[-1].energy
    def return_average_scf_time(self):
        total_scf_time  =0
        for geom in self.geoms:
            total_scf_time += geom.scf_time
        return total_scf_time/len(self.geoms)

    def read_opt_from_file(self,filename):
        self.filename = filename
        if self.optimizer == "default":
            out_file = open(filename,"r").read().split("OPTIMIZATION CYCLE")
            for text in out_file[0].split("\n"):
                if "geom_opt_driver = 1996" in text:
                    self.optimizer = "1996"
                    break
        if self.optimizer == "1996":
            out_file = open(filename,"r").read().split("Optimization Cycle:")

        for i in range(2,len(out_file)): #check every split geoms
            geom_texts = out_file[i].split("\n")

            geom = geoms()
            geom.index = i-1
            internal_index = int(geom_texts[0][2:])

            detected_data = 0
            molecule_check = -1
            detected_molecule = 0
            datas = []
            molecule_cart = []
            geom_have_energy = 0
            for text_line in geom_texts:
                if "Step {}".format(internal_index) in text_line:
                    detected_data = 1
                    continue
                elif "Step Taken" in text_line:
                    detected_data = 1
                    continue
                if self.read_scf_time == True:
                    if " SCF time" in text_line:
                        scf_time = text_line.split(" ")
                        while '' in scf_time:
                            scf_time.remove('')
                        geom.scf_time = float(scf_time[3][:-1])


                if "Energy is" in text_line and self.optimizer=="1996":
                    energy = float(text_line.split(" ")[-1])
                    geom_have_energy = 1
                    geom.energy = energy
                if detected_data == 1:
                    datas.append((text_line))

                if "Standard Nuclear Orientation (Angstroms)" in text_line:
                    detected_molecule = 1
                    continue
                if detected_molecule == 1:
                    if " ----------------------------------------------------------------" in text_line:
                        molecule_check += 1
                        continue
                if molecule_check == 0:
                    molecule_cart.append(text_line)
                if "**  OPTIMIZATION CONVERGED  **" in text_line:
                    self.opt_converged = True

            for cart in molecule_cart:
                cart = cart.split(" ")
                while "" in cart:
                    cart.remove("")
                geom.molecule.carti.append(cart[1:])
            for data in datas:
                if self.optimizer == "1996":
                    break
                if "Energy is " in data:
                    energy = float(data.split(" ")[-1])
                    geom_have_energy = 1
                    geom.energy = energy
                if "Gradient" in data:

                    data = data.split(" ")
                    while '' in data:
                        data.remove('')
                    data = data[1].split("\t")
                    gradient = float(data[0])
                    geom.gradient = gradient
                if "Displacement" in data:
                    if self.optimizer == "1996":
                        continue
                    data = data.split(" ")
                    while '' in data:
                        data.remove('')
                    data = data[1].split("\t")
                    displacement = float(data[0])
                    geom.displacement = displacement

                if i == len(out_file)-1:
                    if self.read_final_cpu_time == True:
                        if " Total job time" in data:
                            jobtime = data.split(" ")
                            while '' in jobtime:
                                jobtime.remove('')
                            self.final_cpu_time = float(jobtime[-1][:-6])
                    if "OPTIMIZATION DID NOT CONVERGE" in data:
                        self.opt_converged = False
                    if "END OF GEOMETRY OPTIMIZER USING LIBOPT3" in data:
                        self.opt_converged = True




            if geom_have_energy == 0:
                a=1
            else:
                self.geoms.append(geom)
        self.geom_number = len(self.geoms)



class geoms(object):
    def __init__(self):
        self.index = 0
        self.energy = 0
        self.gradient = 0
        self.displacement = 0
        self.molecule = molecule()




