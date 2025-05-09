import sys
import numpy as np

def judge(filename): #!this function no longer used. classification of different input files, return 0 for qchem reference inp
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

    def read_from_file(self,filename):# read standard inp file, require filename that consist of path and suffix. use class.check to determine which part you want to modify
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
                    self.remain_texts += "\n$molecule\nread\n$end\n\n"
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










class molecule(object):
    def __init__(self):
        self.charge = 0
        self.multistate = 1
        self.carti = []
        self.check = False
    def return_atom_xyz(self,atom):

        return np.array([self.carti[atom][1], self.carti[atom][2], self.carti[atom][3]]).astype(float)
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

class rem(object):
    def __init__(self):
        self.texts = ""
        self.check = False
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

    def return_output_format(self):
        out = "\n$opt\nCONSTRAINT\n"
        text = ""
        for i in self.constraint.stre[0]:
            text += i + " "
        text += "\n"
        out += text
        out += "ENDCONSTRAINT\n$end\n\n"
        return out

class constraint(object):
    def __init__(self):
        self.stre =[]
    def modify_stre(self,which_line,j):
        self.stre[which_line][3] = str(j)

class opt2(object):
    def __init__(self):
        self.check = False
        self.r12 = []
        self.r12mr34 = []
        self.r12pr34 = []

    def modify_r12(self,which_line,j):
        self.r12[which_line][2] = j
    def modify_r12mr34(self,which_line,j):
        self.r12mr34[which_line][4] = j
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
        if self.opt_converged == False:
            print("{} don't have energy".format(self.filename))
        for i in range(len(self.geoms)):

            if self.geoms[-1-i].have_energy == 1:
                return self.geoms[-1-i].energy
    def return_average_scf_time(self):
        total_scf_time  =0
        for geom in self.geoms:
            total_scf_time += geom.scf_time
        return total_scf_time/len(self.geoms)
    def return_opt_soc_ene(self):
        return self.opt_ene_list[-1]
    def read_ene_job(self,filename):
        file = open(filename, "r").read().split("\n")
        for text in file:
            if " Total energy" in text:
                data = float(text.split("=")[1])
        self.ene = data
    def read_multiple_jobs_out(self,filename):
        self.filename = filename
        file = open(filename,"r").read().split(" Welcome to Q-Chem")
        self.out_texts = file
        self.total_jobs = len(file)-1
    def read_pes_scan_from_file(self,filename,return_type = "list"):
        file = open(filename, "r").read().split("\n")
        scan_value_list = []
        scan_ene_list = []

        for i in range(len(file)):
            if "PES scan" in file[i]:
                data = file[i].split(":")
                scan_value_list.append(float(data[1].split(" ")[-2]))
                scan_ene_list.append(float(data[2]))
        if return_type == "list":
            a = 1
        elif return_type == "numpy" or return_type == "np":
            scan_value_list = np.array(scan_value_list)
            scan_ene_list = np.array(scan_ene_list)
        self.scan_value_list = scan_value_list
        self.scan_ene_list = scan_ene_list
        return scan_value_list, scan_ene_list

    def read_soc_opt(self,filename,return_type): #这个函数是给peizheng的soc单独写的！ this function can only use in soc opt jobs written by peizheng!
        file = open(filename, "r").read().split("\n")
        num = 1
        opt_time_list = []
        opt_ene_list = []
        for i in range(len(file)):
            if "E_adiab: " in file[i]:
                data = file[i].split(":")
                opt_time_list.append(num)
                opt_ene_list.append(float(data[-1]))
                num += 1
        if return_type == "list":
            a = 1
        elif return_type == "numpy" or return_type == "np":
            opt_time_list = np.array(opt_time_list)
            opt_ene_list = np.array(opt_ene_list)
        self.opt_time_list = opt_time_list
        self.opt_ene_list = opt_ene_list
        return opt_time_list, opt_ene_list

    def read_opt_from_file(self,filename,out_text=""):
        if filename != "":
            self.filename = filename

        if out_text == "":
            out_files = open(filename,"r").read()
        else:
            out_files = out_text
        if self.optimizer == "default":
            out_file = out_files.split("OPTIMIZATION CYCLE")
            for text in out_file[0].split("\n"):
                if "geom_opt_driver = 1996" in text:
                    self.optimizer = "1996"
                    break
        if self.optimizer == "1996":
            out_file = out_files.split("Optimization Cycle:")

        for i in range(1,len(out_file)): #check every split geoms

            geom_texts = out_file[i].split("\n")

            geom = geoms()
            geom.index = i-1
            internal_index = int(geom_texts[0][2:])

            detected_data = 0
            molecule_check = -1
            detected_molecule = 0
            datas = []
            molecule_cart = []
            self.geom_have_energy = 0
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
                    self.geom_have_energy = 1
                    geom.have_energy = 1
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
                    self.geom_have_energy = 1
                    geom.have_energy = 1
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




            if self.geom_have_energy == 0:
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




