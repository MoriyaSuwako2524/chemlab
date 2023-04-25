import numpy as np
import matplotlib.pyplot as plt





class reaction(object):
    def __init__(self,rate=0,A=None,B=None,C=None):
        self.rate = rate
        self.conc_of_A = A
        self.conc_of_B = B
        self.conc_of_C = C
        self.level = 0
    def judge_level(self,conc_of_A,conc_of_B,conc_of_C):
        if conc_of_A == None:
            return "error"
        else:
            if conc_of_B == None:
                self.level = 1
            else:
                if conc_of_C == None:
                    self.level = 2
                else:
                    self.level = 3

class compound(object):
    def __init__(self,init_conc):
        self.conc = 0



class model(object):
    def __init__(self):
        self.time = 0
        self.S1 = compound(0.1)
        self.S2 = compound(0.1)
        self.S3 = compound(0.0)
        self.reaction1 = reaction(rate=0,A=self.S1,B=self.S2)


