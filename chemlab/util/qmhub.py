import numpy as np
from chemlab.util.file_system import molecule,ELEMENT_DICT,NUM2ELEMENT


class mm_molecule(molecule):
    def __init__(self):
        super(mm_molecule,self).__init__()
        self.coord = []
        self.charges = []
    @property
    def external_charges(self):
        return np.column_stack((self.coord, self.charges))
class qm_molecule(molecule):
    def __init__(self):
        super(qm_molecule,self).__init__()
        self.coord = []
        self.qm_type = []
    @property
    def _atoms(self):
        return np.vectorize(NUM2ELEMENT.get)(self.qm_type)
    @property
    def _carti(self):
        return np.column_stack((self._atoms,self.coord))

class qmmm_molecule(molecule):
    def __init__(self):
        super(qmmm_molecule, self).__init__()
        self.mm_molecule = mm_molecule()
        self.qm_molecule = qm_molecule()

    def load_qmmm(self, filename):
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        header = lines[0].split()

        n_qm = int(header[0])
        n_mm = int(header[1])
        qm_charge = int(header[2])
        qm_spin = int(header[3])

        self.qm_molecule.charge = qm_charge
        self.qm_molecule.spin = qm_spin

        qm_lines = lines[1:1 + n_qm]

        for line in qm_lines:
            cols = line.split()
            x, y, z = map(float, cols[0:3])
            Z = int(cols[-1])

            self.qm_molecule.coord.append([x, y, z])
            self.qm_molecule.qm_type.append(Z)

        mm_lines = lines[1 + n_qm:1 + n_qm + n_mm]

        for line in mm_lines:
            cols = line.split()
            x, y, z = map(float, cols[0:3])
            q_mm = float(cols[3])

            self.mm_molecule.coord.append([x, y, z])
            self.mm_molecule.charges.append(q_mm)

        assert len(self.qm_molecule.coord) == n_qm, "QM atom count mismatch"
        assert len(self.mm_molecule.coord) == n_mm, "MM atom count mismatch"
        self.qm_molecule.coord = np.array(self.qm_molecule.coord)
        self.mm_molecule.coord = np.array(self.mm_molecule.coord)
        self.qm_molecule.qm_type = np.array(self.qm_molecule.qm_type)
        self.mm_molecule.charges = np.array(self.mm_molecule.charges)


