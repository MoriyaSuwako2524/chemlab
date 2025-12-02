from chemlab.util.modify_inp import qchem_out_opt, single_spin_job, qchem_file, molecule
from chemlab.scripts.base import QchemBaseScript
from chemlab.config.config_loader import ConfigBase

class ThreePointFiniteDifferenceConfig(QchemBaseScript):
    section_name = "TPFD"

class ThreePointFiniteDifference(QchemBaseScript):

    name = "TPFD"
    config = ThreePoinFiniteDifferenceConfig
    def run(self,cfg):
        cfg_env = QchemEnvConfig()
        env_script = cfg_env.env_script.strip()
        self.path = cfg.path
        self.ref = qchem_file()
        self.ref.read_from_file(cfg.ref)
        self.molecule = self.ref.molecule
    for i in range(self.molecule.xyz.shape[0]):
        for j in range(self.molecule.xyz.shape[1]):
            self.point(cfg,i,j)

    def point(self,cfg,i,j):
        tem_molecule = molecule()
        tem_molecule.xyz = self.molecule.xyz
        tem_molecule.xyz[i,j] + cfg.distance
        tem_molecule.xyz[i,j] - cfg.distance


