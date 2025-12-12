import os
from chemlab.util.file_system import qchem_file, qchem_out_force, molecule, SPIN_REF, Hartree_to_kcal
import numpy as np

class mecp(object):
    def __init__(self): #usually spin_1 < spin_2
        self.energy_gap_list = []
        self.iteration_list = []
        self.state_1 = mol_state()
        self.state_2 = mol_state()
        self.state_1.ene_list = []
        self.state_2.ene_list = []
        self.state_1.gradient_list = []
        self.state_2.gradient_list = []
        self.state_1.inp = qchem_file()
        self.state_2.inp = qchem_file()
        self.state_1.spin = 1
        self.state_2.spin = 3
        self.ref_path = ""
        self.ref_filename = ""
        self.prefix=""
        self.different_type="analytical"
        self.job_num = 0
        self.job_max = 100
        self.max_stepsize = 1
        self.out_path=""
        self.converge_limit = 1e-4
        self.inv_hess = None
        self.last_structure = None
        self.last_gradient = None
        self.restrain = False
        self.restrain_list = []
        self.hessian_coefficient = 0.01
        self.step_size = 0.01

    @property
    def energy_tol(self):
        return self.converge_limit
    @property
    def grad_tol(self):
        return self.converge_limit *5
    @property
    def disp_tol(self):
        return self.converge_limit *10
    def initialize_bfgs(self):
        """Initialize inverse Hessian for quasi-Newton update."""
        natom = self.state_1.inp.molecule.natom
        self.inv_hess = np.eye(3 * natom)
        self.last_structure = None
        self.last_gradient = None
    def add_restrain(self,atom_i, atom_j, R0, K=1000.0):
        self.restrain_list.append([atom_i,atom_j,R0,K])
        self.restrain = True
    def read_init_structure(self):
        path = self.ref_path
        filename = self.ref_filename
        self.state_1.inp.molecule.check = True
        self.state_2.inp.molecule.check = True
        self.state_1.inp.read_from_file(f"{path}/{filename}")
        self.state_2.inp.read_from_file(f"{path}/{filename}")
        if self.prefix == "":
            self.prefix = filename[:-4]
        self.state_1.inp.molecule.multistate =   self.state_1.spin 
        self.state_2.inp.molecule.multistate =   self.state_2.spin 
        self.structure_list = [self.state_1.inp.molecule.return_xyz_list()]
    def read_output(self):
        if self.out_path == "":
            path = self.ref_path
        else:
            path = self.out_path
        self.state_1.out = qchem_out_force()
        self.state_2.out = qchem_out_force()
        self.state_1.job_name = "{}{}_job{}.out".format(self.prefix,self.state_1._spin,self.job_num)
        self.state_2.job_name = "{}{}_job{}.out".format(self.prefix,self.state_2._spin,self.job_num)
        print(f"Reading Qchem outpur file:{os.path.join(path,self.state_1.job_name)},{os.path.join(path,self.state_2.job_name)}, gradient_type={self.different_type}")
        self.state_1.out.read_file(os.path.join(path, self.state_1.job_name),self_check=False,different_type=self.different_type)
        self.state_2.out.read_file(os.path.join(path, self.state_2.job_name),self_check=False,different_type=self.different_type)
        self.job_num +=1
        print(f"state1 ene: {self.state_1.out.ene},state2 ene: {self.state_2.out.ene}")
        self.state_1.ene_list.append(self.state_1.out.ene)
        self.state_2.ene_list.append(self.state_2.out.ene)
        self.state_1.gradient_list.append(self.state_1.out.force)
        self.state_2.gradient_list.append(self.state_2.out.force)

    def calc_new_gradient(self):
        grad_1 = self.state_1.out.force
        grad_2 = self.state_2.out.force
        from chemlab.util.unit import GRADIENT
        grad_1 = GRADIENT(grad_1).convert_to({"energy": ("Hartree", 1), "distance": ("Ang", -1)})
        grad_2 = GRADIENT(grad_2).convert_to({"energy": ("Hartree", 1), "distance": ("Ang", -1)})
        self.grad_1 = grad_1.flatten()
        self.grad_2 = grad_2.flatten()

    def update_structure(self):
        structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T
        natom = self.state_1.inp.molecule.natom
        x_k = structure.flatten()
        nvar = 3 * natom

        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        Ga = self.grad_1  # 态1的梯度
        Gb = self.grad_2  # 态2的梯度

        # ==================== Harvey有效梯度 ====================
        # facPP = 140 是Harvey的经验值，用于平衡两个方向的Hessian量级
        facPP = 140.0
        facP = 1.0

        # PerpG = Ga - Gb (垂直于seam的梯度差)
        PerpG = Ga - Gb
        npg = np.linalg.norm(PerpG)

        if npg < 1e-10:
            print("⚠️ 梯度差接近零，已接近MECP")
            ParG = Ga.copy()
            G_eff = ParG
        else:
            # pp = Ga在PerpG方向的投影长度
            pp = np.dot(Ga, PerpG) / npg

            # ParG = Ga减去其在PerpG方向的分量（Ga在seam上的投影）
            ParG = Ga - (PerpG / npg) * pp

            # 有效梯度 = 能量差项 + 切向项
            G_eff = (E1 - E2) * facPP * PerpG + facP * ParG

        # ==================== BFGS更新 ====================
        if getattr(self, 'inv_hess', None) is None:
            print("ℹ️ 初始化逆Hessian (对角0.7)")
            self.inv_hess = 0.7 * np.eye(nvar)

        if getattr(self, 'last_structure', None) is not None and \
                getattr(self, 'last_G_eff', None) is not None:

            DelX = x_k - self.last_structure
            DelG = G_eff - self.last_G_eff

            # 计算BFGS所需的量
            fac = np.dot(DelG, DelX)
            HDelG = self.inv_hess @ DelG
            fae = np.dot(DelG, HDelG)

            print(f"BFGS: fac={fac:.6e}, fae={fae:.6e}")

            if abs(fac) > 1e-10 and abs(fae) > 1e-10:
                fac_inv = 1.0 / fac
                fad = 1.0 / fae

                # w向量
                w = fac_inv * DelX - fad * HDelG

                # BFGS公式 (DFP-BFGS混合)
                H_new = self.inv_hess.copy()
                for i in range(nvar):
                    for j in range(nvar):
                        H_new[i, j] += fac_inv * DelX[i] * DelX[j] \
                                       - fad * HDelG[i] * HDelG[j] \
                                       + fae * w[i] * w[j]

                self.inv_hess = H_new
                print("✓ BFGS更新成功")
            else:
                print("⚠️ BFGS跳过: fac或fae太小")

        # ==================== 计算步长 ====================
        if getattr(self, 'is_first_step', True):
            # 第一步：简单的最速下降
            ChgeX = -0.7 * G_eff
            self.is_first_step = False
        else:
            # BFGS步
            ChgeX = -self.inv_hess @ G_eff

        # ==================== 步长限制 (Harvey方法) ====================
        STPMX = 0.1  # 单个坐标最大位移
        stpmax = STPMX * nvar  # 总步长限制

        # 限制总步长
        stpl = np.linalg.norm(ChgeX)
        if stpl > stpmax:
            ChgeX = ChgeX / stpl * stpmax
            print(f"🔻 总步长截断: {stpl:.4f} → {stpmax:.4f}")

        # 限制单个坐标最大位移
        lgstst = np.max(np.abs(ChgeX))
        if lgstst > STPMX:
            ChgeX = ChgeX / lgstst * STPMX
            print(f"🔻 单坐标截断: {lgstst:.4f} → {STPMX:.4f}")

        # ==================== 更新结构 ====================
        new_structure_vec = x_k + ChgeX
        new_structure = new_structure_vec.reshape((3, natom))

        self.state_1.inp.molecule.replace_new_xyz(new_structure)
        if hasattr(self.state_2.inp.molecule, 'carti'):
            self.state_2.inp.molecule.carti = self.state_1.inp.molecule.carti

        # ==================== 保存历史 ====================
        self.last_structure = x_k.copy()
        self.last_G_eff = G_eff.copy()

        # 保存用于收敛检查
        self.ParG = ParG
        self.PerpG = PerpG
        self.G_eff = G_eff
        self.ChgeX = ChgeX

        # ==================== 诊断输出 ====================
        print("=" * 60)
        print(f"E1 = {E1:.6f}, E2 = {E2:.6f}, ΔE = {E1 - E2:.6f}")
        print(f"‖PerpG‖ = {npg:.6f} (梯度差)")
        print(f"‖ParG‖ = {np.linalg.norm(ParG):.6f} (切向梯度)")
        print(f"‖G_eff‖ = {np.linalg.norm(G_eff):.6f} (有效梯度)")
        print(f"‖ChgeX‖ = {np.linalg.norm(ChgeX):.6f} (位移)")
        print(f"max|ChgeX| = {np.max(np.abs(ChgeX)):.6f}")
        print("=" * 60)

    def generate_new_inp(self):
        path = self.out_path
        self.state_1.job_name = "{}{}_job{}.inp".format(self.prefix, self.state_1._spin, self.job_num)
        self.state_2.job_name = "{}{}_job{}.inp".format(self.prefix, self.state_2._spin, self.job_num)
        with open(os.path.join(path, self.state_1.job_name), "w") as out:
            out.write(self.state_1.inp.molecule.return_output_format() + self.state_1.inp.remain_texts)
        with open(os.path.join(path, self.state_2.job_name), "w") as out:
            out.write(self.state_2.inp.molecule.return_output_format() + self.state_2.inp.remain_texts)

    def check_convergence(self):
        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        DE = abs(E1 - E2)

        natom = self.state_1.inp.molecule.natom
        nvar = 3 * natom

        # Harvey收敛标准
        G_eff = self.G_eff
        ChgeX = self.ChgeX

        GMax = np.max(np.abs(G_eff))
        GRMS = np.sqrt(np.mean(G_eff ** 2))
        DXMax = np.max(np.abs(ChgeX))
        DXRMS = np.sqrt(np.mean(ChgeX ** 2))

        # 默认阈值 (与easyMECP一致)
        TDE = getattr(self, 'energy_tol', 5e-5)
        TGMax = getattr(self, 'grad_tol', 7e-4)
        TGRMS = 5e-4
        TDXMax = 4e-3
        TDXRMS = 2.5e-3

        flags = [
            DE < TDE,
            GMax < TGMax,
            GRMS < TGRMS,
            DXMax < TDXMax,
            DXRMS < TDXRMS
        ]

        is_converged = all(flags)

        print(f"\n{'=' * 60}")
        print(f"收敛检查:")
        print(f"  Energy diff:  {DE:.6e}  ({'YES' if flags[0] else 'NO '}) (阈值: {TDE})")
        print(f"  Max Gradient: {GMax:.6e}  ({'YES' if flags[1] else 'NO '}) (阈值: {TGMax})")
        print(f"  RMS Gradient: {GRMS:.6e}  ({'YES' if flags[2] else 'NO '}) (阈值: {TGRMS})")
        print(f"  Max Delta X:  {DXMax:.6e}  ({'YES' if flags[3] else 'NO '}) (阈值: {TDXMax})")
        print(f"  RMS Delta X:  {DXRMS:.6e}  ({'YES' if flags[4] else 'NO '}) (阈值: {TDXRMS})")
        print(f"  总体收敛: {'是 ✓' if is_converged else '否'}")
        print(f"{'=' * 60}\n")

        return is_converged
    def restrain_ene(self, atom_i, atom_j, R0, K=1000.0):
        R_vec = self.state_1.inp.molecule.calc_array_from_atom_1_to_atom_2(atom_i, atom_j)
        Rij = np.linalg.norm(R_vec)
        delta = Rij - R0
        ene = K * delta**2
        self.EI = ene
        return ene
    def restrain_force(self, atom_i, atom_j, R0, K=1000.0):
        R_vec = self.state_1.inp.molecule.calc_array_from_atom_1_to_atom_2(atom_i, atom_j)
        Rij = np.linalg.norm(R_vec)
        delta = Rij - R0
        grad = np.zeros((3, self.state_1.inp.molecule.natom))  # shape (3, N)
        
        if Rij > 1e-8:  # avoid divide-by-zero
            dR_dqi = R_vec / Rij
            grad[:, atom_i] += dR_dqi
            grad[:, atom_j] -= dR_dqi
    
        F_restrain = 2 * K * delta * grad
        self.F_EI = F_restrain
        return F_restrain

    
    def plot_energy_progress(self):
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
        import numpy as np
    
        clear_output(wait=True)  # ✅ 每次清空之前的图像输出
        self.iteration_list.append(self.job_num)
    
        # 获取能量信息
        e1 = np.array(self.state_1.ene_list)
        e2 = np.array(self.state_2.ene_list)
        gap = np.abs(e1 - e2)
        self.energy_gap_list = gap
    
        # 初始化轨迹记录
        if not hasattr(self, 'grad_norm_list'):
            self.grad_norm_list = []
        if not hasattr(self, 'displacement_list'):
            self.displacement_list = []
    
        # 计算梯度范数
        g_k = (self.parallel_gradient + self.orthogonal_gradient).flatten()
        grad_norm = np.linalg.norm(g_k)
        self.grad_norm_list.append(grad_norm)
    
        # 计算结构位移
        natom = self.state_1.inp.molecule.natom
        current_structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T
        if self.last_structure is not None:
            last_structure = self.last_structure.reshape((3, natom))
            displacement = np.linalg.norm(current_structure - last_structure)
        else:
            displacement = np.nan
        self.displacement_list.append(displacement)
    
        # 图1：能量差
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(self.iteration_list, self.energy_gap_list, label='|Energy Gap|', color='red', linestyle='--', marker='x')
        ax1.set_xlabel('Optimization Step')
        ax1.set_ylabel('Energy Gap (Hartree)')
        ax1.set_title('Energy Gap vs. Optimization Step')
        ax1.grid(True)
        ax1.legend()
        fig1.tight_layout()
        display(fig1)
        plt.close(fig1)
    
        # 图2：两个态的能量
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(self.iteration_list, e1, label='State 1 Energy', marker='o')
        ax2.plot(self.iteration_list, e2, label='State 2 Energy', marker='o')
        ax2.set_xlabel('Optimization Step')
        ax2.set_ylabel('Energy (Hartree)')
        ax2.set_title('State Energies vs. Optimization Step')
        ax2.grid(True)
        ax2.legend()
        fig2.tight_layout()
        display(fig2)
        plt.close(fig2)
    
        # 图3：梯度范数和位移
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(self.iteration_list, self.grad_norm_list, label='Gradient Norm', linestyle=':', color='green')
        ax3.plot(self.iteration_list, self.displacement_list, label='Displacement (Å)', linestyle='-.', color='purple')
        ax3.set_xlabel('Optimization Step')
        ax3.set_ylabel('Gradient Norm / Displacement')
        ax3.set_title('Gradient Norm and Displacement')
        ax3.grid(True)
        ax3.legend()
        fig3.tight_layout()
        display(fig3)
        plt.close(fig3)

class mecp_soc(mecp):
    def __init__(self):
        super(mecp_soc, self).__init__()
        self.different_type = "soc"
    def generate_new_inp(self):
        if self.out_path == "":
            path = self.ref_path
        else:
            path = self.out_path
        self.state_1.job_name = "{}{}_job{}.inp".format(self.prefix,self.state_1._spin,self.job_num)
        out = open(path+self.state_1.job_name,"w")
        out.write(self.state_1.inp.molecule.return_output_format()+self.state_1.inp.remain_texts)

    def check_converge(self):
        """
        Check convergence for SOC MECP optimization based on:
        - Energy change in spin-adiabatic energy (E_adiab)
        - Gradient norm (total gradient)
        - Structure displacement
        """
        # Current energy
        current_energy = self.state_1.out.final_adiabatic_ene  # spin-adiabatic energy from output
        natom = self.state_1.inp.molecule.natom
        current_structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T

        # Energy change
        if hasattr(self, "last_adiabatic_energy"):
            delta_E = abs(current_energy - self.last_adiabatic_energy)
        else:
            delta_E = np.inf

        # Gradient norm
        grad_norm = np.linalg.norm(self.parallel_gradient + self.orthogonal_gradient)

        # Structure displacement
        if self.last_structure is not None:
            last_structure = self.last_structure.reshape((3, natom))
            displacement = np.linalg.norm(current_structure - last_structure)
        else:
            displacement = np.inf

        # Update memory for next step
        self.last_adiabatic_energy = current_energy

        # Convergence logic
        converged_flags = [
            delta_E < self.energy_tol,
            grad_norm < self.grad_tol,
            displacement < self.disp_tol,
        ]
        is_converged = sum(converged_flags) >= 2

        print(f"[SOC] Energy change: {delta_E:.5e}, Converged? {delta_E < self.energy_tol};")
        print(f"[SOC] Gradient norm: {grad_norm:.5e}, Converged? {grad_norm < self.grad_tol};")
        print(f"[SOC] Displacement: {displacement:.5e}, Converged? {displacement < self.disp_tol}.\n")
        return is_converged

    def read_output(self):
        path = self.out_path
        self.state_1.out = qchem_out_force()
        self.state_2.out = qchem_out_force()
        self.state_1.job_name = "{}{}_job{}.inp.out".format(self.prefix, self.state_1._spin, self.job_num)
        print(f"Reading output file: {path}{self.state_1.job_name}")
        self.state_1.out.read_file(os.path.join(path,self.state_1.job_name), self_check=False, different_type=self.different_type)
        self.job_num += 1
        self.state_1.out.ene = self.state_1.out.final_adiabatic_ene
        self.state_2.out.ene = self.state_1.out.final_adiabatic_ene + self.state_1.out.final_soc_ene
        self.state_1.ene_list.append(self.state_1.out.final_adiabatic_ene)
        self.state_2.ene_list.append(self.state_1.out.final_adiabatic_ene + self.state_1.out.final_soc_ene)

        self.state_1.out.force = self.state_1.out.force
        self.state_2.out.force = -self.state_1.out.force + self.state_1.out.force_e1 + self.state_1.out.force_e2

        self.state_1.gradient_list.append(self.state_1.out.force)
        self.state_2.gradient_list.append(self.state_2.out.force)

class mol_state(object):
    def __init__(self):
        self.read = False
        self.spin = 1
    @property
    def _spin(self):
        return SPIN_REF[self.spin]
