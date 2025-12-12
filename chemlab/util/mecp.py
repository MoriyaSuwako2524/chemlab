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
        
        # ========== Harvey MECP 相关参数 (来自 easymecp.py) ==========
        self.nstep = 0           # 当前步数
        self.ffile = 0           # 是否从ProgFile恢复 (0=否)
        
        # 收敛阈值 (与 easymecp.py 一致)
        self.TDE = 5.0e-5        # 能量差阈值
        self.TDXMax = 4.0e-3     # 最大位移阈值
        self.TDXRMS = 2.5e-3     # RMS位移阈值
        self.TGMax = 7.0e-4      # 最大梯度阈值
        self.TGRMS = 5.0e-4      # RMS梯度阈值
        
        # Harvey有效梯度参数
        self.facPP = 140.0       # 垂直梯度系数 (经验值)
        self.facP = 1.0          # 平行梯度系数
        self.STPMX = 0.1         # 单坐标最大步长 (Angstrom)
        
        # 历史数据存储 (用于BFGS)
        self.X_1 = None          # 前前一步坐标
        self.X_2 = None          # 前一步坐标
        self.G_1 = None          # 前一步有效梯度
        self.G_2 = None          # 当前有效梯度
        self.HI_1 = None         # 前一步逆Hessian
        self.HI_2 = None         # 当前逆Hessian

    @property
    def energy_tol(self):
        return self.TDE
    @property
    def grad_tol(self):
        return self.TGMax
    @property
    def disp_tol(self):
        return self.TDXMax
        
    def initialize_bfgs(self):
        """Initialize inverse Hessian for quasi-Newton update.
        
        按照 easymecp.py Fortran代码中的 Initialize 子程序:
        逆Hessian对角元素初始化为0.7 (对应Hessian约1.4 Hartree/Angstrom^2)
        """
        natom = self.state_1.inp.molecule.natom
        nx = 3 * natom
        
        # 初始化逆Hessian (对角0.7)
        self.inv_hess = np.zeros((nx, nx))
        for i in range(nx):
            self.inv_hess[i, i] = 0.7
            
        self.HI_1 = self.inv_hess.copy()
        self.HI_2 = None
        
        self.last_structure = None
        self.last_gradient = None
        self.X_1 = None
        self.X_2 = None
        self.G_1 = None
        self.G_2 = None
        self.nstep = 0
        self.ffile = 0
        
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
        """
        计算有效梯度 (Effective Gradient)
        
        按照 easymecp.py Fortran代码中的 Effective_Gradient 子程序:
        G = (Ea - Eb) * facPP * PerpG + facP * ParG
        
        其中:
        - PerpG = Ga - Gb (垂直于seam的梯度差)
        - ParG = Ga - (PerpG/|PerpG|) * (Ga·PerpG/|PerpG|) (平行于seam的梯度)
        """
        # 读取梯度并转换单位
        grad_1 = self.state_1.out.force
        grad_2 = self.state_2.out.force
        from chemlab.util.unit import GRADIENT
        grad_1 = GRADIENT(grad_1).convert_to({"energy": ("Hartree", 1), "distance": ("Ang", -1)})
        grad_2 = GRADIENT(grad_2).convert_to({"energy": ("Hartree", 1), "distance": ("Ang", -1)})
        
        # 展平为一维数组
        Ga = grad_1.flatten()
        Gb = grad_2.flatten()
        
        # 保存原始梯度
        self.grad_1 = Ga
        self.grad_2 = Gb
        
        # 获取能量
        Ea = self.state_1.out.ene
        Eb = self.state_2.out.ene
        
        n = len(Ga)
        
        # ========== Harvey有效梯度计算 (来自easymecp.py Fortran代码) ==========
        # PerpG = Ga - Gb (垂直于seam的梯度差)
        PerpG = Ga - Gb
        
        # npg = |PerpG|
        npg = np.sqrt(np.sum(PerpG**2))
        
        # pp = Ga在PerpG方向的投影
        pp = np.dot(Ga, PerpG)
        
        if npg > 1e-10:
            pp = pp / npg
            # ParG = Ga - (PerpG/npg) * pp (Ga在seam上的投影)
            ParG = Ga - (PerpG / npg) * pp
            # 有效梯度 G = (Ea-Eb) * facPP * PerpG + facP * ParG
            G_eff = (Ea - Eb) * self.facPP * PerpG + self.facP * ParG
        else:
            # 梯度差太小，已接近MECP
            print("⚠️ 梯度差接近零 (npg < 1e-10)")
            ParG = Ga.copy()
            G_eff = self.facP * ParG
        
        # 保存结果
        self.PerpG = PerpG      # 垂直梯度 (差分梯度)
        self.ParG = ParG        # 平行梯度
        self.G_eff = G_eff      # 有效梯度
        self.npg = npg          # 梯度差范数
        
        # 用于兼容旧代码
        self.orthogonal_gradient = PerpG
        self.parallel_gradient = ParG

    def update_structure(self):
        """
        更新分子结构
        
        按照 easymecp.py Fortran代码中的 UpdateX 子程序:
        使用BFGS准牛顿方法更新坐标
        """
        # 获取当前结构
        structure = self.state_1.inp.molecule.return_xyz_list().astype(float)
        natom = self.state_1.inp.molecule.natom
        nx = 3 * natom
        
        # 当前坐标 (展平)
        X_2 = structure.flatten()
        
        # 当前有效梯度
        G_2 = self.G_eff.copy()
        
        # ========== BFGS更新 (来自easymecp.py Fortran代码 UpdateX子程序) ==========
        
        if (self.nstep == 0) and (self.ffile == 0):
            # 第一步：简单的最速下降，步长因子0.7
            ChgeX = -0.7 * G_2
            
            # 复制逆Hessian
            if self.HI_1 is None:
                self.HI_1 = np.eye(nx) * 0.7
            self.HI_2 = self.HI_1.copy()
            
        else:
            # 后续步骤：BFGS更新
            
            # 梯度差和坐标差
            DelG = G_2 - self.G_1
            DelX = X_2 - self.X_1
            
            # 计算 HDelG = H * DelG
            HDelG = self.HI_1 @ DelG
            
            # 计算点积
            fac = np.dot(DelG, DelX)      # DelG · DelX
            fae = np.dot(DelG, HDelG)     # DelG · H · DelG
            
            if abs(fac) > 1e-10 and abs(fae) > 1e-10:
                fac_inv = 1.0 / fac
                fad = 1.0 / fae
                
                # w向量
                w = fac_inv * DelX - fad * HDelG
                
                # BFGS逆Hessian更新公式:
                # H_new = H + (DelX⊗DelX)/fac - (HDelG⊗HDelG)/fae + fae*(w⊗w)
                self.HI_2 = self.HI_1.copy()
                for i in range(nx):
                    for j in range(nx):
                        self.HI_2[i, j] += (fac_inv * DelX[i] * DelX[j] 
                                           - fad * HDelG[i] * HDelG[j]
                                           + fae * w[i] * w[j])
            else:
                print(f"⚠️ BFGS跳过: fac={fac:.2e}, fae={fae:.2e}")
                self.HI_2 = self.HI_1.copy()
            
            # 计算步长: ChgeX = -H * G
            ChgeX = np.zeros(nx)
            for i in range(nx):
                for j in range(nx):
                    ChgeX[i] -= self.HI_2[i, j] * G_2[j]
        
        # ========== 步长限制 (来自easymecp.py Fortran代码) ==========
        stpmax = self.STPMX * nx  # 总步长限制
        
        # 计算总步长
        stpl = np.sqrt(np.sum(ChgeX**2))
        
        # 限制总步长
        if stpl > stpmax:
            ChgeX = ChgeX / stpl * stpmax
            print(f"🔻 总步长截断: {stpl:.4f} → {stpmax:.4f}")
        
        # 限制单坐标最大位移
        lgstst = np.max(np.abs(ChgeX))
        if lgstst > self.STPMX:
            ChgeX = ChgeX / lgstst * self.STPMX
            print(f"🔻 单坐标截断: {lgstst:.4f} → {self.STPMX:.4f}")
        
        # ========== 更新坐标 ==========
        X_3 = X_2 + ChgeX
        new_structure = X_3.reshape((3, natom))
        print(f"self.grad_1:{self.grad_1}")
        print(f"X2:{X_2}")
        print(f"structure:{structure}")
        print(f"new_structure:{new_structure}")
        # 写入新坐标
        self.state_1.inp.molecule.replace_new_xyz(new_structure)
        if hasattr(self.state_2.inp.molecule, 'carti'):
            self.state_2.inp.molecule.carti = self.state_1.inp.molecule.carti
        
        # ========== 保存历史数据 (用于下一步BFGS) ==========
        self.X_1 = X_2.copy()       # 保存当前坐标为"前一步"
        self.G_1 = G_2.copy()       # 保存当前梯度为"前一步"
        self.HI_1 = self.HI_2.copy() if self.HI_2 is not None else self.HI_1.copy()
        
        # 更新步数
        self.nstep += 1
        
        # 保存用于收敛检查
        self.ChgeX = ChgeX
        self.X_2 = X_2
        self.X_3 = X_3
        
        # 兼容旧代码
        self.last_structure = X_2.copy()
        self.last_gradient = G_2.copy()
        self.last_G_eff = G_2.copy()
        
        # ========== 诊断输出 ==========
        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        print("=" * 60)
        print(f"Step {self.nstep}")
        print(f"E1 = {E1:.10f}, E2 = {E2:.10f}")
        print(f"ΔE = {abs(E1 - E2):.6e} Hartree ({abs(E1 - E2) * Hartree_to_kcal:.4f} kcal/mol)")
        print(f"‖PerpG‖ = {self.npg:.6f} (梯度差)")
        print(f"‖ParG‖ = {np.linalg.norm(self.ParG):.6f} (切向梯度)")
        print(f"‖G_eff‖ = {np.linalg.norm(G_2):.6f} (有效梯度)")
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
        """
        检查收敛
        
        按照 easymecp.py Fortran代码中的 TestConvergence 子程序:
        检查5个收敛标准 (全部满足才算收敛)
        """
        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        
        natom = self.state_1.inp.molecule.natom
        nx = 3 * natom
        
        # 获取有效梯度和位移
        G = self.G_eff
        DeltaX = self.ChgeX
        
        # ========== 计算收敛指标 (来自easymecp.py Fortran代码) ==========
        
        # 能量差
        DE = abs(E1 - E2)
        
        # 位移统计
        DXMax = np.max(np.abs(DeltaX))
        DXRMS = np.sqrt(np.mean(DeltaX**2))
        
        # 梯度统计
        GMax = np.max(np.abs(G))
        GRMS = np.sqrt(np.mean(G**2))
        
        # 垂直/平行梯度统计 (用于诊断输出)
        PpGRMS = np.sqrt(np.mean(self.PerpG**2))
        PGRMS = np.sqrt(np.mean(self.ParG**2))
        
        # ========== 收敛判断 ==========
        flags = {
            'TGMax': GMax < self.TGMax,
            'TGRMS': GRMS < self.TGRMS,
            'TDXMax': DXMax < self.TDXMax,
            'TDXRMS': DXRMS < self.TDXRMS,
            'TDE': DE < self.TDE
        }
        
        is_converged = all(flags.values())
        
        # ========== 输出收敛信息 ==========
        print(f"\n{'=' * 70}")
        print(f"Energy of First State:  {E1:.10f}")
        print(f"Energy of Second State: {E2:.10f}")
        print()
        print("Convergence Check (Actual Value, then Threshold, then Status):")
        print(f"Max Gradient El.: {GMax:11.6f} ({self.TGMax:8.6f})  {'YES' if flags['TGMax'] else ' NO'}")
        print(f"RMS Gradient El.: {GRMS:11.6f} ({self.TGRMS:8.6f})  {'YES' if flags['TGRMS'] else ' NO'}")
        print(f"Max Change of X:  {DXMax:11.6f} ({self.TDXMax:8.6f})  {'YES' if flags['TDXMax'] else ' NO'}")
        print(f"RMS Change of X:  {DXRMS:11.6f} ({self.TDXRMS:8.6f})  {'YES' if flags['TDXRMS'] else ' NO'}")
        print(f"Difference in E:  {DE:11.6f} ({self.TDE:8.6f})  {'YES' if flags['TDE'] else ' NO'}")
        print()
        print(f"Difference Gradient: (RMS * DE: {PpGRMS:.6f})")
        print(f"Parallel Gradient: (RMS: {PGRMS:.6f})")
        print()
        
        if is_converged:
            print("The MECP Optimization has CONVERGED at that geometry !!!")
            print("Goodbye and fly with us again...")
        else:
            print(f"Not converged. Proceeding to step {self.nstep + 1}...")
        print(f"{'=' * 70}\n")
        
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
