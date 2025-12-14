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
        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        gradient_1 = -self.state_1.out.force
        gradient_2 = -self.state_2.out.force

        # Difference vector between the two gradients
        delta_gradient = gradient_1 - gradient_2
        norm_dg = np.linalg.norm(delta_gradient)

        # Handle degenerate case
        if norm_dg < 1e-8:
            print("⚠️ Warning: gradient difference norm is near zero!")
            unit_delta_gradient = delta_gradient
        else:
            unit_delta_gradient = delta_gradient / norm_dg
        delta_E = E1 - E2

        # Orthogonal gradient component (perpendicular to crossing surface)
        self.orthogonal_gradient = 10 * (E1 - E2) * delta_gradient / norm_dg

        # Project gradient_1 onto unit direction
        projection_scalar = np.sum(gradient_1 * unit_delta_gradient)
        projection_vector = projection_scalar * unit_delta_gradient

        # Parallel gradient component (tangent to crossing surface)
        self.parallel_gradient = gradient_1 - projection_vector
        if self.different_type == "smd":
            self.parallel_gradient = self.parallel_gradient.T
            self.orthogonal_gradient = self.orthogonal_gradient.T
        if self.restrain:
            for restrain in self.restrain_list:
                grad = self.restrain_force(restrain[0], restrain[1], restrain[2], restrain[3])
                self.parallel_gradient += grad

    def update_structure(self):
        structure = self.state_1.inp.molecule.return_xyz_list().astype(float)
        natom = self.state_1.inp.molecule.natom

        x_k = structure.flatten()
        g_k = (self.parallel_gradient + self.orthogonal_gradient).flatten()

        if self.last_structure is not None:

            x_km1 = self.last_structure.flatten()
            g_km1 = self.last_gradient.flatten()

            dx = x_k - x_km1
            dg = g_k - g_km1

            dg_dx = np.dot(dg, dx)
            if abs(dg_dx) < 1e-12:
                print("⚠️ BFGS skipped: Δg·Δx too small")
            else:
                H = self.inv_hess

                Hdg = H @ dg
                dg_H_dg = np.dot(dg, Hdg)

                if abs(dg_H_dg) < 1e-12:
                    print("⚠️ BFGS skipped: Δgᵀ H Δg too small")
                else:
                    rho = 1.0 / dg_dx
                    sigma = 1.0 / dg_H_dg

                    w = rho * dx - sigma * Hdg

                    self.inv_hess = (
                            H
                            + rho * np.outer(dx, dx)
                            - sigma * np.outer(Hdg, Hdg)
                            + dg_H_dg * np.outer(w, w)
                    )
                    self.inv_hess = (self.inv_hess +self.inv_hess.T)/2

        else:
            self.inv_hess = np.eye(len(g_k))
            print("⚠️  BFGS update skipped: first step")

        # === Newton step ===
        print(f"inv_hessian:{self.inv_hess}")
        step_vector = -self.inv_hess @ g_k
        step_vector = step_vector.reshape((natom, 3))

        step_norm = np.linalg.norm(step_vector)


        # 保存历史
        self.last_structure = x_k.reshape((natom, 3))
        self.last_gradient = g_k.reshape((natom, 3))


        if step_norm > self.max_stepsize:
            print(f"🔻 Step clipped from {step_norm:.4f} Å to {self.max_stepsize:.4f} Å")
            step_vector *= self.max_stepsize / step_norm
        # Update structure
        new_structure = structure + step_vector
        print(f"new structure: {new_structure},shape={new_structure.shape}")
        self.state_1.inp.molecule.replace_new_xyz(new_structure)
        self.state_2.inp.molecule.carti = self.state_1.inp.molecule.carti


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
        delta_E = abs(E1 - E2)
        # Norm of orthogonal gradient
        grad_norm = np.linalg.norm(self.orthogonal_gradient + self.parallel_gradient)
        natom = self.state_1.inp.molecule.natom
        # Structure shift
        current_structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T
        if self.last_structure is not None:
            last_structure = self.last_structure.reshape((3, natom))
            displacement = np.linalg.norm(current_structure - last_structure)
        else:
            displacement = np.inf

            #  Converge check
        converged_flags = [delta_E < self.energy_tol, grad_norm < self.grad_tol, displacement < self.disp_tol, ]
        is_converged = sum(converged_flags) >= 2
        print(
            f"Energy gap: {delta_E:.5e}, Converged? {delta_E < self.energy_tol}; \n Gradient norm: {grad_norm:.5e}, Converged? {grad_norm < self.grad_tol};\n Displacement: {displacement:.5e}, Converged? {displacement < self.disp_tol}. \n")
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
