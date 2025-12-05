import os
from chemlab.util.file_system import qchem_file,qchem_out_force,molecule,SPIN_REF
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
        self.stepsize = 1
        self.out_path=""
        self.converge_limit = 1e-4
        self.inv_hess = None
        self.last_structure = None
        self.last_gradient = None
        self.restrain = False
        self.restrain_list = []
        self.hessian_coefficient = 0.01  

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
        """
        计算并存储梯度信息。
        假设 self.state_1.out.force 是梯度 (Gradient, ∇E)，指向能量升高的方向。
        """
        grad_1 = self.state_1.out.force
        grad_2 = self.state_2.out.force

        # 保存梯度供 update_structure 使用
        self.grad_1 = grad_1
        self.grad_2 = grad_2

        # 差分梯度向量 (Difference Gradient Vector)
        # x = g1 - g2，这是能隙 (E1 - E2) 变化最剧烈的方向
        self.d_grad = grad_1 - grad_2

        # 目标梯度：通常使用 State 1 的梯度或平均梯度在缝上最小化能量
        self.target_grad = grad_1.copy()

    def update_structure(self):
        """
        基于 MECP 算法 (Projected BFGS + Newton Gap Step) 更新分子结构。
        """
        # 1. 获取当前坐标
        structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T
        natom = self.state_1.inp.molecule.natom
        x_k = structure.flatten()
        nvar = 3 * natom

        # 2. 准备向量
        dg_vec = self.d_grad.flatten()  # 差分梯度向量 x
        g_target_vec = self.target_grad.flatten()  # 目标梯度 g

        norm_dg = np.linalg.norm(dg_vec)
        norm_dg_sq = norm_dg ** 2

        # 3. 构建投影算符 P (Projector)
        # P = I - (x * x^T) / |x|^2，用于滤除垂直于交叉缝的分量
        if norm_dg < 1e-8:
            print("⚠️ 梯度差过小，无法构建投影，使用单位矩阵。")
            P = np.eye(nvar)
            # 防止除零错误，置零
            step_orth = np.zeros(nvar)
        else:
            u = dg_vec / norm_dg
            P = np.eye(nvar) - np.outer(u, u)

        # 4. 计算当前切向梯度 (Projected Gradient)
        g_tan = P @ g_target_vec

        # 定义 g_k 用于 BFGS 历史记录 (只记录切向梯度的变化)
        g_k = g_tan

        # 5. 初始化或更新 Hessian (BFGS)
        if getattr(self, 'inv_hess', None) is None:
            print("ℹ️ 初始化 Inverse Hessian 为投影单位矩阵。")
            self.inv_hess = P.copy()

        # 如果有上一步的信息，进行 BFGS 更新
        if getattr(self, 'last_structure', None) is not None and getattr(self, 'last_gradient', None) is not None:
            # 计算位移差 s 和梯度差 y
            s_k = (x_k - self.last_structure).reshape(-1, 1)
            y_k = (g_k - self.last_gradient).reshape(-1, 1)

            # 投影到切空间 (理论上 g_k 已经是切向的，但 s_k 需要投影)
            # 我们只利用切向的信息来更新 Hessian，避免正交步长(Gap Step)污染曲率信息
            s_tan = P @ s_k
            y_tan = P @ y_k

            # 计算点积 s^T * y
            sty = float(s_tan.T @ y_tan)

            if sty > 1e-10:
                rho = 1.0 / sty
                I = np.eye(nvar)
                H = self.inv_hess

                # BFGS 更新公式 (Updating Inverse Hessian)
                term1 = I - rho * (s_tan @ y_tan.T)
                term2 = I - rho * (y_tan @ s_tan.T)
                H_new = term1 @ H @ term2 + rho * (s_tan @ s_tan.T)

                # 强制对称化并再次投影，确保 Hessian 仅作用于切空间
                H_new = P @ H_new @ P
                self.inv_hess = 0.5 * (H_new + H_new.T)
            else:
                print("⚠️ s^T*y 太小，跳过 BFGS 更新。")
                # 可选：如果更新失败，重置为 P
                # self.inv_hess = P.copy()

        # 6. 计算步长 (Step Calculation)

        # --- A. 正交步长 (Orthogonal Step) ---
        # 目的：消除能隙 (E1 - E2 -> 0)
        # 使用 Newton Step: dx = - (E1-E2) / |g1-g2|^2 * (g1-g2)
        # 输入必须确认为梯度 (Gradient)
        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        delta_E = E1 - E2

        if norm_dg >= 1e-8:
            # 负号至关重要：逆着梯度差方向走
            step_orth = - (delta_E / norm_dg_sq) * dg_vec
        self.orthogonal_gradient = step_orth
        # --- B. 切向步长 (Tangent Step) ---
        # 目的：在缝上寻找能量极小值
        # 使用 Quasi-Newton: dx = - H^-1 * g_tan
        step_tan = - self.inv_hess @ g_tan
        self.parallel_gradient = g_tan
        # 7. 合并总步长与限幅
        total_step = -(step_tan + step_orth)

        step_norm = np.linalg.norm(total_step)
        if step_norm > self.stepsize:
            scale = self.stepsize / step_norm
            print(f"🔻 步长过大 ({step_norm:.4f} Å)，缩放比例: {scale:.4f}")
            total_step *= scale

        # 8. 更新坐标
        new_structure_vec = x_k + total_step
        new_structure = new_structure_vec.reshape((3, natom))

        # 将新坐标写入对象
        self.state_1.inp.molecule.replace_new_xyz(new_structure)
        # 假设 state_2 共享相同的几何结构对象或需要同步
        if hasattr(self.state_2.inp.molecule, 'carti'):
            self.state_2.inp.molecule.carti = self.state_1.inp.molecule.carti

        # 9. 保存当前状态用于下一次迭代
        self.last_structure = x_k.copy()
        self.last_gradient = g_k.copy()

    def generate_new_inp(self):
        path = self.out_path
        self.state_1.job_name = "{}{}_job{}.inp".format(self.prefix,self.state_1._spin,self.job_num)
        self.state_2.job_name = "{}{}_job{}.inp".format(self.prefix,self.state_2._spin,self.job_num)
        out = open(os.path.join(path, self.state_1.job_name),"w")
        out.write(self.state_1.inp.molecule.return_output_format()+self.state_1.inp.remain_texts)
        out = open(os.path.join(path, self.state_2.job_name),"w")
        out.write(self.state_2.inp.molecule.return_output_format()+self.state_2.inp.remain_texts)


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
        converged_flags = [delta_E < self.energy_tol,grad_norm < self.grad_tol,displacement < self.disp_tol,]
        is_converged = sum(converged_flags) >= 2
        print(f"Energy gap: {delta_E:.5e}, Converged? {delta_E < self.energy_tol}; \n Gradient norm: {grad_norm:.5e}, Converged? {grad_norm < self.grad_tol};\n Displacement: {displacement:.5e}, Converged? {displacement < self.disp_tol}. \n")
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
