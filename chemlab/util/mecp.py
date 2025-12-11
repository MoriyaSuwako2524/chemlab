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
        self.grad_1 = grad_1
        self.grad_2 = grad_2
        # 注意：d_grad = grad_1 - grad_2，用于正交步方向
        self.d_grad = grad_1 - grad_2
        self.target_grad = 0.5 * (grad_1 + grad_2).copy()

    def update_structure(self):
        structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T
        natom = self.state_1.inp.molecule.natom
        x_k = structure.flatten()
        nvar = 3 * natom

        dg_vec = self.d_grad.flatten()
        g_target_vec = self.target_grad.flatten()

        norm_dg = np.linalg.norm(dg_vec)
        norm_dg_sq = norm_dg ** 2

        if norm_dg < 1e-8:
            print("⚠️ Too small gradient difference")
            P = np.eye(nvar)
            u = np.zeros(nvar)
        else:
            u = dg_vec / norm_dg
            P = np.eye(nvar) - np.outer(u, u)

        g_tan = P @ g_target_vec
        g_k = g_tan

        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        delta_E = E1 - E2

        # ==================== BFGS 更新 ====================
        if getattr(self, 'inv_hess', None) is None:
            print("ℹ️ Initializing Inverse Hessian")
            H0 = 0.5 * np.eye(nvar)
            self.inv_hess = P @ H0 @ P

        if getattr(self, 'last_structure', None) is not None and \
                getattr(self, 'last_gradient', None) is not None:

            s_full = x_k - self.last_structure
            y_full = g_k - self.last_gradient

            # 关键修改：只使用切向分量
            # 用当前的投影矩阵P投影位移向量
            s_tan = P @ s_full
            y_tan = y_full  # y已经是切向梯度的差

            s_tan = s_tan.reshape(-1, 1)
            y_tan = y_tan.reshape(-1, 1)

            sty = float(s_tan.T @ y_tan)
            s_tan_norm = np.linalg.norm(s_tan)
            y_tan_norm = np.linalg.norm(y_tan)

            print(f"BFGS诊断: sty={sty:.6e}, ||s_tan||={s_tan_norm:.4f}, ||y_tan||={y_tan_norm:.4f}")

            # 检查Hessian条件数
            eigvals = np.linalg.eigvalsh(self.inv_hess)
            cond = eigvals.max() / max(eigvals.min(), 1e-10)

            if cond > 1000:
                print(f"⚠️ inv_hess 病态 (条件数={cond:.1f})，重置")
                self.inv_hess = P @ (0.5 * np.eye(nvar)) @ P
            elif sty > 1e-10 and s_tan_norm > 1e-4:
                # 标准BFGS更新
                rho = 1.0 / sty
                I = np.eye(nvar)
                H = self.inv_hess

                term1 = I - rho * (s_tan @ y_tan.T)
                term2 = I - rho * (y_tan @ s_tan.T)
                H_new = term1 @ H @ term2 + rho * (s_tan @ s_tan.T)

                # 对称化并投影到切空间
                H_new = 0.5 * (H_new + H_new.T)
                self.inv_hess = P @ H_new @ P
                print("✓ BFGS 更新成功")
            elif sty < -1e-10:
                # sty为负，说明不满足曲率条件，可以尝试SR1更新或跳过
                print(f"⚠️ sty < 0 ({sty:.2e})，尝试阻尼BFGS")
                # 阻尼BFGS：修正y使sty > 0
                theta = 0.8
                Hs = self.inv_hess @ s_tan.flatten()
                sHs = float(s_tan.T @ Hs.reshape(-1, 1))

                if sty < theta * sHs:
                    # Powell阻尼
                    if abs(sHs - sty) > 1e-12:
                        phi = (theta * sHs) / (sHs - sty + theta * sHs)
                        y_damped = phi * y_tan.flatten() + (1 - phi) * Hs
                        y_tan = y_damped.reshape(-1, 1)
                        sty = float(s_tan.T @ y_tan)
                        print(f"  阻尼后 sty={sty:.6e}")

                        if sty > 1e-10:
                            rho = 1.0 / sty
                            I = np.eye(nvar)
                            H = self.inv_hess
                            term1 = I - rho * (s_tan @ y_tan.T)
                            term2 = I - rho * (y_tan @ s_tan.T)
                            H_new = term1 @ H @ term2 + rho * (s_tan @ s_tan.T)
                            H_new = 0.5 * (H_new + H_new.T)
                            self.inv_hess = P @ H_new @ P
                            print("✓ 阻尼BFGS 更新成功")
            else:
                print(f"⚠️ sty太小或s_tan太小，跳过BFGS更新")

        # ==================== 计算步长 ====================

        # 正交步（减小能量差）
        # step_orth 方向应该使 (E1-E2)^2 减小
        # d/dx (E1-E2)^2 = 2(E1-E2)(g1-g2) = 2*delta_E*d_grad
        # 所以下降方向是 -delta_E * d_grad
        if norm_dg >= 1e-8:
            step_orth = -(delta_E / norm_dg_sq) * dg_vec
        else:
            step_orth = np.zeros(nvar)

        # 切向步（最小化seam上的能量）
        step_tan = -self.inv_hess @ g_tan

        # ==================== 步长限制 ====================
        # 根据当前状态动态调整最大步长
        gap_threshold = 0.005  # 5 mHartree

        if abs(delta_E) > 0.01:  # gap > 6 kcal/mol
            max_step_tan = 0.08
            max_step_orth = 0.15
        elif abs(delta_E) > gap_threshold:
            max_step_tan = 0.12
            max_step_orth = 0.10
        else:  # gap已经很小
            max_step_tan = 0.15
            max_step_orth = 0.05

        norm_tan = np.linalg.norm(step_tan)
        norm_orth = np.linalg.norm(step_orth)

        if norm_tan > max_step_tan:
            step_tan = step_tan * (max_step_tan / norm_tan)
            print(f"🔻 step_tan 截断: {norm_tan:.4f} → {max_step_tan:.4f}")

        if norm_orth > max_step_orth:
            step_orth = step_orth * (max_step_orth / norm_orth)
            print(f"🔻 step_orth 截断: {norm_orth:.4f} → {max_step_orth:.4f}")

        # ==================== 权重调整 ====================
        # 平滑过渡权重
        ratio = min(abs(delta_E) / gap_threshold, 1.0)
        alpha_orth = 0.5 + 0.5 * ratio
        alpha_tan = 1.0 - 0.7 * ratio

        # 如果能量差符号翻转（过零），减小正交步权重避免振荡
        if getattr(self, 'last_delta_E', None) is not None:
            if self.last_delta_E * delta_E < 0:
                print("⚠️ 能量差符号翻转，减小正交步权重")
                alpha_orth *= 0.5

        total_step = alpha_tan * step_tan + alpha_orth * step_orth

        # 最终步长限制
        step_norm = np.linalg.norm(total_step)
        final_max_step = 0.15

        if step_norm > final_max_step:
            total_step = total_step * (final_max_step / step_norm)
            print(f"🔻 total_step 截断: {step_norm:.4f} → {final_max_step:.4f}")

        # ==================== 诊断输出 ====================
        print("=" * 50)
        print(f"E1 = {E1:.6f}, E2 = {E2:.6f}, ΔE = {delta_E:.6f}")
        print(f"‖g_tan‖ = {np.linalg.norm(g_tan):.6f}")
        print(f"‖d_grad‖ = {norm_dg:.6f}")
        print(f"‖step_tan‖ = {np.linalg.norm(step_tan):.6f}")
        print(f"‖step_orth‖ = {np.linalg.norm(step_orth):.6f}")
        print(f"‖total_step‖ = {np.linalg.norm(total_step):.6f}")
        print(f"权重: α_tan={alpha_tan:.4f}, α_orth={alpha_orth:.4f}")

        eigvals = np.linalg.eigvalsh(self.inv_hess)
        print(f"inv_hess 特征值范围: [{eigvals.min():.4f}, {eigvals.max():.4f}]")

        # 检查step_tan的方向
        if np.linalg.norm(step_tan) > 1e-10 and np.linalg.norm(g_tan) > 1e-10:
            cos_angle = np.dot(step_tan, -g_tan) / (np.linalg.norm(step_tan) * np.linalg.norm(g_tan))
            print(f"cos(step_tan, -g_tan) = {cos_angle:.4f}")

        # 检查正交步方向
        if np.linalg.norm(step_orth) > 1e-10:
            # step_orth应该使delta_E^2减小，即与delta_E*d_grad反向
            expected_change = 2 * delta_E * np.dot(dg_vec, step_orth)
            print(f"预测 Δ(E1-E2)² 变化: {expected_change:.6f} (应为负)")

        print("=" * 50)

        # ==================== 更新结构 ====================
        new_structure_vec = x_k + total_step
        new_structure = new_structure_vec.reshape((3, natom))

        self.state_1.inp.molecule.replace_new_xyz(new_structure)
        if hasattr(self.state_2.inp.molecule, 'carti'):
            self.state_2.inp.molecule.carti = self.state_1.inp.molecule.carti

        # ==================== 保存历史 ====================
        self.last_structure = x_k.copy()
        self.last_gradient = g_k.copy()
        self.last_delta_E = delta_E
        self.parallel_gradient = g_tan
        self.orthogonal_gradient = (np.eye(nvar) - P) @ g_target_vec

    def generate_new_inp(self):
        path = self.out_path
        self.state_1.job_name = "{}{}_job{}.inp".format(self.prefix, self.state_1._spin, self.job_num)
        self.state_2.job_name = "{}{}_job{}.inp".format(self.prefix, self.state_2._spin, self.job_num)
        out = open(os.path.join(path, self.state_1.job_name), "w")
        out.write(self.state_1.inp.molecule.return_output_format() + self.state_1.inp.remain_texts)
        out.close()
        out = open(os.path.join(path, self.state_2.job_name), "w")
        out.write(self.state_2.inp.molecule.return_output_format() + self.state_2.inp.remain_texts)
        out.close()

    def check_convergence(self):
        E1 = self.state_1.out.ene
        E2 = self.state_2.out.ene
        delta_E = abs(E1 - E2)

        # 切向梯度范数（MECP点应该为0）
        tan_grad_norm = np.linalg.norm(self.parallel_gradient)

        # 正交梯度范数（可选监控）
        orth_grad_norm = np.linalg.norm(self.orthogonal_gradient)

        natom = self.state_1.inp.molecule.natom

        # 结构位移
        current_structure = self.state_1.inp.molecule.return_xyz_list().astype(float).T
        if self.last_structure is not None:
            last_structure = self.last_structure.reshape((3, natom))
            displacement = np.linalg.norm(current_structure - last_structure)
        else:
            displacement = np.inf

        # 收敛判据
        energy_converged = delta_E < self.energy_tol
        grad_converged = tan_grad_norm < self.grad_tol
        disp_converged = displacement < self.disp_tol

        converged_flags = [energy_converged, grad_converged, disp_converged]
        is_converged = sum(converged_flags) >= 2

        print(f"Energy gap: {delta_E:.5e}, Converged? {energy_converged}")
        print(f"Tangent gradient norm: {tan_grad_norm:.5e}, Converged? {grad_converged}")
        print(f"Orthogonal gradient norm: {orth_grad_norm:.5e} (for reference)")
        print(f"Displacement: {displacement:.5e}, Converged? {disp_converged}")
        print(f"Overall converged: {is_converged} ({sum(converged_flags)}/3 criteria met)\n")

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
