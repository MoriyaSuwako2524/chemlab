

import numpy as np
import math as ma
import matplotlib.pyplot as plt


def F0(t):
    if t < 1e-8:
        return 1 - t / 3
    else:
        return 0.5 * ((ma.pi / t) ** 0.5) * ma.erf(t ** 0.5)


A_R = 0.529177249  # 1波尔半径 = 0.529177249 埃
U = np.array([[2 ** -0.5, 2 ** -0.5], [2 ** -0.5, -2 ** -0.5]])


class basis_set(object):  # 基组计算
    '''
    基组计算详细内容:
    对于单电子积分

    '''

    def __init__(self):
        self.set_name = ""
        self.contraction_coefficient = []
        self.gaussian_index = []
        self.zeta = {1: 1.24, 2: 2.095}

    def psi_GF_1s(self, a, za, rR):
        a = a * (self.zeta[za] ** 2)
        out = ((2 * a / ma.pi) ** (3 / 4)) * ma.exp(-a * (rR ** 2))
        return out

    def psi_CGF_1s_nG(self, za, rR):
        out = 0
        for i in range(len(self.contraction_coefficient)):
            out += self.contraction_coefficient[i] * self.psi_GF_1s(self.gaussian_index[i], za, rR)
        return out

    def original_gaussian_overlap_integral(self, a, b, za, zb, RR):  # 原初高斯重叠积分 (a|b)
        a = a * (self.zeta[za] ** 2)
        b = b * (self.zeta[zb] ** 2)
        out = ((ma.pi / (a + b)) ** (3 / 2)) * (ma.exp((((-a * b) / (a + b))) * RR ** 2)) * (
                    (2 * a / ma.pi) ** (3 / 4)) * ((2 * b / ma.pi) ** (3 / 4))
        return out

    def kinetic_energy_integral(self, a, b, za, zb, RR):  # 动能积分(a|-0.5delta^2|b)
        a = a * (self.zeta[za] ** 2)
        b = b * (self.zeta[zb] ** 2)
        out = (a * b / (a + b)) * (3 - (2 * a * b * (RR ** 2)) / (a + b)) * ((ma.pi / (a + b)) ** (3 / 2)) * (
            ma.exp((((-a * b) / (a + b))) * RR ** 2)) * ((2 * a / ma.pi) ** (3 / 4)) * ((2 * b / ma.pi) ** (3 / 4))
        return out

    def nuclear_attraction_integral(self, a, b, za, zb, Ra, Rb, Rc, Z):  # 核吸引积分 (a|Zc/r1c|b)
        a = a * (self.zeta[za] ** 2)
        b = b * (self.zeta[zb] ** 2)
        Rp = (a * Ra + b * Rb) / (a + b)
        ra_rb = np.linalg.norm(Ra - Rb)
        rp_rc = np.linalg.norm(Rp - Rc)
        # print(a,b,Ra,Rb,Rc,Rp,rp_rc)
        out = (-2 * ma.pi / (a + b)) * Z * (ma.exp((((-a * b) / (a + b))) * (ra_rb ** 2))) * F0(
            (a + b) * (rp_rc ** 2)) * ((2 * a / ma.pi) ** (3 / 4)) * ((2 * b / ma.pi) ** (3 / 4))
        return out

    def two_electron_integral(self, a, b, c, d, za, zb, zc, zd, Ra, Rb, Rc, Rd):  # 双电子积分(ij|kl)
        a = a * (self.zeta[za] ** 2)
        b = b * (self.zeta[zb] ** 2)
        c = c * (self.zeta[zc] ** 2)
        d = d * (self.zeta[zd] ** 2)
        Rp = (a * Ra + b * Rb) / (a + b)
        Rq = (c * Rc + d * Rd) / (c + d)
        ra_rb = np.linalg.norm(Ra - Rb)
        rc_rd = np.linalg.norm(Rc - Rd)
        rp_rq = np.linalg.norm(Rp - Rq)
        K = (ma.exp((((-a * b) / (a + b))) * (ra_rb ** 2))) * (ma.exp((((-c * d) / (c + d))) * (rc_rd ** 2)))
        out = (2 * (ma.pi ** (5 / 2))) / ((a + b) * (c + d) * ((a + b + c + d) ** 0.5)) * K * F0(
            ((a + b) * (c + d) / (a + b + c + d)) * (rp_rq ** 2))
        out *= ((2 * a / ma.pi) ** (3 / 4)) * ((2 * b / ma.pi) ** (3 / 4)) * ((2 * c / ma.pi) ** (3 / 4)) * (
                    (2 * d / ma.pi) ** (3 / 4))
        return out


class molecular(object):  # 分子类
    def __init__(self):
        self.electron_number = 2
        self.orbital_number = self.electron_number
        self.nuclear_list = [2, 1]
        self.nuclear_number = len(self.nuclear_list)
        self.nuclear_position_cartesian = np.array([[1.4632, 0, 0], [0, 0, 0]])
        self.overlap_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.core_hamiltonian_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.kinetic_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.nuclear_attract_matrix = np.zeros((self.nuclear_number, self.orbital_number, self.orbital_number))
        self.basis_set_name = "STO-3G"
        self.basis = basis_set()
        self.RR_matrix = np.array([[0, 1.4], [1.4, 0]])
        self.two_electron_integral_matrix = np.zeros((self.orbital_number ** 2, self.orbital_number ** 2))
        self.density_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.g_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.thoery = "RHF"
        self.charge = 0

        if self.basis_set_name == "STO-3G":
            self.basis.contraction_coefficient = [0.444635, 0.535328, 0.154329]
            self.basis.gaussian_index = [0.109818, 0.405771, 2.22766]
            self.zeta_1s = {1: 1.24, 2: 2.0925}
            self.basis.zeta = self.zeta_1s
    def update_molecular(self):
        self.electron_number = sum(self.nuclear_list) - self.charge
        self.orbital_number = len(self.nuclear_list)
        self.nuclear_number = len(self.nuclear_list)
        self.overlap_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.core_hamiltonian_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.kinetic_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.nuclear_attract_matrix = np.zeros((self.nuclear_number, self.orbital_number, self.orbital_number))
        self.two_electron_integral_matrix = np.zeros((self.orbital_number ** 2, self.orbital_number ** 2))
        self.density_matrix = np.zeros((self.orbital_number, self.orbital_number))
        self.g_matrix = np.zeros((self.orbital_number, self.orbital_number))

    def calculate_RR_matrix(self):  # 距离矩阵 R_ij = |P[i] - P[j]|
        on = len(self.nuclear_position_cartesian)
        P = self.nuclear_position_cartesian
        R_t = P[:, None, :] - P[None, :, :]
        R = np.linalg.norm(R_t, axis=-1)
        self.RR_matrix = R
        E_RR = 0
        for i in range(on):
            if i == 0 :
                continue
            for j in range(on):
                if j>= i :
                    continue
                E_RR += 1/R[i][j]
        self.nuclear_energy = E_RR


    def calculate_overlap_matrix(self):  # 重叠矩阵计算 S[i][j] = \sum p \sum q \psi_pq
        S = self.overlap_matrix
        #print(self.RR_matrix)
        length = len(self.basis.contraction_coefficient)
        # Over_int = np.zeros(length,length)
        for i in range(len(S[0])):
            for j in range(len(S[0])):
                RR = self.RR_matrix[i][j]
                if j < i:
                    continue
                if i == j:
                    S[i][j] = 1.0
                else:
                    s_i_j = 0

                    za = self.nuclear_list[i]
                    zb = self.nuclear_list[j]
                    for p in range(length):
                        for q in range(length):
                            a = self.basis.gaussian_index[p]
                            b = self.basis.gaussian_index[q]
                            c = self.basis.contraction_coefficient[p]
                            d = self.basis.contraction_coefficient[q]
                            s_i_j += self.basis.original_gaussian_overlap_integral(a, b, za, zb, RR) * c * d
                    S[i][j] = s_i_j
                    S[j][i] = s_i_j
        self.overlap_matrix = S

    def calculate_overlap_matrix_2(self):
        S = self.overlap_matrix

    def calculate_kinetic_matrix(self):  # 动能矩阵计算
        T = self.kinetic_matrix
        for i in range(len(T[0])):
            for j in range(len(T[0])):
                RR = self.RR_matrix[i][j]
                if j < i:
                    continue
                t_i_j = 0
                length = len(self.basis.contraction_coefficient)
                za = self.nuclear_list[i]
                zb = self.nuclear_list[j]
                for p in range(length):
                    for q in range(length):
                        a = self.basis.gaussian_index[p]
                        b = self.basis.gaussian_index[q]
                        c = self.basis.contraction_coefficient[p]
                        d = self.basis.contraction_coefficient[q]
                        t_i_j += self.basis.kinetic_energy_integral(a, b, za, zb, RR) * c * d
                T[i][j] = t_i_j
                T[j][i] = t_i_j
        self.kinetic_matrix = T

    def calculate_nuclear_attract_matrix(self):  # 核吸引矩阵
        for I in range(len(self.nuclear_attract_matrix)):
            V = self.nuclear_attract_matrix[I]
            z = self.nuclear_list[I]
            Rc = self.nuclear_position_cartesian[I]
            for i in range(len(V[0])):
                for j in range(len(V[0])):
                    RR = self.RR_matrix[i][j]
                    if j < i:
                        continue
                    v_i_j = 0
                    length = len(self.basis.contraction_coefficient)
                    Ra = self.nuclear_position_cartesian[i]
                    Rb = self.nuclear_position_cartesian[j]
                    za = self.nuclear_list[i]
                    zb = self.nuclear_list[j]
                    for p in range(length):
                        for q in range(length):
                            a = self.basis.gaussian_index[p]
                            b = self.basis.gaussian_index[q]
                            c = self.basis.contraction_coefficient[p]
                            d = self.basis.contraction_coefficient[q]

                            v_i_j += self.basis.nuclear_attraction_integral(a, b, za, zb, Ra, Rb, Rc,
                                                                            z) * c * d  # a,b,Ra,Rb,Rc,zc)
                    V[i][j] = v_i_j
                    V[j][i] = v_i_j
            self.nuclear_attract_matrix[I] = V

    def calculate_matrixs(self):  # 计算重叠，动能与核吸引等单电子积分
        self.calculate_RR_matrix()
        self.calculate_overlap_matrix()
        self.calculate_kinetic_matrix()
        self.calculate_nuclear_attract_matrix()

    def calculate_core_hamiltonian_matrix(self):  # 将计算得到的动能，核吸引积分相加得到芯哈密顿矩阵
        self.calculate_matrixs()
        H = self.kinetic_matrix.copy()
        for i in range(len(self.nuclear_attract_matrix)):
            H += self.nuclear_attract_matrix[i]
        self.core_hamiltonian_matrix = H

    def calculate_two_electron_integral_matrix(self):  # 双电子积分的计算
        on = self.orbital_number
        N = np.zeros((on ** 2, on ** 2))
        I = -1
        N_dict = {}
        for i in range(on):
            for j in range(on):
                I += 1
                J = -1
                for k in range(on):
                    for l in range(on):
                        J += 1
                        if J < I:
                            continue
                        n_i_j_k_l = 0
                        length = len(self.basis.contraction_coefficient)
                        Ra = self.nuclear_position_cartesian[i]
                        Rb = self.nuclear_position_cartesian[j]
                        Rc = self.nuclear_position_cartesian[k]
                        Rd = self.nuclear_position_cartesian[l]
                        za = self.nuclear_list[i]
                        zb = self.nuclear_list[j]
                        zc = self.nuclear_list[k]
                        zd = self.nuclear_list[l]
                        for p in range(length):
                            for q in range(length):
                                for n in range(length):
                                    for m in range(length):
                                        a = self.basis.gaussian_index[p]
                                        b = self.basis.gaussian_index[q]
                                        c = self.basis.gaussian_index[n]
                                        d = self.basis.gaussian_index[m]
                                        e = self.basis.contraction_coefficient[p]
                                        f = self.basis.contraction_coefficient[q]
                                        g = self.basis.contraction_coefficient[n]
                                        h = self.basis.contraction_coefficient[m]
                                        n_i_j_k_l += self.basis.two_electron_integral(a, b, c, d, za, zb, zc, zd, Ra,
                                                                                      Rb, Rc, Rd) * e * f * g * h
                        # print([I,J],[i+1,j+1,k+1,l+1],n_i_j_k_l)
                        N_dict[i, j, k, l] = [I, J]
                        N_dict[i, j, l, k] = [I, J]
                        N_dict[l, k, i, j] = [I, J]
                        N_dict[k, l, i, j] = [I, J]
                        N[I][J] = n_i_j_k_l
                        N[J][I] = n_i_j_k_l
        self.two_electron_integral_matrix = N
        self.two_electron_integral_matrix_index = N_dict

    def calculate_canonical(self):  # 正交化重叠矩阵获得正交X矩阵
        S = self.overlap_matrix.copy()
        s12 = S[0][1]
        s1 = s12 + 1
        s2 = 1 - s12
        S_12 = np.array([[s1 ** -0.5, 0], [0, s2 ** -0.5]])
        X = np.linalg.inv(np.linalg.cholesky(S).T)
        #self.canonical = np.dot(U, S_12)
        self.canonical = X

    def guess_fock_matrix(self):  # 初猜Fock方程，这里直接猜的密度矩阵所有矩阵元全为0
        self.fock_matrix = self.core_hamiltonian_matrix.copy()
        F = self.fock_matrix.copy()
        self.calculate_canonical()
        X = self.canonical.copy()
        X_pin = X.T.conjugate()

        F2 = np.dot(np.dot(X_pin, F), X)
        self.fock_matrix = F2

    def diagonalize_fock_matrix(self):  # 根据X矩阵对角化Fock矩阵，使得HFR方程转为标准矩阵方程
        self.guess_fock_matrix()
        F = self.fock_matrix.copy()
        X = self.canonical.copy()
        C0 = np.linalg.eig(F)[1]
        epsilon = np.linalg.eig(F)[0]
        C = np.dot(X, C0)
        # print(F,C0,C,epsilon)
        return [epsilon, C, C0]

    def SCF(self, out=False):  # SCF迭代手续
        self.update_molecular()
        self.calculate_core_hamiltonian_matrix()  # 初始化
        H = self.core_hamiltonian_matrix.copy()
        l = self.diagonalize_fock_matrix()
        F = self.fock_matrix.copy()
        X = self.canonical.copy()
        C = l[1]
        on = self.electron_number
        self.calculate_two_electron_integral_matrix()
        N = self.two_electron_integral_matrix.copy()
        N_dict = self.two_electron_integral_matrix_index.copy()

        def density_matrix(n, C):
            P = C.copy()
            for p in range(len(P[0])):
                for q in range(len(P[0])):
                    P_p_q = 0
                    for i in range(int(n / 2)):
                        P_p_q += C[p][i] * C[q][i]
                    P_p_q = P_p_q * 2
                    P[p][q] = P_p_q
            return P

        P = np.zeros(F.shape)
        self.density_matrix = P.copy()

        def g_matrix(P, N, N_dict):
            G = P.copy()
            for i in range(len(P[0])):
                for j in range(len(P[0])):  # i=u j=v
                    g_i_j = 0
                    for p in range(len(P[0])):
                        for q in range(len(P[0])):  # p =lambda q= sigma

                            index_i_j_p_q = N_dict[(i, j, p, q)]
                            index_i_q_p_j = N_dict[(i, q, p, j)]
                            g_i_j += P[p][q] * (N[index_i_j_p_q[0], index_i_j_p_q[1]] - 0.5 * N[
                                index_i_q_p_j[0], index_i_q_p_j[1]])
                    G[i][j] = g_i_j
                    G[j][i] = g_i_j
            return G

        def energy(P, F, H):
            e = 0
            for i in range(len(P[0])):
                for j in range(len(P[0])):
                    e += P[i][j] * (H[i][j] + F[i][j])
            return e * 0.5

        energylist = [energy(P, F, H)]

        def converse_fock_matrix(F, X):
            X_pin = X.T.conjugate()
            F2 = np.dot(np.dot(X_pin, F), X)
            return F2

        def diagonalize_fock_matrix(F, X):
            C0 = np.linalg.eig(F)[1]
            epsilon = np.linalg.eig(F)[0]
            C = np.dot(X, C0)
            return [epsilon, C, C0]

        P = self.density_matrix.copy()
        # print(P)
        for i in range(1000):
            if i == 999:
                self.SCF_converge = False
                break
            G = g_matrix(P, N, N_dict)
            F = G + H
            energylist.append(energy(P, F, H))
            F2 = converse_fock_matrix(F, X)
            C = diagonalize_fock_matrix(F2, X)[1]
            # C= (np.array([C[0] * [-1, 1], C[1] * [-1, 1]]))
            epsilon = np.diag(diagonalize_fock_matrix(F, X)[0])
            if out:
                print("这是第{}次迭代".format(i + 1), "\nP=", P, "\nF=", F, "\nC=", C, "\nepsilon=", epsilon,
                      "\nE_ele=", energylist[-1])
            if abs(energylist[-1] - energylist[-2]) <= 1e-7 and i != 0:
                self.converge_time = i
                self.SCF_converge = True
                break
            P = density_matrix(on, C)
        if self.SCF_converge:
            print("SCF Done")
        elif not self.SCF_converge:
            print("SCF false")
        self.density_matrix = P.copy()
        self.fock_matrix = F.copy()
        self.canonical = C  # C为变换后的矩阵
        self.energy = energylist[-1] +self.nuclear_energy

    def guess_fock_matrix_UHF(self):  # 初猜UHF-Fock矩阵，这里直接猜的密度矩阵所有矩阵元全为0
        self.fock_matrix_alpha = self.core_hamiltonian_matrix.copy() * (0.5)
        self.fock_matrix_beta = self.core_hamiltonian_matrix.copy() * (0.5)

        F_a = self.fock_matrix_alpha.copy()
        F_b = self.fock_matrix_beta.copy()
        self.calculate_canonical()
        X = self.canonical.copy()
        X_pin = X.T.conjugate()
        F2 = np.dot(np.dot(X_pin, F_a), X)
        self.fock_matrix_alpha = F2
        F2 = np.dot(np.dot(X_pin, F_b), X)
        self.fock_matrix_beta = F2

    def diagonalize_fock_matrix_UHF(self):  # 根据X矩阵对角化Fock矩阵，使得HFR方程转为标准矩阵方程
        self.guess_fock_matrix_UHF()
        F_a = self.fock_matrix_alpha.copy()
        F_b = self.fock_matrix_beta.copy()
        X = self.canonical.copy()
        C0_a = np.linalg.eig(F_a)[1]
        C0_b = np.linalg.eig(F_b)[1]
        # print(F_a,F_b,C0_a,C0_b)
        epsilon_a = np.linalg.eig(F_a)[0]
        epsilon_b = np.linalg.eig(F_b)[0]
        C_a = np.dot(X, C0_a)
        C_b = np.dot(X, C0_b)
        # print(C_a,C_b)
        # print(epsilon_a,epsilon_b)
        return [[epsilon_a, C_a, C0_a], [epsilon_b, C_b, C0_b]]

    def UHF_SCF(self, out=False):
        self.update_molecular()
        self.calculate_core_hamiltonian_matrix()  # 初始化
        H = self.core_hamiltonian_matrix.copy()
        l = self.diagonalize_fock_matrix_UHF()
        l_a = l[0]
        l_b = l[1]

        F_a = self.fock_matrix_alpha.copy()
        F_b = self.fock_matrix_beta.copy()

        X = self.canonical.copy()

        C_a = l_a[1]
        C_b = l_b[1]
        if self.electron_number % 2 == 0:
            on_a = self.electron_number / 2
            on_b = self.electron_number / 2
        else:
            on_a = self.electron_number // 2 + 1
            on_b = self.electron_number // 2
        self.calculate_two_electron_integral_matrix()
        N = self.two_electron_integral_matrix.copy()
        N_dict = self.two_electron_integral_matrix_index.copy()

        def density_matrix(n, C):
            P = C.copy()
            for p in range(len(P[0])):
                for q in range(len(P[0])):
                    P_p_q = 0
                    for i in range(int(n)):
                        P_p_q += C[p][i] * C[q][i]
                    P_p_q = P_p_q * 2
                    P[p][q] = P_p_q
            return P

        P_a = density_matrix(on_a, C_a)
        P_b = density_matrix(on_b, C_b)
        # print(P_a,P_b)
        self.density_matrix_alpha = P_a.copy()
        self.density_matrix_beta = P_b.copy()
        P = P_a + P_b
        self.density_matrix = P.copy()

        def g_matrix(P_a, P_b, N, N_dict):
            P = P_a + P_b
            G_a = P_a.copy()
            G_b = P_b.copy()
            for i in range(len(P[0])):
                for j in range(len(P[0])):  # i=u j=v
                    g_a_i_j = 0
                    g_b_i_j = 0
                    for p in range(len(P[0])):
                        for q in range(len(P[0])):  # p =lambda q= sigma
                            index_i_j_p_q = N_dict[(i, j, p, q)]
                            index_i_q_p_j = N_dict[(i, q, p, j)]
                            g_a_i_j += P[p][q] * (N[index_i_j_p_q[0], index_i_j_p_q[1]]) - P_a[p][q] * N[
                                index_i_q_p_j[0], index_i_q_p_j[1]]
                            g_b_i_j += P[p][q] * (N[index_i_j_p_q[0], index_i_j_p_q[1]]) - P_b[p][q] * N[
                                index_i_q_p_j[0], index_i_q_p_j[1]]
                    G_a[i][j] = g_a_i_j
                    G_a[j][i] = g_a_i_j
                    G_b[i][j] = g_b_i_j
                    G_b[j][i] = g_b_i_j

            return [G_a, G_b]

        def energy(P, H, P_a, P_b, F_a, F_b):
            e = 0
            for i in range(len(P[0])):
                for j in range(len(P[0])):
                    e += P[i][j] * H[i][j] + P_a[i][j] * F_a[i][j] + P_b[i][j] * F_b[i][j]
            return e * 0.5

        energylist = [energy(P, H, P_a, P_b, F_a, F_b)]  # 随机计算一次F

        def converse_fock_matrix(F, X):
            X_pin = X.T.conjugate()
            F2 = np.dot(np.dot(X_pin, F), X)
            return F2

        def diagonalize_fock_matrix(F, X):
            C0 = np.linalg.eig(F)[1]
            epsilon = np.linalg.eig(F)[0]
            C = np.dot(X, C0)
            return [epsilon, C, C0]

        # 以下为SCF手续正式迭代部分
        P_a = np.zeros(F_a.shape)
        # P_b = np.zeros(F_b.shape)
        P_b = np.random.random(F_a.shape)
        P = P_a + P_b

        for i in range(1000):
            if i == 999:
                self.SCF_converge = False
                break
            G = g_matrix(P_a, P_b, N, N_dict)
            G_a = G[0]
            G_b = G[1]
            F_a = G_a + H
            F_b = G_b + H
            energylist.append(energy(P, H, P_a, P_b, F_a, F_b))
            F2_a = converse_fock_matrix(F_a, X)
            F2_b = converse_fock_matrix(F_b, X)
            C_a = diagonalize_fock_matrix(F2_a, X)[1]
            C_b = diagonalize_fock_matrix(F2_b, X)[1]
            # C= (np.array([C[0] * [-1, 1], C[1] * [-1, 1]]))

            if out:
                print("这是第{}次迭代".format(i + 1), "\nP=", P, "\nP_a=", P_a, "\nP_b=", P_b, "\nF_a=", F_a, "\nF_b=",
                      F_b, "\nE_ele=", energylist[-1])
            if abs(energylist[-1] - energylist[-2]) <= 1e-7 and i != 0:
                self.SCF_converge = True
                break
            P_a = 0.5 * density_matrix(on_a, C_a)
            P_b = 0.5 * density_matrix(on_b, C_b)
            P = P_a + P_b
        print(C_a, C_b)
        self.density_matrix = P.copy()
        self.fock_matrix_alpha = F_a.copy()
        self.fock_matrix_beta = F_b.copy()
        self.canonical = X  # C为变换后的矩阵
        self.energy = energylist[-1] +self.nuclear_energy

    def molecular_orbital(self):
        self.SCF()


'''
tem_mol = molecular()
def read_mol(input_mol):
    read_set = False
    tem_mol.__init__()
    tem_mol_car = []
    tem_mol_nuc = []
    tem_dic = {"H":1,"He":2}
    for i in range(len(input_mol)):
        if read_set :
            if input_mol[i] == "":
                continue
            tem_line = input_mol[i].split(" ")
            while "" in tem_line:
                tem_line.remove("")
            if len(tem_line) == 2:
                tem_mol.charge = int(tem_line[0])
            elif len(tem_line) == 4 :
                tem_mol_car.append([float(tem_line[1]),float(tem_line[2]),float(tem_line[3])])
                tem_mol_nuc.append(tem_dic[tem_line[0]])
        if "Title" in input_mol[i]:
            read_set = True
    tem_mol.nuclear_position_cartesian = np.divide(np.array(tem_mol_car),A_R)
    tem_mol.nuclear_list = np.array(tem_mol_nuc)
'''
H2 = molecular()
H2.nuclear_position_cartesian = np.array([[1.5 / A_R, 0, 0], [0, 0, 0]])
H2.nuclear_list = [1,1]
H2.UHF_SCF()
#H2.SCF()
#H2.molecular_orbital()
print(H2.energy + A_R / 1.5)

mol_name  = "H2-2.gjf"
input_mol = open(mol_name,"r").read().split("\n")

#read_mol(input_mol)
#tem_mol.SCF(out=True)
#print(tem_mol.energy)
'''
rand_car = np.zeros((4,3))

out_file = open("test.txt","w")
Done = 0
for i in range(250):
    if Done == 100:
        break
    print("num:"+str(i))
    np.random.seed(10000+i)
    rand_car = np.random.random((4,3))*3
    tem_mol.charge = 0
    tem_mol.nuclear_list = [1,1,1,1]
    tem_mol.nuclear_position_cartesian = rand_car
    tem_mol.SCF()
    if tem_mol.SCF_converge:
        Done +=1
        out_file.write("No." + str(i+1))
        out_file.write("\n")
        out_file.write(str(tem_mol.converge_time))
        out_file.write("\n")
        out_file.write(str(tem_mol.core_hamiltonian_matrix))
        out_file.write("\n")
        out_file.write(str(tem_mol.density_matrix))
        out_file.write("\n")

'''