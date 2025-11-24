/*
 * =====================================================================================
 *
 *       Filename:  spin_adiabatic_state.C
 *
 *    Description:  Construt SAC using spin-orbit coupling between
 *                  two spin states
 *
 *        Version:  4.0
 *        Created:  06/16/2019 04:31:45 PM
 *       Revision:  02/20/2020, 01/28/2023
 *
 *         Author:  Zheng, Yihan 
 *
 * =====================================================================================
 */

#ifndef _SPIN_ADIABATIC_STATE_H_
#define _SPIN_ADIABATIC_STATE_H_

//#include <libgscf/util/arma_extra.h>
//#include <cassert>
#include "gen_scfman_overlap.h"
//#include "rohf.h"

#include "qchem.h"
#include "BSetMgr.hh"
#include "BasisSet.hh"
#include <gen_scfman/gen_scfman_main.h>
#include <gen_scfman/gen_scfman_common.h>
#include <gen_scfman/get_scf_params.h>
#include <gen_scfman/scf_params.h>
#include <gen_scfman/scf.h>
#include <gen_scfman/rhf.h>
#include <gen_scfman/rohf.h>
#include <gen_scfman/gen_scfman_hybrid_algorithm.h>
#include <libgscf/fock/fock_desc.h>
#include <libfock/hso/hso1.h>
#include <libfock/hso/hso1_deriv1.h>
#include <libqints/qchem/aobasis.h>
#include <libqints/algorithms/gto/gto_order.h>
#include <include/operators.h>
#include <libgscf/opt/lin_minres.h>
#include <libgscf/opt/lin_block_cg.h>
#include <libgscf/opt/hessian_linear_problem.h>
#include <gen_scfman/ks_hess_on_vecs_r_u.h>
#include <libalmo/cpscf/cpscf_bl_r.h>
//#include <drvman.h>
#include "../../drvman/drvman.h"
#include "../../setman/setman_codes.h"
//#include "../../drvman/soc_singles_grad.h"
#include "gscf_cpscf_serial_logic.h"
#include <libgscf/qchem/deriv/cpscf_driver_qchem.h>
#include <libgscf/util/arma_extra.h>

void DrvMan_omp(void);
using libgscf::fock_desc;
using namespace arma;   


struct MOpair { 
    mat C_alpha;   
    mat C_beta;   // !caution, the MOs here are already biothornal
    mat U;
    mat V;
    vec lambda;
    mat effect_C_o_alpha;
    mat effect_C_o_beta;
    mat effect_C_v_alpha;
    mat effect_C_v_beta;
    mat E_a;
    mat E_b;

    vector<int> flips;

    size_t n_svd,n_ao,n_occ_a,n_occ_b,n_vir_a,n_vir_b;

    // gradient
    mat sigma_u, sigma_v;
    mat sigma_aa, sigma_bb, sigma_ab, sigma_ba;
};

struct OrbitalPair {
    double Ms1;
    double Ms2;
    int direction;
    int nvir1_a;
    int nvir1_b;
    int nvir2_a;
    int nvir2_b;
    int Ms_idx;
    MOpair block1;
    MOpair block2;
    mat C1_flipped_alpha;
    mat C1_flipped_beta;
    mat C2_flipped_alpha;
    mat C2_flipped_beta;

    vec psi1;   // MO from state 1 (e.g. singlet, doublet)
    vec psi2;   // MO from state 2 (e.g. triplet, quartet)
    mat U_a;
    mat U_b;
    mat V_a;
    mat V_b;
    vec lambda_a;
    vec lambda_b;
    int slater_phase1; // slater determinant phase factor
    int slater_phase2;
    double phase;    // SVD determinant phase factor
    vec psi1_alpha;   
    vec psi2_alpha;  
    vec psi1_beta;   
    vec psi2_beta;  
    double vsoc_x,vsoc_y;
    double val_a,val_b;
    double vsoc_z;
    double phase_alpha,phase_beta; 

    // gradients

    mat deriv_explicit_lxy;
    mat deriv_explicit_lz;
    mat deriv_explicit_s;
    mat y1_a,y1_b,y2_a,y2_b;

    mat sigma_ua, sigma_va,sigma_ub, sigma_vb;
    mat pi_aa_1,pi_bb_1,pi_ab_1,pi_ba_1;
    mat pi_aa_2,pi_bb_2,pi_ab_2,pi_ba_2;
    mat L_a_1,L_a_2,L_b_1,L_b_2;
    mat L_aa_1,L_ab_1,L_ba_1,L_bb_1;
    mat L_aa_2,L_ab_2,L_ba_2,L_bb_2;
    vec explicit_derivatives;
    vec implicit_derivatives_1;
    vec implicit_derivatives_2;
    bool ene_check=false;
    bool gradient_check=false;

};
struct vsoc_value{
    double vsoc_x = 0;
    double vsoc_y = 0;
    double vsoc_z = 0;
};





class spin_adiabatic_state {
   public:

   // basis set info
   const size_t NOrb = rem_read(REM_NLINOR);
   const size_t NB2 = rem_read(REM_NB2);
   const size_t NBas = bSetMgr.crntShlsStats(STAT_NBASIS);
   const size_t NBas2 = NBas * NBas;
   const size_t NB2car = rem_read(REM_NB2CAR);
   const size_t NBas6D = bSetMgr.crntShlsStats(STAT_NBAS6D);
   const size_t NBas6D2 = NBas6D * NBas6D;
   const double scaling_factor = -1.0/137.036/137.036/2.0;
   // have to write this to file
   int NAtoms = rem_read(REM_NATOMS);
   const size_t Nuclear = 3 * NAtoms;

   libqints::array_view<double> m_nucchg; // (effective) nuclear charges

   // scf classes for two states
   scf *myscf_1, *myscf_2;
   rhf *myrhf_1, *myrhf_2;
   uhf *myuhf_1, *myuhf_2;
   rohf *myrohf_1, *myrohf_2;

   int unrestricted;
   size_t nalpha1, nbeta1, nvir1_a, nvir1_b; // first state
   size_t nalpha2, nbeta2, nvir2_a, nvir2_b; // second state
   int n_total;
   double S1, S2;
   int n_Ms1, n_Ms2;
   int Ms_dimension;
   mat AOS;
   mat C1_alpha, C1_beta, C2_alpha, C2_beta;
   mat P1_alpha, P1_beta, P2_alpha, P2_beta;
   mat F1_alpha, F1_beta, F2_alpha, F2_beta;

   //mat S_MO_alpha, S_MO_beta;
   cube L_AO; // coupling matrices in AO
   int get_index(double Ms1, double Ms2);

   double E1, E2, E_soc, E_adiab; // energy variables
   size_t lagrangian;
   vec v_soc;
   double current_Vsoc;
   vector<vsoc_value> vsoc_values;
   uvec indices;
   
   // svd of core orbital overlap
   vector<mat> U, V;
   vector<vec> lambda;
   vec prod_s;

   //alpha-beta biorthonal mat of state1 and state 2
   MOpair S1_orthonal;
   MOpair S2_orthonal;
   vector<vector<OrbitalPair>> vsoc_pairs;

   vector<vec> C1_prime, C2_prime; // effective one-orbitals 
   mat C2_alpha_new, C2_beta_new; // transformed orbitals for z component
   mat y1_vo_alpha;
   mat y1_vo_beta;
   // second high-spin state
   mat y2_ov_alpha;
   mat y2_ov_beta;
   mat s_oo_inv_1;
   mat s_oo_inv_2;
   scf* run_scf(double& finalEnergy, const int spin);
   void run_two_states();
   void collect_scf_orbitals();
   void lsoc_xyz();
   vec vsoc_vector();
   void spin_orbit_coupling();
   
   double calc_phase_factor(double S, double Ms);
   vec vsoc_vector_modular();  
   int state_phase(const int n_1, const int n_2, const vector<int>& Index);
   void build_spin_blocks_for_state(const MOpair& ortho, double S, size_t nalpha, size_t nbeta,map<double, vector<MOpair>>& Ms_blocks);
   double compute_coupling_component(const OrbitalPair& pair, int direction);

    vec E_adiab_gradients();

   // gradients

   size_t need_gradient = 0; // init
   
   vec grad_1;
   vec grad_2;
   double* jGrad_vsoc;
   mat S_inv;

   mat S_MO_alpha;
   mat S_MO_beta;
   void getdL(double* vderiv, int l_vderiv);
   vec state_gradients();
   vec Esoc_gradients();
   void total_gradients();

    vec gradient_explicit_Ms();
    void gradient_explicit_s(OrbitalPair& pair);
    void gradient_explicit_lxy(OrbitalPair& pair);
    void gradient_explicit_lz(OrbitalPair& pair);
    void pseudo_density_explicit_xy(OrbitalPair& pair);
    void pseudo_density_explicit_z(OrbitalPair& pair);
    void gradient_implicit_rhs_Ms();


    void sigma_overlap(MOpair& block);
    void pi_matrix(OrbitalPair& pair);
    void k_matrix_null(OrbitalPair& pair); //  +- vsoc
    void k_matrix_last(OrbitalPair& pair); //  0 vsoc

   //void pseudo_density_explicit(mat &deriv_explicit_s, mat &deriv_explicit_l);
   void pseudo_density_explicit(mat &deriv_explicit_s, vector<mat> &deriv_explicit_l);
   vec gradient_explicit();
   void gradient_implicit_rhs(mat& y1_vo_alpha, mat& y1_vo_beta, mat& y2_ov_alpha, mat& y2_ov_beta);
   void gradient_implicit_xy(mat& y1_vo_alpha, mat& y1_vo_beta, mat& y2_ov_alpha, mat& y2_ov_beta);
   void gradient_implicit_z(mat& y1_vo_alpha, mat& y1_vo_beta, mat& y2_ov_alpha, mat& y2_ov_beta);
   vec gradient_implicit_rhf(scf* myscf, const mat& y_vo, const mat& C);
   vec gradient_implicit_uhf(scf* myscf, const mat& y_ov_alpha, const mat& y_ov_beta, const mat& Ca, const mat& Cb);
   vec gradient_implicit_rohf(scf* myscf, const mat& y_ov_alpha, const mat& y_ov_beta, const mat& C);

   //mat zvector_form_rhs_s(const mat& s_mo_oo, const size_t o, int type);
   //mat zvector_form_rhs_t(const mat& s_mo_oo, const size_t o, int type);
   vec zvector_solve(scf* thescf, vec& rhs);
   vec zvector_gradient(const vec& z, mat& PzA, mat& PzB, int spin=1);

   void check_rohf_orbital_response(int sort);

   vec gradient_one_electron(const double* jPv, int spin);


   spin_adiabatic_state();
   ~spin_adiabatic_state() { };
   static spin_adiabatic_state& instance() {
      static spin_adiabatic_state instance;
      return instance;
   }

   private:
   spin_adiabatic_state(const spin_adiabatic_state &);

};

#endif
