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

#define DEBUG_K_LAST 1
#ifdef DEBUG_K_LAST
#define DBG(msg) std::cout << "Zexuan Check " << msg << std::endl
#else
#define DBG(msg)
#endif

#include "spin_adiabatic_state.h"
void matrix_print_2d(const double* matrix, const size_t nrow, const size_t ncol, const char* keyword)
{
//   const size_t nrow = matrix.n_rows;
//   const size_t ncol = matrix.n_cols;

   const size_t nwidth = 6;

   size_t nbatch = ncol/nwidth;
   if (nbatch*nwidth < ncol) nbatch += 1;

   printf("%s\n", keyword);
   for (size_t k=0; k<nbatch; ++k){
      printf("    ");
      size_t j1 = nwidth*k;
      size_t j2 = nwidth*(k+1);
      if (k == nbatch-1) j2 = ncol;

     for (size_t j=j1; j<j2; ++j) 
         printf("%10d        ", (j+1));
      printf("\n");
      for (size_t i=0; i<nrow; ++i){
         printf("%4d", (i+1));
         for (size_t j=j1; j<j2; ++j) 
            printf("%18.10e", matrix[i + j*nrow]);
         printf("\n");
      }
   }
}

void print_mat_size(const arma::mat& M, const std::string& name = "") {
    std::cout << "Matrix " << name << " size: "
              << M.n_rows << " rows * " << M.n_cols << " cols" << std::endl;
}

int spin_adiabatic_state::get_index(double Ms1, double Ms2) {
    int i = round(Ms1 + S1);
    int j = round(Ms2 + S2);
    return i * (static_cast<int>(2*S2+1)) + j;
}

spin_adiabatic_state::spin_adiabatic_state()
{
   // see include/JobTypes.h
   const size_t jobtype = rem_read(REM_JOBTYP);
   // geom_opt, td_opt, reaction_path, force
   if (jobtype==2 || jobtype==3 || jobtype==6 || jobtype==7)
      need_gradient = 1;

   
   int *inucchg, NAtoms;
   get_carts(NULL,NULL,&inucchg,&NAtoms);
   double *jnucchg = QAllocDouble(NAtoms);
   for (int n=0; n<NAtoms; ++n){
      jnucchg[n] = (double) inucchg[n];
   }
      MatPrint(jnucchg, 1, NAtoms, "jnucchg");
   /*
    * parameters taken from 10.1080/0144235032000101743
    * 10.1002/jcc
    */
   if (rem_read(REM_YS_DEBUG)==-16){ // effective charge
      for (int n=0; n<NAtoms; ++n){
         if (inucchg[n]==3) jnucchg[n]=1.35;        // Li
         else if (inucchg[n]==6) jnucchg[n]=3.60;   // C
         else if (inucchg[n]==7) jnucchg[n]=4.55;   // N
         else if (inucchg[n]==8) jnucchg[n]=5.60;   // O
         else if (inucchg[n]==9) jnucchg[n]=6.75;   // F
         else if (inucchg[n]==12) jnucchg[n]=10.80;   // Mg
         else if (inucchg[n]==13) jnucchg[n]=11.53;   // Al
         else if (inucchg[n]==16) jnucchg[n]=13.60; // S
         else if (inucchg[n]==17) jnucchg[n]=14.24; // Cl

         // ECP SBKJC is used
         else if (inucchg[n]==25) jnucchg[n]=12.8; // Mn
         else if (inucchg[n]==26) jnucchg[n]=13.9; // Fe
         else if (inucchg[n]==27) jnucchg[n]=15.1; // Co
         else if (inucchg[n]==28) jnucchg[n]=16.4; // Ni
         else if (inucchg[n]==29) jnucchg[n]=17.7; // Cu
      }
      MatPrint(jnucchg, 1, NAtoms, "jnucchg");
   }
   m_nucchg = libqints::array_view<double>(jnucchg, NAtoms);
   //QFree(jnucchg); // can't free it
}


scf* spin_adiabatic_state::run_scf(double& finalEnergy, const int spin)
{
   //rem_write(justal, REM_JUSTAL);
   rem_write(spin, REM_SET_SPIN);

   int iter = 0;
   int maxIter = rem_read(REM_MAXSCF);
   
   scf_params::Orbtype ot = get_orbtype();
   libgscf::fock_desc aop;
   get_ao_params(aop);
   scf* the_scf = gen_scfman::orbital_selection(ot,aop);
   
   int IGUESS = rem_read(REM_IGUESS);
   cout << "zheng IGUESS: " << IGUESS << endl;
   the_scf->obtain_guess(IGUESS);
   
   int max_stab_iter = rem_read(REM_INTERNAL_STABILITY_ITER);
   bool do_stab = rem_read(REM_INTERNAL_STABILITY)>0;
   hybrid_algorithm(the_scf,true,do_stab,max_stab_iter,true);
   
   //if we didn't crash, then we have converged the scf
   rem_write(1,REM_SCF_CONVERGED);//so that drvman recognizes that we are a converged scf 
   
   finalEnergy = the_scf->total_energy();
   FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_CRNT_TOTAL_ENERGY,FM_BEG,&finalEnergy);
   FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_SCF_ENERGY,FM_BEG,&finalEnergy);
   the_scf->post_convergence_wrapup();

   return the_scf;
}


vec spin_adiabatic_state::state_gradients()
{
   SCF_Grad();
   vec grad = zeros<vec>(Nuclear);
   FileMan(FM_READ,FILE_NUCLEAR_GRADIENT,FM_DP,Nuclear,0,FM_BEG,grad.memptr());
   return grad;
}


void spin_adiabatic_state::lsoc_xyz()
{
   using namespace libqints;
   using libqints::qchem::aobasis;
   dev_omp dev; dev.init(0); 
   size_t memreq = libfock::hso1<double>(aobasis.b1, aobasis.bnuc1, dev).memreq();
   dev.init(memreq/dev.nthreads);
   cout << "memreq: " << memreq << " " << memreq/dev.nthreads << endl;

   L_AO = cube(NBas, NBas, 3);
   multi_array<double> ma_hso(3);
   for (size_t i=0; i<3; ++i){
      ma_hso.set(i, array_view<double>(L_AO.slice(i).memptr(), NBas2));
   }
   
   //size_t errcode = libfock::hso1<double>(aobasis.b1, aobasis.bnuc1, dev).compute(aobasis.nucchg, ma_hso);
   size_t errcode = libfock::hso1<double>(aobasis.b1, aobasis.bnuc1, dev).compute(m_nucchg, ma_hso);

   for (size_t i=0; i<3; ++i){
      mat Hsox(arrays<double>::ptr(ma_hso[i]), NBas, NBas, false, true);
      gto::reorder_cc(Hsox, aobasis.b1, true, true, gto::lex, gto::korder);
   }
   
   //MatPrint(L_AO.slice(0).memptr(), NBas, NBas, "L_AO x");
   //MatPrint(L_AO.slice(1).memptr(), NBas, NBas, "L_AO y");
   //MatPrint(L_AO.slice(2).memptr(), NBas, NBas, "L_AO z");
}


// copy from drvman/soc_singles_grad.C with customized nuclear charges.
void spin_adiabatic_state::getdL(double* vderiv, int l_vderiv)
{
   using namespace libqints;
   using libqints::qchem::aobasis;

   array_view<double> av_vderiv(vderiv,3*l_vderiv);

   //make dev
   dev_omp dev; dev.init(8UL * 1024*1024);

   //make ma_vderiv
   multi_array<double> ma_vderiv(3);
   for (size_t i=0; i<3; ++i)
      ma_vderiv.set(i, array_view<double>(av_vderiv, i*l_vderiv, l_vderiv));

   libfock::hso1_deriv1<double>(aobasis.b1, aobasis.bnuc1, dev).compute(m_nucchg, ma_vderiv);
 
   size_t nbsf = aobasis.b1.get_nbsf();
   cout << "zheng getdL nbsf: " << nbsf << " l_vderiv: " << l_vderiv << endl;
   assert(l_vderiv%(nbsf*nbsf) == 0);
   size_t l_v = 3*l_vderiv/(nbsf*nbsf);
   for(size_t i = 0; i < l_v; i++){
      arma::mat dHso(vderiv + i*nbsf*nbsf, nbsf, nbsf, false, true);
      gto::reorder_cc(dHso, aobasis.b1, true, true, gto::lex, gto::korder);
   }
   cout << "Zexuan Wei: function getdL end" << endl;
   return;
}

double spin_adiabatic_state::calc_phase_factor(double S, double Ms){

   double diff = S - Ms;

    if (abs(diff) < 1e-6) {
        return 1.0;
    }

    int nflip = static_cast<int>(round(diff));

    return 1.0 / sqrt(nflip);
}

vector<vector<int>> generate_combinations(int start, int end, int k) {
    int n = end - start;
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), start);

    vector<bool> select(n, false);
    std::fill(select.begin(), select.begin() + k, true);

    vector<vector<int>> result;
    do {
        vector<int> comb;
        for (int i = 0; i < n; ++i) {
            if (select[i]) comb.push_back(indices[i]);
        }
        result.push_back(comb);
    } while (prev_permutation(select.begin(), select.end()));
    return result;
}



mat make_E_a(int n_alpha, int n_beta, const vector<int>& flip_idxs) {

    vector<char> is_flip(n_alpha, 0);
    for (int idx : flip_idxs)
        if (idx >= 0 && idx < n_alpha)
            is_flip[idx] = 1;


    int n_unflipped = 0;
    for (int i = 0; i < n_alpha; ++i)
        if (!is_flip[i]) ++n_unflipped;


    mat J_alpha(n_alpha, n_unflipped, fill::zeros);
    int col = 0;
    for (int i = 0; i < n_alpha; ++i)
        if (!is_flip[i])
            J_alpha(i, col++) = 1.0;


    mat E_alpha_full(n_alpha + n_beta, n_unflipped, fill::zeros);
    E_alpha_full.submat(0, 0, n_alpha - 1, n_unflipped - 1) = J_alpha;

    return E_alpha_full;
}

mat make_E_b(int n_alpha, int n_beta, const vector<int>& flip_idxs) {

    mat E_alpha(n_alpha, static_cast<int>(flip_idxs.size()), fill::zeros);
    for (int j = 0; j < (int)flip_idxs.size(); ++j) {
        int idx = flip_idxs[j];
        if (idx >= 0 && idx < n_alpha)
            E_alpha(idx, j) = 1.0;
    }

    int nflip = static_cast<int>(flip_idxs.size());

    mat E_beta_full(n_alpha + n_beta, n_beta + nflip, fill::zeros);


    if (n_beta > 0)
        E_beta_full.submat(n_alpha, 0,
                           n_alpha + n_beta - 1, n_beta - 1)
            = eye(n_beta, n_beta);


    if (nflip > 0)
        E_beta_full.submat(0, n_beta,
                           n_alpha - 1, n_beta + nflip - 1)
            = E_alpha;

    return E_beta_full;
}


void spin_adiabatic_state::build_spin_blocks_for_state(
    const MOpair& ortho, double S, size_t nalpha, size_t nbeta,
    map<double, vector<MOpair>>& Ms_blocks)
{
    int twoS = static_cast<int>(std::lround(2 * S));
    int n_Ms = twoS + 1;

    for (int i = 0; i < n_Ms; ++i) {
        int twoMs_int = -twoS + 2 * i;
        double Ms = 0.5 * twoMs_int;
        int tem_nalpha = (n_total + std::abs(twoMs_int)) / 2;
        int tem_nbeta  = n_total - tem_nalpha;
        int nflip = static_cast<int>(nalpha) - tem_nalpha;

        DBG("fabs(Ms):" << std::fabs(Ms) << " Ms:" << Ms
            << " tem_nalpha:" << tem_nalpha
            << " tem_nbeta:"  << tem_nbeta
            << " nflip:"      << nflip);

        vector<vector<int>> flip_indices_list =
            generate_combinations(nbeta, nalpha, nflip);

        for (const auto& flips : flip_indices_list) {
	    //matrix_print_2d(flips.memptr(),flips.n_rows,flips.n_cols,"Zexuan Check flips:");
            MOpair block = ortho; 
            block.Ms = Ms;
            const mat& U0 = ortho.U;   // (n_ori_alpha, nalpha)
            const mat& V0 = ortho.V;   // (n_ori_beta,  nbeta)
            block.flips = flips;


            mat E_a = make_E_a(static_cast<int>(U0.n_cols),
                               static_cast<int>(V0.n_cols),
                               flips);
            mat E_b = make_E_b(static_cast<int>(U0.n_cols),
                               static_cast<int>(V0.n_cols),
                               flips);
            //matrix_print_2d(E_a.memptr(),E_a.n_rows, E_a.n_cols,"E_a:");
            //matrix_print_2d(E_b.memptr(),E_b.n_rows, E_b.n_cols,"E_b:");
            if (Ms >= -0.01){
                block.E_a = E_a;
                block.E_b = E_b;
                block.U = ortho.U;
                block.V = ortho.V;
            }
            else {
                block.E_a = E_b;
                block.E_b = E_a;
                block.U = ortho.V;
                block.V = ortho.U;
            }
            mat C_all = join_rows(ortho.C_alpha, ortho.C_beta);
            block.C_alpha = C_all * block.E_a;
            block.C_beta  = C_all * block.E_b;

            Ms_blocks[Ms].push_back(block);
        }
    }

}


int spin_adiabatic_state::state_phase(const int n_1, const int n_2, const vector<int>& Index){
    int k = Index.size();
    int sum_Index = accumulate(Index.begin(),Index.end(),0LL);
    int S = k*(n_1+n_2-1) - sum_Index - k*(k-1)/2;
    return (S % 2 == 0) ? +1: -1;

}

vec spin_adiabatic_state::vsoc_vector_modular() {
   /*
   Variable check reference

    */
   mat U1,V1;
   mat U2,V2;
   vec lambda1,lambda2;
   nvir1_a = NOrb - nalpha1;
   nvir1_b = NOrb - nbeta1;
   nvir2_a = NOrb - nalpha2;
   nvir2_b = NOrb - nbeta2;
   double cf = ConvFac(HARTREES_TO_WAVENUMBERS);
   S1 = 0.5 * (nalpha1 - nbeta1);
   S2 = 0.5 * (nalpha2 - nbeta2);
   n_total = nalpha1 + nbeta1;
   n_Ms1 = static_cast<int>(2 * S1) + 1;
   n_Ms2 = static_cast<int>(2 * S2) + 1;
   Ms_dimension = n_Ms1 *n_Ms2;
   vec vsoc = zeros<vec>(Ms_dimension);
   vsoc_pairs.resize(Ms_dimension);
   vsoc_values.resize(Ms_dimension);

   // split occupied&virtual orbitals
   mat C1_o_alpha = C1_alpha.head_cols(nalpha1);
   mat C1_o_beta = C1_beta.head_cols(nbeta1);
   mat C2_o_alpha = C2_alpha.head_cols(nalpha2);
   mat C2_o_beta = C2_beta.head_cols(nbeta2);
   mat C1_v_alpha = C1_alpha.tail_cols(nvir1_a);
   mat C1_v_beta = C1_beta.tail_cols(nvir1_b);
   mat C2_v_alpha = C2_alpha.tail_cols(nvir2_a);
   mat C2_v_beta = C2_beta.tail_cols(nvir2_b);

   // construct biorthonal alpha-beta orbital for state 1 and state 2
   mat S1_oo = C1_o_alpha.t()  * AOS * C1_o_beta;
   mat S2_oo = C2_o_alpha.t()  * AOS * C2_o_beta;
   //matrix_print_2d(S1_oo.memptr(),S1_oo.n_rows,S1_oo.n_cols,"S1_oo");
   //matrix_print_2d(S2_oo.memptr(),S2_oo.n_rows,S2_oo.n_cols,"S2_oo");
   
   svd(U1,lambda1,V1,S1_oo);
   svd(U2,lambda2,V2,S2_oo);
   vec lambda2_inv = lambda2;
   for (size_t i = 0; i < lambda2.n_elem; ++i) {
      lambda2_inv[i] = 1/lambda2[i];
   }

   mat tem_du = diagmat(lambda2_inv) * U2.t();
   mat Sooinv2 = V2 * tem_du;
   // SVD in qchem gives V in stead of Vt:
   // matrix_print_2d(Sooinv2.memptr(),Sooinv2.n_rows,Sooinv2.n_cols,"Sooinv2");
   mat S_check = S2_oo * Sooinv2;
   //matrix_print_2d(S_check.memptr(),S_check.n_rows,S_check.n_cols,"S_check");
   mat U_n0 = U2.tail_cols(1);
   mat V_n0 = V2.tail_cols(1);

   //matrix_print_2d(U_n0.memptr(),U_n0.n_rows,U_n0.n_cols,"U_n0");
   //matrix_print_2d(V_n0.memptr(),V_n0.n_rows,V_n0.n_cols,"V_n0");
   //matrix_print_2d(U2.memptr(),U2.n_rows,U2.n_cols,"U2");
   //matrix_print_2d(V2.memptr(),V2.n_rows,V2.n_cols,"V2");
   S1_orthonal.C_alpha = C1_o_alpha * U1;
   S1_orthonal.C_beta = C1_o_beta * V1;
   S1_orthonal.U = U1;
   S1_orthonal.V = V1;
   S1_orthonal.effect_C_o_alpha = C1_o_alpha;
   S1_orthonal.effect_C_o_beta = C1_o_beta;
   S1_orthonal.effect_C_v_alpha = C1_v_alpha;
   S1_orthonal.effect_C_v_beta = C1_v_beta;
   S1_orthonal.lambda = lambda1;

   S2_orthonal.C_alpha = C2_o_alpha * U2;
   S2_orthonal.C_beta = C2_o_beta * V2;
   S2_orthonal.effect_C_o_alpha = C2_o_alpha;
   S2_orthonal.effect_C_o_beta = C2_o_beta;
   S2_orthonal.effect_C_v_alpha = C2_v_alpha;
   S2_orthonal.effect_C_v_beta = C2_v_beta;
   S2_orthonal.U = U2;
   S2_orthonal.V = V2;
   S2_orthonal.lambda = lambda2;
   MatPrint(lambda1.memptr(), 1, lambda1.n_elem, "Zexuan Wei S1 lambda:");
   MatPrint(lambda2.memptr(), 1, lambda2.n_elem, "Zexuan Wei S2 lambda:");


   map<double, vector<MOpair>> MO_state1_by_Ms1;
   map<double, vector<MOpair>> MO_state2_by_Ms2;
   build_spin_blocks_for_state(S1_orthonal, S1, nalpha1, nbeta1, MO_state1_by_Ms1);
   build_spin_blocks_for_state(S2_orthonal, S2, nalpha2, nbeta2, MO_state2_by_Ms2);

   int vsoc_idx;

   for (auto it = MO_state1_by_Ms1.begin() ; it !=  MO_state1_by_Ms1.end(); ++it) {
      double Ms1 = it->first;
      const vector<MOpair>& blocks1 = it->second;        
      cout << "Zexuan Wei vsoc Ms1 :" << Ms1 << endl;
      for (int dir = 0; dir < 3; ++dir) {
         cout << "Zexuan Wei vsoc direction :" << dir << endl;
         double delta_Ms = (dir == 0) ? +1 :
         (dir == 1) ? -1 : 0;
         double Ms2 = Ms1 + delta_Ms;
         if (MO_state2_by_Ms2.count(Ms2) == 0) continue;

         const auto& blocks2 = MO_state2_by_Ms2[Ms2];

	      cout << "Zexuan Wei Ms1 block size :" << blocks1.size() << endl;
         cout << "Zexuan Wei Ms2 block size :" << blocks2.size() << endl;
         double tem_x = 0;
         double tem_y = 0;
         double tem_z = 0;
         for (const auto& b1 : blocks1) {
            for (const auto& b2 : blocks2) {
               mat S_alpha = b1.C_alpha.t() * AOS * b2.C_alpha;
               mat S_beta  = b1.C_beta.t()  * AOS * b2.C_beta;

               mat Ua, Va, Ub, Vb;
               vec la, lb;
               svd(Ua, la, Va, S_alpha);
               svd(Ub, lb, Vb, S_beta);
               MatPrint(la.memptr(), 1, la.n_elem, "Zexuan Wei vsoc la:");
               MatPrint(lb.memptr(), 1, lb.n_elem, "Zexuan Wei vsoc lb:");
               std::complex<double> val_complex;
               if (dir == 0) 
               {
                  OrbitalPair pair;
                  pair.Ms1 = Ms1;
                  pair.Ms2 = Ms2;
                  pair.direction = dir;
                  pair.block1 = b1;
                  pair.block2 = b2;
                  pair.C1_flipped_alpha = b1.C_alpha;
                  pair.C1_flipped_beta  = b1.C_beta;
                  pair.C2_flipped_alpha = b2.C_alpha;
                  pair.C2_flipped_beta  = b2.C_beta;
                  pair.U_a = Ua; pair.V_a = Va;
                  pair.U_b = Ub; pair.V_b = Vb;
                  pair.lambda_a = la; pair.lambda_b = lb;
                  pair.phase = prod(la) * prod(lb)* det(Ua) * det(Va)* det(Ub) * det(Vb)
                  / sqrt(blocks1.size() * blocks2.size());
                  pair.slater_phase1 = state_phase(nalpha1,nbeta1,pair.block1.flips);
                  pair.slater_phase2 = state_phase(nalpha2,nbeta2,pair.block2.flips);
                  pair.phase *= pair.slater_phase1 * pair.slater_phase2;
                  pair.psi1 = b1.C_beta * Ub.tail_cols(1);
                  pair.psi2 = b2.C_alpha * Va.tail_cols(1);

                  double val_x = pair.phase * dot(pair.psi1, L_AO.slice(0) * pair.psi2);
                  double val_y = pair.phase * dot(pair.psi1, L_AO.slice(1) * pair.psi2);
                  cout << "Zexuan Wei vsoc val_x:" << val_x << "val_x:" << val_y << endl;
                  pair.phase *=scaling_factor;
                  val_x *= scaling_factor;
                  val_y *= scaling_factor;
                  pair.vsoc_x = val_x;
                  pair.vsoc_y = val_y;
                  tem_x += val_x;
                  tem_y += val_y;
                  vsoc_idx = get_index(Ms1, Ms2);
                  pair.Ms_idx = vsoc_idx;
                  vsoc_pairs[vsoc_idx].push_back(pair);
               } 
               else if (dir == 1) 
               {
                  OrbitalPair pair;
                  pair.Ms1 = Ms1;
                  pair.Ms2 = Ms2;
                  pair.direction = dir;
                  pair.block1 = b1;
                  pair.block2 = b2;
                  pair.C1_flipped_alpha = b1.C_alpha;
                  pair.C1_flipped_beta  = b1.C_beta;
                  pair.C2_flipped_alpha = b2.C_alpha;
                  pair.C2_flipped_beta  = b2.C_beta;
                  pair.U_a = Ua; pair.V_a = Va;
                  pair.U_b = Ub; pair.V_b = Vb;
                  pair.lambda_a = la; pair.lambda_b = lb;
                  pair.phase = prod(la) * prod(lb)* det(Ua) * det(Va)* det(Ub) * det(Vb)
                  / sqrt(blocks1.size() * blocks2.size());
                  pair.slater_phase1 = state_phase(nalpha1,nbeta1,pair.block1.flips);
                  pair.slater_phase2 = state_phase(nalpha2,nbeta2,pair.block2.flips);
                  pair.phase *= pair.slater_phase1 * pair.slater_phase2;

                  pair.psi1 = b1.C_alpha * Ua.tail_cols(1);
                  pair.psi2 = b2.C_beta * Vb.tail_cols(1);
                  double val_x = pair.phase * dot(pair.psi1, L_AO.slice(0) * pair.psi2);
                  double val_y = pair.phase * dot(pair.psi1, L_AO.slice(1) * pair.psi2);
                  pair.phase *=scaling_factor;
                  // val_y should have minus sign due to L- = Lx-iLy. However, in gradient calculation, there's no minus sign for val_x and val_y
                  cout << "Zexuan Wei vsoc val_x:" << val_x << "val_y:" << -val_y << endl;
                  val_x *= scaling_factor;
                  val_y *= scaling_factor;
                  pair.vsoc_x = val_x;
                  pair.vsoc_y = val_y;
                  tem_x += val_x;
                  tem_y += val_y;

                  vsoc_idx = get_index(Ms1, Ms2);
                  pair.Ms_idx = vsoc_idx;
                  vsoc_pairs[vsoc_idx].push_back(pair);
               }
               else if (dir == 2) 
               {
                  OrbitalPair pair;
                  pair.Ms1 = Ms1;
                  pair.Ms2 = Ms2;
                  pair.direction = dir;
                  pair.block1 = b1;
                  pair.block2 = b2;
                  pair.C1_flipped_alpha = b1.C_alpha;
                  pair.C1_flipped_beta  = b1.C_beta;
                  pair.C2_flipped_alpha = b2.C_alpha;
                  pair.C2_flipped_beta  = b2.C_beta;
                  pair.U_a = Ua; pair.V_a = Va;
                  pair.U_b = Ub; pair.V_b = Vb;
                  pair.lambda_a = la; pair.lambda_b = lb;

	               cout << "Zexuan Wei vsoc phase:" <<  det(Ua) * det(Va)* det(Ub) * det(Vb)  << endl;

                  pair.phase_alpha = prod(la.head(la.n_elem-1)) * prod(lb)* det(Ua) * det(Va)* det(Ub) * det(Vb)
                     / sqrt(blocks1.size() * blocks2.size());
                  pair.phase_beta  = prod(lb.head(lb.n_elem-1)) * prod(la)* det(Ua) * det(Va)* det(Ub) * det(Vb)
                  / sqrt(blocks1.size() * blocks2.size());
                  pair.slater_phase1 = state_phase(nalpha1,nbeta1,pair.block1.flips);
                  pair.slater_phase2 = state_phase(nalpha2,nbeta2,pair.block2.flips);
                  pair.phase_alpha *= pair.slater_phase1 * pair.slater_phase2;
                   pair.phase_beta *= pair.slater_phase1 * pair.slater_phase2;

 
                  pair.psi1_alpha = b1.C_alpha * Ua.tail_cols(1);
                  pair.psi2_alpha = b2.C_alpha * Va.tail_cols(1);
                  pair.psi1_beta  = b1.C_beta  * Ub.tail_cols(1);
                  pair.psi2_beta  = b2.C_beta  * Vb.tail_cols(1);

                  cout << "Zexuan Wei vsoc zpure alpha:" << dot(pair.psi1_alpha, L_AO.slice(2) * pair.psi2_alpha)  << endl;
                  cout << "Zexuan Wei vsoc zpure beta:" << dot(pair.psi1_beta, L_AO.slice(2) * pair.psi2_beta)  << endl;
                  cout << "Zexuan Wei vsoc z phase alpha:" << pair.phase_alpha  << endl;
                  cout << "Zexuan Wei vsoc z phase beta:" << pair.phase_beta  << endl;

                  double val_a = pair.phase_alpha * dot(pair.psi1_alpha, L_AO.slice(2) * pair.psi2_alpha)  * 0.5;
                  double val_b = pair.phase_beta * dot(pair.psi1_beta, L_AO.slice(2) * pair.psi2_beta) * -0.5;
				  double val = val_a + val_b;
                  cout << "Zexuan Wei vsoc val:" << val << endl;
                  val *= scaling_factor;
                  vsoc_idx = get_index(Ms1, Ms2);
                  pair.Ms_idx = vsoc_idx;
                  pair.val_a = val_a * scaling_factor;
                  pair.val_b = val_b * scaling_factor;
                  pair.vsoc_z = val;
                  tem_z += val;
                  pair.phase_alpha *=scaling_factor;
                  pair.phase_beta *=scaling_factor;
                  vsoc_pairs[vsoc_idx].push_back(pair);
               }
            }
         }
         vsoc_idx = get_index(Ms1, Ms2);
         if (dir != 2){
            vsoc_values[vsoc_idx].vsoc_x += tem_x;
            vsoc_values[vsoc_idx].vsoc_y += tem_y;
            vsoc(vsoc_idx) += sqrt(tem_x * tem_x + tem_y * tem_y);
            cout << "current Vsoc of Ms1:" << Ms1 << "-> MS2:"<< Ms2 << "index:" << vsoc_idx <<"vsoc_x:" << cf * tem_x  << endl;
            cout << "current Vsoc of Ms1:" << Ms1 << "-> MS2:"<< Ms2 << "index:" << vsoc_idx <<"vsoc_y:" << cf * tem_y  << endl;
         }
         else if (dir == 2){
            vsoc_values[vsoc_idx].vsoc_z += tem_z;
            vsoc(vsoc_idx) += tem_z;
            cout << "current Vsoc of Ms1:" << Ms1 << "-> MS2:"<< Ms2 << "index:" << vsoc_idx <<"vsoc_z:" << cf * tem_z << endl;
         }
         
         cout << "current Vsoc of Ms1:" << Ms1 << "-> MS2:"<< Ms2 << "index:" << vsoc_idx <<"vsoc:" << cf * vsoc(vsoc_idx) << endl;


      }
   }


   //vsoc *= scaling_factor;
   cout << "[SOC] Final V_SOC vector: " << vsoc.t();
   return vsoc;
}



vec spin_adiabatic_state::vsoc_vector()
{
   const size_t nleft = nalpha2 - nbeta2;
   cout << "nleft: " << nleft << endl;
   int ndim;
   if (nleft == 2){ // triplet 
      ndim = 3;
      prod_s = vec(5);
   }
   else if (nleft == 3){ // quartet
      ndim = 5;
      prod_s = vec(10);
   }
   vec tv_soc = zeros<vec>(ndim); // final goal of this function

   int ys_debug = rem_read(REM_YS_DEBUG);

   // read overlap matrix
   if (ys_debug > 1){
      //matrix_print_2d(AOS.memptr(), NBas, NBas, "overlap matrix");
      //MatPrint(C1_alpha.memptr(), NBas, NOrb, "jC1");
      //MatPrint(C2_alpha.memptr(), NBas, NOrb, "jC2");
   }

   // two half-transformed overlap matrices with C1 orbitals
   mat CS_alpha = C1_alpha.head_cols(nalpha1).t() * AOS;
   mat CS_beta  = C1_beta.head_cols(nbeta1).t()   * AOS;

   // alpha and beta orbital overlap matrices between two states
   mat S_MO_alpha = CS_alpha * C2_alpha.head_cols(nalpha2);
   mat S_MO_beta  = CS_beta  * C2_beta.head_cols(nbeta2);

   // temporary variables
   mat Ua, Va, Ub, Vb;
   vec lambda_a, lambda_b;
   vec c1_p, c2_p;

   svd(Ua,lambda_a,Va,S_MO_alpha);
   svd(Ub,lambda_b,Vb,S_MO_beta);
   double phase = det(Ua)*det(Va)*det(Ub)*det(Vb);
   U.push_back(Ua); V.push_back(Va); lambda.push_back(lambda_a); // alpha 0
   U.push_back(Ub); V.push_back(Vb); lambda.push_back(lambda_b); // beta  1
   cout << "sa: " << lambda_a.t();
   cout << "sb: " << lambda_b.t();

   // the effective one-orbitals to be coupled by SO operator
   c1_p = C1_beta.head_cols(nbeta1) * Ub.tail_cols(1);
   c2_p = C2_alpha.head_cols(nalpha2) * Va.tail_cols(1);
   C1_prime.push_back(c1_p); C2_prime.push_back(c2_p); // 0 for Lx and Ly

   for (size_t i=0; i<2; ++i){ // only for Lx and Ly components
//   for (size_t i=0; i<3; ++i){ // only for Lx and Ly components
      tv_soc(i) = dot(c1_p, L_AO.slice(i) * c2_p);
   }
   prod_s(0) = prod(lambda_a) * prod(lambda_b) * phase;
   tv_soc *= prod_s(0);


   // Lz componet: same numbers of alpha and beta electrons for two states.
   // alpha-beta overlap matrix of the second state
   mat S_MO_t = C2_alpha.head_cols(nalpha2).t() * AOS * C2_beta.head_cols(nbeta2);

   svd(Ua,lambda_a,Vb,S_MO_t);
   U.push_back(Ua); V.push_back(Vb); lambda.push_back(lambda_a); // alpha and beta 2
   //if (ys_debug > 0){
      MatPrint(lambda_a.memptr(), 1, lambda_a.n_elem, " alpha beta s for z=0:");
   //}
      
   //MatPrint(S_MO_t.memptr(), S_MO_t.n_rows, S_MO_t.n_cols, "S_MO_t");
   vec sum_proj = zeros<vec>(nalpha2);
   for (size_t i=0; i<nalpha2; ++i){
      sum_proj(i) = dot(S_MO_t.row(i), S_MO_t.row(i));
   }
   MatPrint(sum_proj.memptr(), 1, sum_proj.n_elem, "sum_proj");
   indices = stable_sort_index(sum_proj);
   indices = indices.head(nleft);
   //cout << "indices: " << indices.t();
   printf("indices: ");
   for (size_t i=0; i<nleft; ++i) printf("%4d ", indices(i));
   printf("\n");

   if (unrestricted){
      // new occupied orbitals for Ms=0 triplet or Ms=1/2 quartet
      C2_alpha_new = C2_alpha.head_cols(nalpha2) * Ua; 
      C2_beta_new  = C2_beta.head_cols(nbeta2) * Vb;

      //MatPrint(C2_alpha.memptr(), C2_alpha.n_rows, nalpha2, "C2_alpha");
      //MatPrint(C2_beta.memptr(), C2_beta.n_rows, nbeta2, "C2_beta");
      //MatPrint(C2_alpha_new.memptr(), C2_alpha_new.n_rows, C2_alpha_new.n_cols, "C2_alpha_new");
      //MatPrint(C2_beta_new.memptr(), C2_beta_new.n_rows, C2_beta_new.n_cols, "C2_beta_new");
      mat S_MO_tt = C2_alpha_new.t() * AOS * C2_beta_new;
      //MatPrint(S_MO_tt.memptr(), S_MO_tt.n_rows, S_MO_tt.n_cols, "S_MO_tt");
   }
   else {
      // use original orbitals
      C2_alpha_new = C2_alpha.head_cols(nalpha2); 
      C2_beta_new  = C2_beta.head_cols(nbeta2);
   }


   for (size_t i=0; i<nleft; ++i){
//      size_t j = indices(i);
      size_t j = nbeta2 + i; // !!! TODO check do we need indices?
      mat C2_alpha_t = C2_alpha_new;
      C2_alpha_t.shed_col(j);
      mat C2_beta_t = join_rows(C2_beta_new, C2_alpha_new.col(j));

      S_MO_alpha = CS_alpha * C2_alpha_t;
      S_MO_beta = CS_beta * C2_beta_t;

      svd(Ua,lambda_a,Va,S_MO_alpha);
      svd(Ub,lambda_b,Vb,S_MO_beta);
      phase = det(Ua)*det(Va)*det(Ub)*det(Vb);
      cout << "Zheng phase factor:" << phase << endl;
      U.push_back(Ua); V.push_back(Va); lambda.push_back(lambda_a); // alpha 3, 5, 7
      U.push_back(Ub); V.push_back(Vb); lambda.push_back(lambda_b); // beta  4, 6, 8
      //if (ys_debug > 0){
         MatPrint(lambda_a.memptr(), 1, lambda_a.n_elem, "z sa");
         MatPrint(lambda_b.memptr(), 1, lambda_b.n_elem, "z sb");
      //}

      double soc_z;
      // alpha-spin SOC
      c1_p = C1_alpha.head_cols(nalpha1) * Ua.tail_cols(1);
      c2_p = C2_alpha_t * Va.tail_cols(1);
      C1_prime.push_back(c1_p); C2_prime.push_back(c2_p); // 1, 3, 5 for Lz
      soc_z = dot(c1_p, L_AO.slice(2) * c2_p);
      cout << "Zheng check socz pure:" << soc_z << endl;
      double d = prod(lambda_a.head(nalpha1-1)) * prod(lambda_b) * pow(-1, j) * phase;
      prod_s(1+i*2) = d;
      tv_soc(2) += soc_z * d;
      cout << "Zheng check socz:" << soc_z * d /(2 * sqrt(nleft)) << endl;
      //printf("soc_z: %15.8f\n", soc_z);

      // beta-spin SOC
      c1_p = C1_beta.head_cols(nbeta1) * Ub.tail_cols(1);
      c2_p = C2_beta_t * Vb.tail_cols(1);
      C1_prime.push_back(c1_p); C2_prime.push_back(c2_p); // 2, 4, 6 for Lz
      soc_z = dot(c1_p, L_AO.slice(2) * c2_p);
      cout << "Zheng check socz pure:" << soc_z << endl;

      d = -prod(lambda_a) * prod(lambda_b.head(nbeta1-1)) * pow(-1, j) * phase;
      prod_s(2+i*2) = d;
      tv_soc(2) += soc_z * d;
      cout << "Zheng check socz:" << soc_z/(2 * sqrt(nleft)) * d << endl;

      //printf("soc_z: %15.8f\n", soc_z);
   }
   tv_soc(2) /= (2 * sqrt(nleft)); // spin is 1/2; 1/sqrt(n) for n configurations
   //MatPrint(tv_soc.memptr(), 1, tv_soc.n_elem, "tv_soc");

   tv_soc *= pow(-1, nbeta2); // phase factor for the first three components

//tv_soc(2) = 0.0 /////////////

   // +-
   if (nleft == 3){
      for (int i=0; i<nleft; ++i){
         mat tmp = C2_alpha_new.tail_cols(nleft);
         mat C2_alpha_t = join_rows(C2_alpha_new.head_cols(nbeta2), tmp.col(i));
         tmp.shed_col(i);
         mat C2_beta_t = join_rows(C2_beta_new, tmp);

         S_MO_alpha = CS_alpha * C2_alpha_t;
         S_MO_beta = CS_beta * C2_beta_t;

         svd(Ua,lambda_a,Va,S_MO_alpha);
         svd(Ub,lambda_b,Vb,S_MO_beta);
         phase = det(Ua)*det(Va)*det(Ub)*det(Vb);
         U.push_back(Ua); V.push_back(Va); lambda.push_back(lambda_a); // alpha  9, 11, 13
         U.push_back(Ub); V.push_back(Vb); lambda.push_back(lambda_b); // beta  10, 12, 14
         if (ys_debug > 0){
            //MatPrint(lambda_a.memptr(), 1, lambda_a.n_elem, "- sa");
            //MatPrint(lambda_b.memptr(), 1, lambda_b.n_elem, "- sb");
         }

         vec soc_tmp(2);
         // the effective one-orbitals to be coupled by SO operator
         c1_p = C1_alpha.head_cols(nalpha1) * Ua.tail_cols(1);
         c2_p = C2_beta_t * Vb.tail_cols(1);
         C1_prime.push_back(c1_p); C2_prime.push_back(c2_p); // 7, 8, 9 for Lx and Ly
         for (int x=0; x<2; ++x){
            soc_tmp(x) = dot(c1_p, L_AO.slice(x) * c2_p);
         }
         double d = prod(lambda_a) * prod(lambda_b) * pow(-1, i) * phase;
         tv_soc.tail(2) += soc_tmp * d;
      }
      tv_soc.tail(2) /= sqrt(nleft); // 1/sqrt(n) for n configurations
   }


   /* 
   // test
   mat t_ovlp_all(nalpha1+nbeta1, nalpha2+nbeta2);
   t_ovlp_all.submat(0,0,nalpha1-1,nalpha2-1) = C1_alpha.head_cols(nalpha1).t() * AOS * C2_alpha.head_cols(nalpha2);
   t_ovlp_all.submat(nalpha1,0,nalpha1+nbeta1-1,nalpha2-1) = C1_beta.head_cols(nbeta1).t() * AOS * C2_alpha.head_cols(nalpha2);
   t_ovlp_all.submat(0,nalpha2,nalpha1-1,nalpha2+nbeta2-1) = C1_alpha.head_cols(nalpha1).t() * AOS * C2_beta.head_cols(nbeta2);
   t_ovlp_all.submat(nalpha1,nalpha2,nalpha1+nbeta1-1,nalpha2+nbeta2-1) = C1_beta.head_cols(nbeta1).t() * AOS * C2_beta.head_cols(nbeta2);

   mat t_U, t_V;
   vec t_lambda;
   svd(t_U,t_lambda,t_V,t_ovlp_all);
   MatPrint(t_lambda.memptr(), 1, t_lambda.n_elem, "t_lambda");
   vec t_U_last = t_U.cols(nalpha1-1);
   vec t_V_last = t_V.cols(nalpha1-1);
   vec t_C1_prime = C1_alpha.head_cols(nalpha1) * t_U_last.head(nalpha1);
   vec t_C2_prime = C2_alpha.head_cols(nalpha2) * t_V_last.head(nalpha2);
   double t_soc_z = dot(t_C1_prime, L_AO.slice(2) * t_C2_prime);
   cout << "t_soc_z: " << t_soc_z << endl;
   t_C1_prime = C1_beta.head_cols(nbeta1) * t_U_last.tail(nbeta1);
   t_C2_prime = C2_beta.head_cols(nbeta2) * t_V_last.tail(nbeta2);
   t_soc_z -= dot(t_C1_prime, L_AO.slice(2) * t_C2_prime);
   cout << "t_soc_z: " << t_soc_z << endl;
   t_soc_z *= prod(t_lambda.head(nalpha1));
   cout << "t_soc_z: " << t_soc_z << endl;
   // end test
   */

   tv_soc.head(2) *= sqrt(2);
   tv_soc *= scaling_factor ;
   prod_s *= scaling_factor ;

   return tv_soc;
}


void spin_adiabatic_state::run_two_states()
{
   unrestricted = rem_read(REM_JUSTAL);
   const size_t alpha = rem_read(REM_NALPHA);
   const size_t beta = rem_read(REM_NBETA);
   cout << "alpha: " << alpha << " beta: " << beta << " electrons"<< endl;
   
   if (alpha == beta || alpha == beta + 1){ // singlet or doublet
      nalpha1 = alpha;
      nalpha2 = alpha + 1;
      nbeta1  = beta;
      nbeta2  = beta - 1;
   }
   else if (alpha == beta + 2 || alpha == beta + 3){ // triplet or quartet
      nalpha1 = alpha - 1;
      nalpha2 = alpha;
      nbeta1  = beta + 1;
      nbeta2  = beta;
      
      // calculate singlet/doublet first
      rem_write(nalpha1, REM_NALPHA);
      rem_write(nbeta1, REM_NBETA);
   }

   // singlet state (Ms=0) or doublet state (Ms=+1/2)
   if (unrestricted){ // uks
      cout << "Zexuan Wei check scf alpha" << REM_NALPHA << "beta" << REM_NBETA << endl;
      myscf_1 = run_scf(E1, 1);
      myuhf_1 = dynamic_cast<uhf*>(myscf_1);
   }
   else { 
      if (nalpha1 == nbeta1){ // rks
         myscf_1 = run_scf(E1, 0);
         myrhf_1 = dynamic_cast<rhf*>(myscf_1);
      }
      else { // roks
         myscf_1 = run_scf(E1, 2);
         myrohf_1 = dynamic_cast<rohf*>(myscf_1);
      }
   }

   if (need_gradient > 0){
      grad_1  = state_gradients();
   }

   // triplet state (Ms=+1) or quartet state (Ms=+3/2)
   rem_write(nalpha2, REM_NALPHA);
   rem_write(nbeta2, REM_NBETA);
   if (unrestricted){ // uks
      cout << "Zexuan Wei check scf alpha" << REM_NALPHA << "beta" << REM_NBETA << endl;

      myscf_2 = run_scf(E2, 3);
      myuhf_2 = dynamic_cast<uhf*>(myscf_2);
   }
   else { // roks
      myscf_2 = run_scf(E2, 2);
      myrohf_2 = dynamic_cast<rohf*>(myscf_2);
   }

   if (need_gradient > 0){
      grad_2 = state_gradients();
   }

}

void spin_adiabatic_state::collect_scf_orbitals()
{
   if (unrestricted){
      C1_alpha = myuhf_1->Alpha->C;
      C1_beta  = myuhf_1->Beta->C;
      P1_alpha = myuhf_1->Alpha->P;
      P1_beta  = myuhf_1->Beta->P;
      F1_alpha = myuhf_1->Alpha->Fock;
      F1_beta  = myuhf_1->Beta->Fock;

      C2_alpha = myuhf_2->Alpha->C;
      C2_beta  = myuhf_2->Beta->C;
      P2_alpha = myuhf_2->Alpha->P;
      P2_beta  = myuhf_2->Beta->P;
      F2_alpha = myuhf_2->Alpha->Fock;
      F2_beta  = myuhf_2->Beta->Fock;

      AOS      = myuhf_2->Alpha->AOS;
   }
   else {
      if (nalpha1 == nbeta1){
         C1_alpha = myrhf_1->C;
         C1_beta  = myrhf_1->C;
         P1_alpha = myrhf_1->P;
         P1_beta  = myrhf_1->P;
         F1_alpha = myrhf_1->Fock;
         F1_beta  = myrhf_1->Fock;
      }
      else {
         C1_alpha = myrohf_1->C;
         C1_beta  = myrohf_1->C;
         P1_alpha = myrohf_1->PA;
         P1_beta  = myrohf_1->PB;
         F1_alpha = myrohf_1->FockA;
         F1_beta  = myrohf_1->FockB;
      }

      C2_alpha = myrohf_2->C;
      C2_beta  = myrohf_2->C;
      P2_alpha = myrohf_2->PA;
      P2_beta  = myrohf_2->PB;
      F2_alpha = myrohf_2->FockA;
      F2_beta  = myrohf_2->FockB;

      AOS      = myrohf_2->AOS;
   }
}


void spin_adiabatic_state::spin_orbit_coupling()
{
   cout << "In spin_orbit_coupling MICHARGE: " << rem_read(REM_MICHARGE) << endl;

   run_two_states();
   collect_scf_orbitals();

   QTimer TimeVsoc;
   TimeVsoc.On();
   string vsoc_type = "zexuan";
   lsoc_xyz();
   double cf = ConvFac(HARTREES_TO_WAVENUMBERS);
   vec vsoc_complex;   
   if (rem_read(2096) == 2 or rem_read(2096) == 3){
      v_soc = vsoc_vector_modular();
      current_Vsoc = norm(v_soc, 2);
      vsoc_complex = cf * v_soc;
   }
   else if (rem_read(2096) == 1){
      v_soc = vsoc_vector();
      current_Vsoc = norm(v_soc, 2);
      vsoc_complex = cf * v_soc;
   }
   current_Vsoc /= v_soc.size();
   TimeVsoc.Off();
   TimeVsoc.Print("Time spent on soc energy");

   if (rem_read(2096) == 3){
   E_soc = sqrt( (E1-E2) * (E1-E2) + 4.0*current_Vsoc*current_Vsoc);
  E_adiab = 0.5 * (E1 + E2 - E_soc);
  lagrangian = 0;
  if (rem_read(2095)>0) lagrangian = rem_read(2095);
  cout << "energy restrain lagrangian" << lagrangian << endl;
  double E_r = 0.5 * lagrangian * (E1-E2) * (E1-E2);
  E_adiab += E_r;

  FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_CRNT_TOTAL_ENERGY,FM_BEG,&current_Vsoc);
  FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_SCF_ENERGY,FM_BEG,&current_Vsoc);
  printf("E1: %20.15f  E2: %20.15f E_r: %20.15f  E_soc: %18.15f E_adiab: %20.15f\n", E1, E2, E_r, E_soc, E_adiab);
   }
   
   if (rem_read(2096) == 2){


   for (double Ms1 = -S1; Ms1 <= S1; Ms1 += 1.0) {
      for (int dir = 0; dir < 3; ++dir) {
         double delta_Ms = (dir == 0) ? +1 :
                           (dir == 1) ? -1 : 0;
         double Ms2 = Ms1 + delta_Ms;

         int idx = get_index(Ms1, Ms2);
         if (dir == 0){
         cout << "Ms1 = " << Ms1 << ", Ms2 = " << Ms2 
               << ", direction = " << dir << ", vsoc_x =   " << cf * vsoc_values[idx].vsoc_x << ", vsoc_y=   "<< cf * vsoc_values[idx].vsoc_y << "i"  << endl;
         }
         else if (dir == 1){
         cout << "Ms1 = " << Ms1 << ", Ms2 = " << Ms2 
               << ", direction = " << dir << ", vsoc_x =   " << cf * vsoc_values[idx].vsoc_x << ", vsoc_y=   "<< -cf * vsoc_values[idx].vsoc_y << "i" << endl;
         }
         else if (dir == 2){
         cout << "Ms1 = " << Ms1 << ", Ms2 = " << Ms2 
               << ", direction = " << dir << ", vsoc_z =   " << cf * vsoc_values[idx].vsoc_z << endl;
         }

         }
      }
   

      E_soc = sqrt( (E1-E2) * (E1-E2) + 4.0*current_Vsoc*current_Vsoc);
      E_adiab = 0.5 * (E1 + E2 - E_soc);
      lagrangian = 0;
      if (rem_read(2095)>0) lagrangian = rem_read(2095);
      cout << "energy restrain lagrangian" << lagrangian << endl;
      double E_r = 0.5 * lagrangian * (E1-E2) * (E1-E2);
      E_adiab += E_r;

      FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_CRNT_TOTAL_ENERGY,FM_BEG,&E_adiab);
      FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_SCF_ENERGY,FM_BEG,&E_adiab);
      printf("E1: %20.15f  E2: %20.15f E_r: %20.15f  E_soc: %18.15f E_adiab: %20.15f\n", E1, E2, E_r, E_soc, E_adiab);
   }
   else {

      printf("--------\n");
      printf("current_Vsoc: %20.12e or %.3f cm^-1\n", current_Vsoc, cf*current_Vsoc);
      printf("V^soc = %.3f %.3f %.3f", vsoc_complex(0), vsoc_complex(1), vsoc_complex(2));
      if (v_soc.n_elem == 3) printf(" cm^-1\n");
      else if (v_soc.n_elem == 5) printf(" %.3f %.3f cm^-1\n", vsoc_complex(3), vsoc_complex(4));

      printf("ms=0,  0: %12.3f %+13.3fi cm^-1\n", 0.0, vsoc_complex(2));
      printf("ms=0, +1: %12.3f %+13.3fi cm^-1\n", vsoc_complex(1)/2, vsoc_complex(0)/2);
      printf("ms=0, -1: %12.3f %+13.3fi cm^-1\n", vsoc_complex(1)/2, vsoc_complex(0)/2);

      E_soc = sqrt( (E1-E2) * (E1-E2) + 4.0*current_Vsoc*current_Vsoc);
      E_adiab = 0.5 * (E1 + E2 - E_soc);
      lagrangian = 0;
      if (rem_read(2095)>0) lagrangian = rem_read(2095);
      cout << "energy restrain lagrangian" << lagrangian << endl;
      double E_r = 0.5 * lagrangian * (E1-E2) * (E1-E2);
      E_adiab += E_r;

      FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_CRNT_TOTAL_ENERGY,FM_BEG,&E_adiab);
      FileMan(FM_WRITE,FILE_ENERGY,FM_DP,1,FILE_POS_SCF_ENERGY,FM_BEG,&E_adiab);
      printf("E1: %20.15f  E2: %20.15f E_r: %20.15f  E_soc: %18.15f E_adiab: %20.15f\n", E1, E2, E_r, E_soc, E_adiab);
   }
}


void spin_adiabatic_state::total_gradients()
{
   bool restrain = true;
   // get virtual orbital numbers
   nvir1_a = NOrb - nalpha1;
   nvir1_b = NOrb - nbeta1;
   nvir2_a = NOrb - nalpha2;
   nvir2_b = NOrb - nbeta2;

   // pack terms up
   cout << "gradient restrain lagrangian:" << lagrangian << endl;
   double prefactor_1 = 0.5 * (1 - (E1-E2) / E_soc) + lagrangian * (E1-E2);
   double prefactor_2 = 0.5 * (1 + (E1-E2) / E_soc) - lagrangian * (E1-E2);
   double prefactor_v = - 2.0 / E_soc;
   printf("prefactors: %15.12f %15.12f %17.12f\n", prefactor_1, prefactor_2, prefactor_v);

   grad_1 *= prefactor_1;

   grad_2 *= prefactor_2;

   vec grad_soc;
   if (rem_read(2096)== 1){
      grad_soc = Esoc_gradients();
   }
   else if (rem_read(2096) == 2) {
      grad_soc = E_adiab_gradients();
   }
	grad_soc /= v_soc.size();
   grad_soc *= prefactor_v;

   matrix_print_2d(grad_soc.memptr(), 3, NAtoms, "zheng ASG soc");
   matrix_print_2d(grad_1.memptr(), 3, NAtoms, "zheng ASG first state");
   matrix_print_2d(grad_2.memptr(), 3, NAtoms, "zheng ASG second state");
   vec grad_total = grad_1 + grad_2 + grad_soc;

   if (rem_read(REM_N_GEOM_REST)>0){ // geometric restraint
      cout << "factoring restrain gradient" << endl;
      vec grad_rest = zeros<vec>(Nuclear);
      FileMan(FM_READ,FILE_NUCLEAR_GRADIENT,FM_DP,Nuclear,Nuclear,FM_BEG,grad_rest.memptr());
      grad_total += grad_rest * (1.0 - prefactor_1 - prefactor_2);
   }



   matrix_print_2d(grad_total.memptr(), 3, NAtoms, "zheng adiabatic surface gradient");
   FileMan(FM_WRITE,FILE_NUCLEAR_GRADIENT,FM_DP,Nuclear,0,FM_BEG,grad_total.memptr());
   writegrad(&NAtoms,&E_adiab,grad_total.memptr());
}

vec spin_adiabatic_state::gradient_explicit_Ms()
{
   vec derivatives = zeros<vec>(Nuclear);
   S_inv = C1_alpha  * C1_alpha.t();
   // AO overlap derivatives
   int size_jSxv = NB2car*Nuclear*2;
   vector<double> jOrigin(3,0.0);
   vector<double> jSxv(size_jSxv,0.0);
   vector<double> jHx(size_jSxv,0.0);

   MakeOv(NULL,1,-1,0,0,jOrigin.data(),jSxv.data(),jHx.data()); // need the jHx space!

   // AO soc derivatives in 3N, 3N, 3N order
   int size_jdL = Nuclear*NBas6D2;
   vector<double> jdL(3*size_jdL,0.0);
   getdL(jdL.data(), size_jdL);


   for (double Ms1 = -S1; Ms1 <= S1; Ms1 += 1.0) {
      for (int dir = 0; dir < 3; ++dir) {
         double delta_Ms = (dir == 0) ? +1 :
                           (dir == 1) ? -1 : 0;
         double Ms2 = Ms1 + delta_Ms;

         int idx = get_index(Ms1, Ms2);

         auto& pair_list = vsoc_pairs[idx];
         cout << "Processing Ms1 = " << Ms1 << ", Ms2 = " << Ms2 
               << ", direction = " << dir << ", num pairs = " << pair_list.size() << endl;

         for (auto& pair : pair_list) {
            pair.explicit_derivatives = zeros<vec>(Nuclear);
            if (dir != 2){ //xy
			  //components: 1. (C1U1U0)^T dLC2V2V0D ,deriv of L, need to claim that L,S are all AO basis here
              //2.- 0.5 (C1U1U0)^T dS^T(CC^T)  L                           C2V2V0D explicit part in deriv of C1
              //3.- 0.5 (C1U1U0)^T             L    CC^TdS                 C2V2V0D explicit part in deriv of C2

			  //
               mat tem_psi2_phase = pair.psi2 * pair.phase; // C2V2V0D
			   mat tem_psi1_T = pair.psi1.t();              // (C1U1U0)^T

               mat dL_AO,tem_psi1T_dL_psi2_phase;
               tem_psi1T_dL_psi2_phase = zeros<vec>(Nuclear);
               for (size_t x=0; x<Nuclear; ++x){
                  dL_AO = zeros<mat>(NBas, NBas);
                  mat tem_mat_x(jdL.data()+NBas2*(0*Nuclear+x),NBas,NBas,true);
                  mat tem_mat_y(jdL.data()+NBas2*(1*Nuclear+x),NBas,NBas,true);
                  dL_AO += tem_mat_x * vsoc_values[pair.Ms_idx].vsoc_x;
                  dL_AO += tem_mat_y * vsoc_values[pair.Ms_idx].vsoc_y; //vsoc_y should be positive as vsoc_y in Vsoc is square
                  mat tem_dL_psi2_phase = dL_AO * tem_psi2_phase;
                  tem_psi1T_dL_psi2_phase(x) += as_scalar(pair.psi1.t() * tem_dL_psi2_phase);
               }

               pair.explicit_derivatives += tem_psi1T_dL_psi2_phase; // component 1
				//matrix_print_2d(pair.explicit_derivatives.memptr(), 3, NAtoms, "Zexuan Wei soc gradient_explicit_of pair component 1");



              //2.- 0.5 (C1U1U0)^T dS^T(CC^T)  L                           C2V2V0D explicit part in deriv of C1
              //3.- 0.5 (C1U1U0)^T             L    CC^TdS                 C2V2V0D explicit part in deriv of C2

               mat L_AO_Vsoc_xy; // x,y are equal in equation.
               L_AO_Vsoc_xy = zeros<mat>(NBas, NBas);

               L_AO_Vsoc_xy += L_AO.slice(0) * vsoc_values[pair.Ms_idx].vsoc_x;
               L_AO_Vsoc_xy += L_AO.slice(1) * vsoc_values[pair.Ms_idx].vsoc_y;
               // x and y components


               mat tem_L_psi2_phase = L_AO_Vsoc_xy * tem_psi2_phase;
               mat tem_SinvT_L_psi2_phase = S_inv.t() * tem_L_psi2_phase; 
               mat tem_psi1T_L = pair.psi1.t() * L_AO_Vsoc_xy;
               mat tem_psi1T_L_CCT = tem_psi1T_L * S_inv;


               mat dS_AO(NBas, NBas),tem_psi1T_dST_SinvT_L_psi2_phase;
               tem_psi1T_dST_SinvT_L_psi2_phase = zeros<vec>(Nuclear);
               for (size_t x=0; x<Nuclear; ++x){
                  ScaV2M(dS_AO.memptr(), jSxv.data()+x*NB2car, 1, 1);
                  mat tem_dST_SinvT_L_psi2_phase = dS_AO.t() * tem_SinvT_L_psi2_phase * 0.5;
                  tem_psi1T_dST_SinvT_L_psi2_phase(x) += as_scalar(pair.psi1.t()* tem_dST_SinvT_L_psi2_phase);
               }

               pair.explicit_derivatives -= tem_psi1T_dST_SinvT_L_psi2_phase;
				//matrix_print_2d(pair.explicit_derivatives.memptr(), 3, NAtoms, "Zexuan Wei soc gradient_explicit_of pair component 2");

               //state2 deriv
               mat tem_psi1T_L_Sinv_dS_psi2_phase;
               tem_psi1T_L_Sinv_dS_psi2_phase = zeros<vec>(Nuclear);
               for (size_t x=0; x<Nuclear; ++x){
                  ScaV2M(dS_AO.memptr(), jSxv.data()+x*NB2car, 1, 1);
                  mat tem_dS_psi2_phase = dS_AO * tem_psi2_phase *0.5;
                  tem_psi1T_L_Sinv_dS_psi2_phase(x) += as_scalar(tem_psi1T_L_CCT * tem_dS_psi2_phase);
               }
               pair.explicit_derivatives -= tem_psi1T_L_Sinv_dS_psi2_phase;
				//matrix_print_2d(pair.explicit_derivatives.memptr(), 3, NAtoms, "Zexuan Wei soc gradient_explicit_of pair component 3");

            }
            else{ //z
			  //components: 1. (C1U1U0)^T dLC2V2V0D ,deriv of L, need to claim that L,S are all AO basis here
              //2.- 0.5 (C1U1U0)^T dS^T(CC^T)  L                           C2V2V0D explicit part in deriv of C1
              //3.- 0.5 (C1U1U0)^T             L    CC^TdS                 C2V2V0D explicit part in deriv of C2
               // dL part
               mat tem_psi2_alpha_phase,tem_psi2_beta_phase;
               tem_psi2_alpha_phase =  pair.psi2_alpha  * pair.phase_alpha  * 0.5;
               tem_psi2_beta_phase  = -pair.psi2_beta   * pair.phase_beta   * 0.5;

               mat dL_AO,tem_psi1_dL_psi2_alpha_phase,tem_psi1_dL_psi2_beta_phase;
               tem_psi1_dL_psi2_alpha_phase = zeros<vec>(Nuclear);
               tem_psi1_dL_psi2_beta_phase = zeros<vec>(Nuclear);
               for (size_t x=0; x<Nuclear; ++x){
                  dL_AO = zeros<mat>(NBas, NBas);
                  mat tem_mat_z(jdL.data()+NBas2*(2*Nuclear+x),NBas,NBas,true);
                  dL_AO += tem_mat_z * vsoc_values[pair.Ms_idx].vsoc_z;
                  mat tem_dL_psi2_alpha_phase = dL_AO * tem_psi2_alpha_phase;
                  mat tem_dL_psi2_beta_phase = dL_AO * tem_psi2_beta_phase;
                  tem_psi1_dL_psi2_alpha_phase(x) += as_scalar(pair.psi1_alpha.t() * tem_dL_psi2_alpha_phase);
                  tem_psi1_dL_psi2_beta_phase(x)  += as_scalar(pair.psi1_beta.t()  * tem_dL_psi2_beta_phase);
               }

               pair.explicit_derivatives += tem_psi1_dL_psi2_alpha_phase; // caution, 0.5 and minus sign has been added in tem_psi2_beta_phase
               pair.explicit_derivatives += tem_psi1_dL_psi2_beta_phase;

               // dS part

               mat L_AO_Vsoc_z = L_AO.slice(2) * vsoc_values[pair.Ms_idx].vsoc_z;
               mat tem_L_psi2_alpha_phase = L_AO_Vsoc_z * tem_psi2_alpha_phase;
               mat tem_L_psi2_beta_phase  = L_AO_Vsoc_z * tem_psi2_beta_phase;


               mat tem_SinvT_L_psi2_alpha_phase = S_inv.t() * tem_L_psi2_alpha_phase; // minus sign
               mat tem_SinvT_L_psi2_beta_phase = S_inv.t() * tem_L_psi2_beta_phase;

               mat dS_AO(NBas, NBas),tem_psi1_dST_SinvT_L_psi2_alpha_phase,tem_psi1_dST_SinvT_L_psi2_beta_phase;
               tem_psi1_dST_SinvT_L_psi2_alpha_phase = zeros<vec>(Nuclear);
               tem_psi1_dST_SinvT_L_psi2_beta_phase = zeros<vec>(Nuclear);
               for (size_t x=0; x<Nuclear; ++x){
                  ScaV2M(dS_AO.memptr(), jSxv.data()+x*NB2car, 1, 1);
                  mat tem_dST_SinvT_L_psi2_alpha_phase = dS_AO.t() * tem_SinvT_L_psi2_alpha_phase * 0.5;
                  mat tem_dST_SinvT_L_psi2_beta_phase  = dS_AO.t() * tem_SinvT_L_psi2_beta_phase  * 0.5;
                  tem_psi1_dST_SinvT_L_psi2_alpha_phase(x) += as_scalar(pair.psi1_alpha.t() *  tem_dST_SinvT_L_psi2_alpha_phase);
                  tem_psi1_dST_SinvT_L_psi2_beta_phase(x) += as_scalar(pair.psi1_beta.t() *  tem_dST_SinvT_L_psi2_beta_phase);
               }

               pair.explicit_derivatives -= tem_psi1_dST_SinvT_L_psi2_alpha_phase;
               pair.explicit_derivatives -= tem_psi1_dST_SinvT_L_psi2_beta_phase;



               //state2 deriv
               mat tem_psi1T_alpha_L = pair.psi1_alpha.t() * L_AO_Vsoc_z;
               mat tem_psi1T_beta_L  = pair.psi1_beta.t()  * L_AO_Vsoc_z;

               mat tem_psi1T_alpha_L_CCT = tem_psi1T_alpha_L * S_inv;
               mat tem_psi1T_beta_L_CCT  = tem_psi1T_beta_L  * S_inv;

               mat tem_psi1_L_Sinv_dS_psi2_alpha_phase,tem_psi1_L_Sinv_dS_psi2_beta_phase;
               tem_psi1_L_Sinv_dS_psi2_alpha_phase = zeros<vec>(Nuclear);
               tem_psi1_L_Sinv_dS_psi2_beta_phase  = zeros<vec>(Nuclear);
               for (size_t x=0; x<Nuclear; ++x){
                  ScaV2M(dS_AO.memptr(), jSxv.data()+x*NB2car, 1, 1);
                  mat tem_dS_psi2_alpha_phase = dS_AO * tem_psi2_alpha_phase * 0.5;
                  mat tem_dS_psi2_beta_phase  = dS_AO * tem_psi2_beta_phase * 0.5;
                  tem_psi1_L_Sinv_dS_psi2_alpha_phase(x) += as_scalar(tem_psi1T_alpha_L_CCT * tem_dS_psi2_alpha_phase);
                  tem_psi1_L_Sinv_dS_psi2_beta_phase(x)  += as_scalar(tem_psi1T_beta_L_CCT  * tem_dS_psi2_beta_phase);
               }



               pair.explicit_derivatives -= tem_psi1_L_Sinv_dS_psi2_alpha_phase;
               pair.explicit_derivatives -= tem_psi1_L_Sinv_dS_psi2_beta_phase;


            }
            matrix_print_2d(pair.explicit_derivatives.memptr(), 3, NAtoms, "Zexuan Wei soc gradient_explicit_of pair");
            derivatives += pair.explicit_derivatives;
         }
      }
   }
   cout << "Zexuan Wei gradient_explicit_Ms end" << endl;
   return derivatives;
}


mat sigma_u(MOpair& block) {
    mat sigma(block.n_occ_a * block.n_occ_a, block.n_occ_a * block.n_occ_b, fill::zeros);
    DBG("block.n_occ_a = " << block.n_occ_a << " block.n_occ_b = " << block.n_occ_b);
    mat Ainv, U_null, UnUnT, Vdinv;
    if (block.n_occ_a > block.n_svd){
        mat U_core = block.U.cols(0, block.n_svd - 1);
        mat V_core = block.V.cols(0, block.n_svd - 1);
        Ainv = V_core * diagmat(1.0/block.lambda) * U_core.t();
        mat lambda_inv = 1.0 / block.lambda;
        U_null = block.U.cols(block.n_svd, block.n_occ_a - 1);
        DBG("U_null size = " << U_null.n_rows << " x " << U_null.n_cols);
        UnUnT = U_null * U_null.t();
        Vdinv = block.V * diagmat(lambda_inv);

    }

    for (size_t i = 0; i < block.n_occ_a; ++i){
        for (size_t ip = 0; ip < block.n_occ_a; ++ip){
            for (size_t jp = 0; jp < block.n_occ_b; ++jp){
                for (size_t l = 0; l < block.n_occ_a; ++l){
                    if (l < block.n_svd){
                        for (size_t lp = 0; lp < block.n_svd; ++lp){
                            if (lp == l){
                                continue;
                            }
                            if (abs(block.lambda(l) - block.lambda(l)) < 1e-7){
                                continue;
                            }
                            double dl2 = (block.lambda(l) * block.lambda(l));
                            double dlp2 = (block.lambda(lp) * block.lambda(lp));
                            double denom = dl2 - dlp2;



                            sigma(i * block.n_occ_a +l, ip + jp *block.n_occ_a) += block.U(i,lp) *(block.U(ip,lp) * block.V(jp,l) * block.lambda(l) + block.U(ip,l) * block.V(jp,lp) * block.lambda(lp) )/denom;
                        }
                        if (block.n_occ_a > block.n_svd){
                            sigma(i * block.n_occ_a +l, ip + jp *block.n_occ_a) += UnUnT(i,ip) * Vdinv(jp,l);
                        }
                    }
                    else{
                        sigma(i * block.n_occ_a +l, ip + jp *block.n_occ_a) -= block.U(ip,l) * Ainv(jp,i);
                    }

                }
            }
        }
    }

    DBG("sigma_u size=" << sigma.n_rows << sigma.n_cols);
    DBG("sigma_u max abs=" <<  abs(sigma).max());
    return sigma;
}

mat sigma_v(MOpair& block) {
    mat sigma(block.n_occ_b * block.n_occ_b,
              block.n_occ_a * block.n_occ_b,
              fill::zeros);

    DBG("block.n_occ_a = " << block.n_occ_a
        << " block.n_occ_b = " << block.n_occ_b);

    bool has_null = (block.n_occ_b > block.n_svd);
    mat Ainv, V_null, VnVnT, Udinv;

    if (has_null) {
        vec lambda_inv = 1.0 / block.lambda;
        mat U_core = block.U.cols(0, block.n_svd - 1);
        mat V_core = block.V.cols(0, block.n_svd - 1);
        Ainv = V_core * diagmat(1.0/block.lambda) * U_core.t();
        V_null = block.V.cols(block.n_svd, block.n_occ_b - 1);
        VnVnT = V_null * V_null.t();
        Udinv = block.U * diagmat(lambda_inv);
    }

    for (size_t j = 0; j < block.n_occ_b; ++j) {
        for (size_t ip = 0; ip < block.n_occ_a; ++ip) {
            for (size_t jp = 0; jp < block.n_occ_b; ++jp) {
                for (size_t l = 0; l < block.n_occ_b; ++l) {

                    if (l < block.n_svd) {

                        for (size_t lp = 0; lp < block.n_svd; ++lp) {
                            if (lp == l) continue;
                            if (block.lambda(l) == block.lambda(l)){
                                continue;
                            }
                            double numer =
                                block.V(j, lp) *
                                (block.lambda(lp) * block.U(ip, lp) * block.V(jp, l)
                               + block.lambda(l)  * block.V(jp, lp) * block.U(ip, l));

                            double denom =
                                block.lambda(l) * block.lambda(l)
                              - block.lambda(lp) * block.lambda(lp);
                            if (std::abs(denom) < 1e-8){
                                denom = (denom >= 0 ? +1 : -1) * 1e-8;
                            }
                            sigma(j * block.n_occ_b + l,
                                  ip + jp * block.n_occ_a) += numer / denom;
                        }

                        // null correction
                        if (has_null) {
                            sigma(j * block.n_occ_b + l,
                                  ip + jp * block.n_occ_a) +=
                                VnVnT(j, jp) * Udinv(ip, l);
                        }
                    }

                    // -------- null block: l >= n_svd --------
                    else {
                        sigma(j * block.n_occ_b + l,
                              ip + jp * block.n_occ_a) -=
                            block.V(jp, l) * Ainv(ip, j);
                    }

                }
            }
        }
    }
    DBG("sigma_v size=" << sigma.n_rows << sigma.n_cols);
    DBG("sigma_v max abs=" << abs(sigma).max());
    return sigma;
}








void spin_adiabatic_state::sigma_overlap(MOpair& block) {
    DBG("===== sigma_overlap START =====");
   /*
   * size of sigma_aa: (n_ao(mu) * r , n_vir_a(a) * n_occ_a(i))
   *sigma_ab: (n_ao(mu) * r , n_vir_b(b) * n_occ_b(j))
   *sigma_ba: (n_ao(mu) * r , n_vir_a(a) * n_occ_a(i))
   *sigma_bb: (n_ao(mu) * r , n_vir_b(b) * n_occ_b(j))
    */
    block.n_svd = block.lambda.n_elem;
    block.n_ao = block.effect_C_o_alpha.n_rows;
    block.n_occ_a = block.effect_C_o_alpha.n_cols;
    block.n_occ_b = block.effect_C_o_beta.n_cols;
    block.n_vir_b = block.n_ao - block.n_occ_b;
    block.n_vir_a = block.n_ao - block.n_occ_a;
   mat sigma_aa(block.n_ao * block.n_occ_a, block.n_vir_a * block.n_occ_a, fill::zeros);
   mat sigma_ab(block.n_ao * block.n_occ_a, block.n_vir_b * block.n_occ_b, fill::zeros);
   mat sigma_ba(block.n_ao * block.n_occ_b, block.n_vir_a * block.n_occ_a, fill::zeros);
   mat sigma_bb(block.n_ao * block.n_occ_b, block.n_vir_b * block.n_occ_b, fill::zeros);
   mat s_vo_ab = block.effect_C_v_alpha.t() * AOS * block.effect_C_o_beta;
   mat s_ov_ab = block.effect_C_o_alpha.t() * AOS * block.effect_C_v_beta;
   block.sigma_u = sigma_u(block);
   block.sigma_v = sigma_v(block);

       //最怀念einsum的一集
    for (size_t mu = 0; mu < block.n_ao; ++mu) {
        // ---------- αα ----------
        for (size_t l = 0; l < block.n_occ_a; ++l) {
            for (size_t a = 0; a < block.n_vir_a; ++a){
                for (size_t i = 0; i < block.n_occ_a; ++i) {
                    double val = block.effect_C_v_alpha(mu, a) * block.U(i, l);
                    for (size_t ip = 0; ip < block.n_occ_a; ++ip) {
                        double Coi = block.effect_C_o_alpha(mu, ip);
                        if (Coi == 0.0) continue;
                        for (size_t j = 0; j < block.n_occ_b; ++j) {
                            double su = block.sigma_u(ip * block.n_occ_a + l,i + j * block.n_occ_a);
                            if (su == 0.0) continue;
                            val += Coi * su * s_vo_ab(a, j);
                        }
                    }
                    sigma_aa(mu * block.n_occ_a + l , a + i * block.n_vir_a) = val;
                }
            }
        }
        // ---------- αβ ----------
        for (size_t l = 0; l < block.n_occ_a; ++l) {
            for (size_t b = 0; b < block.n_vir_b; ++b){
                for (size_t j = 0; j < block.n_occ_b; ++j) {
                    double val = 0.0;
                    for (size_t i = 0; i < block.n_occ_a; ++i) {
                        double Coi = block.effect_C_o_alpha(mu, i);
                        if (Coi == 0.0) continue;
                        for (size_t ip = 0; ip < block.n_occ_a; ++ip) {
                            double su = block.sigma_u(i * block.n_occ_a + l,ip + j * block.n_occ_a);
                            if (su == 0.0) continue;
                            val += Coi * su * s_ov_ab(ip, b);
                        }
                    }
                    sigma_ab(mu * block.n_occ_a + l, b + j * block.n_vir_b) = val;
                }
            }
        }
        // ---------- βα ----------
        for (size_t l = 0; l < block.n_occ_b; ++l) {
            for (size_t a = 0; a < block.n_vir_a; ++a){
                for (size_t i = 0; i < block.n_occ_a; ++i) {
                    double val = 0.0;
                    for (size_t j = 0; j < block.n_occ_b; ++j) {
                        double Coj = block.effect_C_o_beta(mu, j);
                        if (Coj == 0.0) continue;
                        for (size_t jp = 0; jp < block.n_occ_b; ++jp) {
                            double sv = block.sigma_v(j * block.n_occ_b + l,i + jp * block.n_occ_a);
                            if (sv == 0.0) continue;
                            val += Coj * sv * s_vo_ab(a, jp);
                        }
                    }
                    sigma_ba(mu * block.n_occ_b + l, a + i * block.n_vir_a) = val;
                }
            }
        }
        // ---------- ββ ----------
        for (size_t l = 0; l < block.n_occ_b; ++l) {
            for (size_t b = 0; b < block.n_vir_b; ++b){
                for (size_t j = 0; j < block.n_occ_b; ++j) {
                    double val = block.effect_C_v_beta(mu, b) * block.V(j, l);
                    for (size_t jp = 0; jp < block.n_occ_b; ++jp) {
                        double Coj = block.effect_C_o_beta(mu, jp);
                        if (abs(Coj) < 1e-7) continue;
                        for (size_t i = 0; i < block.n_occ_a; ++i) {
                            double sv = block.sigma_v(j * block.n_occ_b + l,i + jp * block.n_occ_a);
                            if (sv == 0.0) continue;
                            val += Coj * sv * s_ov_ab(i, b);
                        }
                    }
                    sigma_bb(mu * block.n_occ_b + l, b + j * block.n_vir_b) = val;
                }
            }
        }
    }

    block.sigma_aa = sigma_aa;
    block.sigma_ba = sigma_ba;
    block.sigma_ab = sigma_ab;
    block.sigma_bb = sigma_bb;


    DBG("sigma_aa max abs =" << sigma_aa.max());
    DBG("sigma_ba max abs =" << sigma_ba.max());
    DBG("sigma_ab max abs =" << sigma_ab.max());
    DBG("sigma_bb max abs =" << sigma_bb.max());
    DBG("===== sigma_overlap END =====");
    return;
}




void spin_adiabatic_state::sigma_matrix(MOpair& block) {
    DBG("===== sigma_overlap START (Optimized) =====");


    block.n_svd = block.lambda.n_elem;
    block.n_ao = block.effect_C_o_alpha.n_rows;
    block.n_occ_a = block.effect_C_o_alpha.n_cols;
    block.n_occ_b = block.effect_C_o_beta.n_cols;
    block.n_vir_a = block.n_ao - block.n_occ_a;
    block.n_vir_b = block.n_ao - block.n_occ_b;

    size_t na = block.n_occ_a;
    size_t nb = block.n_occ_b;
    size_t nva = block.n_vir_a;
    size_t nvb = block.n_vir_b;
    size_t nao = block.n_ao;


    mat sigma_aa(nao * na, nva * na, fill::zeros);
    mat sigma_ab(nao * na, nvb * nb, fill::zeros);
    mat sigma_ba(nao * nb, nva * na, fill::zeros);
    mat sigma_bb(nao * nb, nvb * nb, fill::zeros);

    mat s_vo_ab = block.effect_C_v_alpha.t() * AOS * block.effect_C_o_beta; // (nva, nb)
    mat s_ov_ab = block.effect_C_o_alpha.t() * AOS * block.effect_C_v_beta; // (na, nvb)
    mat s_vo_ab_t = s_vo_ab.t(); // (nb, nva)
    mat s_ov_ab_t = s_ov_ab.t(); // (nvb, na)

    block.sigma_u = sigma_u(block);
    block.sigma_v = sigma_v(block);

    mat st_u = block.sigma_u.t();
    mat st_v = block.sigma_v.t();

//sigma_aa
    #pragma omp parallel for schedule(dynamic)
    for (size_t l = 0; l < na; ++l) {
        for (size_t i = 0; i < na; ++i) {
            double u_val = block.U(i, l);
            if (std::abs(u_val) > 1e-12) {
                for (size_t mu = 0; mu < nao; ++mu) {
                    for (size_t a = 0; a < nva; ++a) {
                         sigma_aa(mu * na + l, i * nva + a) = block.effect_C_v_alpha(mu, a) * u_val;
                    }
                }
            }
        }


        mat T_mat(na, na * nva, fill::zeros);

        for (size_t ip = 0; ip < na; ++ip) {
            mat M(st_u.colptr(ip * na + l), na, nb, false, true);

            mat Res = M * s_vo_ab_t;

            T_mat.row(ip) = vectorise(Res.t()).t();
        }

        mat FinalBlock = block.effect_C_o_alpha * T_mat; // (nao, na*nva)

        for (size_t mu = 0; mu < nao; ++mu) {
             sigma_aa.row(mu * na + l) += FinalBlock.row(mu);
        }
    }

//sigma_ab
    #pragma omp parallel for schedule(dynamic)
    for (size_t l = 0; l < na; ++l) {
        for (size_t j = 0; j < nb; ++j) {
            mat M_u(na, na);
            for(size_t i=0; i<na; ++i) {
                const double* col_ptr = st_u.colptr(i * na + l);
                std::memcpy(M_u.colptr(i), col_ptr + j * na, na * sizeof(double));
            }

            mat Z = M_u.t() * s_ov_ab;

            mat R = block.effect_C_o_alpha * Z;

            for(size_t mu=0; mu<nao; ++mu) {
                sigma_ab(mu * na + l, span(j * nvb, (j + 1) * nvb - 1)) = R.row(mu);
            }
        }
    }

//sigma_ba
    #pragma omp parallel for schedule(dynamic)
    for (size_t l = 0; l < nb; ++l) {
        for (size_t i = 0; i < na; ++i) {
            mat K(nb, nb);
            for(size_t j=0; j<nb; ++j) {
                const double* src_col = st_v.colptr(j * nb + l);
                for(size_t jp=0; jp<nb; ++jp) {
                    K(j, jp) = src_col[i + jp * na];
                }
            }
            mat Z = K * s_vo_ab_t;

            mat R = block.effect_C_o_beta * Z;

            for(size_t mu=0; mu<nao; ++mu) {
                sigma_ba(mu * nb + l, span(i * nva, (i + 1) * nva - 1)) = R.row(mu);
            }
        }
    }

    // sigma_bb
    #pragma omp parallel for schedule(dynamic)
    for (size_t l = 0; l < nb; ++l) {
        for (size_t j = 0; j < nb; ++j) {
            double v_val = block.V(j, l);
            if (std::abs(v_val) > 1e-12) {
                for (size_t mu = 0; mu < nao; ++mu) {
                    for (size_t b = 0; b < nvb; ++b) {
                        sigma_bb(mu * nb + l, j * nvb + b) = block.effect_C_v_beta(mu, b) * v_val;
                    }
                }
            }
        }


        for (size_t j = 0; j < nb; ++j) {

            mat M(st_v.colptr(j * nb + l), na, nb, false, true);


            mat Res = M.t() * s_ov_ab;


            mat Block_Res = block.effect_C_o_beta * Res;

            for(size_t mu=0; mu<nao; ++mu) {
                sigma_bb(mu * nb + l, span(j * nvb, (j + 1) * nvb - 1)) += Block_Res.row(mu);
            }
        }
    }

    block.sigma_aa = sigma_aa;
    block.sigma_ba = sigma_ba;
    block.sigma_ab = sigma_ab;
    block.sigma_bb = sigma_bb;

    DBG("sigma_aa max abs =" << sigma_aa.max());
    DBG("sigma_ba max abs =" << sigma_ba.max());
    DBG("sigma_ab max abs =" << sigma_ab.max());
    DBG("sigma_bb max abs =" << sigma_bb.max());
    DBG("===== sigma_overlap END (Optimized) =====");
}
/*
void spin_adiabatic_state::sigma_matrix_test(MOpair& block){

    DBG("===== sigma_overlap TEST MODE START =====");
    mat sigma_aa_orig, sigma_ab_orig, sigma_ba_orig, sigma_bb_orig;

    auto t1 = std::chrono::high_resolution_clock::now();
    sigma_overlap(block);
    sigma_aa_orig = block.sigma_aa;
    sigma_ab_orig = block.sigma_ab;
    sigma_ba_orig = block.sigma_ba;
    sigma_bb_orig = block.sigma_bb;
    auto t2 = std::chrono::high_resolution_clock::now();
    double t_original = std::chrono::duration<double>(t2 - t1).count();
    DBG("[TIME] original sigma_overlap = " << t_original << " sec");

    mat sigma_aa_fast, sigma_ab_fast, sigma_ba_fast, sigma_bb_fast;

    auto t3 = std::chrono::high_resolution_clock::now();
    sigma_matrix(block);
    sigma_aa_fast = block.sigma_aa;
    sigma_ab_fast = block.sigma_ab;
    sigma_ba_fast = block.sigma_ba;
    sigma_bb_fast = block.sigma_bb;
    auto t4 = std::chrono::high_resolution_clock::now();
    double t_fast = std::chrono::duration<double>(t4 - t3).count();
    DBG("[TIME] fast sigma_overlap     = " << t_fast << " sec");
    DBG("[SPEEDUP] = " << t_original / t_fast << " × faster");
    double diff_aa = abs(sigma_aa_orig - sigma_aa_fast).max();
    double diff_ab = abs(sigma_ab_orig - sigma_ab_fast).max();
    double diff_ba = abs(sigma_ba_orig - sigma_ba_fast).max();
    double diff_bb = abs(sigma_bb_orig - sigma_bb_fast).max();
    DBG("===== Consistency check =====");
    DBG("diff σ_aa max abs = " << diff_aa);
    DBG("diff σ_ab max abs = " << diff_ab);
    DBG("diff σ_ba max abs = " << diff_ba);
    DBG("diff σ_bb max abs = " << diff_bb);
    double diff_max = std::max({diff_aa, diff_ab, diff_ba, diff_bb});
    if (diff_max < 1e-10)
        DBG(">>> sigma_overlap FAST VERSION VALID (diff < 1e-10)");
    else
        DBG(">>> WARNING: sigma_overlap fast and original differ! max diff = "
            << diff_max);
    DBG("===== sigma_overlap TEST MODE END =====");

}


void spin_adiabatic_state::pi_matrix(OrbitalPair& pair) {

    DBG("===== pi_matrix START =====");
    MOpair b1 = S1_orthonal;
    MOpair b2 = S2_orthonal;
    pair.n_ao = b1.n_ao;
    pair.n_occ_a1 = pair.C1_flipped_alpha.n_cols;
    pair.n_occ_b1 = pair.C1_flipped_beta.n_cols;
    pair.n_vir_b1 = pair.n_ao - pair.n_occ_b1;
    pair.n_vir_a1 = pair.n_ao - pair.n_occ_a1;
    pair.n_occ_a2 = pair.C2_flipped_alpha.n_cols;
    pair.n_occ_b2 = pair.C2_flipped_beta.n_cols;
    pair.n_vir_b2 = pair.n_ao - pair.n_occ_b2;
    pair.n_vir_a2 = pair.n_ao - pair.n_occ_a2;
    mat pi_aa_1(pair.n_occ_a1 * pair.n_occ_a2, b1.n_vir_a * b1.n_occ_a, fill::zeros);
    mat pi_ab_1(pair.n_occ_a1 * pair.n_occ_a2, b1.n_vir_b * b1.n_occ_b, fill::zeros);
    mat pi_aa_2(pair.n_occ_a1 * pair.n_occ_a2, b2.n_vir_a * b2.n_occ_a, fill::zeros);
    mat pi_ab_2(pair.n_occ_a1 * pair.n_occ_a2, b2.n_vir_b * b2.n_occ_b, fill::zeros);

    mat pi_ba_1(pair.n_occ_b1 * pair.n_occ_b2, b1.n_vir_a * b1.n_occ_a, fill::zeros);
    mat pi_bb_1(pair.n_occ_b1 * pair.n_occ_b2, b1.n_vir_b * b1.n_occ_b, fill::zeros);
    mat pi_ba_2(pair.n_occ_b1 * pair.n_occ_b2, b2.n_vir_a * b2.n_occ_a, fill::zeros);
    mat pi_bb_2(pair.n_occ_b1 * pair.n_occ_b2, b2.n_vir_b * b2.n_occ_b, fill::zeros);

    mat S_Cpa = AOS * pair.C2_flipped_alpha;
    mat S_Cpb  = AOS * pair.C2_flipped_beta;

    mat CaT_S = pair.C1_flipped_alpha.t() * AOS;
    mat CbT_S = pair.C1_flipped_beta.t() * AOS;

    // aa 1 & ab 1
    DBG("Computing Pi^aa_1 & ab1");
    for (size_t i = 0; i < pair.n_occ_a1; ++i) {
        for (size_t j = 0; j < pair.n_occ_a2; ++j) {
            size_t row_idx = i * pair.n_occ_a2 + j;
            for (size_t mu = 0; mu < b1.n_ao; ++mu) {
                double coeff = S_Cpa(mu,j);
                for (size_t l = 0; l < b1.n_occ_a; ++l) {
                    double e_aa = pair.block1.E_a(l, i);
                    if (e_aa == 0) continue;

                    for (size_t aip = 0; aip < b1.n_vir_a * b1.n_occ_a; ++aip){
                        pi_aa_1(row_idx, aip) += coeff * b1.sigma_aa(mu * b1.n_occ_a + l, aip) * e_aa;
                    }
                    for (size_t bjp = 0; bjp < b1.n_vir_b * b1.n_occ_b; ++bjp) {
                        pi_ab_1(row_idx, bjp) += coeff * b1.sigma_ab(mu * b1.n_occ_a + l, bjp) * e_aa;
                    }
                }
                for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                    double e_ba = pair.block1.E_a(lp+b1.n_occ_a, i);
                    for (size_t aip = 0; aip < b1.n_vir_a * b1.n_occ_a; ++aip) {
                        pi_aa_1(row_idx, aip) += coeff * b1.sigma_ba(mu * b1.n_occ_b + lp, aip) * e_ba;
                    }
                    for (size_t bjp = 0; bjp < b1.n_vir_b * b1.n_occ_b; ++bjp) {
                        pi_ab_1(row_idx, bjp) += coeff * b1.sigma_bb(mu * b1.n_occ_b + lp, bjp) * e_ba;
                    }
                }
            }
        }
    }

    // ba 1 and bb 1
    DBG("Computing Pi^ba_1 & bb1");
    for (size_t i = 0; i < pair.n_occ_b1; ++i) {
        for (size_t j = 0; j < pair.n_occ_b2; ++j) {
            size_t row_idx = i * pair.n_occ_b2 + j;
            for (size_t mu = 0; mu < b1.n_ao; ++mu) {
                double coeff = S_Cpb(mu,j);
                for (size_t l = 0; l < b1.n_occ_a; ++l) {
                    double e_ab = pair.block1.E_b(l, i);
                    for (size_t aip = 0; aip < b1.n_vir_a * b1.n_occ_a; ++aip){
                        pi_ba_1(row_idx, aip) += coeff * b1.sigma_aa(mu * b1.n_occ_a + l, aip) * e_ab ;
                    }
                    for (size_t bjp = 0; bjp < b1.n_vir_b * b1.n_occ_b; ++bjp){
                        pi_bb_1(row_idx, bjp) += coeff * b1.sigma_ab(mu * b1.n_occ_a + l, bjp) * e_ab ;
                    }
                }

                for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                    double e_bb = pair.block1.E_b(lp+b1.n_occ_a, i);

                    for (size_t aip = 0; aip < b1.n_vir_a * b1.n_occ_a; ++aip){
                        pi_ba_1(row_idx, aip) += coeff * b1.sigma_ba(mu * b1.n_occ_b + lp, aip) * e_bb;
                    }
                    for (size_t bjp = 0; bjp < b1.n_vir_b * b1.n_occ_b; ++bjp){
                        pi_bb_1(row_idx, bjp) += coeff * b1.sigma_bb(mu * b1.n_occ_b + lp, bjp) * e_bb;
                    }
                }
            }
        }
    }

    // aa 2 and ab 2
    DBG("Computing Pi^aa_2 & ab2");
    for (size_t i = 0; i < pair.n_occ_a1; ++i) {
        for (size_t j = 0; j < pair.n_occ_a2; ++j) {
            size_t row_idx = i * pair.n_occ_a2 + j;
            for (size_t mu = 0; mu < b1.n_ao; ++mu) {
                double coeff = CaT_S(i, mu);
                for (size_t l = 0; l < b2.n_occ_a; ++l) {
                    double e_aa = pair.block2.E_a(l, j);

                    for (size_t aip = 0; aip < b2.n_vir_a * b2.n_occ_a; ++aip){
                        pi_aa_2(row_idx, aip) += coeff * b2.sigma_aa(mu * b2.n_occ_a + l, aip) * e_aa;
                    }
                    for (size_t bjp = 0; bjp < b2.n_vir_b * b2.n_occ_b; ++bjp){
                        pi_ab_2(row_idx, bjp) += coeff * b2.sigma_ab(mu * b2.n_occ_a + l, bjp) * e_aa ;
                    }
                }

                for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                    double e_ba = pair.block2.E_a(lp+b2.n_occ_a, j);

                    for (size_t aip = 0; aip < b2.n_vir_a * b2.n_occ_a; ++aip){
                        pi_aa_2(row_idx, aip) += coeff * b2.sigma_ba(mu * b2.n_occ_b + lp, aip) * e_ba;
                    }
                    for (size_t bjp = 0; bjp < b2.n_vir_b * b2.n_occ_b; ++bjp){
                        pi_ab_2(row_idx, bjp) += coeff * b2.sigma_bb(mu * b2.n_occ_b + lp, bjp) * e_ba;
                    }
                }
            }
        }
    }
    // ba 2 and bb 2
    DBG("Computing Pi^ba2 & bb2");
    for (size_t i = 0; i < pair.n_occ_b1; ++i) {
        for (size_t j = 0; j < pair.n_occ_b2; ++j) {
            size_t row_idx = i * pair.n_occ_b2 + j;
            for (size_t mu = 0; mu < b1.n_ao; ++mu) {
                double coeff = CbT_S(i, mu);
                for (size_t l = 0; l < b2.n_occ_a; ++l) {
                    double e_ab = pair.block2.E_b(l, j);
                    for (size_t aip = 0; aip < b2.n_vir_a * b2.n_occ_a; ++aip)
                        pi_ba_2(row_idx, aip) += coeff * b2.sigma_aa(mu * b2.n_occ_a + l, aip) * e_ab ;

                    for (size_t bjp = 0; bjp < b2.n_vir_b * b2.n_occ_b; ++bjp)
                        pi_bb_2(row_idx, bjp) += coeff * b2.sigma_ab(mu * b2.n_occ_a + l, bjp) * e_ab ;
                }
                for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                    double e_bb = pair.block2.E_b(lp+b2.n_occ_a, j);

                    for (size_t aip = 0; aip < b2.n_vir_a * b2.n_occ_a; ++aip){
                        pi_ba_2(row_idx, aip) += coeff * b2.sigma_ba(mu * b2.n_occ_b + lp, aip) * e_bb;
                    }
                    for (size_t bjp = 0; bjp < b2.n_vir_b * b2.n_occ_b; ++bjp){
                        pi_bb_2(row_idx, bjp) += coeff * b2.sigma_bb(mu * b2.n_occ_b + lp, bjp) * e_bb;
                    }
                }
            }
        }
    }

    pair.pi_aa_1 = pi_aa_1;
    pair.pi_ab_1 = pi_ab_1;
    pair.pi_aa_2 = pi_aa_2;
    pair.pi_ab_2 = pi_ab_2;
    pair.pi_ba_1 = pi_ba_1;
    pair.pi_bb_1 = pi_bb_1;
    pair.pi_ba_2 = pi_ba_2;
    pair.pi_bb_2 = pi_bb_2;
    DBG("Pi_aa_1 max abs =" << pi_aa_1.max());
    DBG("Pi_ba_1 max abs =" << pi_ba_1.max());
    DBG("Pi_ab_1 max abs =" << pi_ab_1.max());
    DBG("Pi_bb_1 max abs =" << pi_bb_1.max());
    DBG("Pi_aa_2 max abs =" << pi_aa_2.max());
    DBG("Pi_ba_2 max abs =" << pi_ba_2.max());
    DBG("Pi_ab_2 max abs =" << pi_ab_2.max());
    DBG("Pi_bb_2 max abs =" << pi_bb_2.max());
    DBG("===== pi_matrix END =====");
}
*/
/*
void spin_adiabatic_state::k_matrix_null(OrbitalPair& pair)
{

    pi_matrix(pair);

    MOpair b1 = S1_orthonal;
    MOpair b2 = S2_orthonal;
    size_t n_ao = b1.n_ao;
    DBG("===== k_matrix_null START =====");
    DBG("Notice: For Ms<0, the alpha-beta should exchange here:block1: "
        << " n_occ_a=" << b1.n_occ_a
        << " n_occ_b=" << b1.n_occ_b
        << " n_vir_a=" << b1.n_vir_a
        << " n_vir_b=" << b1.n_vir_b
        << " n_svd="   << b1.n_svd);

    DBG("block2: "
        << " n_occ_a=" << b2.n_occ_a
        << " n_occ_b=" << b2.n_occ_b
        << " n_vir_a=" << b2.n_vir_a
        << " n_vir_b=" << b2.n_vir_b
        << " n_svd="   << b2.n_svd);

    DBG("n_ao = " << n_ao);
    mat k_a_1(b1.n_vir_a, b1.n_occ_a, fill::zeros);
    mat k_b_1(b1.n_vir_b, b1.n_occ_b, fill::zeros);
    mat k_a_2(b2.n_vir_a, b2.n_occ_a, fill::zeros);
    mat k_b_2(b2.n_vir_b, b2.n_occ_b, fill::zeros);

    vec lambda_a_inv = 1.0 / pair.lambda_a;
    vec lambda_b_inv = 1.0 / pair.lambda_b;
    mat Sooinva = pair.V_a * diagmat(lambda_a_inv) * pair.U_a.t();
    mat Sooinvb = pair.V_b * diagmat(lambda_b_inv) * pair.U_b.t();

    mat L_vsocxy = L_AO.slice(0) + L_AO.slice(1);
    mat tem_L_psi2 = L_vsocxy * pair.psi2;
    mat tem_psi1_L = pair.psi1.t() * L_vsocxy;//I forgot to add T here, this should be tem_psi1T_L
    mat U_null_b   = pair.U_b.tail_cols(1);
    mat V_null_a   = pair.V_a.tail_cols(1);
    mat C1bT_L_psi2 = pair.C1_flipped_beta.t() * tem_L_psi2;
    mat psi1_L_C2a = tem_psi1_L * pair.C2_flipped_alpha ;


    // ============ K^α (block1) ============
    DBG("Computing K^a_1");
    mat tem_Sooinvb_C1bT_L_psi2 = Sooinvb * C1bT_L_psi2;
    mat tem_psi1_L_C2a_Sooinva = psi1_L_C2a * Sooinva;
    for (size_t a = 0; a < b1.n_vir_a; ++a) {
        for (size_t i = 0; i < b1.n_occ_a; ++i) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t j = 0; j < pair.n_occ_b1; ++j) {
                    for (size_t l = 0; l < b1.n_occ_a; ++l) {
                        double s_aa = b1.sigma_aa(mu * b1.n_occ_a + l, a + i * b1.n_vir_a);
                        double e_ab = pair.block1.E_b(l, j);
                        term1 += U_null_b(j) * s_aa * e_ab  * tem_L_psi2(mu);
                    }
                    for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                        double s_ba = b1.sigma_ba(mu * b1.n_occ_b + lp, a + i * b1.n_vir_a);
                        double e_bb = pair.block1.E_b(lp+b1.n_occ_a, j);
                        term1 += U_null_b(j) * s_ba * e_bb * tem_L_psi2(mu);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t j = 0; j < pair.n_occ_b2; ++j)
                    term2 += U_null_b(ip) * pair.pi_ba_1(ip * pair.n_occ_b2 + j ,a + i * b1.n_vir_a) * tem_Sooinvb_C1bT_L_psi2(j);
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t j = 0; j < pair.n_occ_a2; ++j)
                    term3 += tem_psi1_L_C2a_Sooinva(ip) * pair.pi_aa_1(ip * pair.n_occ_a2 + j ,a + i * b1.n_vir_a) * V_null_a(j);
            }

            k_a_1(a, i) = term1 - term2 - term3;
        }
    }


    // ============ K^β (block1) ============
    DBG("Computing K^b_1");
    for (size_t b = 0; b < b1.n_vir_b; ++b) {
        for (size_t j = 0; j < b1.n_occ_b; ++j) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t jp = 0; jp < pair.n_occ_b1; ++jp) {
                    for (size_t l = 0; l < b1.n_occ_a; ++l) {
                        double s_ab = b1.sigma_ab(mu * b1.n_occ_a + l, b + j * b1.n_vir_b);
                        double e_ab = pair.block1.E_b(l, jp);
                        term1 += U_null_b(jp) * s_ab * e_ab  * tem_L_psi2(mu);
                    }
                    for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                        double s_bb = b1.sigma_bb(mu * b1.n_occ_b + lp, b + j * b1.n_vir_b);
                        double e_bb = pair.block1.E_b(lp+b1.n_occ_a, jp);
                        term1 += U_null_b(jp) *  s_bb * e_bb * tem_L_psi2(mu);
                    }

                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp)
                    term2 += U_null_b(ip) * pair.pi_bb_1(ip * pair.n_occ_b2 + jp ,b + j * b1.n_vir_b) * tem_Sooinvb_C1bT_L_psi2(jp);
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp)
                    term3 += tem_psi1_L_C2a_Sooinva(ip) * pair.pi_ab_1(ip * pair.n_occ_a2 + jp ,b + j * b1.n_vir_b) * V_null_a(jp);
            }

            k_b_1(b, j) = term1 - term2 - term3;
        }
    }


    // ============ K'^α (block2) ============
    DBG("Computing K^a_2");
    for (size_t a = 0; a < b2.n_vir_a; ++a) {
        for (size_t i = 0; i < b2.n_occ_a; ++i) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t j = 0; j < pair.n_occ_b2; ++j) {
                    for (size_t l = 0; l < b2.n_occ_a; ++l) {
                        double s_aa = b2.sigma_aa(mu * b2.n_occ_a + l, a + i * b2.n_vir_a);
                        double e_ab = pair.block2.E_b(l, j);
                        term1 += tem_psi1_L(mu) * s_aa * e_ab  * V_null_a(j);
                    }
                    for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                        double s_ba = b2.sigma_ba(mu * b2.n_occ_b + lp, a + i * b2.n_vir_a);
                        double e_bb = pair.block2.E_b(lp+b2.n_occ_a, j);
                        term1 += tem_psi1_L(mu) * s_ba * e_bb * V_null_a(j);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t j = 0; j < pair.n_occ_b2; ++j)
                    term2 += U_null_b(ip) * pair.pi_ba_2(ip * pair.n_occ_b2 + j ,a + i * b2.n_vir_a) * tem_Sooinvb_C1bT_L_psi2(j);
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t j = 0; j < pair.n_occ_a2; ++j)
                    term3 += tem_psi1_L_C2a_Sooinva(ip) * pair.pi_aa_2(ip * pair.n_occ_a2 + j ,a + i * b2.n_vir_a) * V_null_a(j);
            }

            k_a_2(a, i) = term1 - term2 - term3;
        }
    }


    // ============ K'^β (block2) ============
    DBG("Computing K^b_2");

    for (size_t b = 0; b < b2.n_vir_b; ++b) {
        for (size_t j = 0; j < b2.n_occ_b; ++j) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp) {
                    for (size_t l = 0; l < b2.n_occ_a; ++l) {
                        double s_ab = b2.sigma_ab(mu * b2.n_occ_a + l, b + j * b2.n_vir_b);
                        double e_ab = pair.block2.E_b(l, jp);
                        term1 += tem_psi1_L(mu) * s_ab * e_ab  * V_null_a(jp);
                    }
                    for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                        double s_bb = b2.sigma_bb(mu * b2.n_occ_b + lp, b + j * b2.n_vir_b);
                        double e_bb = pair.block2.E_b(lp+b2.n_occ_a, jp);
                        term1 += tem_psi1_L(mu) * s_bb * e_bb * V_null_a(jp);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp)
                    term2 += U_null_b(ip) * pair.pi_bb_2(ip * pair.n_occ_b2 + jp ,b + j * b2.n_vir_b) * tem_Sooinvb_C1bT_L_psi2(jp);
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp)
                    term3 += tem_psi1_L_C2a_Sooinva(ip) * pair.pi_ab_2(ip * pair.n_occ_a2 + jp ,b + j * b2.n_vir_b) * V_null_a(jp);
            }

            k_b_2(b, j) = term1 - term2 - term3;
        }
    }
    DBG("||k_a_1|| = " << norm(k_a_1, "fro"));
    DBG("||k_b_1|| = " << norm(k_b_1, "fro"));
    DBG("||k_a_2|| = " << norm(k_a_2, "fro"));
    DBG("||k_b_2|| = " << norm(k_b_2, "fro"));


    pair.L_a_1 = k_a_1 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_aa_1 + Sooinvb * pair.pi_ba_1);
    pair.L_b_1 = k_b_1 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_ab_1 + Sooinvb * pair.pi_bb_1);
    pair.L_a_2 = k_a_2 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_aa_2 + Sooinvb * pair.pi_ba_2);
    pair.L_b_2 = k_b_2 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_ab_2 + Sooinvb * pair.pi_bb_2);
    DBG("===== k_matrix_null END =====");
}

void spin_adiabatic_state::k_matrix_last(OrbitalPair& pair)
{

    DBG("Notice: For Ms<0, the alpha-beta should exchange here(it shouldn't change inside code, but exchange in physic):===== k_matrix_last START =====");
    pi_matrix(pair);

    MOpair b1 = S1_orthonal;
    MOpair b2 = S2_orthonal;
    size_t n_ao = b1.n_ao;

    DBG("block1: "
        << " n_occ_a=" << b1.n_occ_a
        << " n_occ_b=" << b1.n_occ_b
        << " n_vir_a=" << b1.n_vir_a
        << " n_vir_b=" << b1.n_vir_b
        << " n_svd="   << b1.n_svd);

    DBG("block2: "
        << " n_occ_a=" << b2.n_occ_a
        << " n_occ_b=" << b2.n_occ_b
        << " n_vir_a=" << b2.n_vir_a
        << " n_vir_b=" << b2.n_vir_b
        << " n_svd="   << b2.n_svd);

    DBG("n_ao = " << n_ao);

    mat k_aa_1(b1.n_vir_a, b1.n_occ_a, fill::zeros);
    mat k_ba_1(b1.n_vir_a, b1.n_occ_a, fill::zeros);
    mat k_aa_2(b2.n_vir_a, b2.n_occ_a, fill::zeros);
    mat k_ba_2(b2.n_vir_a, b2.n_occ_a, fill::zeros);

    mat k_ab_1(b1.n_vir_b, b1.n_occ_b, fill::zeros);
    mat k_bb_1(b1.n_vir_b, b1.n_occ_b, fill::zeros);
    mat k_ab_2(b2.n_vir_b, b2.n_occ_b, fill::zeros);
    mat k_bb_2(b2.n_vir_b, b2.n_occ_b, fill::zeros);


    mat L_vsocz = L_AO.slice(2) ;
    mat tem_L_psi2a = L_vsocz * pair.psi2_alpha;
    mat tem_psi1aT_L = pair.psi1_alpha.t() * L_vsocz;
    mat tem_L_psi2b = L_vsocz * pair.psi2_beta;
    mat tem_psi1bT_L = pair.psi1_beta.t() * L_vsocz;
    mat U_last_b   = pair.U_b.tail_cols(1);
    mat V_last_a   = pair.V_a.tail_cols(1);
    mat U_last_a   = pair.U_a.tail_cols(1);
    mat V_last_b   = pair.V_b.tail_cols(1);
    mat C1bT_L_psi2b = pair.C1_flipped_beta.t() * tem_L_psi2b;
    mat C1aT_L_psi2a = pair.C1_flipped_alpha.t() * tem_L_psi2a;
    mat psi1a_L_C2a = tem_psi1aT_L * pair.C2_flipped_alpha ;
    mat psi1b_L_C2b = tem_psi1bT_L * pair.C2_flipped_beta ;
    MOpair temblocka,temblockb;
    temblocka.U = pair.U_a;
    temblocka.V = pair.V_a;
    temblockb.U = pair.U_b;
    temblockb.V = pair.V_b;
    temblocka.lambda = pair.lambda_a;
    temblockb.lambda = pair.lambda_b;
    temblocka.n_ao = n_ao;
    temblockb.n_ao = n_ao;
    temblocka.n_occ_a = pair.n_occ_a1;
    temblocka.n_occ_b = pair.n_occ_a2;
    temblocka.n_svd = pair.lambda_a.size();
    temblockb.n_occ_a = pair.n_occ_b1;
    temblockb.n_occ_b = pair.n_occ_b2;
    temblockb.n_svd = pair.lambda_b.size();
    pair.sigma_ua = sigma_u(temblocka);
    pair.sigma_va = sigma_v(temblocka);
    pair.sigma_ub = sigma_u(temblockb);
    pair.sigma_vb = sigma_v(temblockb);


    // ============ K^αα  1============
    DBG("Computing K^aa_1");
    for (size_t a = 0; a < b1.n_vir_a; ++a) {
        for (size_t i = 0; i < b1.n_occ_a; ++i) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t j = 0; j < pair.n_occ_a1; ++j) {
                    for (size_t l = 0; l < b1.n_occ_a; ++l) {
                        double s_aa = b1.sigma_aa(mu * b1.n_occ_a + l, a + i * b1.n_vir_a);
                        double e_aa = pair.block1.E_a(l, j);
                        term1 += U_last_a(j) * s_aa * e_aa  * tem_L_psi2a(mu);
                    }
                    for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                        double s_ba = b1.sigma_ba(mu * b1.n_occ_b + lp, a + i * b1.n_vir_a);
                        double e_ba = pair.block1.E_a(lp+b1.n_occ_a, j);
                        term1 += U_last_a(j) *  s_ba * e_ba * tem_L_psi2a(mu);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_a1; ++ipp)
                    term2 +=  pair.pi_aa_1(ip * pair.n_occ_a2 + jp ,a + i * b1.n_vir_a) * pair.sigma_ua((ipp+1) * pair.n_occ_a1 -1, ip + jp * pair.n_occ_a1) * C1aT_L_psi2a(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t jpp = 0; jpp < pair.n_occ_a2; ++jpp)
                    term3 +=   psi1a_L_C2a(jpp) * pair.pi_aa_1(ip * pair.n_occ_a2 + jp ,a + i * b1.n_vir_a) * pair.sigma_va((jpp+1) * pair.n_occ_a2 -1, ip + jp * pair.n_occ_a1) ;
                }
            }


            k_aa_1(a, i) = term1 + term2 + term3;
        }
    }


    // ============ K^βα 1============
    DBG("Computing K^ba_1...");
    for (size_t a = 0; a < b1.n_vir_a; ++a) {
        for (size_t i = 0; i < b1.n_occ_a; ++i) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t j = 0; j < pair.n_occ_b1; ++j) {
                    for (size_t l = 0; l < b1.n_occ_a; ++l) {
                        double s_aa = b1.sigma_aa(mu * b1.n_occ_a + l, a + i * b1.n_vir_a);
                        double e_ab = pair.block1.E_b(l, j);
                        term1 += U_last_b(j) * s_aa * e_ab * tem_L_psi2b(mu);
                    }
                    for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                        double s_ba = b1.sigma_ba(mu * b1.n_occ_b + lp, a + i * b1.n_vir_a);
                        double e_bb = pair.block1.E_b(lp+b1.n_occ_a, j);
                        term1 += U_last_b(j) *  s_ba * e_bb * tem_L_psi2b(mu);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_b1; ++ipp)
                    term2 +=  pair.pi_ba_1(ip * pair.n_occ_b2 + jp ,a + i * b1.n_vir_a) * pair.sigma_ub((ipp+1) * pair.n_occ_b1 -1, ip + jp * pair.n_occ_b1) * C1bT_L_psi2b(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t jpp = 0; jpp < pair.n_occ_b2; ++jpp)
                    term3 +=   psi1b_L_C2b(jpp) * pair.pi_ba_1(ip * pair.n_occ_b2 + jp ,a + i * b1.n_vir_a) * pair.sigma_vb((jpp+1) * pair.n_occ_b2 -1, ip + jp * pair.n_occ_b1) ;

                }
            }


            k_ba_1(a, i) = term1 + term2 + term3;
        }
    }

    // ============ K^αb 1  ============
    DBG("Computing K^ab_1...");
    for (size_t b = 0; b < b1.n_vir_b; ++b) {
        for (size_t j = 0; j < b1.n_occ_b; ++j) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t jp = 0; jp < pair.n_occ_a1; ++jp) {
                    for (size_t l = 0; l < b1.n_occ_a; ++l) {
                        double s_ab = b1.sigma_ab(mu * b1.n_occ_a + l, b + j * b1.n_vir_b);
                        double e_aa = pair.block1.E_a(l, jp);
                        term1 += U_last_a(j) * s_ab * e_aa * tem_L_psi2a(mu);
                    }
                    for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                        double s_bb = b1.sigma_bb(mu * b1.n_occ_b + lp, b + j * b1.n_vir_b);
                        double e_ba = pair.block1.E_a(lp+b1.n_occ_a, jp);
                        term1 += U_last_a(j) *  s_bb * e_ba * tem_L_psi2a(mu);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_a1; ++ipp)
                    term2 +=  pair.pi_ab_1(ip * pair.n_occ_a2 + jp ,b + j * b1.n_vir_b) * pair.sigma_ua((ipp+1) * pair.n_occ_a1 -1, ip + jp * pair.n_occ_a1) * C1aT_L_psi2a(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t jpp = 0; jpp < b2.n_occ_a; ++jpp)
                    term3 +=   psi1a_L_C2a(jpp) * pair.pi_ab_1(ip * pair.n_occ_a2 + jp ,b + j * b1.n_vir_b) * pair.sigma_va((jpp+1) * pair.n_occ_a2 -1, ip + jp * pair.n_occ_a1) ;
                }
            }
            k_ab_1(b, j) = term1 + term2 + term3;
        }
    }

    // ============ K^ββ 1============
    DBG("Computing K^bb_1...");
    for (size_t b = 0; b < b1.n_vir_b; ++b) {
        for (size_t j = 0; j < b1.n_occ_b; ++j) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t jp = 0; jp < pair.n_occ_b1; ++jp) {
                    for (size_t l = 0; l < b1.n_occ_a; ++l) {
                        double s_ab = b1.sigma_ab(mu * b1.n_occ_a + l, b + j * b1.n_vir_b);
                        double e_ab = pair.block1.E_b(l, jp);
                        term1 += U_last_b(j) * s_ab * e_ab  * tem_L_psi2b(mu);
                    }
                    for (size_t lp = 0; lp < b1.n_occ_b; ++lp) {
                        double s_bb = b1.sigma_bb(mu * b1.n_occ_b + lp, b + j * b1.n_vir_b);
                        double e_bb = pair.block1.E_b(lp+b1.n_occ_a, jp);
                        term1 += U_last_b(j) *  s_bb * e_bb * tem_L_psi2b(mu);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_b1; ++ipp)
                    term2 +=  pair.pi_bb_1(ip * pair.n_occ_b2 + jp ,b + j * b1.n_vir_b) * pair.sigma_ub((ipp+1) * pair.n_occ_b1 -1, ip + jp * pair.n_occ_b1) * C1bT_L_psi2b(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t jpp = 0; jpp < pair.n_occ_b2; ++jpp)
                    term3 +=   psi1b_L_C2b(jpp) * pair.pi_bb_1(ip * pair.n_occ_b2 + jp ,b + j * b1.n_vir_b) * pair.sigma_vb((jpp+1) * pair.n_occ_b2 -1, ip + jp * pair.n_occ_b1) ;
                }
            }
            k_bb_1(b, j) = term1 + term2 + term3;
        }
    }
    // block2
    // ============ K^αα  2============
    DBG("Computing K^aa_2...");
    for (size_t a = 0; a < b2.n_vir_a; ++a) {
        for (size_t i = 0; i < b2.n_occ_a; ++i) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t j = 0; j < pair.n_occ_a2; ++j) {
                    for (size_t l = 0; l < b2.n_occ_a; ++l) {
                        double s_aa = b2.sigma_aa(mu * b2.n_occ_a + l, a + i * b2.n_vir_a);
                        double e_aa = pair.block2.E_a(l, j);
                        term1 += tem_psi1aT_L(mu) * s_aa * e_aa * V_last_a(j);
                    }
                    for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                        double s_ba = b2.sigma_ba(mu * b2.n_occ_b + lp, a + i * b2.n_vir_a);
                        double e_ba = pair.block2.E_a(lp+b2.n_occ_a, j);
                        term1 += tem_psi1aT_L(mu) *  s_ba * e_ba * V_last_a(j);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_a1; ++ipp)
                    term2 +=  pair.pi_aa_2(ip * pair.n_occ_a2 + jp ,a + i * b2.n_vir_a) * pair.sigma_ua((ipp+1) * pair.n_occ_a1 -1, ip + jp * pair.n_occ_a1) * C1aT_L_psi2a(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t jpp = 0; jpp < pair.n_occ_a2; ++jpp)
                    term3 +=   psi1a_L_C2a(jpp) * pair.pi_aa_2(ip * pair.n_occ_a2 + jp ,a + i * b2.n_vir_a) * pair.sigma_va((jpp+1) * pair.n_occ_a2 -1, ip + jp * pair.n_occ_a1) ;
                }
            }


            k_aa_2(a, i) = term1 + term2 + term3;
        }
    }


    // ============ K^βα 2============
    DBG("Computing K^ba_2...");
    for (size_t a = 0; a < b2.n_vir_a; ++a) {
        for (size_t i = 0; i < b2.n_occ_a; ++i) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t j = 0; j < pair.n_occ_b2; ++j) {
                    for (size_t l = 0; l < b2.n_occ_a; ++l) {
                        double s_aa = b2.sigma_aa(mu * b2.n_occ_a + l, a + i * b2.n_vir_a);
                        double e_ab = pair.block2.E_b(l, j);
                        term1 += tem_psi1bT_L(mu) * s_aa * e_ab  * V_last_b(j);
                    }
                    for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                        double s_ba = b2.sigma_ba(mu * b2.n_occ_b + lp, a + i * b2.n_vir_a);
                        double e_bb = pair.block2.E_b(lp+b2.n_occ_a, j);
                        term1 += tem_psi1bT_L(mu) * s_ba * e_bb * V_last_b(j);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_b2; ++ipp)
                    term2 +=  pair.pi_ba_2(ip * pair.n_occ_b2 + jp ,a + i * b2.n_vir_a) * pair.sigma_ub((ipp+1) * pair.n_occ_b1 -1, ip + jp * pair.n_occ_b1) * C1bT_L_psi2b(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t jpp = 0; jpp < pair.n_occ_b1; ++jpp)
                    term3 +=   psi1b_L_C2b(jpp) * pair.pi_ba_2(ip * pair.n_occ_b2 + jp ,a + i * b2.n_vir_a) * pair.sigma_vb((jpp+1) * pair.n_occ_b2 -1, ip + jp * pair.n_occ_b1) ;

                }
            }


            k_ba_2(a, i) = term1 + term2 + term3;
        }
    }

    // ============ K^αb 2  ============
    DBG("Computing K^ab_2...");
    for (size_t b = 0; b < b2.n_vir_b; ++b) {
        for (size_t j = 0; j < b2.n_occ_b; ++j) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {

                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp) {
                    for (size_t l = 0; l < b2.n_occ_a; ++l) {
                        double s_ab = b2.sigma_ab(mu * b2.n_occ_a + l, b + j * b2.n_vir_b);
                        double e_aa = pair.block2.E_a(l, jp);
                        term1 += tem_psi1aT_L(mu) * s_ab * e_aa * V_last_a(j);
                        }
                    for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                        double s_bb = b2.sigma_bb(mu * b2.n_occ_b + lp, b + j * b2.n_vir_b);
                        double e_ba = pair.block2.E_a(lp+b2.n_occ_a, jp);
                        term1 += tem_psi1aT_L(mu) * s_bb * e_ba * V_last_a(j);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_a1; ++ipp)
                    term2 +=  pair.pi_ab_2(ip * pair.n_occ_a2 + jp ,b + j * b2.n_vir_b) * pair.sigma_ua((ipp+1) * pair.n_occ_a1 -1, ip + jp * pair.n_occ_a1) * C1aT_L_psi2a(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_a1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_a2; ++jp){
                    for  (size_t jpp = 0; jpp < pair.n_occ_a2; ++jpp)
                    term3 +=   psi1a_L_C2a(jpp) * pair.pi_ab_2(ip * pair.n_occ_a2 + jp ,b + j * b2.n_vir_b) * pair.sigma_va((jpp+1) * pair.n_occ_a2 -1, ip + jp * pair.n_occ_a1) ;
                }
            }
            k_ab_2(b, j) = term1 + term2 + term3;
        }
    }

    // ============ K^ββ 2============
    DBG("Computing K^bb_2...");
    for (size_t b = 0; b < b2.n_vir_b; ++b) {
        for (size_t j = 0; j < b2.n_occ_b; ++j) {
            double term1 = 0.0;
            for (size_t mu = 0; mu < n_ao; ++mu) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp) {
                    for (size_t l = 0; l < b2.n_occ_a; ++l) {
                        double s_ab = b2.sigma_ab(mu * b2.n_occ_a + l, b + j * b2.n_vir_b);
                        double e_ab = pair.block2.E_b(l, jp);
                        term1 += tem_psi1bT_L(mu) * s_ab * e_ab  * V_last_b(j);
                    }
                    for (size_t lp = 0; lp < b2.n_occ_b; ++lp) {
                        double s_bb = b2.sigma_bb(mu * b2.n_occ_b + lp, b + j * b2.n_vir_b);
                        double e_bb = pair.block2.E_b(lp+b2.n_occ_a, jp);
                        term1 += tem_psi1bT_L(mu) * s_bb * e_bb * V_last_b(j);
                    }
                }
            }


            double term2 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t ipp = 0; ipp < pair.n_occ_b1; ++ipp)
                    term2 +=  pair.pi_bb_2(ip * pair.n_occ_b2 + jp ,b + j * b2.n_vir_b) * pair.sigma_ub((ipp+1) * pair.n_occ_b1 -1, ip + jp * pair.n_occ_b1) * C1bT_L_psi2b(ipp);
                }
            }


            double term3 = 0.0;
            for (size_t ip = 0; ip < pair.n_occ_b1; ++ip) {
                for (size_t jp = 0; jp < pair.n_occ_b2; ++jp){
                    for  (size_t jpp = 0; jpp < b2.n_occ_b; ++jpp)
                    term3 +=   psi1b_L_C2b(jpp) * pair.pi_bb_2(ip * pair.n_occ_b2 + jp ,b + j * b2.n_vir_b) * pair.sigma_vb((jpp+1) * pair.n_occ_b2 -1, ip + jp * pair.n_occ_b1) ;
                }
            }
            k_bb_2(b, j) = term1 + term2 + term3;
        }
    }


    DBG("||K_aa_1|| = " << norm(k_aa_1, "fro"));
    DBG("||K_ba_1|| = " << norm(k_ba_1, "fro"));
    DBG("||K_ab_1|| = " << norm(k_ab_1, "fro"));
    DBG("||K_bb_1|| = " << norm(k_bb_1, "fro"));

    DBG("||K_aa_2|| = " << norm(k_aa_2, "fro"));
    DBG("||K_ba_2|| = " << norm(k_ba_2, "fro"));
    DBG("||K_ab_2|| = " << norm(k_ab_2, "fro"));
    DBG("||K_bb_2|| = " << norm(k_bb_2, "fro"));

    mat Vca = pair.V_a.cols(0,pair.lambda_a.size()-1);
    mat Vcb = pair.V_b.cols(0,pair.lambda_b.size()-1);
    mat Uca = pair.U_a.cols(0,pair.lambda_a.size()-1);
    mat Ucb = pair.U_b.cols(0,pair.lambda_b.size()-1);


    vec lambda_a_m1_inv = 1.0 / pair.lambda_a;
    vec lambda_b_m1_inv = 1.0 / pair.lambda_b;
    lambda_a_m1_inv(lambda_a_m1_inv.n_elem - 1) = 0.0;
    lambda_b_m1_inv(lambda_b_m1_inv.n_elem - 1) = 0.0;


    mat Sooinva = Vca * diagmat(lambda_a_m1_inv) * Uca.t();
    mat Sooinvb = Vcb * diagmat(lambda_b_m1_inv) * Ucb.t();
    DBG("===== Matrix size check =====");

    DBG("K_aa_1 size = " << k_aa_1.n_rows << " x " << k_aa_1.n_cols);
    DBG("K_ba_1 size = " << k_ba_1.n_rows << " x " << k_ba_1.n_cols);
    DBG("K_ab_1 size = " << k_ab_1.n_rows << " x " << k_ab_1.n_cols);
    DBG("K_bb_1 size = " << k_bb_1.n_rows << " x " << k_bb_1.n_cols);

    DBG("K_aa_2 size = " << k_aa_2.n_rows << " x " << k_aa_2.n_cols);
    DBG("K_ba_2 size = " << k_ba_2.n_rows << " x " << k_ba_2.n_cols);
    DBG("K_ab_2 size = " << k_ab_2.n_rows << " x " << k_ab_2.n_cols);
    DBG("K_bb_2 size = " << k_bb_2.n_rows << " x " << k_bb_2.n_cols);

    // --- pi matrices
    DBG("pi_aa_1 size = " << pair.pi_aa_1.n_rows << " x " << pair.pi_aa_1.n_cols);
    DBG("pi_ba_1 size = " << pair.pi_ba_1.n_rows << " x " << pair.pi_ba_1.n_cols);
    DBG("pi_ab_1 size = " << pair.pi_ab_1.n_rows << " x " << pair.pi_ab_1.n_cols);
    DBG("pi_bb_1 size = " << pair.pi_bb_1.n_rows << " x " << pair.pi_bb_1.n_cols);

    DBG("pi_aa_2 size = " << pair.pi_aa_2.n_rows << " x " << pair.pi_aa_2.n_cols);
    DBG("pi_ba_2 size = " << pair.pi_ba_2.n_rows << " x " << pair.pi_ba_2.n_cols);
    DBG("pi_ab_2 size = " << pair.pi_ab_2.n_rows << " x " << pair.pi_ab_2.n_cols);
    DBG("pi_bb_2 size = " << pair.pi_bb_2.n_rows << " x " << pair.pi_bb_2.n_cols);

    // --- U, V, lambda
    DBG("U_a size = " << pair.U_a.n_rows << " x " << pair.U_a.n_cols);
    DBG("V_a size = " << pair.V_a.n_rows << " x " << pair.V_a.n_cols);
    DBG("lambda_a size = " << pair.lambda_a.n_elem);

    DBG("U_b size = " << pair.U_b.n_rows << " x " << pair.U_b.n_cols);
    DBG("V_b size = " << pair.V_b.n_rows << " x " << pair.V_b.n_cols);
    DBG("lambda_b size = " << pair.lambda_b.n_elem);

    // --- core / null split results
    DBG("Uca size = " << Uca.n_rows << " x " << Uca.n_cols);
    DBG("Vca size = " << Vca.n_rows << " x " << Vca.n_cols);

    DBG("Ucb size = " << Ucb.n_rows << " x " << Ucb.n_cols);
    DBG("Vcb size = " << Vcb.n_rows << " x " << Vcb.n_cols);

    // --- Sooinv blocks
    DBG("Sooinva size = " << Sooinva.n_rows << " x " << Sooinva.n_cols);
    DBG("Sooinvb size = " << Sooinvb.n_rows << " x " << Sooinvb.n_cols);

    // --- vector sizes used in diagmat
    DBG("lambda_a_m1_inv size = " << lambda_a_m1_inv.n_elem);
    DBG("lambda_b_m1_inv size = " << lambda_b_m1_inv.n_elem);

    // --- scalars
    DBG("scaling_factor = " << scaling_factor);
    DBG("val_a = " << pair.val_a);
    DBG("val_b = " << pair.val_b);

    pair.L_aa_1 = k_aa_1 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_aa_1 + Sooinvb * pair.pi_ba_1);
    pair.L_ba_1 = k_ba_1 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_ba_1 + Sooinva * pair.pi_aa_1);
    pair.L_ab_1 = k_ab_1 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_ab_1 + Sooinvb * pair.pi_bb_1);
    pair.L_bb_1 = k_bb_1 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_bb_1 + Sooinva * pair.pi_ab_1);
    pair.L_aa_2 = k_aa_2 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_aa_2 + Sooinvb * pair.pi_ba_2);
    pair.L_ba_2 = k_ba_2 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_ba_2 + Sooinva * pair.pi_aa_2);
    pair.L_ab_2 = k_ab_2 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_ab_2 + Sooinvb * pair.pi_bb_2);
    pair.L_bb_2 = k_bb_2 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_bb_2 + Sooinva * pair.pi_ab_2);

    DBG("===== Final L-block size check =====");

    DBG("L_aa_1 size = " << pair.L_aa_1.n_rows << " x " << pair.L_aa_1.n_cols);
    DBG("L_ba_1 size = " << pair.L_ba_1.n_rows << " x " << pair.L_ba_1.n_cols);
    DBG("L_ab_1 size = " << pair.L_ab_1.n_rows << " x " << pair.L_ab_1.n_cols);
    DBG("L_bb_1 size = " << pair.L_bb_1.n_rows << " x " << pair.L_bb_1.n_cols);

    DBG("L_aa_2 size = " << pair.L_aa_2.n_rows << " x " << pair.L_aa_2.n_cols);
    DBG("L_ba_2 size = " << pair.L_ba_2.n_rows << " x " << pair.L_ba_2.n_cols);
    DBG("L_ab_2 size = " << pair.L_ab_2.n_rows << " x " << pair.L_ab_2.n_cols);
    DBG("L_bb_2 size = " << pair.L_bb_2.n_rows << " x " << pair.L_bb_2.n_cols);


}

*/


static void compute_intermediate_T(
    const mat& E, size_t n_occ_top, size_t n_occ_bot,
    const mat& Sigma_top_full, const mat& Sigma_bot_full,
    size_t mu, size_t n_occ_sigma_top, size_t n_occ_sigma_bot,
    mat& T_out)
{
    uword start_row_top = mu * n_occ_sigma_top;
    uword end_row_top = start_row_top + n_occ_sigma_top - 1;

    uword start_row_bot = mu * n_occ_sigma_bot;
    uword end_row_bot = start_row_bot + n_occ_sigma_bot - 1;

    T_out = E.rows(0, n_occ_top - 1).t() * Sigma_top_full.rows(start_row_top, end_row_top);

    if (n_occ_bot > 0) {
        T_out += E.rows(n_occ_top, n_occ_top + n_occ_bot - 1).t() * Sigma_bot_full.rows(start_row_bot, end_row_bot);
    }
}

void spin_adiabatic_state::pi_matrix(OrbitalPair& pair) {
    DBG("===== pi_matrix START (Optimize) =====");

    MOpair& b1 = S1_orthonal;
    MOpair& b2 = S2_orthonal;

    pair.n_ao = b1.n_ao;
    pair.n_occ_a1 = pair.C1_flipped_alpha.n_cols;
    pair.n_occ_b1 = pair.C1_flipped_beta.n_cols;
    pair.n_vir_b1 = pair.n_ao - pair.n_occ_b1;
    pair.n_vir_a1 = pair.n_ao - pair.n_occ_a1;
    pair.n_occ_a2 = pair.C2_flipped_alpha.n_cols;
    pair.n_occ_b2 = pair.C2_flipped_beta.n_cols;
    pair.n_vir_b2 = pair.n_ao - pair.n_occ_b2;
    pair.n_vir_a2 = pair.n_ao - pair.n_occ_a2;

    size_t na1 = pair.n_occ_a1;
    size_t nb1 = pair.n_occ_b1;
    size_t na2 = pair.n_occ_a2;
    size_t nb2 = pair.n_occ_b2;
    size_t nao = pair.n_ao;

    mat pi_aa_1(na1 * na2, b1.n_vir_a * b1.n_occ_a, fill::zeros);
    mat pi_ab_1(na1 * na2, b1.n_vir_b * b1.n_occ_b, fill::zeros);
    mat pi_ba_1(nb1 * nb2, b1.n_vir_a * b1.n_occ_a, fill::zeros);
    mat pi_bb_1(nb1 * nb2, b1.n_vir_b * b1.n_occ_b, fill::zeros);

    mat pi_aa_2(na1 * na2, b2.n_vir_a * b2.n_occ_a, fill::zeros);
    mat pi_ab_2(na1 * na2, b2.n_vir_b * b2.n_occ_b, fill::zeros);
    mat pi_ba_2(nb1 * nb2, b2.n_vir_a * b2.n_occ_a, fill::zeros);
    mat pi_bb_2(nb1 * nb2, b2.n_vir_b * b2.n_occ_b, fill::zeros);

    mat S_Cpa = AOS * pair.C2_flipped_alpha;
    mat S_Cpb = AOS * pair.C2_flipped_beta;
    mat CaT_S = pair.C1_flipped_alpha.t() * AOS;
    mat CbT_S = pair.C1_flipped_beta.t() * AOS;

    mat S_Cpa_t = S_Cpa.t();
    mat S_Cpb_t = S_Cpb.t();

    mat T_aa, T_ab;
    mat T_ba, T_bb;
    mat T2_aa, T2_ab;
    mat T2_ba, T2_bb;

    #pragma omp parallel
    {
        for (size_t mu = 0; mu < nao; ++mu) {
            #pragma omp single
            {
                compute_intermediate_T(
                    pair.block1.E_a, b1.n_occ_a, b1.n_occ_b,
                    b1.sigma_aa, b1.sigma_ba,
                    mu, b1.n_occ_a, b1.n_occ_b,
                    T_aa
                );
                compute_intermediate_T(
                    pair.block1.E_a, b1.n_occ_a, b1.n_occ_b,
                    b1.sigma_ab, b1.sigma_bb,
                    mu, b1.n_occ_a, b1.n_occ_b,
                    T_ab
                );
            }

            vec s_vec = S_Cpa_t.col(mu);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < na1; ++i) {
                pi_aa_1.rows(i * na2, (i + 1) * na2 - 1) += s_vec * T_aa.row(i);
                pi_ab_1.rows(i * na2, (i + 1) * na2 - 1) += s_vec * T_ab.row(i);
            }
        }

        //Pi_ba_1 & Pi_bb_1
        for (size_t mu = 0; mu < nao; ++mu) {
            #pragma omp single
            {
                compute_intermediate_T(
                    pair.block1.E_b, b1.n_occ_a, b1.n_occ_b,
                    b1.sigma_aa, b1.sigma_ba,
                    mu, b1.n_occ_a, b1.n_occ_b,
                    T_ba
                );
                compute_intermediate_T(
                    pair.block1.E_b, b1.n_occ_a, b1.n_occ_b,
                    b1.sigma_ab, b1.sigma_bb,
                    mu, b1.n_occ_a, b1.n_occ_b,
                    T_bb
                );
            }

            vec s_vec = S_Cpb_t.col(mu);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < nb1; ++i) {
                pi_ba_1.rows(i * nb2, (i + 1) * nb2 - 1) += s_vec * T_ba.row(i);
                pi_bb_1.rows(i * nb2, (i + 1) * nb2 - 1) += s_vec * T_bb.row(i);
            }
        }
        //  Pi_aa_2 & Pi_ab_2
        for (size_t mu = 0; mu < nao; ++mu) {
            #pragma omp single
            {
                compute_intermediate_T(
                    pair.block2.E_a, b2.n_occ_a, b2.n_occ_b,
                    b2.sigma_aa, b2.sigma_ba,
                    mu, b2.n_occ_a, b2.n_occ_b,
                    T2_aa
                );
                compute_intermediate_T(
                    pair.block2.E_a, b2.n_occ_a, b2.n_occ_b,
                    b2.sigma_ab, b2.sigma_bb,
                    mu, b2.n_occ_a, b2.n_occ_b,
                    T2_ab
                );
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < na1; ++i) {
                double val = CaT_S(i, mu);
                if (std::abs(val) > 1e-12) {
                    pi_aa_2.rows(i * na2, (i + 1) * na2 - 1) += val * T2_aa;
                    pi_ab_2.rows(i * na2, (i + 1) * na2 - 1) += val * T2_ab;
                }
            }
        }

        //: Pi_ba_2 & Pi_bb_2
        for (size_t mu = 0; mu < nao; ++mu) {
            #pragma omp single
            {
                compute_intermediate_T(
                    pair.block2.E_b, b2.n_occ_a, b2.n_occ_b,
                    b2.sigma_aa, b2.sigma_ba,
                    mu, b2.n_occ_a, b2.n_occ_b,
                    T2_ba
                );
                compute_intermediate_T(
                    pair.block2.E_b, b2.n_occ_a, b2.n_occ_b,
                    b2.sigma_ab, b2.sigma_bb,
                    mu, b2.n_occ_a, b2.n_occ_b,
                    T2_bb
                );
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < nb1; ++i) {
                double val = CbT_S(i, mu);
                if (std::abs(val) > 1e-12) {
                    pi_ba_2.rows(i * nb2, (i + 1) * nb2 - 1) += val * T2_ba;
                    pi_bb_2.rows(i * nb2, (i + 1) * nb2 - 1) += val * T2_bb;
                }
            }
        }

    }

    pair.pi_aa_1 = pi_aa_1;
    pair.pi_ab_1 = pi_ab_1;
    pair.pi_aa_2 = pi_aa_2;
    pair.pi_ab_2 = pi_ab_2;
    pair.pi_ba_1 = pi_ba_1;
    pair.pi_bb_1 = pi_bb_1;
    pair.pi_ba_2 = pi_ba_2;
    pair.pi_bb_2 = pi_bb_2;

    DBG("===== pi_matrix END (Optimized & Fixed) =====");
}

static mat compute_k_block(
    const mat& sigma_aa, const mat& sigma_ba,
    const mat& pi_cross, const mat& pi_diag,
    const mat& E_block,
    const vec& scale_vec,
    const vec& u_vec,
    const vec& w_pi_cross,
    const vec& w_pi_diag,
    size_t n_occ_top, size_t n_occ_bot,
    size_t n_vir, size_t n_occ
) {

    vec e_top = E_block.rows(0, n_occ_top - 1) * u_vec;

    vec e_bot;
    if (n_occ_bot > 0) {
        e_bot = E_block.rows(n_occ_top, n_occ_top + n_occ_bot - 1) * u_vec;
    }


    vec w_aa = kron(scale_vec, e_top);

    rowvec res_vec = w_aa.t() * sigma_aa;

    if (n_occ_bot > 0) {
        vec w_ba = kron(scale_vec, e_bot);
        res_vec += w_ba.t() * sigma_ba;
    }

    res_vec -= w_pi_cross.t() * pi_cross;

    res_vec -= w_pi_diag.t() * pi_diag;

    return reshape(res_vec, n_vir, n_occ);
}

void spin_adiabatic_state::k_matrix_null(OrbitalPair& pair)
{
    pi_matrix(pair);

    DBG("===== k_matrix_null START (Optimized) =====");

    MOpair& b1 = S1_orthonal;
    MOpair& b2 = S2_orthonal;
    size_t n_ao = b1.n_ao;
    mat k_a_1(b1.n_vir_a, b1.n_occ_a, fill::zeros);
    mat k_b_1(b1.n_vir_b, b1.n_occ_b, fill::zeros);
    mat k_a_2(b2.n_vir_a, b2.n_occ_a, fill::zeros);
    mat k_b_2(b2.n_vir_b, b2.n_occ_b, fill::zeros);

    vec lambda_a_inv = 1.0 / pair.lambda_a;
    vec lambda_b_inv = 1.0 / pair.lambda_b;
    mat Sooinva = pair.V_a * diagmat(lambda_a_inv) * pair.U_a.t();
    mat Sooinvb = pair.V_b * diagmat(lambda_b_inv) * pair.U_b.t();

    mat L_vsocxy = L_AO.slice(0) + L_AO.slice(1);
    vec tem_L_psi2 = L_vsocxy * pair.psi2;
    vec tem_psi1_L = (pair.psi1.t() * L_vsocxy).t();

    vec U_null_b   = pair.U_b.tail_cols(1);
    vec V_null_a   = pair.V_a.tail_cols(1);
    vec C1bT_L_psi2 = pair.C1_flipped_beta.t() * tem_L_psi2;
    vec psi1_L_C2a = pair.C2_flipped_alpha.t() * tem_psi1_L;

    vec tem_psi1_L_C2a_vec = (tem_psi1_L.t() * pair.C2_flipped_alpha).t();

    vec tem_Sooinvb_C1bT_L_psi2 = Sooinvb * C1bT_L_psi2;
    vec tem_psi1_L_C2a_Sooinva = (tem_psi1_L_C2a_vec.t() * Sooinva).t();

    vec w_term2_b1b2 = kron(U_null_b, tem_Sooinvb_C1bT_L_psi2);

    vec w_term3_a1a2 = kron(tem_psi1_L_C2a_Sooinva, V_null_a);

    #pragma omp parallel sections
    {
        // ============ K^α (block1) ============
        #pragma omp section
        {
            // Term 1 specific: Scale=tem_L_psi2, U=U_null_b, E=block1.E_b
            // Term 2 specific: Pi = pi_ba_1 (indices ip*nb2+j)
            // Term 3 specific: Pi = pi_aa_1 (indices ip*na2+j)
            k_a_1 = compute_k_block(
                b1.sigma_aa, b1.sigma_ba,
                pair.pi_ba_1, pair.pi_aa_1,
                pair.block1.E_b,
                tem_L_psi2, U_null_b,
                w_term2_b1b2, w_term3_a1a2,
                b1.n_occ_a, b1.n_occ_b,
                b1.n_vir_a, b1.n_occ_a
            );
        }

        // ============ K^β (block1) ============
        #pragma omp section
        {
            k_b_1 = compute_k_block(
                b1.sigma_ab, b1.sigma_bb,
                pair.pi_bb_1, pair.pi_ab_1,
                pair.block1.E_b,
                tem_L_psi2, U_null_b,
                w_term2_b1b2, w_term3_a1a2,
                b1.n_occ_a, b1.n_occ_b,
                b1.n_vir_b, b1.n_occ_b
            );
        }

        // ============ K'^α (block2) ============
        #pragma omp section
        {
            // Term 1 specific: Scale=tem_psi1_L, U=V_null_a, E=block2.E_b
            // Term 2: Pi = pi_ba_2
            // Term 3: Pi = pi_aa_2
            k_a_2 = compute_k_block(
                b2.sigma_aa, b2.sigma_ba,
                pair.pi_ba_2, pair.pi_aa_2,
                pair.block2.E_b,
                tem_psi1_L, V_null_a,
                w_term2_b1b2, w_term3_a1a2,
                b2.n_occ_a, b2.n_occ_b,
                b2.n_vir_a, b2.n_occ_a
            );
        }

        // ============ K'^β (block2) ============
        #pragma omp section
        {
            k_b_2 = compute_k_block(
                b2.sigma_ab, b2.sigma_bb,
                pair.pi_bb_2, pair.pi_ab_2,
                pair.block2.E_b,
                tem_psi1_L, V_null_a,
                w_term2_b1b2, w_term3_a1a2,
                b2.n_occ_a, b2.n_occ_b,
                b2.n_vir_b, b2.n_occ_b
            );
        }
    }

    DBG("||k_a_1|| = " << norm(k_a_1, "fro"));
    DBG("||k_b_1|| = " << norm(k_b_1, "fro"));
    DBG("||k_a_2|| = " << norm(k_a_2, "fro"));
    DBG("||k_b_2|| = " << norm(k_b_2, "fro"));


    pair.L_a_1 = k_a_1 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_aa_1 + Sooinvb * pair.pi_ba_1);
    pair.L_b_1 = k_b_1 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_ab_1 + Sooinvb * pair.pi_bb_1);
    pair.L_a_2 = k_a_2 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_aa_2 + Sooinvb * pair.pi_ba_2);
    pair.L_b_2 = k_b_2 * scaling_factor + (pair.vsoc_x +pair.vsoc_y) * (Sooinva * pair.pi_ab_2 + Sooinvb * pair.pi_bb_2);

    DBG("===== k_matrix_null END (Optimized) =====");
}


static uvec get_strided_indices(size_t stride, size_t count) {
    uvec indices(count);
    for(size_t i = 0; i < count; ++i) {
        indices(i) = (i + 1) * stride - 1;
    }
    return indices;
}


static vec compute_sigma_weight(const mat& sigma_full, const vec& c_vec, const uvec& row_indices) {

    mat sub_sigma = sigma_full.rows(row_indices);

    return (c_vec.t() * sub_sigma).t();
}


static mat compute_k_last_block(
    const mat& sigma_top, const mat& sigma_bot,
    const mat& pi_matrix,
    const mat& E_block,
    const vec& scale_vec_t1,
    const vec& u_vec_t1,
    const vec& w_vec_t2,
    const vec& w_vec_t3,
    size_t n_occ_top, size_t n_occ_bot,
    size_t n_vir, size_t n_occ
) {

    vec e_top = E_block.rows(0, n_occ_top - 1) * u_vec_t1;
    vec w_top = kron(scale_vec_t1, e_top);

    rowvec res_vec = w_top.t() * sigma_top;

    if (n_occ_bot > 0) {
        vec e_bot = E_block.rows(n_occ_top, n_occ_top + n_occ_bot - 1) * u_vec_t1;
        vec w_bot = kron(scale_vec_t1, e_bot);
        res_vec += w_bot.t() * sigma_bot;
    }


    res_vec += w_vec_t2.t() * pi_matrix;


    res_vec += w_vec_t3.t() * pi_matrix;


    return reshape(res_vec, n_vir, n_occ);
}

void spin_adiabatic_state::k_matrix_last(OrbitalPair& pair)
{
    DBG("Notice: For Ms<0, the alpha-beta should exchange here:===== k_matrix_last START (Optimized) =====");

    pi_matrix(pair);

    MOpair b1 = S1_orthonal;
    MOpair b2 = S2_orthonal;
    size_t n_ao = b1.n_ao;

    mat k_aa_1(b1.n_vir_a, b1.n_occ_a, fill::zeros);
    mat k_ba_1(b1.n_vir_a, b1.n_occ_a, fill::zeros);
    mat k_ab_1(b1.n_vir_b, b1.n_occ_b, fill::zeros);
    mat k_bb_1(b1.n_vir_b, b1.n_occ_b, fill::zeros);

    mat k_aa_2(b2.n_vir_a, b2.n_occ_a, fill::zeros);
    mat k_ba_2(b2.n_vir_a, b2.n_occ_a, fill::zeros);
    mat k_ab_2(b2.n_vir_b, b2.n_occ_b, fill::zeros);
    mat k_bb_2(b2.n_vir_b, b2.n_occ_b, fill::zeros);


    mat L_vsocz = L_AO.slice(2);
    vec tem_L_psi2a = L_vsocz * pair.psi2_alpha;
    vec tem_psi1aT_L = (pair.psi1_alpha.t() * L_vsocz).t();
    vec tem_L_psi2b = L_vsocz * pair.psi2_beta;
    vec tem_psi1bT_L = (pair.psi1_beta.t() * L_vsocz).t();

    vec U_last_b = pair.U_b.tail_cols(1);
    vec V_last_a = pair.V_a.tail_cols(1);
    vec U_last_a = pair.U_a.tail_cols(1);
    vec V_last_b = pair.V_b.tail_cols(1);


    vec C1bT_L_psi2b = pair.C1_flipped_beta.t() * tem_L_psi2b;
    vec C1aT_L_psi2a = pair.C1_flipped_alpha.t() * tem_L_psi2a;

    vec psi1a_L_C2a = (pair.psi1_alpha.t() * L_vsocz * pair.C2_flipped_alpha).t();
    vec psi1b_L_C2b = (pair.psi1_beta.t() * L_vsocz * pair.C2_flipped_beta).t();


    MOpair temblocka, temblockb;
    temblocka.U = pair.U_a;
    temblocka.V = pair.V_a;
    temblockb.U = pair.U_b;
    temblockb.V = pair.V_b;
    temblocka.lambda = pair.lambda_a;
    temblockb.lambda = pair.lambda_b;
    temblocka.n_ao = n_ao;
    temblockb.n_ao = n_ao;

    temblocka.n_occ_a = pair.n_occ_a1;
    temblocka.n_occ_b = pair.n_occ_a2;
    temblocka.n_svd = pair.lambda_a.size();
    temblockb.n_occ_a = pair.n_occ_b1;
    temblockb.n_occ_b = pair.n_occ_b2;
    temblockb.n_svd = pair.lambda_b.size();

    pair.sigma_ua = sigma_u(temblocka);
    pair.sigma_va = sigma_v(temblocka);
    pair.sigma_ub = sigma_u(temblockb);
    pair.sigma_vb = sigma_v(temblockb);


    uvec idx_ua = get_strided_indices(pair.n_occ_a1, pair.n_occ_a1);
    uvec idx_va = get_strided_indices(pair.n_occ_a2, pair.n_occ_a2);
    uvec idx_ub = get_strided_indices(pair.n_occ_b1, pair.n_occ_b1);
    uvec idx_vb = get_strided_indices(pair.n_occ_b2, pair.n_occ_b2);


    vec w_ua = compute_sigma_weight(pair.sigma_ua, C1aT_L_psi2a, idx_ua);
    vec w_va = compute_sigma_weight(pair.sigma_va, psi1a_L_C2a,  idx_va);
    vec w_ub = compute_sigma_weight(pair.sigma_ub, C1bT_L_psi2b, idx_ub);
    vec w_vb = compute_sigma_weight(pair.sigma_vb, psi1b_L_C2b,  idx_vb);


    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // K_aa_1
            k_aa_1 = compute_k_last_block(
                b1.sigma_aa, b1.sigma_ba, pair.pi_aa_1,
                pair.block1.E_a, tem_L_psi2a, U_last_a,
                w_ua, w_va,
                b1.n_occ_a, b1.n_occ_b, b1.n_vir_a, b1.n_occ_a
            );
        }
        #pragma omp section
        {
            // K_ba_1
            k_ba_1 = compute_k_last_block(
                b1.sigma_aa, b1.sigma_ba, pair.pi_ba_1,
                pair.block1.E_b, tem_L_psi2b, U_last_b,
                w_ub, w_vb,
                b1.n_occ_a, b1.n_occ_b, b1.n_vir_a, b1.n_occ_a
            );
        }
        #pragma omp section
        {
            // K_ab_1
            k_ab_1 = compute_k_last_block(
                b1.sigma_ab, b1.sigma_bb, pair.pi_ab_1,
                pair.block1.E_a, tem_L_psi2a, U_last_a,
                w_ua, w_va,
                b1.n_occ_a, b1.n_occ_b, b1.n_vir_b, b1.n_occ_b
            );
        }
        #pragma omp section
        {
            // K_bb_1
            k_bb_1 = compute_k_last_block(
                b1.sigma_ab, b1.sigma_bb, pair.pi_bb_1,
                pair.block1.E_b, tem_L_psi2b, U_last_b,
                w_ub, w_vb, // Weights for pi_bb
                b1.n_occ_a, b1.n_occ_b, b1.n_vir_b, b1.n_occ_b
            );
        }

        #pragma omp section
        {
            // K_aa_2
            k_aa_2 = compute_k_last_block(
                b2.sigma_aa, b2.sigma_ba, pair.pi_aa_2,
                pair.block2.E_a, tem_psi1aT_L, V_last_a,
                w_ua, w_va,
                b2.n_occ_a, b2.n_occ_b, b2.n_vir_a, b2.n_occ_a
            );
        }
        #pragma omp section
        {
            // K_ba_2
            k_ba_2 = compute_k_last_block(
                b2.sigma_aa, b2.sigma_ba, pair.pi_ba_2,
                pair.block2.E_b, tem_psi1bT_L, V_last_b,
                w_ub, w_vb,
                b2.n_occ_a, b2.n_occ_b, b2.n_vir_a, b2.n_occ_a
            );
        }
        #pragma omp section
        {
            // K_ab_2
            k_ab_2 = compute_k_last_block(
                b2.sigma_ab, b2.sigma_bb, pair.pi_ab_2,
                pair.block2.E_a, tem_psi1aT_L, V_last_a,
                w_ua, w_va,
                b2.n_occ_a, b2.n_occ_b, b2.n_vir_b, b2.n_occ_b
            );
        }
        #pragma omp section
        {
            // K_bb_2
            k_bb_2 = compute_k_last_block(
                b2.sigma_ab, b2.sigma_bb, pair.pi_bb_2,
                pair.block2.E_b, tem_psi1bT_L, V_last_b,
                w_ub, w_vb,
                b2.n_occ_a, b2.n_occ_b, b2.n_vir_b, b2.n_occ_b
            );
        }
    }


    DBG("||K_aa_1|| = " << norm(k_aa_1, "fro"));
    DBG("||K_ba_1|| = " << norm(k_ba_1, "fro"));
    DBG("||K_ab_1|| = " << norm(k_ab_1, "fro"));
    DBG("||K_bb_1|| = " << norm(k_bb_1, "fro"));
    DBG("||K_aa_2|| = " << norm(k_aa_2, "fro"));
    DBG("||K_ba_2|| = " << norm(k_ba_2, "fro"));
    DBG("||K_ab_2|| = " << norm(k_ab_2, "fro"));
    DBG("||K_bb_2|| = " << norm(k_bb_2, "fro"));


    mat Vca = pair.V_a.cols(0, pair.lambda_a.size() - 1);
    mat Vcb = pair.V_b.cols(0, pair.lambda_b.size() - 1);
    mat Uca = pair.U_a.cols(0, pair.lambda_a.size() - 1);
    mat Ucb = pair.U_b.cols(0, pair.lambda_b.size() - 1);

    vec lambda_a_m1_inv = 1.0 / pair.lambda_a;
    vec lambda_b_m1_inv = 1.0 / pair.lambda_b;
    lambda_a_m1_inv(lambda_a_m1_inv.n_elem - 1) = 0.0;
    lambda_b_m1_inv(lambda_b_m1_inv.n_elem - 1) = 0.0;

    mat Sooinva = Vca * diagmat(lambda_a_m1_inv) * Uca.t();
    mat Sooinvb = Vcb * diagmat(lambda_b_m1_inv) * Ucb.t();


    pair.L_aa_1 = k_aa_1 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_aa_1 + Sooinvb * pair.pi_ba_1);
    pair.L_ba_1 = k_ba_1 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_ba_1 + Sooinva * pair.pi_aa_1);
    pair.L_ab_1 = k_ab_1 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_ab_1 + Sooinvb * pair.pi_bb_1);
    pair.L_bb_1 = k_bb_1 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_bb_1 + Sooinva * pair.pi_ab_1);
    pair.L_aa_2 = k_aa_2 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_aa_2 + Sooinvb * pair.pi_ba_2);
    pair.L_ba_2 = k_ba_2 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_ba_2 + Sooinva * pair.pi_aa_2);
    pair.L_ab_2 = k_ab_2 * scaling_factor + (pair.val_a) * (Sooinva * pair.pi_ab_2 + Sooinvb * pair.pi_bb_2);
    pair.L_bb_2 = k_bb_2 * scaling_factor + (pair.val_b) * (Sooinvb * pair.pi_bb_2 + Sooinva * pair.pi_ab_2);

    DBG("===== k_matrix_last END (Optimized) =====");
}







void spin_adiabatic_state::gradient_implicit_rhs_Ms()
{
   S_MO_alpha = C1_alpha.t() * AOS * C2_alpha;
   S_MO_beta  = C1_beta.t() * AOS * C2_beta;
   y1_vo_alpha = zeros<mat>(nvir1_a, nalpha1);
   y1_vo_beta  = zeros<mat>(nvir1_b, nbeta1);
   y2_ov_alpha = zeros<mat>(nvir2_a, nalpha2); // (D+S)*V
   y2_ov_beta  = zeros<mat>(nvir2_b,  nbeta2); // D*(S+V)
   DBG("Zexuan Wei gradient_implicit_rhs start");
   DBG("y1_vo_alpha size = " << y1_vo_alpha.n_rows << " x " << y1_vo_alpha.n_cols);
   DBG("y1_vo_beta size = " << y1_vo_beta.n_rows << " x " << y1_vo_beta.n_cols);
   DBG("y2_ov_alpha size = " << y2_ov_alpha.n_rows << " x " << y2_ov_alpha.n_cols);
   DBG("y2_ov_beta size = " << y2_ov_beta.n_rows << " x " << y2_ov_beta.n_cols);
   sigma_matrix(S1_orthonal);
   sigma_matrix(S2_orthonal);
   //sigma_matrix_test(S1_orthonal);
   for (double Ms1 = -S1; Ms1 <= S1; Ms1 += 1.0) {
      for (int dir = 0; dir < 3; ++dir) {
         double delta_Ms = (dir == 0) ? +1 :
                           (dir == 1) ? -1 : 0;
         double Ms2 = Ms1 + delta_Ms;

         int idx = get_index(Ms1, Ms2);


         auto& pair_list = vsoc_pairs[idx];
         cout << "Processing Ms1 = " << Ms1 << ", Ms2 = " << Ms2
               << ", direction = " << dir << ", num pairs = " << pair_list.size() << endl;

         for (auto& pair : pair_list) {;

            if (dir != 2){
               k_matrix_null(pair);
               int vsoc_idx = get_index(Ms1, Ms2);
               DBG("pair.L_a_1 size = " << pair.L_a_1.n_rows << " x " << pair.L_a_1.n_cols);
               DBG("pair.L_b_1 size = " << pair.L_b_1.n_rows << " x " << pair.L_b_1.n_cols);
               DBG("pair.L_a_2 size = " << pair.L_a_2.n_rows << " x " << pair.L_a_2.n_cols);
               DBG("pair.L_b_2 size = " << pair.L_b_2.n_rows << " x " << pair.L_b_2.n_cols);
               y1_vo_alpha += v_soc(vsoc_idx) * pair.L_a_1 * pair.phase ;
               y1_vo_beta  += v_soc(vsoc_idx) * pair.L_b_1 * pair.phase ;
               y2_ov_alpha += v_soc(vsoc_idx) * pair.L_a_2 * pair.phase ;
               y2_ov_beta  += v_soc(vsoc_idx) * pair.L_b_2 * pair.phase ;


            }
            else{
               k_matrix_last(pair);
               int vsoc_idx = get_index(Ms1, Ms2);
               y1_vo_alpha += v_soc(vsoc_idx) * (pair.L_aa_1 * pair.phase_alpha + pair.L_ba_1 * pair.phase_beta);
               y1_vo_beta += v_soc(vsoc_idx) *  (pair.L_ab_1 * pair.phase_alpha + pair.L_bb_1 * pair.phase_beta);
               y2_ov_alpha += v_soc(vsoc_idx) * (pair.L_aa_2 * pair.phase_alpha + pair.L_ba_2 * pair.phase_beta);
               y2_ov_beta += v_soc(vsoc_idx) *  (pair.L_ab_2 * pair.phase_alpha + pair.L_bb_2 * pair.phase_beta);
            }
         }
      }
   }

   y2_ov_alpha = y2_ov_alpha.t();
   y2_ov_beta = y2_ov_beta.t();
   cout << "Zexuan Wei gradient_implicit_rhs end" << endl;
   return;
}




vec spin_adiabatic_state::E_adiab_gradients()
{

   cout << "Zexuan Wei start Eadiab_gradients" << endl;

   vec deriv_slx = gradient_explicit_Ms();
   matrix_print_2d(deriv_slx.memptr(), 3, NAtoms, "Zexuan Wei soc gradient_explicit");



      // then orbital rotation response

   cout << "orbtype: ";
   if (unrestricted) 
      cout << myuhf_1->orbtype << " " << myuhf_2->orbtype << endl;
   else{
      if (nalpha1 == nbeta1) cout << myrhf_1->orbtype;
      else cout << myrohf_1->orbtype;
      cout << " " << myrohf_2->orbtype << endl; 
   }
   gradient_implicit_rhs_Ms();


   int pe_rootsTemp = rem_read(REM_SET_PE_ROOTS);
   rem_write(1, REM_SET_PE_ROOTS);

   // first low-spin state
   rem_write(nalpha1, REM_NALPHA);
   rem_write(nbeta1,  REM_NBETA);
   rem_write(0, REM_SET_SPIN);
   rem_write(1, REM_SET_3OR1); // singlet // setman_codes.h
   double *jPv = QAllocDouble(2*NB2+2*NB2car+1);
   jPv[2*NB2car+2*NB2] = 1.14514;
   ScaV2M(P1_alpha.memptr(), jPv, 1, 0);
   ScaV2M(P1_beta.memptr(),  jPv+NB2car, 1, 0);
   cout << "Zexuan Wei jPv test" << jPv[2*NB2+2*NB2car] << endl;
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_BEG,C1_alpha.memptr());
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_CUR,C1_beta.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,P1_alpha.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,P1_beta.memptr());
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_BEG,jPv);
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_CUR,jPv+NB2car);
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,F1_alpha.memptr());
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,F1_beta.memptr());
//   VRscale(jPv, NB2, 2.0);
//   vec deriv_o1 = gradient_one_electron(jPv, 1);
   vec deriv_imp_1;
   if (unrestricted)
      deriv_imp_1 = gradient_implicit_uhf(myscf_1, y1_vo_alpha.t(), y1_vo_beta.t(), C1_alpha, C1_beta);
   else{
      if (nalpha1 == nbeta1)
         deriv_imp_1 = gradient_implicit_rhf(myscf_1, y1_vo_alpha+y1_vo_beta, C1_alpha);
      else
         deriv_imp_1 = gradient_implicit_rohf(myscf_1, y1_vo_alpha.t(), y1_vo_beta.t(), C1_alpha);
   }




   // state 2
   rem_write(nalpha2, REM_NALPHA);
   rem_write(nbeta2,  REM_NBETA);

   rem_write(0, REM_SET_3OR1); // triplet
   rem_write(1, REM_SET_SPIN);

   ScaV2M(P2_alpha.memptr(), jPv, 1, 0);
   ScaV2M(P2_beta.memptr(),  jPv+NB2car, 1, 0);
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_BEG,C2_alpha.memptr());
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_CUR,C2_beta.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,P2_alpha.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,P2_beta.memptr());
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_BEG,jPv);
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_CUR,jPv+NB2car);
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,F2_alpha.memptr());
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,F2_beta.memptr());
//   vec deriv_o2 = gradient_one_electron(jPv, 2);
   vec deriv_imp_2;
   if (unrestricted){
      deriv_imp_2 = gradient_implicit_uhf(myscf_2, y2_ov_alpha, y2_ov_beta, C2_alpha, C2_beta);
   }
   else {
      deriv_imp_2 = gradient_implicit_rohf(myscf_2, y2_ov_alpha, y2_ov_beta, C2_alpha);
   }


   QFree(jPv);

   rem_write(pe_rootsTemp, REM_SET_PE_ROOTS);
   cout << "Zexuan finished Eadiab_gradients" << endl;

   // int t_spin = unrestricted? 2 : 3;
   // check_rohf_orbital_response(t_spin);

   double prefactor_v = - 2.0 / E_soc;
   deriv_slx *= prefactor_v;
   matrix_print_2d(deriv_slx.memptr(), 3, NAtoms, "zheng ASG soc slx");
   deriv_imp_1 *= prefactor_v;
   matrix_print_2d(deriv_imp_1.memptr(), 3, NAtoms, "zheng ASG soc imp_1");
   deriv_imp_2 *= prefactor_v;
   matrix_print_2d(deriv_imp_2.memptr(), 3, NAtoms, "zheng ASG soc imp_2");

   vec grad_soc = deriv_slx - deriv_imp_1 + deriv_imp_2;
   grad_soc /= prefactor_v;
   return grad_soc;


}

vec spin_adiabatic_state::Esoc_gradients()
{
   cout << "zheng start Esoc_gradients" << endl;
   // first comes orbital rotation free contributions
   vec deriv_slx = gradient_explicit();
   
   //vec deriv_slx = zeros<vec>(Nuclear);

   // then orbital rotation response
   cout << "orbtype: ";
   if (unrestricted) 
      cout << myuhf_1->orbtype << " " << myuhf_2->orbtype << endl;
   else{
      if (nalpha1 == nbeta1) cout << myrhf_1->orbtype;
      else cout << myrohf_1->orbtype;
      cout << " " << myrohf_2->orbtype << endl; 
   }

   // locate right-hand-side vectors
   // first low-spin state
   mat y1_vo_alpha = zeros<mat>(nvir1_a, nalpha1);
   mat y1_vo_beta  = zeros<mat>(nvir1_b, nbeta1);
   // second high-spin state
   mat y2_ov_alpha = zeros<mat>(nalpha2, nvir2_a); // (D+S)*V
   mat y2_ov_beta  = zeros<mat>(nbeta2,  nvir2_b); // D*(S+V)

   gradient_implicit_rhs(y1_vo_alpha, y1_vo_beta, y2_ov_alpha, y2_ov_beta);


   int pe_rootsTemp = rem_read(REM_SET_PE_ROOTS);
   rem_write(1, REM_SET_PE_ROOTS);

   // first low-spin state
   rem_write(nalpha1, REM_NALPHA);
   rem_write(nbeta1,  REM_NBETA);
   rem_write(0, REM_SET_SPIN);
   rem_write(1, REM_SET_3OR1); // singlet // setman_codes.h
   double *jPv = QAllocDouble(2*NB2+2*NB2car+1);
   jPv[2*NB2car+2*NB2] = 1.14514;
   ScaV2M(P1_alpha.memptr(), jPv, 1, 0);
   ScaV2M(P1_beta.memptr(),  jPv+NB2car, 1, 0);
   cout << "Zexuan Wei jPv test" << jPv[2*NB2+2*NB2car] << endl;
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_BEG,C1_alpha.memptr());
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_CUR,C1_beta.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,P1_alpha.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,P1_beta.memptr());
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_BEG,jPv);
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_CUR,jPv+NB2car);
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,F1_alpha.memptr());
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,F1_beta.memptr());
//   VRscale(jPv, NB2, 2.0);
//   vec deriv_o1 = gradient_one_electron(jPv, 1);

   vec deriv_imp_1;
   if (unrestricted)
      deriv_imp_1 = gradient_implicit_uhf(myscf_1, y1_vo_alpha.t(), y1_vo_beta.t(), C1_alpha, C1_beta);
   else{
      if (nalpha1 == nbeta1)
         deriv_imp_1 = gradient_implicit_rhf(myscf_1, y1_vo_alpha+y1_vo_beta, C1_alpha);
      else
         deriv_imp_1 = gradient_implicit_rohf(myscf_1, y1_vo_alpha.t(), y1_vo_beta.t(), C1_alpha);
   }

   // check_rohf_orbital_response(1);

   // second high-spin state
   rem_write(nalpha2, REM_NALPHA);
   rem_write(nbeta2,  REM_NBETA);
   rem_write(0, REM_SET_3OR1); // triplet
   if (unrestricted){
      //rem_write(1, REM_JUSTAL);
      rem_write(1, REM_SET_SPIN);
   }
   else {
      //rem_write(0, REM_JUSTAL);
      rem_write(1, REM_SET_SPIN);
   }
   ScaV2M(P2_alpha.memptr(), jPv, 1, 0);
   ScaV2M(P2_beta.memptr(),  jPv+NB2car, 1, 0);
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_BEG,C2_alpha.memptr());
   FileMan(FM_WRITE,FILE_MO_COEFS,FM_DP,NBas*NOrb,0,FM_CUR,C2_beta.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,P2_alpha.memptr());
   FileMan(FM_WRITE,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,P2_beta.memptr());
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_BEG,jPv);
   FileMan(FM_WRITE,FILE_SPARSE_DENSITY_MATRIX,FM_DP,NB2,0,FM_CUR,jPv+NB2car);
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,F2_alpha.memptr());
   FileMan(FM_WRITE,FILE_FOCK_MATRIX,FM_DP,NBas*NBas,0,FM_CUR,F2_beta.memptr());
//   vec deriv_o2 = gradient_one_electron(jPv, 2);
   vec deriv_imp_2;
   if (unrestricted){
      deriv_imp_2 = gradient_implicit_uhf(myscf_2, y2_ov_alpha, y2_ov_beta, C2_alpha, C2_beta);
   }
   else {
      deriv_imp_2 = gradient_implicit_rohf(myscf_2, y2_ov_alpha, y2_ov_beta, C2_alpha);
   }
   QFree(jPv);

   rem_write(pe_rootsTemp, REM_SET_PE_ROOTS);
   cout << "zheng finished Esoc_gradients" << endl;

   // int t_spin = unrestricted? 2 : 3;
   // check_rohf_orbital_response(t_spin);

   double prefactor_v = - 2.0 / E_soc;
   deriv_slx *= prefactor_v;
   matrix_print_2d(deriv_slx.memptr(), 3, NAtoms, "zheng ASG soc slx");
   deriv_imp_1 *= prefactor_v;
   matrix_print_2d(deriv_imp_1.memptr(), 3, NAtoms, "zheng ASG soc imp_1");
   deriv_imp_2 *= prefactor_v;
   matrix_print_2d(deriv_imp_2.memptr(), 3, NAtoms, "zheng ASG soc imp_2");

   vec grad_soc = deriv_slx - deriv_imp_1 + deriv_imp_2;
   grad_soc /= prefactor_v;
   return grad_soc;
}


//void spin_adiabatic_state::pseudo_density_explicit(mat &deriv_explicit_s, mat &deriv_explicit_l)
void spin_adiabatic_state::pseudo_density_explicit(mat &deriv_explicit_s, vector<mat> &deriv_explicit_l)
{
   mat tmp = C2_prime[0] * (C1_prime[0].t() * prod_s(0)); // for Lx and Ly

   deriv_explicit_l.push_back(tmp);

   tmp = zeros<mat>(NBas, NBas);
   const size_t nleft = nalpha2 - nbeta2;
   for (size_t j=0; j<2; ++j){ // alpha and beta electrons
      for (size_t i=0; i<nleft; ++i){ // orbitals
         tmp += C2_prime[1+j+i*2] * (C1_prime[1+j+i*2].t() * prod_s(1+j+i*2));
      }
   }
   deriv_explicit_l.push_back(tmp / (2 * sqrt(nleft))); // for Lz
   //matrix_print_2d(deriv_explicit_l.memptr(), deriv_explicit_l.n_rows, deriv_explicit_l.n_cols, "deriv_explicit_l");
   cout << "norm(deriv_explicit_l[0]) = " << norm(deriv_explicit_l[0], "fro") << endl;
   cout << "norm(deriv_explicit_l[1]) = " << norm(deriv_explicit_l[1], "fro") << endl;

   // S^-1 = CCT matrix
   S_inv = C1_alpha  * C1_alpha.t();

   tmp = zeros<mat>(NBas, NBas);
   for (size_t i=0; i<2; ++i){
//   for (size_t i=0; i<3; ++i){
      tmp += L_AO.slice(i) * v_soc(i);
   }

   // x and y components
   mat tmp2 = tmp * deriv_explicit_l[0];
   deriv_explicit_s =- S_inv * tmp2; // minus sign

   tmp2 = deriv_explicit_l[0] * tmp; // TODO: this should be -transpose
   deriv_explicit_s -= tmp2 * S_inv; // minus sign

   // z component
   tmp = L_AO.slice(2) * v_soc(2);

   tmp2 = tmp * deriv_explicit_l[1];
   deriv_explicit_s -= S_inv * tmp2; // alpha low-spin

   tmp2 = deriv_explicit_l[1] * tmp;
   deriv_explicit_s -= tmp2 * S_inv; // alpha high-spin
   //matrix_print_2d(deriv_explicit_s.memptr(), deriv_explicit_s.n_rows, deriv_explicit_s.n_cols, "deriv_explicit_s");
	cout << "norm(C2_prime[0]) = " << norm(C2_prime[0]) << ", norm(C1_prime[0].t()) = " << norm(C1_prime[0].t()) << ", phase = " << prod_s(0) << endl;
   cout << "norm(deriv_explicit_s) = " << norm(deriv_explicit_s, "fro") << endl;
   // svd parts for Sx are zeros
   // eigenvalues for Sx are zeros
   
   return;
}


vec spin_adiabatic_state::gradient_explicit()
{
   vec derivatives = zeros<vec>(Nuclear);

   // pseudo-densitis for Sx and Lx integrals
   //mat deriv_explicit_s, deriv_explicit_l;
   mat deriv_explicit_s;
   vector<mat> deriv_explicit_l;
   pseudo_density_explicit(deriv_explicit_s, deriv_explicit_l);


   // AO overlap derivatives
   int size_jSxv = NB2car*Nuclear*2+1;
   std::vector<double> jOrigin(3,0.0);
   std::vector<double> jSxv(size_jSxv,0.0);
   std::vector<double> jHx(size_jSxv,0.0);
   jSxv[NB2car*Nuclear*2] = 1.14514;
   
   MakeOv(NULL,1,-1,0,0,jOrigin.data(),jSxv.data(),jHx.data()); // need the jHx space!
   cout << "Zexuan Wei jSxv memory test:" << jSxv[NB2car*Nuclear*2] << endl;
   cout << "Zexuan Wei jSxv" << jSxv.data() << endl;
   cout << "zheng gradient_explicit 002 NBas: " << NBas << " NBas6D2: " << NBas6D2 << endl;

   // combine deriv_explicit_s with Sx
   mat dS_AO(NBas, NBas);
   for (size_t x=0; x<Nuclear; ++x){
      ScaV2M(dS_AO.memptr(), jSxv.data()+x*NB2car, 1, 1);
      derivatives(x) = dot(dS_AO, deriv_explicit_s);
   }
   

   derivatives *= 0.5;
   //matrix_print_2d(derivatives.memptr(), 3, NAtoms, "zheng gradient_explicit s");


   // AO soc derivatives in 3N, 3N, 3N order
   int size_jdL = Nuclear*NBas6D2;
   std::vector<double> jdL(3*size_jdL,0.0);
   getdL(jdL.data(), size_jdL);

   // combine deriv_explicit_l with Lx
   mat dL_AO;
   for (size_t x=0; x<Nuclear; ++x){
      dL_AO = zeros<mat>(NBas, NBas);
      for (size_t i=0; i<2; ++i){
         mat tem_mat(jdL.data()+NBas2*(i*Nuclear+x),NBas,NBas,true);
//      for (size_t i=0; i<3; ++i){
         dL_AO += tem_mat * v_soc(i);
         // matrix_print_2d(dL_AO.memptr(), NBas, NBas, "dL_AO");
      }
      // note the dL_AO is anti-symmetric
      derivatives(x) += dot(dL_AO, deriv_explicit_l[0]);

      // z component
      mat tem_mat_z(jdL.data()+NBas2*(2*Nuclear+x),NBas,NBas,true);
      dL_AO = tem_mat_z * v_soc(2);
      derivatives(x) += dot(dL_AO, deriv_explicit_l[1]);
   }
   cout << "Zexuan Wei gradient explicit end" << endl;
   //matrix_print_2d(derivatives.memptr(), 3, NAtoms, "zheng gradient_explicit l");
   return derivatives;
}


void spin_adiabatic_state::gradient_implicit_rhs(mat& y1_vo_alpha, mat& y1_vo_beta,
        mat& y2_ov_alpha, mat& y2_ov_beta)
{
   gradient_implicit_xy(y1_vo_alpha, y1_vo_beta, y2_ov_alpha, y2_ov_beta);
   gradient_implicit_z(y1_vo_alpha, y1_vo_beta, y2_ov_alpha, y2_ov_beta);

   //MatPrint(y1_vo_alpha.memptr(), y1_vo_alpha.n_rows, y1_vo_alpha.n_cols, "y1_vo_alpha", 4);
   //MatPrint(y1_vo_beta.memptr(), y1_vo_beta.n_rows, y1_vo_beta.n_cols, "y1_vo_beta", 4);
   //MatPrint(y2_ov_alpha.memptr(), y2_ov_alpha.n_rows, y2_ov_alpha.n_cols, "y2_ov_alpha", 4);
   //MatPrint(y2_ov_beta.memptr(), y2_ov_beta.n_rows, y2_ov_beta.n_cols, "y2_ov_beta", 4);
   cout << "Zexuan Wei gradient_implicit_rhs end" << endl;
   return;
}



void spin_adiabatic_state::gradient_implicit_xy(mat& y1_vo_alpha, mat& y1_vo_beta,
        mat& y2_ov_alpha, mat& y2_ov_beta)
{
   // TODO: should put these in class later
   mat S_MO_alpha = C1_alpha.t() * AOS * C2_alpha;
   mat S_MO_beta  = C1_beta.t() * AOS * C2_beta;

   mat s_vo_a = S_MO_alpha.submat(nalpha1, 0, NOrb-1, nalpha2-1);
   mat s_vo_b = S_MO_beta.submat(nbeta1, 0, NOrb-1, nbeta2-1);
   mat s_ov_a = S_MO_alpha.submat(0, nalpha2, nalpha1-1, NOrb-1);
   mat s_ov_b = S_MO_beta.submat(0, nbeta2, nbeta1-1, NOrb-1);

   // svd matrix inverse S^-1 = V 1/d U^T
   // alpha
   mat s_oo_inv_a = zeros<mat>(nalpha2, nalpha1);
   for (size_t k=0; k<nalpha1; ++k){
      s_oo_inv_a += (V[0].col(k) / lambda[0](k)) * U[0].col(k).t();
      //s_oo_inv_a += Va.col(k) * (Ua.col(k).t() / lambda_a(k));
   }
   //s_oo_inv_a = pinv(S_MO_alpha.submat(0, 0, nalpha1-1, nalpha2-1));
   //MatPrint(s_oo_inv_a.memptr(), s_oo_inv_a.n_rows, s_oo_inv_a.n_cols, "s_oo_inv_a");
   mat test_iden = s_oo_inv_a * S_MO_alpha.submat(0, 0, nalpha1-1, nalpha2-1);
   //MatPrint(test_iden.memptr(), test_iden.n_rows, test_iden.n_cols, "test_iden");
   test_iden = S_MO_alpha.submat(0, 0, nalpha1-1, nalpha2-1) * s_oo_inv_a;
   //MatPrint(test_iden.memptr(), test_iden.n_rows, test_iden.n_cols, "test_iden");

   // beta
   mat s_oo_inv_b = zeros<mat>(nbeta2, nbeta1);
   for (size_t k=0; k<nbeta2; ++k){
      s_oo_inv_b += V[1].col(k) * (U[1].col(k).t() / lambda[1](k));
   }
   //MatPrint(s_oo_inv_b.memptr(), s_oo_inv_b.n_rows, s_oo_inv_b.n_cols, "s_oo_inv_b");
   test_iden = s_oo_inv_b * S_MO_beta.submat(0, 0, nbeta1-1, nbeta2-1);
   //MatPrint(test_iden.memptr(), test_iden.n_rows, test_iden.n_cols, "test_iden");
   test_iden = S_MO_beta.submat(0, 0, nbeta1-1, nbeta2-1) * s_oo_inv_b;
   //MatPrint(test_iden.memptr(), test_iden.n_rows, test_iden.n_cols, "test_iden");

   //const double v_soc_square = dot(v_soc, v_soc);
   double v_soc_square = v_soc(0)*v_soc(0) + v_soc(1)*v_soc(1);

   mat tmp = zeros<mat>(NBas, NBas);
   for (size_t i=0; i<2; ++i){
   //for (size_t i=0; i<3; ++i){
      tmp += L_AO.slice(i) * (v_soc(i) * prod_s(0));
   }
   
   mat L_MO_tot = C1_beta.t() * tmp * C2_alpha;
   mat l_mo_oo = L_MO_tot.submat(0, 0, nbeta1-1, nalpha2-1);
   mat l_mo_ov = L_MO_tot.submat(0, nalpha2, nbeta1-1, NOrb-1);
   mat l_mo_vo = L_MO_tot.submat(nbeta1, 0, NOrb-1, nalpha2-1);

   const mat VUt = V[0].tail_cols(1) * U[1].tail_cols(1).t();

   // orbital response from orbital derivative
   // low-spin state
   y1_vo_beta  += l_mo_vo * VUt; // beta electron
   // high-spin state
   y2_ov_alpha += VUt * l_mo_ov; // alpha electron


   // determinant derivatives
   // low-spin state
   tmp = s_vo_a * s_oo_inv_a; // alpha electron
   y1_vo_alpha += tmp * v_soc_square;
   tmp = s_vo_b * s_oo_inv_b; // beta electron
   y1_vo_beta += tmp * v_soc_square;

   // high-spin state
   tmp = s_oo_inv_a * s_ov_a; // alpha electron
   y2_ov_alpha += tmp * v_soc_square; // beta electron

   tmp = s_oo_inv_b * s_ov_b;
   y2_ov_beta += tmp * v_soc_square;


   // U and V single column derivatives
   tmp = VUt * l_mo_oo;
   mat vul = tmp * s_oo_inv_a;

   tmp = l_mo_oo * VUt;
   mat lvu = s_oo_inv_b * tmp;

   // low-spin state
   y1_vo_alpha -= s_vo_a * vul; // alpha electron
   y1_vo_beta -= s_vo_b * lvu; // beta electron

   // high-spin state
   y2_ov_alpha -= vul * s_ov_a; // alpha electron
   y2_ov_beta -= lvu * s_ov_b; // beta electron

   return;
}


void spin_adiabatic_state::gradient_implicit_z(mat& y1_vo_alpha, mat& y1_vo_beta,
        mat& y2_ov_alpha, mat& y2_ov_beta)
{  // z component
   double v_soc_square = v_soc(2)*v_soc(2);
   const mat l_ao_t   = L_AO.slice(2) * v_soc(2);
   const mat CL_alpha = C1_alpha.t()  * l_ao_t;
   const mat CL_beta  = C1_beta.t()   * l_ao_t;

   const mat CS_alpha = C1_alpha.t() * AOS;
   const mat CS_beta  = C1_beta.t()  * AOS;

   const size_t nleft = nalpha2 - nbeta2;
   for (size_t i=0; i<nleft; ++i){ // orbitals
      mat y1_a = zeros<mat>(nvir1_a, nalpha1);
      mat y1_b = zeros<mat>(nvir1_b, nbeta1);
      mat y2_a = zeros<mat>(nalpha1, nvir2_a);
      mat y2_b = zeros<mat>(nbeta1,  nvir2_b);

      size_t j = nbeta2 + 1; //indices(i);
      mat C2_alpha_t = C2_alpha_new;
      C2_alpha_t.shed_col(j);
      mat C2_beta_t = join_rows(C2_beta_new, C2_alpha_new.col(j));

      mat s_vo_a = CS_alpha.tail_rows(nvir1_a) * C2_alpha_t;
      mat s_vo_b = CS_beta.tail_rows(nvir1_b)  * C2_beta_t;
      mat s_ov_a = CS_alpha.head_rows(nalpha1) * C2_alpha.tail_cols(nvir2_a);
      mat s_ov_b = CS_beta.head_rows(nbeta1)   * C2_beta.tail_cols(nvir2_b);

      // svd matrix inverse S^-1 = V 1/d U^T
      // alpha
      mat s_oo_inv_a = zeros<mat>(nalpha1, nalpha1);
      for (size_t k=0; k<nalpha1-1; ++k){
         s_oo_inv_a += (V[3+i*2].col(k) / lambda[3+i*2](k)) * U[3+i*2].col(k).t();
      }
      // beta
      mat s_oo_inv_b = zeros<mat>(nbeta1, nbeta1);
      for (size_t k=0; k<nbeta1-1; ++k){
         s_oo_inv_b += V[4+i*2].col(k) * (U[4+i*2].col(k).t() / lambda[4+i*2](k));
      }


      mat l_mo_oo_alpha = CL_alpha.head_rows(nalpha1) * C2_alpha_t;
      mat l_mo_ov_alpha = CL_alpha.head_rows(nalpha1) * C2_alpha.tail_cols(nvir2_a);
      mat l_mo_vo_alpha = CL_alpha.tail_rows(nvir1_a) * C2_alpha_t;
      mat l_mo_oo_beta  = CL_beta.head_rows(nbeta1)   * C2_beta_t;
      mat l_mo_ov_beta  = CL_beta.head_rows(nbeta1)   * C2_beta.tail_cols(nvir2_b);
      mat l_mo_vo_beta  = CL_beta.tail_rows(nvir1_b)  * C2_beta_t;

      mat VUt_alpha = (prod_s(1+i*2) * V[3+i*2].tail_cols(1)) * U[3+i*2].tail_cols(1).t();
      mat VUt_beta = (prod_s(2+i*2) * V[4+i*2].tail_cols(1)) * U[4+i*2].tail_cols(1).t();

      // orbital response from orbital derivative
      // low-spin state
      y1_a += l_mo_vo_alpha * VUt_alpha; // alpha electron
      y1_b -= l_mo_vo_beta  * VUt_beta; // beta electron
      // high-spin state
      y2_a += VUt_alpha * l_mo_ov_alpha; // alpha electron
      y2_b -= VUt_beta  * l_mo_ov_beta;  // beta electron


      // determinant derivatives
      // low-spin state
      mat tmp = s_vo_a * s_oo_inv_a; // alpha electron
      y1_a += tmp * v_soc_square;
      tmp = s_vo_b * s_oo_inv_b; // beta electron
      y1_b += tmp * v_soc_square;

      // high-spin state
      tmp = s_oo_inv_a * s_ov_a; // alpha electron
      y2_a += tmp * v_soc_square; // beta electron

      tmp = s_oo_inv_b * s_ov_b;
      y2_b += tmp * v_soc_square;


      // U and V single column derivatives
      tmp = VUt_alpha * l_mo_oo_alpha;
      mat vul = tmp * s_oo_inv_a;

      tmp = l_mo_oo_alpha * VUt_alpha;
      mat lvu = s_oo_inv_a * tmp;

      // low-spin state
      y1_a -= s_vo_a * (lvu + vul); // alpha electron

      // high-spin state
      y2_a -= (lvu + vul) * s_ov_a; // alpha electron


      // U and V single column derivatives
      tmp = VUt_beta * l_mo_oo_beta;
      vul = tmp * s_oo_inv_b;

      tmp = l_mo_oo_beta * VUt_beta;
      lvu = s_oo_inv_b * tmp;

      // low-spin state
      y1_b += s_vo_b * (lvu + vul); // beta electron

      // high-spin state
      y2_b += (lvu + vul) * s_ov_b; // beta electron

      //cout << "dimension: " << size(y1_a) << " " << size(y2_a) << " " << size(y2_b) << "    " << size(y1_vo_alpha) << " " << size(y2_ov_alpha) << " " << size(y2_ov_beta) << endl;
      y1_vo_alpha += y1_a;
      y1_vo_beta += y1_b;
      y2_a.insert_rows(j-1, y2_b.row(nbeta2));
      y2_ov_alpha += y2_a;
      y2_ov_beta  += y2_b.head_rows(nbeta2);
   }


   // derivatives of SVD eigenvectors of second-state alpha-beta overlap
   if (unrestricted){
      const mat L_mo_alpha = CL_alpha.head_rows(nalpha1) * C2_alpha.head_cols(nalpha2);
      const mat L_mo_beta  = CL_beta.head_rows(nbeta1) * C2_beta.head_cols(nbeta2);
      const mat L_mo_ba    = CL_beta.head_rows(nbeta1) * C2_alpha.head_cols(nalpha2);

      // alpha
      mat proj        = eye<mat>(nalpha2, nalpha2) - (U[2] * U[2].t());
      const mat l_a_1 = L_mo_alpha * U[2];
      const mat l_a_2 = L_mo_alpha * proj;
      // beta
      proj            = eye<mat>(nbeta2, nbeta2) - (V[2] * V[2].t());
      const mat l_b_1 = L_mo_beta * V[2];
      const mat l_b_2 = L_mo_beta * proj;

      // beta-alpha
      mat s_oo_inv = zeros<mat>(nbeta2, nalpha2);
      for (size_t k=0; k<nbeta2; ++k){
         s_oo_inv += V[2].col(k) * (U[2].col(k) / lambda[2](k));
      }
      const mat ls_u_a = L_mo_alpha * s_oo_inv.t();
      const mat ls_u_b = L_mo_ba * s_oo_inv.t();


      mat A = zeros<mat>(nbeta2, nalpha2);
      for (size_t i=0; i<nleft; ++i){ // orbitals
         size_t j = nbeta2 + i;

         // alpha
         mat tmp_a   = (prod_s(1+i*2) * V[3+i*2].tail_cols(1)) * U[3+i*2].tail_cols(1).t();
         mat tmp_u_1 = tmp_a * l_a_1;
         mat tmp_u_2 = tmp_a * l_a_2;
         // beta
         mat tmp_b   = (prod_s(2+i*2) * V[4+i*2].tail_cols(1)) * U[4+i*2].tail_cols(1).t();
         mat tmp_v_1 = tmp_b * l_b_1;
         mat tmp_v_2 = tmp_b * l_b_2;

         for (size_t k1=0; k1<nbeta2; ++k1){
            double ek1 = lambda[2][k1];
            // U response for alpha
            A += V[2].col(k1) * (tmp_u_2.row(k1) / ek1);
            // V response for beta
            A += (tmp_v_2.row(k1).t() / ek1) * U[2].col(k1).t();

            for (size_t k=0; k<nbeta2; ++k){
               if (k1 != k){
                  double ek = lambda[2][k];
                  double dot = (ek1*ek1 - ek*ek);

                  mat tmp_VUjk = V[2].col(k1) * U[2].col(k).t();
                  mat tmp_VUkj = V[2].col(k) * U[2].col(k1).t();

                  // U response for alpha
                  double dota = tmp_u_1[k1,k] / dot;
                  A += (dota * ek1) * tmp_VUjk;
                  A += (dota * ek) * tmp_VUkj;

                  // V response for beta
                  double dotb = tmp_v_1[k1,k] / dot;
                  A += (dotb * ek) * tmp_VUjk;
                  A += (dotb * ek1) * tmp_VUkj;
               }
            }
         }

         // eigenvectors of zero singular values
         for (size_t k1=nbeta2, k2=nbeta2; k1<nalpha2; ++k1){
            mat tmp;
            if (k1 == j){ // beta
               tmp = tmp_b.row(nbeta2) * ls_u_b;
            }
            else{ // alpha
               std::cout << "1 i: " << i << " j " << j << " k1: " << k1 << " k2: " << k2 << std::endl;
               tmp = tmp_a.row(k2) * ls_u_a;
               k2 ++;
            }
            A -= tmp.t() * U[2].col(k1).t();
         }

      }

      const mat S_2_vo = C2_alpha.tail_cols(nvir2_a).t() * AOS * C2_beta.head_cols(nbeta2);
      const mat S_2_ov = C2_alpha.head_cols(nalpha2).t() * AOS * C2_beta.tail_cols(nvir2_b);
      y2_ov_alpha += (S_2_vo * A).t();
      y2_ov_beta  += A * S_2_ov;
   }

   return;
}



vec spin_adiabatic_state::gradient_implicit_rhf(scf* myscf, const mat& y_vo_alpha, const mat& C)
{
   // rhf virtual index runs fast
   vec rhs = vectorise(y_vo_alpha);
   vec zvec = zvector_solve(myscf, rhs); // rhs includes alpha and beta

   // form AO P-like zvector
   const size_t nv = y_vo_alpha.n_rows;
   const size_t no = y_vo_alpha.n_cols;
   mat lagrange_vo(zvec.memptr(), nv, no);
   mat PzA = C.tail_cols(nv) * lagrange_vo * trans(C.head_cols(no));
   PzA += trans(PzA);
   //matrix_print_2d(lagrange_vo.memptr(), lagrange_vo.n_rows, lagrange_vo.n_cols, "lagrange_vo s");

   vec derivatives = zvector_gradient(zvec, PzA, PzA, 1);
   //matrix_print_2d(derivatives.memptr(), 3, NAtoms, "implicit gradient singlet");

   return derivatives;
}


vec spin_adiabatic_state::gradient_implicit_uhf(scf* myscf, const mat& y_ov_alpha, const mat& y_ov_beta, const mat& Ca, const mat& Cb)
{
   cout << "Zexuan Wei gradient_implicit_uhf start" << endl;
   const size_t noa = y_ov_alpha.n_rows;
   const size_t nva = y_ov_alpha.n_cols;
   const size_t nob = y_ov_beta.n_rows;
   const size_t nvb = y_ov_beta.n_cols;

   // alpha, beta
   // row-wise: virtual runs fast
   vec rhs(noa*nva + nob*nvb);
   rhs.head(noa*nva) = vectorise(y_ov_alpha, 1);
   rhs.tail(nob*nvb) = vectorise(y_ov_beta, 1);

   vec zvec = -zvector_solve(myscf, rhs);

   // form AO P-like zvector
   mat lagrange_vo = mat(zvec.memptr(), nva, noa);
   mat PzA = Ca.tail_cols(nva) * lagrange_vo * trans(Ca.head_cols(noa));
   PzA += trans(PzA);
   lagrange_vo = mat(zvec.memptr()+noa*nva, nvb, nob);
   mat PzB = Cb.tail_cols(nvb) * lagrange_vo * trans(Cb.head_cols(nob));
   PzB += trans(PzB);

   vec derivatives = zvector_gradient(zvec, PzA, PzB, 2);
   //matrix_print_2d(derivatives.memptr(), 3, NAtoms, "implicit gradient triplet");

   return derivatives;
}


vec spin_adiabatic_state::gradient_implicit_rohf(scf* myscf, const mat& y_ov_alpha, const mat& y_ov_beta, const mat& C)
{
   const size_t noa = y_ov_alpha.n_rows;
   const size_t nva = y_ov_alpha.n_cols;
   const size_t nob = y_ov_beta.n_rows;
   const size_t nvb = y_ov_beta.n_cols;

   // D*S + D*V + S*V, order follow rohf.C
   const size_t dim_ds = nob * (noa-nob);
   const size_t dim_dv = nob * nva;
   const size_t dim_sv = (noa-nob) * nva;
   vec rhs(dim_ds+dim_dv+dim_sv);

   mat t_yt;
   // D*S
   VRcopy(rhs.memptr(), y_ov_beta.memptr(), dim_ds);
   // D*V
   t_yt = y_ov_alpha.head_rows(nob) + y_ov_beta.tail_cols(nva);
   VRcopy(rhs.memptr()+dim_ds, t_yt.memptr(), dim_dv);
   // S*V
   t_yt = y_ov_alpha.tail_rows(noa-nob);
   VRcopy(rhs.memptr()+dim_ds+dim_dv, t_yt.memptr(), dim_sv);
   // S*V
   t_yt = y_ov_alpha.tail_rows(noa-nob);
   VRcopy(rhs.memptr()+dim_ds+dim_dv, t_yt.memptr(), dim_sv);
   //MatPrint(rhs.memptr(), 1, dim_ds+dim_dv+dim_sv, "zheng rhs", 4);

   vec t_zvec = zvector_solve(myscf, rhs);
   vec zvec(noa*nva + nob*nvb);
   // form MO zvector for alpha and beta
   // (D+S)*V + D*(S+V)
   VRcopy(zvec.memptr(), t_zvec.memptr()+dim_ds, dim_dv); // alpha DV
   VRcopy(zvec.memptr()+dim_dv, t_zvec.memptr()+dim_ds+dim_dv, dim_sv); // alpha SV
   VRcopy(zvec.memptr()+dim_dv+dim_sv, t_zvec.memptr(), dim_ds+dim_dv); // beta DS+DV

   // form AO P-like zvector
   // D*S
   mat lagrange_ov = mat(t_zvec.memptr(), nob, noa-nob);
   mat PzA = C.head_cols(nob) * lagrange_ov * trans(C.cols(nob, noa-1));
   mat PzB = - PzA;
   //MatPrint(lagrange_ov.memptr(), lagrange_ov.n_rows, lagrange_ov.n_cols, "lagrange_ov tds", 4);

   // D*V
   lagrange_ov = mat(t_zvec.memptr()+dim_ds, nob, nva);
   PzB -= C.head_cols(nob) * lagrange_ov * trans(C.tail_cols(nva));
   //MatPrint(lagrange_ov.memptr(), lagrange_ov.n_rows, lagrange_ov.n_cols, "lagrange_ov tdv", 4);

   // S*V
   lagrange_ov = mat(t_zvec.memptr()+dim_ds+dim_dv, noa-nob, nva);
   PzA -= C.cols(nob, noa-1) * lagrange_ov * trans(C.tail_cols(nva));
   //MatPrint(lagrange_ov.memptr(), lagrange_ov.n_rows, lagrange_ov.n_cols, "lagrange_ov tsv", 4);

   PzB += trans(PzB);
   PzA += trans(PzA);
   PzA += PzB;

   //MatPrint(PzA.memptr(), PzA.n_rows, PzA.n_cols, "zheng PzA");
   //MatPrint(PzB.memptr(), PzB.n_rows, PzB.n_cols, "zheng PzB");
   vec derivatives = zvector_gradient(zvec, PzA, PzB, 2);
   //matrix_print_2d(derivatives.memptr(), 3, NAtoms, "implicit gradient triplet");

   return derivatives;
}


vec spin_adiabatic_state::zvector_solve(scf* thescf, vec& rhs)
{
   //Step 1: solve the z-vector eqn
   //we seek the lagrange multipliers for this fragment by solving H*lag = -Y
   cout << "RHS norm: " << norm(rhs, 2) << endl;
   bool do_fd_hvp = rem_read(REM_FD_MAT_VEC_PROD)>0;
   hessian_linear_problem lp(thescf, do_fd_hvp);
   double tolerance = TenMin(rem_read(REM_KONSCF)); //converge as tight as scf
   lin_minres* theSolver = new lin_minres(&lp, tolerance, rhs);
   theSolver->obtain_guess(0); //guess the zero vector //TODO better guess
   theSolver->reset();
   int iter = 0;
   int maxIter = rem_read(REM_MAXSCF);
   //cout<<"solving for lagrange multipliers (restricted) by MINRES"<<endl;
   while (iter<maxIter)
   {
      if(theSolver->next_step())  //We have converged a solution
      {
         //breaks out of while(iter < maxIter)
         break;
      }
      //cout << setw(5)  << iter+1 << setw(14) << setprecision(5) << scientific << theSolver->error << "     00000 " << theSolver->comment << endl;
      iter++; //Increment
   }
   if(iter == maxIter)
   {
      QCrash("MINRES failed to converge");
   }
//   cout<<"MINRES successful"<<endl;

//   matrix_print_2d(theSolver->x.memptr(), 1, theSolver->x.n_elem, "zheng theSolver->x");
   return theSolver->x;
}


vec spin_adiabatic_state::zvector_gradient(const vec& z, mat& PzA, mat& PzB, int spin)
{
   double *jz = QAllocDouble(z.n_elem); //contains z-vector (vo)
   VRcopy(jz, z.memptr(), z.n_elem);

   //contains vectorized: total density, total Pz, zeros (because we use a cis call)
   //expects vectorized P,Q,R (R is zeros for us) see cis_grad
   double *vPtot = QAllocDoubleWithInit(6*NB2car);
   double *jPA = QAllocDoubleWithInit(2*spin*NBas6D2); //contains P
   double *jQA = QAllocDoubleWithInit(2*spin*NBas6D2); //contains Pz
   double *vW = QAllocDoubleWithInit(2*NB2car); //contains the energy-weighted-density-like matrix to be contracted with Sx
   double *jQB;
   double *jCA = QAllocDouble(2*spin*NBas*NOrb);
   double *jGA = QAllocDoubleWithInit(2*spin*NBas6D2);

   if (spin == 1){
      VRcopy(jCA, C1_alpha.memptr(), NBas*NOrb);
      VRcopy(jPA, P1_alpha.memptr(), NBas2);
      VRcopy(jQA, PzA.memptr(), NBas2);
      jQB = jQA;

      ScaV2M(P1_alpha.memptr(), vPtot, 1, 0);
      ScaV2M(PzA.memptr(), vPtot+NB2car, 1, 0);
      VRscale(vPtot, 2*NB2car, 2.0);

      ks_d2EdPdP_dot_D_r_u(jGA,NULL,jz,jPA,NULL,jCA,NULL,1,1);
      mat cct = C1_alpha * C1_alpha.t();
      mat W_like = cct * mat(jGA,NBas,NBas) * P1_alpha;
      W_like += PzA * F1_alpha * cct;
      W_like += trans(W_like);
      ScaV2M(W_like.memptr(), vW, 1, 0);
      //MatPrint(W_like.memptr(), W_like.n_rows, W_like.n_cols, "zheng W_like singlet rhf");
   }
   else{
      VRcopy(jCA, C2_alpha.memptr(), NBas*NOrb);
      double *jCB = jCA + NBas*NOrb;
      VRcopy(jCB, C2_beta.memptr(), NBas*NOrb);
      VRcopy(jPA, P2_alpha.memptr(), NBas2);
      double *jPB = jPA + NBas6D2;
      VRcopy(jPB, P2_beta.memptr(), NBas2);
      VRcopy(jQA, PzA.memptr(), NBas2);
      //double *jQB = jQA + NBas6D2;
      jQB = jQA + NBas6D2;
      VRcopy(jQB, PzB.memptr(), NBas2);

      mat Ptot = P2_alpha + P2_beta; 
      ScaV2M(Ptot.memptr(), vPtot, 1, 0);
      mat Qtot = PzA + PzB; //alpha+beta
      ScaV2M(Qtot.memptr(), vPtot+NB2car, 1, 0);

      double *jGB = jGA + NBas6D2;
      //ks_d2EdPdP_dot_D_r_u(jGA,jGB,jz,jPA,jPB,jCA,jCB,2,1);
      // symmetrize for J and XC is important 
      ks_d2EdPdP_dot_D_r_u(jGA,jGB,PzA.memptr(),PzB.memptr(),jPA,jPB,jCA,jCB, 2,1,true,true,true);
      //matrix_print_2d(jGA, NBas, NBas, "zheng triplet jGA");
      //matrix_print_2d(jGB, NBas, NBas, "zheng triplet jGB");

      mat cct = C2_alpha * C2_alpha.t();
      mat W_like = cct * mat(jGA,NBas,NBas) * P2_alpha;
      W_like += PzA * F2_alpha * cct;

      //cct = C2_beta * C2_beta.t(); // cct are same
      W_like += cct * mat(jGB,NBas,NBas) * P2_beta;
      W_like += PzB * F2_beta * cct;

      W_like += trans(W_like);
      W_like *= 0.5;
      ScaV2M(W_like.memptr(), vW, 1, 0);
      //matrix_print_2d(W_like.memptr(), W_like.n_rows, W_like.n_cols, "zheng W_like triplet");
   }

   QFree(jGA);
   QFree(jCA);

   


   
   //wrap up the H^x and S^x contributions
   vec frag_HxSx;
   {
      double* jGrad = QAllocDoubleWithInit(2*Nuclear);
      GenMatrix Grad(jGrad, 3, NAtoms);
      Grad = 0.0;
      STVu_Grad(Grad, vPtot+NB2car, vW);  //Note: the second passed variable is actually vectorized Pz
      frag_HxSx = vec(jGrad, Nuclear);
      QFree(jGrad);
   }
   //matrix_print_2d(frag_HxSx.memptr(), 3, NAtoms, "frag_HxSx");


   vec frag_IIx;  //Note: it contains contributions from both II^x and Vxc^x
   {
      //see cis_grad.C
      //Q in cis_grad has the lagrange mult/z-vector density + other stuff
      //    from cis, but we just put lagrange density there
      //R is just set to zero in this call because we don't want those 2e terms
      //    related to the cis excitation vector
      //    TODO this is unnecessary work, but I think dual basis grad does this as well
      //P is just the normal scf density in both cis_grad and here
      
      //II^x contribution to the response grad (eq. 19) comes into jE
      double* jE = QAllocDoubleWithInit(2*Nuclear); //will contain grad
      double* jRA =QAllocDoubleWithInit(2*spin*NBas6D2); 

      //make copies of jQA and jPA because it would seem we cannot trust AOints
      //i.e. AOints might modify the passed PA and QA secretly
      double* jQAcopy = QAllocDoubleWithInit(2*spin*NBas6D2); 
      double* jPAcopy = QAllocDoubleWithInit(2*spin*NBas6D2);
      VRcopy(jQAcopy, jQA, spin*NBas6D2); 
      VRcopy(jPAcopy, jPA, spin*NBas6D2); 
      
      //J and K parts
      if (rem_read(REM_LRC_DFT)!=1)
      {
         cout << "PA, QA, RA: " << spin*NBas6D2 << " " << spin << endl;
         //MatPrint(jPA, spin, NBas6D2, "zheng jPA", 4);
         //VRscale(jQA, spin*NBas6D2, 100000);
         //MatPrint(jQA, spin, NBas6D2, "zheng jQA", 4);
         //VRscale(jQA, spin*NBas6D2, 0.00001);
         AOints(jE,NULL,NULL,NULL,vPtot,jPA,jQA,jRA,NULL,127);
         //matrix_print_2d(jE, 1, Nuclear, "zheng jE");
      }
      else
      {
         //zero contributions from non-vectorized to isolate J component
         VRload(jQA, spin*NBas6D2, 0.0);
         VRload(jPA, spin*NBas6D2, 0.0);
         VRload(jRA, spin*NBas6D2, 0.0);
         AOints(jE,NULL,NULL,NULL,vPtot,jPA,jQA,jRA,NULL,127); 

         //Long-range exchange
         double* jELRK = QAllocDoubleWithInit(2*Nuclear);
         //we are done with vPtot, so zero it to avoid adding J again
         VRload(vPtot, 3*NB2car, 0.0);
         //refill P Q R
         VRcopy(jQA, jQAcopy, spin*NBas6D2); 
         VRcopy(jPA, jPAcopy, spin*NBas6D2); 
         VRload(jRA, spin*NBas6D2, 0.0);
         rem_write(OP_ERF, REM_INTEGRAL_2E_OPR);
         int ICoefK = rem_read(REM_COEF_K);
         //Note: usually combine_k is used
         if (rem_read(REM_COMBINE_K) != 1 && rem_read(REM_SRC_DFT) != 1) {
            int hfx_lr_coef = rem_read(REM_HFK_LR_COEF);
            rem_write(hfx_lr_coef, REM_COEF_K);
         }
         else {
            rem_write(100000000, REM_COEF_K);
         }
         AOints(jELRK,NULL,NULL,NULL,vPtot,jPA,jQA,jRA,NULL,127);
         rem_write(ICoefK, REM_COEF_K);
         //add long-range K to J
         VRadd(jE, jE, jELRK, Nuclear);
         QFree(jELRK);
         
         //Short-range exchange
         if (ICoefK>1 && rem_read(REM_COMBINE_K)!=1)
         {
            double* jESRK = QAllocDoubleWithInit(2*Nuclear);
            //evidently AOints can't be trusted. refill
            VRcopy(jQA, jQAcopy, spin*NBas6D2); 
            VRcopy(jPA, jPAcopy, spin*NBas6D2); 
            VRload(jRA, spin*NBas6D2, 0.0);
            VRload(vPtot, 3*NB2car, 0.0);
            rem_write(OP_ERFC, REM_INTEGRAL_2E_OPR);
            AOints(jESRK,NULL,NULL,NULL,vPtot,jPA,jQA,jRA,NULL,127);
            //add short-range K to long-range K and J
            VRadd(jE, jE, jESRK, Nuclear);
            QFree(jESRK);
         }
         //back to the normal coulomb operator
         rem_write(OP_R12, REM_INTEGRAL_2E_OPR);
      }
         
      // undo damage from JK integrals
      VRcopy(jQA, jQAcopy, spin*NBas6D2); 
      VRcopy(jPA, jPAcopy, spin*NBas6D2); 
      VRload(jRA, spin*NBas6D2, 0.0);


      // XC contribution (eq. 20) goes to jEXC
      XCFunctional xcFunc(XCFUNC_SCF);
      if (xcFunc.HasDFT())
      {
         double* jEXC = QAllocDoubleWithInit(2*Nuclear);
         
         //we need d2Exc/dPdx dot Pz
         //follwing in the heroic footsteps of scf_dual_grad.C
         
         //hope you have enough memory 
         //cout << "compute d2Exc/dPdx analytically" << endl;
         double* jFXCxA = QAllocDoubleWithInit(2*spin*NBas6D2*Nuclear);
         //In its current form, this part is VERY slow...efficiency improvements coming soon!
         DFTexplicitNuc1stDrvOfXCMtrx(jFXCxA,jPA,0,NAtoms,1);
         
         //contract with Pz
         for (int j=0; j<Nuclear; ++j){
            jEXC[j] = VRdot(jFXCxA+j*NBas6D2, jQA, NBas*NBas);
         }
         if (spin == 1)
            VRscale(jEXC, Nuclear, 2.0);
         else if (spin == 2){
            double* jFXCxB = jFXCxA + NBas6D2*Nuclear;
            for (int j=0; j<Nuclear; ++j)
               jEXC[j] += VRdot(jFXCxB+j*NBas6D2, jQA+NBas6D2, NBas*NBas);
         }
         QFree(jFXCxA);

         
         //matrix_print_2d(jEXC, 1, Nuclear, "zheng jEXC");
         VRadd(jE, jE, jEXC, Nuclear);
         QFree(jEXC);
      }

      frag_IIx = vec(jE, Nuclear);
      // test will get scf gradient need scale 0.5
      //frag_IIx = 0.5*vec(jE, Nuclear);
      
      QFree(jE);
      QFree(jRA);
      QFree(jQAcopy);   
      QFree(jPAcopy);   
   }
   //matrix_print_2d(frag_IIx.memptr(), 3, NAtoms, "frag_IIx");

   QFree(jPA);
   QFree(jQA);
   QFree(vPtot);
   QFree(vW);

   vec grad = frag_HxSx + frag_IIx;
   return grad;
}


vec spin_adiabatic_state::gradient_one_electron(const double* jPv, int spin)
{
   cout << "zheng gradient_one_electron" << endl;
   double* jGrad = QAllocDoubleWithInit(2*Nuclear);
   Nuc_Grad(jGrad);

   // one electron contributions
   // with gs density and z-vector density
   double *vPtot = QAllocDouble(2*NB2car);
   VRcopy(vPtot, jPv, NB2car);
   if (spin == 2) VRadd(vPtot, jPv+NB2car, NB2car);
   // create the energy-weighted density matrix
   double *jWv = QAllocDouble(2*NB2car);
   Get_W_SCF(jWv);
   GenMatrix Grad(jGrad, 3, NAtoms);
   STVu_Grad(Grad, vPtot, jWv);  //Note: the second passed variable is actually vectorized Pz
   QFree(vPtot); QFree(jWv);

   //MatPrint(jGrad, 3, NAtoms, "zheng one electron gradient");

   XCFunctional XCfunc(XCFUNC_SCF);
   double *jPtot = QAllocDouble(2*spin*NB2car);
   VRcopy(jPtot, jPv, spin*NB2car);
   double *jEx = QAllocDoubleWithInit(2*Nuclear);
   DFTman(jEx,NULL,NULL,NULL,jPtot,NULL,NULL,NULL,101,XCfunc,rem_read(REM_IGRDTY),0);

   GenMatrix Ex(jEx, 3, NAtoms);
   Ex.Print("zheng XC contribution from the ground state");
   Grad += Ex;
   Grad.Print("energy derivatives");

   vec derivatives(jGrad, 3, NAtoms);

   QFree(jPtot);
   QFree(jEx);
   QFree(jGrad);
   
   return derivatives;
}
