"""
    Daniel Claudino - October/2018

    Module to run a projection-based embedding 
    calculation with an open-shell (HF/KS) reference.

"""
import os
import psi4
import numpy as np

def embedding(wfn, mol, nAtomEnv, high_theory, basis):
    nso = wfn.nso()
    nre = mol.nuclear_repulsion_energy()
    alpha_Cocc = wfn.Ca_subset("AO","OCC")
    beta_Cocc = wfn.Cb_subset("AO","OCC")
    alpha_nOccOrbs = wfn.nalpha()
    beta_nOccOrbs = wfn.nbeta()

    outfile = open("input.log", "w")

    def n_active_aos():
        """
        Function that returns the number of AOs from the
        atoms in the active subsystem

        Replace nAtomEnv by the number of atoms in the environment
        """

        nActiveAtoms = mol.natom()-nAtomEnv
        active_atoms = list(range(nActiveAtoms))
        outfile.write("Atoms in the active region %s\n" % active_atoms)
        nActAO = 0
        for ao in range(nso):
            for atom in active_atoms:
                if wfn.basisset().function_to_center(ao) == atom:
                    nActAO += 1

        return nActAO

    def svd(lowerLimit, upperLimit):
        """
        Function that returns the C matrix rotated according to the SVD 
        of the active or environment C coefficients

        wfn.S() = overlap matrix
        """

        X = wfn.S().clone()
        X.power(0.5,1.e-16)
        Xm = wfn.S().clone()
        Xm.power(-0.5,1.e-16)

        alpha_C = alpha_Cocc.clone()
        alpha_C = psi4.core.Matrix.doublet(X, alpha_C)
        alpha_C.np[lowerLimit:upperLimit,:] = 0.0

        u, alpha_s, vh = np.linalg.svd(alpha_C, full_matrices=True)
        alpha_C = psi4.core.Matrix.from_array(alpha_Cocc.np.dot(vh.T))

        beta_C = beta_Cocc.clone()
        beta_C = psi4.core.Matrix.doublet(X, beta_C)
        beta_C.np[lowerLimit:upperLimit,:] = 0.0

        u, beta_s, vh = np.linalg.svd(beta_C, full_matrices=True)
        beta_C = psi4.core.Matrix.from_array(beta_Cocc.np.dot(vh.T))

        return (alpha_C, alpha_s, beta_C, beta_s)

    def print_svd(alpha_s, beta_s, outfile):
        """
        Function to print the singular values of all partitioned matrices
        and partitions the orbital space
        """

        alpha_ds = np.zeros([alpha_nOccOrbs-1])
        outfile.write("\n MO+1, SV(MO) , SV(MO+1)-SV(MO)\n")
        for i in range(1,alpha_nOccOrbs):
            alpha_ds[i-1] = -(alpha_s[i]-alpha_s[i-1])
            outfile.write("%s %s %s \n" % (i, alpha_s[i-1], alpha_ds[i-1]))

        alpha_OrbsInA = np.argpartition(alpha_ds,-1)[-1]+1
        outfile.write("Number of alpha orbitals in the active region: %s\n" % alpha_OrbsInA)
        
        beta_ds = np.zeros([beta_nOccOrbs-1])
        outfile.write("\n MO+1, SV(MO) , SV(MO+1)-SV(MO)\n")
        for i in range(1,beta_nOccOrbs):
            beta_ds[i-1] = -(beta_s[i]-beta_s[i-1])
            outfile.write("%s %s %s \n" % (i, beta_s[i-1], beta_ds[i-1]))

        beta_OrbsInA = np.argpartition(beta_ds,-1)[-1]+1
        outfile.write("Number of beta orbitals in the active region: %s\n" % beta_OrbsInA)
        
        return (alpha_OrbsInA, beta_OrbsInA)

    def density(C):
        """
        Function to compute the AO density matrix for a given C
        """

        D = psi4.core.Matrix.doublet(C, C, False, True)

        return D

    def JK(alpha_C, beta_C):
        """
        Function to compute J and K matrices
        """
         
        jk = psi4.core.JK.build(wfn.basisset(),wfn.get_basisset("DF_BASIS_SCF"),"DF")
        jk.set_memory(int(1.25e8))
        jk.initialize()
        jk.print_header()

        jk.C_left_add(alpha_C)
        jk.C_left_add(beta_C)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        return (jk.J()[0], jk.J()[1], jk.K()[0], jk.K()[1])

    def KS_potential(alpha_D, beta_D):
        """
        Computes the KS XC potential and energy
        """
        wfn.Da().copy(alpha_D)
        wfn.Db().copy(beta_D)
        wfn.form_V()
        ks = psi4.core.VBase.quadrature_values(wfn.V_potential())
        E_ks = ks["FUNCTIONAL"]
        alpha_V = wfn.Va().clone()
        beta_V = wfn.Vb().clone()

        return (alpha_V, beta_V, E_ks)

    def energy(H, alpha_J, beta_J, alpha_K, beta_K, E_xc, alpha_D, beta_D):
        """
        Computes the energy 
        """

        E = H.vector_dot(alpha_D)+H.vector_dot(beta_D)+ \
            0.5*(alpha_J.vector_dot(alpha_D)+beta_J.vector_dot(beta_D)+ \
            beta_J.vector_dot(alpha_D)+alpha_J.vector_dot(beta_D)- \
            wfn.functional().x_alpha()*(alpha_K.vector_dot(alpha_D)+ \
            beta_K.vector_dot(beta_D))) + E_xc

        return E

    outfile.write("Psi4 energy: %s\n" % wfn.energy())
    actAOs = n_active_aos()

    (alpha_Cact, alpha_s, beta_Cact, beta_s) = svd(actAOs, nso)
#    (Cenv, senv) = svd(0, actAOs)

    (alpha_actOrbs, beta_actOrbs) = print_svd(alpha_s, beta_s, outfile)

    alpha_Cenv = alpha_Cact.clone()
    beta_Cenv = beta_Cact.clone()
    alpha_Cact.np[:,alpha_actOrbs:] = 0.0
    beta_Cact.np[:,beta_actOrbs:] = 0.0
    alpha_Cenv.np[:,:alpha_actOrbs] = 0.0
    beta_Cenv.np[:,:beta_actOrbs] = 0.0

    alpha_Dact = density(alpha_Cact)
    beta_Dact = density(beta_Cact)
    alpha_Denv = density(alpha_Cenv)
    beta_Denv = density(beta_Cenv)
    alpha_Dtotal = alpha_Dact.clone()
    alpha_Dtotal.add(alpha_Denv)
    beta_Dtotal = beta_Dact.clone()
    beta_Dtotal.add(beta_Denv)

    (alpha_Jact, beta_Jact, alpha_Kact, beta_Kact) = JK(alpha_Cact, beta_Cact)
    (alpha_Jenv, beta_Jenv, alpha_Kenv, beta_Kenv) = JK(alpha_Cenv, beta_Cenv)

    (alpha_Vact, beta_Vact, E_xc_act)       = KS_potential(alpha_Dact, beta_Dact)
    (alpha_Venv, beta_Venv, E_xc_env)       = KS_potential(alpha_Denv, beta_Denv)
    (alpha_Vtotal, beta_Vtotal, E_xc_total) = KS_potential(alpha_Dtotal, beta_Dtotal)

# E_act = energy of isolated A
# Eenv = energy of isolated B
# G = two-body correction to the energy

    Eact = energy(wfn.H(), alpha_Jact, beta_Jact, alpha_Kact, beta_Kact, E_xc_act, alpha_Dact, beta_Dact)
    Eenv = energy(wfn.H(), alpha_Jenv, beta_Jenv, alpha_Kenv, beta_Kenv, E_xc_env, alpha_Denv, beta_Denv)
    G = 0.5*(alpha_Jact.vector_dot(alpha_Denv)+alpha_Jact.vector_dot(beta_Denv)+ \
        beta_Jact.vector_dot(alpha_Denv)+beta_Jact.vector_dot(beta_Denv)+ \
        alpha_Jenv.vector_dot(alpha_Dact)+alpha_Jenv.vector_dot(beta_Dact)+ \
        beta_Jenv.vector_dot(alpha_Dact)+beta_Jenv.vector_dot(beta_Dact))- \
        0.5*wfn.functional().x_alpha()*(alpha_Kenv.vector_dot(alpha_Dact)+beta_Kenv.vector_dot(beta_Dact)+ \
        alpha_Kact.vector_dot(alpha_Denv)+beta_Kact.vector_dot(beta_Denv))+ \
        E_xc_total-E_xc_act-E_xc_env

    outfile.write("\nE[active] =  %s\n" % Eact)
    outfile.write("E[environment] =  %s\n" % Eenv)
    outfile.write("Non-additive two-electron term = %s\n" % G)
    outfile.write("Nuclear repulsion energy =  %s\n" % nre)
    outfile.write("Total energy =  %s\n" % (Eact+Eenv+G+nre))
    """
        Constructing the embedded core Hamiltonian embed_H
        
        init_H = original core Hamiltonian
        P = projector that ensures orthogonality between subsystems
        1.0e6 is the level-shift parameter 

    """
    alpha_P = psi4.core.Matrix.triplet(wfn.S(),alpha_Denv,wfn.S(), False, False, False)
    beta_P = psi4.core.Matrix.triplet(wfn.S(),beta_Denv,wfn.S(), False, False, False)
    alpha_Vemb = alpha_Jenv.clone()
    alpha_Vemb.add(beta_Jenv)
    alpha_Vemb.add(alpha_Vtotal)
    alpha_Vemb.axpy(-1.0,alpha_Vact)
    alpha_Vemb.axpy(-wfn.functional().x_alpha(),alpha_Kenv)
    alpha_Vemb.axpy(1.0e6,alpha_P)
    beta_Vemb = alpha_Jenv.clone()
    beta_Vemb.add(beta_Jenv)
    beta_Vemb.add(beta_Vtotal)
    beta_Vemb.axpy(-1.0,beta_Vact)
    beta_Vemb.axpy(-wfn.functional().x_alpha(),beta_Kenv)
    beta_Vemb.axpy(1.0e6,beta_P)

    fa = open("Va_emb.dat","w")
    fb = open("Vb_emb.dat","w")
    for i in range(nso):
        for j in range(nso):
            fa.write("%s\n" % alpha_Vemb.get(i,j))
            fb.write("%s\n" % beta_Vemb.get(i,j))
    fa.close()
    fb.close()

    """
        Start of the embedded calculation

        If doing DF-MP2, set frozen_uocc to 0, as the DF-MP2 does not 
        allow freezing virtuals

        If doing DFT-in-DFT embedding, change "hf" by your the chosen functional
    """

    psi4.set_options({"docc": [beta_actOrbs], "socc": [alpha_actOrbs-beta_actOrbs],\
    "frozen_uocc": [beta_nOccOrbs-beta_actOrbs], "guess": "read", "save_jk":\
    "true", 'basis': basis, "reference": "uhf"})

    eng, wfn = psi4.energy('hf', return_wfn=True)

    Eact_embedded = wfn.H().vector_dot(wfn.Da())+wfn.H().vector_dot(wfn.Db())+ \
                    0.5*(wfn.jk().J()[0].vector_dot(wfn.Da())+wfn.jk().J()[1].vector_dot(wfn.Db())+ \
                    wfn.jk().J()[1].vector_dot(wfn.Da())+wfn.jk().J()[0].vector_dot(wfn.Db())- \
                    (wfn.jk().K()[0].vector_dot(wfn.Da())+wfn.jk().K()[1].vector_dot(wfn.Db())))
    """
        Change "method" to the WFT method of choice
        Refer to the Psi4 manual to get the right variable name
        for your choice of correlated method
    """

    corr_eng, corr_wfn = psi4.energy(high_theory, ref_wfn=wfn, return_wfn=True)
    variable = high_theory.upper()+' CORRELATION ENERGY'
    E_correlation = psi4.get_variable(variable)
    outfile.write("Embedded E[active] =  %s\n" % (Eact_embedded+E_correlation))

    """
        Embedded D correction
    """
    wfn.Da().axpy(-1.0, alpha_Dact)
    wfn.Db().axpy(-1.0, beta_Dact)
    correction = wfn.Da().vector_dot(alpha_Vemb)+wfn.Db().vector_dot(beta_Vemb)

    """
        Total embedded energy
    """
    E_embedded = Eact_embedded+Eenv+G+correction+mol.nuclear_repulsion_energy()
    outfile.write("Total embedded energy =  %s\n" % E_embedded)

    outfile.close()
    os.system("rm V*emb.dat")

