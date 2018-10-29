"""
    Daniel Claudino - October/2018

    Module to run a projection-based embedding 
    calculation with a closed-shell (HF/KS) reference.

"""
import os
import psi4
import numpy as np

def embedding(wfn, mol, nAtomEnv, high_theory, basis):
    nso = wfn.nso()
    nre = mol.nuclear_repulsion_energy()
    C_occ = wfn.Ca_subset("AO","OCC")
    nOccOrbs = wfn.nalpha()

    outfile = open("input.log", "w")
    outfile.write("Psi4 non-embedded energy: %s" % wfn.energy())

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

        C = C_occ.clone()
        C = psi4.core.Matrix.doublet(X, C)
        C.np[lowerLimit:upperLimit,:] = 0.0

        u, s, vh = np.linalg.svd(C, full_matrices=True)
        C = psi4.core.Matrix.from_array(C_occ.np.dot(vh.T))

        return (C,s)

    def print_svd(s_A, s_B, outfile):
        """
        Function to print the singular values of all partitioned matrices
        and partitions the orbital space
        """

        ds_A = np.zeros([nOccOrbs-1])
        outfile.write("\n MO+1, SV(MO) , SV(MO+1)-SV(MO)\n")
        for i in range(1,nOccOrbs):
            ds_A[i-1] = -(s_A[i]-s_A[i-1])
            outfile.write("%s %s %s \n" % (i, s_A[i-1], ds_A[i-1]))

        OrbsInA = np.argpartition(ds_A,-1)[-1]+1
        outfile.write("Number of orbitals in the active region: %s\n" % OrbsInA)
        
        return OrbsInA

    def density(orbitals):
        """
        Function to compute the AO density matrix for a given C
        """

        D = psi4.core.Matrix.doublet(orbitals, orbitals, False, True)
        return D

    def JK(orbitals):
        """
        Function to compute J and K matrices
        """
         
        jk = psi4.core.JK.build(wfn.basisset(),wfn.get_basisset("DF_BASIS_SCF"),"DF")
        jk.set_memory(int(1.25e8))
        jk.initialize()
        jk.print_header()

        jk.C_left_add(orbitals)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        return (jk.J()[0], jk.K()[0])

    def KS_potential(density):
        """
        Computes the KS XC potential and energy
        """
        wfn.Da().copy(density)
        wfn.form_V()
        ks = psi4.core.VBase.quadrature_values(wfn.V_potential())
        E_ks = ks["FUNCTIONAL"]
        V = wfn.Va().clone()

        return (V, E_ks)

    def energy(H, J, K, E_xc, density):
        """
        Computes the energy 
        """

        E = 2.0*(H.vector_dot(density)+J.vector_dot(density)) \
            -wfn.functional().x_alpha()*(K.vector_dot(density)) + E_xc

        return E

    outfile.write("Psi4 energy: %s\n" % wfn.energy())
    actAOs = n_active_aos()

    (C_act, s_act) = svd(actAOs, nso)
    (C_env, s_env) = svd(0, actAOs)

    actOrbs = print_svd(s_act, s_env, outfile)

    C_env = C_act.clone()
    C_act.np[:,actOrbs:nOccOrbs] = 0.0
    C_env.np[:,:actOrbs] = 0.0

    D_act = density(C_act)
    D_env = density(C_env)
    D_total = D_act.clone()
    D_total.add(D_env)

    (J_act, K_act) = JK(C_act)
    (J_env, K_env) = JK(C_env)

    (V_act, E_xc_act)     = KS_potential(D_act)
    (V_env, E_xc_env)     = KS_potential(D_env)
    (V_total, E_xc_total) = KS_potential(D_total)

# E_act = energy of isolated A
# E_env = energy of isolated B
# G = two-body correction to the energy

    E_act = energy(wfn.H(), J_act, K_act, E_xc_act, D_act)
    E_env = energy(wfn.H(), J_env, K_env, E_xc_env, D_env)
    G = 2.0*(D_act.vector_dot(J_env)+D_env.vector_dot(J_act))+ \
    E_xc_total-E_xc_act-E_xc_env -\
    wfn.functional().x_alpha()*(K_env.vector_dot(D_act)+K_act.vector_dot(D_env)) 

    outfile.write("\nE[active] =  %s\n" % E_act)
    outfile.write("E[environment] =  %s\n" % E_env)
    outfile.write("Non-additive two-electron term = %s\n" % G)
    outfile.write("Nuclear repulsion energy =  %s\n" % nre)
    outfile.write("Total energy =  %s\n" % (E_act+E_env+G+nre))

    """
        Constructing the embedded core Hamiltonian embed_H
        
        init_H = original core Hamiltonian
        P = projector that ensures orthogonality between subsystems
        1.0e6 is the level-shift parameter 

    """
    P = psi4.core.Matrix.triplet(wfn.S(),D_env,wfn.S(), False, False, False)
    init_H = wfn.H().clone()
    embed_H = wfn.H().clone()
    embed_H.axpy(2.0,J_env)
    embed_H.axpy(-wfn.functional().x_alpha(),K_env)
    embed_H.axpy(1.0,V_total)
    embed_H.axpy(-1.0,V_act)
    embed_H.axpy(1.0e6,P)

    f = open("newH.dat","w")
    for i in range(nso):
        for j in range(nso):
            f.write("%s\n" % embed_H.get(i,j))
    f.close()

    """
        Start of the embedded calculation

        If doing DF-MP2, set frozen_uocc to 0, as the DF-MP2 does not 
        allow freezing virtuals

        If doing DFT-in-DFT embedding, change "hf" by your the chosen functional
    """

    psi4.set_options({"docc": [actOrbs], "frozen_uocc": [nOccOrbs-actOrbs], "guess": "read", "save_jk": "true", 'basis': basis})
    eng, wfn = psi4.energy('hf', return_wfn=True)

    E_act_embedded = 2.0*(wfn.Da().vector_dot(init_H)+ \
                     wfn.Da().vector_dot(wfn.jk().J()[0]))- \
                     wfn.Da().vector_dot(wfn.jk().K()[0])

    """
        Change "method" to the WFT method of choice
        Refer to the Psi4 manual to get the right variable name
        for your choice of correlated method
    """

    corr_eng, corr_wfn = psi4.energy(high_theory, ref_wfn=wfn, return_wfn=True)
    variable = high_theory.upper()+' CORRELATION ENERGY'
    E_correlation = psi4.get_variable(variable)
    outfile.write("Embedded E[active] =  %s\n" % (E_act_embedded+E_correlation))

    """
        Embedded density correction
    """
    wfn.H().axpy(-1.0,init_H)
    wfn.Da().axpy(-1.0,D_act)
    correction = 2.0*wfn.Da().vector_dot(wfn.H())

    """
        Total embedded energy
    """
    E_embedded = E_act_embedded+E_env+G+correction+mol.nuclear_repulsion_energy()
    outfile.write("Total embedded energy =  %s\n" % E_embedded)

    outfile.close()
    os.system("rm newH.dat")



