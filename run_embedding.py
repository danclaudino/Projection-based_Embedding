import os, sys
import psi4
import numpy as np
from embedding import *
from embedding_helper import *
#np.set_printoptions(precision=3, suppress = True)

def run_closed_shell(wfn, 
    mol, 
    n_active_atoms, 
    high_level, 
    partition_method = 'spade', 
    projection_basis = '', 
    operator_name = 'F', 
    n_virtual_shell = None, 
    virt_proj_basis = 'sto-3g_d_p'):

    working_basis = psi4.core.get_global_option('BASIS')

    # Saving some DFT info for the whole system 

    nre = mol.nuclear_repulsion_energy()
    if(wfn.functional().name() != 'HF'):
        V_total = wfn.Va().clone().np
        E_xc_total = psi4.core.VBase.quadrature_values(wfn.V_potential())["FUNCTIONAL"]
    else:
        E_xc_total = 0.0
        V_total = np.zeros([wfn.nso(), wfn.nso()])

    # Partitioning the orbital space 
    
    embed = Embed(wfn, mol, n_active_atoms, projection_basis)

    n_act_aos = embed.count_active_aos()

    # Whether or not to project occupied orbitals onto another (minimal) basis
    if projection_basis != '': 
        S, Cocc = embed.basis_projection(wfn.Ca_subset("AO","OCC").np,\
        embed.proj_wfn.basisset(), wfn.basisset())
    else:
        S = wfn.S().np
        Cocc = wfn.Ca_subset("AO", "OCC").np

    V, sigma = embed.orbital_rotation(S, Cocc, n_act_aos)

    # Partitioning method (spade vs. all AOs in projected basis)
    if partition_method == 'spade':
        ds = [-(sigma[i+1]-sigma[i]) for i in range(len(sigma)-1)]
        n_act_mos = np.argpartition(ds,-1)[-1]+1
    else:
       assert projection_basis != '', '\nDefine a projection (preferably a minimal) basis' 
       n_act_mos = n_act_aos

    # Orbital rotation and partition
    C_A = wfn.Ca_subset("AO", "OCC").np @ V.T[:,:n_act_mos]
    C_B = wfn.Ca_subset("AO", "OCC").np @ V.T[:,n_act_mos:]

    # Retrieving the subsytem energy terms and potential matrices
    E_A, E_xc_A, J_A, K_A, V_A = embed.subsystem(C_A)
    E_B, E_xc_B, J_B, K_B, V_B = embed.subsystem(C_B)
    D_A = C_A @ C_A.T
    D_B = C_B @ C_B.T

    J_AB = 2.0*(matrix_dot(D_A, J_B) + matrix_dot(D_B, J_A))
    K_AB = -wfn.functional().x_alpha()*(matrix_dot(D_A, K_B) + matrix_dot(D_B, K_A))
    XC_AB = E_xc_total - E_xc_A - E_xc_B
    G = J_AB + K_AB + XC_AB

    # Generating molden files for partitioned orbitals before and after pseudocanonicalization 
    wfn.Ca().copy(psi4.core.Matrix.from_array(C_A))
    psi4.driver.molden(wfn, 'before_pseudocanonical.molden')

    v, w = np.linalg.eigh(C_A.T @ wfn.Fa().np @ C_A)
    pseudoC_A = C_A @ w
    wfn.Ca().copy(psi4.core.Matrix.from_array(pseudoC_A))
    psi4.driver.molden(wfn, 'after_pseudocanonical.molden')

    # Defining the embedded core Hamiltonian using the default mu=1e6. 
    P = 1e6*(wfn.S().np @ D_B @ wfn.S().np)
    H_old = wfn.H().np
    H_emb = wfn.H().np + 2.0*J_B - wfn.functional().x_alpha()*K_B + P + V_total - V_A

    # Saving the embedded core Hamiltonian to 'newH.dat', which is read by my version of Psi4
    f = open("newH.dat","w")
    for i in range(wfn.nso()):
        for j in range(wfn.nso()):
            f.write("%s\n" % H_emb[i,j])
    f.close()

    # Running embedded calculation 
    psi4.set_options({"save_jk": "true",\
                    'guess': 'read',\
                    'basis': working_basis,\
                    'docc': [n_act_mos],\
                    'frozen_uocc': [0],\
                    'cc_type': 'df',\
                    'scf_type': 'df'})
    if high_level == 'ccsd' or high_level == 'ccsd(t)':
        psi4.set_options({'df_ints_io' : 'save'})

    scf_eng, scf_wfn = psi4.energy('hf', return_wfn = True)
    psi4.driver.molden(scf_wfn, 'embedded.molden')

    overlap = pseudoC_A.T @ wfn.S().np @ scf_wfn.Ca_subset("AO","OCC").np
    u, s, vh = np.linalg.svd(overlap)
    overlap = np.linalg.det(u)*np.linalg.det(vh)*np.prod(s)

    D = scf_wfn.Da().np 
    embed_E_A = 2.0*(matrix_dot(D, H_old) + matrix_dot(D, scf_wfn.jk().J()[0].np))\
                - matrix_dot(D, scf_wfn.jk().K()[0].np)
    H_emb = H_emb - H_old
    D = D - D_A
    embed_SCF = embed_E_A + E_B + G + nre + 2.0*(matrix_dot(H_emb, D))

    # Printing embedded HF results
    outfile = open('embedding.log', 'w')
    banner(outfile, partition_method)
    outfile.write('\n\n')
    outfile.write('Energy values in atomic units\n')
    outfile.write('Embedded calculation: '+high_level.upper()+'-in-'+wfn.functional().name()+'\n')
    outfile.write('\n')
    if partition_method == 'spade':
        if projection_basis == '':
            outfile.write('Orbital partition method: SPADE\n')
        else:
            outfile.write('Orbital partition method: SPADE from occupied space projected onto '+projection_basis.upper()+ '\n')
    else:
        outfile.write('Orbital partition method: All AOs in '+projection_basis.upper()+' from atoms in A\n')
    
    density_correction = 2.0*(matrix_dot(H_emb, D))
    print_rhf(wfn, n_act_mos, E_A, E_B, G, nre, embed_E_A, density_correction, embed_SCF, overlap, outfile)
    # --------------------------------- Post embedded HF calculation ---------------------------------------

    if isinstance(n_virtual_shell, str) or n_virtual_shell == None:
        emb_e, emb_wfn = psi4.energy(high_level, ref_wfn=scf_wfn, return_wfn=True)
        corr = psi4.core.get_variable("CURRENT CORRELATION ENERGY")
        total_E = embed_SCF + corr
        outfile.write('Embedded {:>5}-in-{:<5} E[A] \t = {:>16.10f}\n'.format(high_level.upper(), wfn.functional().name(), embed_SCF + corr))

    else:
        outfile.write('\nSingular values of '+str(n_virtual_shell+1)+' virtual shells\n')
        outfile.write('Shells constructed with the %s operator\n' % operator_name)

        # First virtual shell
        e_corr = []
        embed.wfn = scf_wfn
        embed.partition_method = 'all'
        embed.projection_basis = virt_proj_basis 
        n_act_aos = embed.count_active_aos()
        shell_size = n_act_aos
        nbf = wfn.basisset().nbf()
        n_env_mos = wfn.nalpha() - n_act_mos 
        Cvir_eff = scf_wfn.Ca().np[:,n_act_mos:(nbf-n_env_mos)]

        mints = psi4.core.MintsHelper(embed.proj_wfn.basisset())
        S_min = mints.ao_overlap().np
        S_AB = mints.ao_overlap(embed.proj_wfn.basisset(), scf_wfn.basisset()).np
        Cvir_proj = np.linalg.inv(S_min[:n_act_aos,:n_act_aos]) @ S_AB[:n_act_aos, :] @ Cvir_eff
        X = Cvir_proj.T @ S_AB[:n_act_aos, :] @ Cvir_eff
        u, s, v = np.linalg.svd(X, full_matrices = True)
        s = s[:n_act_aos]
        shell_size = (s>=1.0e-15).sum()
        C_span = Cvir_eff @ v.T[:,:shell_size]
        C_ker = Cvir_eff @ v.T[:,shell_size:]
        sigma = s[:shell_size]

        eps_span, C_pseudo_span = embed.pseudocanonical(C_span)
        eps_ker, C_pseudo_ker = embed.pseudocanonical(C_ker)
        Cvir_pseudo = np.hstack((C_pseudo_span, C_pseudo_ker))
        eps_pseudo = np.concatenate((eps_span, eps_ker))
        e_corr.append(embed.energy(eps_pseudo, Cvir_pseudo, shell_size, high_level, n_env_mos))
        print_sigma(sigma, 0, outfile)
        outfile.write('{}-in-{} energy of shell # {} with {} orbitals = {:^12.10f}\n'.format(high_level.upper(), wfn.functional().name(), 0, shell_size, embed_SCF + e_corr[0]))

        embed.molden(C_span, C_ker, 0, n_env_mos)

        max_shell = int((nbf-wfn.nalpha())/shell_size)-1
        if n_virtual_shell > int((nbf-n_act_mos)/shell_size):
            n_virtual_shell = max_shell
        elif (nbf-wfn.nalpha()) % shell_size == 0:
            n_virtual_shell = max_shell - 1

        operator = embed.ao_operator(operator_name)
            
        for i in range(1, n_virtual_shell+1):

            mo_operator = C_span.T @ operator @ C_ker
            u, s, v = np.linalg.svd(mo_operator, full_matrices = True)
            sigma = s[:shell_size]
            print_sigma(sigma, i, outfile)

            C_span = np.hstack((C_span, C_ker @ v.T[:,:shell_size]))
            C_ker = C_ker @ v.T[:,shell_size:]
            embed.molden(C_span, C_ker, i, n_env_mos)
            embed.heatmap(C_span, C_ker, i, operator)

            eps_span, C_pseudo_span = embed.pseudocanonical(C_span)
            eps_ker, C_pseudo_ker = embed.pseudocanonical(C_ker)
            Cvir_pseudo = np.hstack((C_pseudo_span, C_pseudo_ker))
            eps_pseudo = np.concatenate((eps_span, eps_ker))
            e_corr.append(embed.energy(eps_pseudo, Cvir_pseudo, (i+1)*shell_size, high_level, n_env_mos))
            outfile.write('{}-in-{} energy of shell # {} with {} orbitals = {:^12.10f}\n'.format(high_level.upper(), wfn.functional().name(), i, shell_size*(i+1), embed_SCF + e_corr[i]))

        if n_virtual_shell == max_shell:

            C_span = np.hstack((C_span, C_ker))
            eps_span, C_pseudo_span = embed.pseudocanonical(C_span)
            e_corr.append(embed.energy(eps_span, C_pseudo_span, len(eps_span), high_level, n_env_mos))
            outfile.write('Energy of all ({}) orbitals = {:^12.10f}\n'.format(C_span.shape[1], embed_SCF + e_corr[-1]))


    # --------------------------------------- More printing  ----------------------------------------------

        outfile.write('\n\nSummary of virtual shell energy convergence\n\n')
        outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format('Shell #', '# active', ' Correlation', 'Total'))
        outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(8*'', 'virtuals', 'energy', 'energy'))
        outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(7*'-', 8*'-', 13*'-', 16*'-'))
        for n in range(n_virtual_shell+1):
            outfile.write('{:^8d} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.format(n, shell_size*(n+1), e_corr[n],  embed_SCF + e_corr[n]))
        if len(eps_span) == nbf - wfn.nalpha():
            outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.format('Eff.', len(eps_span), e_corr[-1], embed_SCF + e_corr[-1]))
            outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.format('Full', len(eps_span)+n_env_mos, e_corr[-1], embed_SCF + e_corr[-1]))
        outfile.write('\n')
        outfile.write('Correction from the projected B\t = {:>16.2e}\n'.format(matrix_dot(P, D)))
        outfile.close()
    os.system('rm newH.dat')   
    

def run_open_shell(wfn, 
    mol, 
    n_active_atoms, 
    high_level, 
    partition_method = 'spade', 
    projection_basis = '', 
    n_virtual_shell = None,
    conf_space = None,
    sf_opts = {},
    delta_a = 0,
    delta_b = 0):

    working_basis = psi4.core.get_global_option('BASIS')

    # Saving some DFT info for the whole system 

    nre = mol.nuclear_repulsion_energy()
    if(wfn.functional().name() != 'HF'):
        Va_total = wfn.Va().clone().np
        Vb_total = wfn.Vb().clone().np
        E_xc_total = psi4.core.VBase.quadrature_values(wfn.V_potential())["FUNCTIONAL"]
    else:
        E_xc_total = 0.0
        Va_total = np.zeros([wfn.nso(), wfn.nso()])
        Vb_total = np.zeros([wfn.nso(), wfn.nso()])

    # Partitioning the orbital space 
    
    embed = Embed(wfn, mol, n_active_atoms, projection_basis)

    n_act_aos = embed.count_active_aos()

    # Whether or not to project occupied orbitals onto another (minimal) basis
    if projection_basis != '': 
        S, Ca_occ = embed.basis_projection(wfn.Ca_subset("AO","OCC").np,\
        embed.proj_wfn.basisset(), wfn.basisset())
        S, Cb_occ = embed.basis_projection(wfn.Cb_subset("AO","OCC").np,\
        embed.proj_wfn.basisset(), wfn.basisset())
    else:
        S = wfn.S().np
        Ca_occ = wfn.Ca_subset("AO", "OCC").np
        Cb_occ = wfn.Cb_subset("AO", "OCC").np

    Va, sigma_a = embed.orbital_rotation(S, Ca_occ, n_act_aos)
    Vb, sigma_b = embed.orbital_rotation(S, Cb_occ, n_act_aos)

    # Partitioning method (spade vs. all AOs in projected basis)
    if partition_method == 'spade':
        ds_a = [-(sigma_a[i+1]-sigma_a[i]) for i in range(wfn.nalpha()-1)]
        n_act_alpha = np.argpartition(ds_a,-1)[-1]+1
        ds_b = [-(sigma_b[i+1]-sigma_b[i]) for i in range(wfn.nbeta()-1)]
        n_act_beta = np.argpartition(ds_b,-1)[-1]+1
    else:
       assert projection_basis != '', '\nDefine a projection (preferably a minimal) basis' 
       n_act_mos = n_act_aos

    # Orbital rotation and partition
    Ca_A = wfn.Ca_subset("AO", "OCC").np @ Va.T[:,:n_act_alpha]
    Cb_A = wfn.Cb_subset("AO", "OCC").np @ Vb.T[:,:n_act_beta]
    Ca_B = wfn.Ca_subset("AO", "OCC").np @ Va.T[:,n_act_alpha:]
    Cb_B = wfn.Cb_subset("AO", "OCC").np @ Vb.T[:,n_act_beta:]

    # Retrieving the subsytem energy terms and potential matrices
    E_A, E_xc_A, Ja_A, Jb_A, Ka_A, Kb_A, Va_A, Vb_A = embed.open_shell_subsystem(Ca_A, Cb_A)
    E_B, E_xc_B, Ja_B, Jb_B, Ka_B, Kb_B, Va_B, Vb_B = embed.open_shell_subsystem(Ca_B, Cb_B)

    Da_A = Ca_A @ Ca_A.T
    Db_A = Cb_A @ Cb_A.T
    Da_B = Ca_B @ Ca_B.T
    Db_B = Cb_B @ Cb_B.T

    J_AB = 0.5*(matrix_dot(Ja_A, Da_B) + matrix_dot(Ja_A, Db_B) +\
                matrix_dot(Jb_A, Da_B) + matrix_dot(Jb_A, Db_B) +\
                matrix_dot(Ja_B, Da_A) + matrix_dot(Ja_B, Db_A) +\
                matrix_dot(Jb_B, Da_A) + matrix_dot(Jb_B, Db_A))

    K_AB = -0.5*wfn.functional().x_alpha()*(matrix_dot(Ka_B, Da_A) + matrix_dot(Kb_B, Db_A)+\
                                            matrix_dot(Ka_A, Da_B) + matrix_dot(Kb_A, Db_B))
    XC_AB = E_xc_total - E_xc_A - E_xc_B
    G = J_AB + K_AB + XC_AB

    wfn.Ca().copy(psi4.core.Matrix.from_array(Ca_A))
    wfn.Cb().copy(psi4.core.Matrix.from_array(Cb_A))
    psi4.driver.molden(wfn, 'before_pseudocanonical.molden')

    v, w = np.linalg.eigh(Ca_A.T @ wfn.Fa().np @ Ca_A)
    pseudoCa_A = Ca_A @ w
    v, w = np.linalg.eigh(Cb_A.T @ wfn.Fb().np @ Cb_A)
    pseudoCb_A = Cb_A @ w

    wfn.Ca().copy(psi4.core.Matrix.from_array(pseudoCa_A))
    wfn.Cb().copy(psi4.core.Matrix.from_array(pseudoCb_A))
    psi4.driver.molden(wfn, 'after_pseudocanonical.molden')

    # Defining the embedded core Hamiltonian. Uses the default mu=1e6. 
    Pa = 1e6*(wfn.S().np @ Da_B @ wfn.S().np)
    Pb = 1e6*(wfn.S().np @ Db_B @ wfn.S().np)

    Va_emb = Ja_B + Jb_B - wfn.functional().x_alpha()*Ka_B + Pa + Va_total - Va_A
    Vb_emb = Ja_B + Jb_B - wfn.functional().x_alpha()*Kb_B + Pb + Vb_total - Vb_A

    # Saving the embedded core Hamiltonian to 'newH.dat', which is read by my version of Psi4
    fa = open("Va_emb.dat","w")
    fb = open("Vb_emb.dat","w")
    for i in range(wfn.nso()):
        for j in range(wfn.nso()):
            fa.write("%s\n" % Va_emb[i,j])
            fb.write("%s\n" % Vb_emb[i,j])
    fa.close()
    fb.close()

    # Running embedded calculation 
    psi4.set_options({"save_jk": "true",\
                    'guess': 'read',\
                    'basis': working_basis,\
                    'docc': [n_act_beta],\
                    'socc': [n_act_alpha-n_act_beta],\
                    'frozen_uocc': [0],\
                    'reference': 'uhf',\
                    'cc_type': 'df',\
                    'scf_type': 'df'})
    if high_level == 'ccsd' or high_level == 'ccsd(t)':
        psi4.set_options({'df_ints_io' : 'save'})

    scf_eng, scf_wfn = psi4.energy('hf', return_wfn = True)
    os.system('rm V*')   
    psi4.driver.molden(scf_wfn, 'embedded.molden')

    # not gonna bother checking the overlap for now
    Sa = pseudoCa_A.T @ wfn.S().np @ scf_wfn.Ca_subset("AO","OCC").np
    u, s, vh = np.linalg.svd(Sa)
    overlap = np.linalg.det(u)*np.linalg.det(vh)*np.prod(s)
    Sb = pseudoCb_A.T @ wfn.S().np @ scf_wfn.Cb_subset("AO","OCC").np
    u, s, vh = np.linalg.svd(Sb)
    overlap += np.linalg.det(u)*np.linalg.det(vh)*np.prod(s)

    Da_emb = scf_wfn.Da().np 
    Db_emb = scf_wfn.Db().np 
    Ja_emb = scf_wfn.jk().J()[0].np
    Jb_emb = scf_wfn.jk().J()[1].np
    Ka_emb = scf_wfn.jk().K()[0].np
    Kb_emb = scf_wfn.jk().K()[1].np

    embed_E_A = matrix_dot(Da_emb + Db_emb, scf_wfn.H().np) + 0.5*matrix_dot(Da_emb + Db_emb, Ja_emb + Jb_emb)\
                -0.5*(matrix_dot(Da_emb, Ka_emb) + matrix_dot(Db_emb, Kb_emb))
    '''
    embed_E_A = matrix_dot(Da, scf_wfn.H().np) + matrix_dot(Db, scf_wfn.H().np) +\
                0.5*(matrix_dot(Da, scf_wfn.jk().J()[0].np) + matrix_dot(Db, scf_wfn.jk().J()[1].np) +\
                matrix_dot(Da, scf_wfn.jk().J()[1].np) + matrix_dot(Db, scf_wfn.jk().J()[0].np) -\
                (matrix_dot(Da, scf_wfn.jk().K()[0].np) + matrix_dot(Db, scf_wfn.jk().K()[1].np)))
    '''

    density_correction = matrix_dot(Va_emb, Da_emb - Da_A) + matrix_dot(Vb_emb, Db_emb - Db_A) 
    embed_SCF = embed_E_A + E_B + G + nre + density_correction

    if high_level == 'mp2' or high_level == 'ccsd':
        # Printing embedded HF results
        outfile = open('embedding.log', 'w')
        banner(outfile, partition_method)
        outfile.write('\n\n')
        outfile.write('Energy values in atomic units\n')
        outfile.write('Embedded calculation: '+high_level.upper()+'-in-'+wfn.functional().name()+'\n')
        outfile.write('\n')
        if partition_method == 'spade':
            if projection_basis == '':
                outfile.write('Orbital partition method: SPADE\n')
            else:
                outfile.write('Orbital partition method: SPADE from occupied space projected onto '+projection_basis.upper()+ '\n')
        else:
            outfile.write('Orbital partition method: All AOs in '+projection_basis.upper()+' from atoms in A\n')
        
        print_uhf(wfn, n_act_alpha, n_act_beta, E_A, E_B, G, nre, embed_E_A, density_correction, embed_SCF, overlap, outfile)
        # --------------------------------- Post embedded HF calculation ---------------------------------------

        if isinstance(n_virtual_shell, str) or n_virtual_shell == None:
            emb_e, emb_wfn = psi4.energy(high_level, ref_wfn=scf_wfn, return_wfn=True)
            corr = psi4.core.get_variable("CURRENT CORRELATION ENERGY")
            total_E = embed_SCF + corr
            outfile.write('Embedded {:>5}-in-{:<5} E[A] \t = {:>16.10f}\n'.format(high_level.upper(), wfn.functional().name(), embed_SCF + corr))

        # not going to try to get the virtual shells working with open shells for now

    elif high_level == 'fock-ci':
        # Calling Shannon's Fock-CI code

        # obtain RAS spaces
        ras1 = scf_wfn.doccpi()[0]
        ras2 = scf_wfn.soccpi()[0]
        ras3 = scf_wfn.basisset().nbf() - (ras1 + ras2)
        # get Fock and MO coefficients
        Ca = psi4.core.Matrix.to_array(scf_wfn.Ca())
        Cb = psi4.core.Matrix.to_array(scf_wfn.Cb())
        Fa, Fb = get_F(scf_wfn)
        # get two-electron integrals
        # I'm going to assume DF is always used
        
        #if(sf_opts['INTEGRAL_TYPE']=="FULL"):
            #tei_int = TEIFull(scf_wfn.Ca(), scf_wfn.basisset(), ras1, ras2, ras3,ref_method='PSI4')
        #if(sf_opts['INTEGRAL_TYPE']=="DF"):
            # if user hasn't defined which aux basis to use, default behavior
            # is to use the one from Psi4 scf_wfn
            #if(sf_opts['AUX_BASIS_NAME'] == ""):


        aux_basis_name = scf_wfn.basisset().name()
        aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", 
                                                 "JKFIT", aux_basis_name)
        tei_int = TEIDF(scf_wfn.Ca(), scf_wfn.basisset(), aux_basis, ras1, ras2,
                            ras3, conf_space, ref_method='PSI4')
        out = do_sf_np(delta_a, delta_b, ras1, ras2, ras3, Fa, Fb, tei_int, scf_eng,
                       conf_space=conf_space, sf_opts=sf_opts)
        # return appropriate values
        if(isinstance(out, tuple)):
            if(sf_opts['RETURN_WFN']):
                out = out + (scf_wfn,)
        elif(sf_opts['RETURN_WFN']):
            out = (out,) + (scf_wfn,)
        return out

