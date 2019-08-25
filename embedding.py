import numpy as np
import psi4
import scipy.linalg
import os
from copy import deepcopy
from decimal import *
#from embedding_functions import *

class Psi4Embed:
    """ Parent class with methods for projection-based embedding calculations."""

    def __init__(self, wfn, mol, n_active_atoms, projection_basis = ''):
        """Initialize the Embed class.

        Args:
            wfn (Psi4 Wavefunction object): the supermolecular wavefunction at the low level of theory.
            mol (Psi4 molecule object): the molecule object.
            n_active_atoms (int): number of atoms in the active subsystem.
            projection_basis (str): label of projection basis

        """

        self.wfn = wfn
        self.mol = mol
        self.n_active_atoms = n_active_atoms
        self.projection_basis = projection_basis

    @property
    def wfn(self):
        return self._wfn

    @wfn.setter
    def wfn(self, new_wfn):
        self._wfn = new_wfn

    @property
    def projection_basis(self):
        return self._projection_basis

    @projection_basis.setter
    def projection_basis(self, new_projection_basis):
        self._projection_basis = new_projection_basis

    @property
    def operator_name(self):
        return self._operator_name

    @operator_name.setter
    def operator_name(self, new_operator_name):
        self._operator_name = new_operator_name

    @staticmethod
    def dot(A, B):
        """ Computes the trace (dot product) of matrices A and B

        Args:
            A, B (numpy.array): matrices to compute tr(A*B)

        Returns:
            The trace (dot product) of A * B
        """

        return np.einsum('ij,ij', A, B) 


    def orbital_rotation(self, S, C, n_active_aos):
        """SVD the orbitals projected onto A in the orthogonal AO basis to rotate the orbitals.
        
        Args:
            S (numpy.array): overlap matrix.
            C (numpy.array): MO coefficient matrix.
            n_active_aos (int): number of atomic orbitals in the active atoms.

        Returns:
            v (numpy.array): right singular vectors of projected orthogonal orbitals (SPADE).
            s (numpy.array): singular vectors.
        """

        S_half = scipy.linalg.fractional_matrix_power(S, 0.5)
        C_orthogonal_ao = (S_half @ C)[:n_active_aos, :]
        u, s, v = np.linalg.svd(C_orthogonal_ao, full_matrices=True)

        return v, s

    def ao_operator(self, operator_name):
        """Returns the matrix of the operator chosen to construct the shells.
        
        Args:
            operator_name (str): name/letter of the operator used to construct the shells.

        Returns:

            K (numpy.array): exchange.
            V (numpy.array): electron-nuclei potential.
            T (numpy.array): kinetic energy.
            H (numpy.array): core Hamiltonian.
            S (numpy.array): overlap matrix.
            F (numpy.array): Fock matrix.
            K_orb (numpy.array): operator for K orbitals (Feller and Davidson, JCP, 74, 3977 (1981)).

        """

        if operator_name == 'K' or operator_name == 'K_orb':
            jk = psi4.core.JK.build(self.wfn.basisset(), self.wfn.get_basisset("DF_BASIS_SCF"),"DF")
            jk.set_memory(int(1.25e9))
            jk.initialize()
            jk.print_header()
            jk.C_left_add(self.wfn.Ca())
            jk.compute()
            jk.C_clear()
            jk.finalize()
            K = jk.K()[0].np
            if operator_name == K:
                return K
            else:
                K_orb = 0.06*self.wfn.Fa().np - K 
                return K_orb

        elif operator_name == 'V':
            mints = psi4.core.MintsHelper(self.wfn.basisset())
            V = mints.ao_potential().np
            return V

        elif operator_name == 'T':
            mints = psi4.core.MintsHelper(self.wfn.basisset())
            T = mints.ao_kinetic().np
            return T

        elif operator_name == 'H':
            H = self.wfn.H().np
            return H

        elif operator_name == 'S':
            S = self.wfn.S().np
            return S

        elif operator_name == 'F':
            F = self.wfn.Fa().np
            return F


    def pseudocanonical(self, C):
        """Returns pseudocanonical orbitals and the corresponding eigenvalues.
        
        Args:
            C (numpy.array): MO coefficients of orbitals to be pseudocanonicalized.

        Returns:
            C_pseudo (numpy.array): pseudocanonical orbitals.
            epsilon (numpy.array): diagonal elements of the Fock matrix in the C_pseudo basis.
        
        """

        moF = C.T @ self.wfn.Fa().np @ C
        epsilon, w = np.linalg.eigh(moF)
        C_pseudo = C @ w

        return epsilon, C_pseudo


    def energy(self, epsilon, C, nfrz, level, n_env_mos):
        """Computes the correlation energy for the current set of active virtual orbitals.
        
        Args:
            epsilon (numpy.array): diagonal elements of the Fock matrix in C basis.
            C (numpy.array): active virtuals orbitals.
            nfrz (int): number of frozen orbitals.
            level (str): correlated level of theory.
            n_env_mos (int): number of orbitals in the environment.

        Returns:
            correlation_energy (float): correlation energy for the active virtuals at the given level of theory. 
        
        """

        # Rearrange the orbitals and orbital energies and pass them to the corresponding Psi4 arrays
        n_shift = self.wfn.nso() - n_env_mos
        orbs = np.hstack((self.wfn.Ca_subset("AO","OCC").np, C, self.wfn.Ca().np[:,n_shift:]))
        eigs = np.concatenate((self.wfn.epsilon_a_subset("AO", "OCC").np, epsilon, self.wfn.epsilon_a().np[n_shift:]))
        self.wfn.Ca().copy(psi4.core.Matrix.from_array(orbs))
        self.wfn.epsilon_a().np[:] = eigs[:]

        # Update the number of frozen orbitals and compute energy
        nfrz = self.wfn.nso() - (self.wfn.nalpha() - n_env_mos) - nfrz
        frzvpi = psi4.core.Dimension.from_list([nfrz])
        self.wfn.new_frzvpi(frzvpi)
        mp2_eng, mp2_wfn = psi4.energy(level,ref_wfn=self.wfn, return_wfn=True)
        correlation_energy = psi4.core.get_variable(level.upper()+" CORRELATION ENERGY")

        return correlation_energy


    def count_active_aos(self):
        """Computes the number of AOs from atoms in the active subsystem (A).
        
        Returns:
            n_active_aos (int): number of atomic orbitals in the active atoms.

        """

        if self.projection_basis == '':
            basisset = self.wfn.basisset()
            nbf = basisset.nbf()

        else:
            self.proj_wfn = psi4.core.Wavefunction.build(self.mol, self.projection_basis)
            basisset = self.proj_wfn.basisset()
            nbf = basisset.nbf()

        active_atoms = list(range(self.n_active_atoms))
        n_active_aos = 0

        for ao in range(nbf):
            for atom in active_atoms:
                if basisset.function_to_center(ao) == atom:
                   n_active_aos += 1
        
        return n_active_aos


    def basis_projection(self, C, projection_basis, working_basis):
        """Defines a projection of orbitals in one basis onto another.
        
        Args: 
            C (numpy.array): MO coefficients to be projected.
            projection_basis (psi4 BasisSet object): basis onto which C is to be projected.
            working_basis (psi4 BasisSet object): basis on which C is currently expanded.

        Returns:
            S (numpy.array): overlap matrix in the projection basis. 
            C_projected (numpy.array): MO coefficients of orbitals projected onto projection_basis.
        
        """

        mints = psi4.core.MintsHelper(projection_basis)
        S = mints.ao_overlap().np
        S_AB = mints.ao_overlap(projection_basis, working_basis).np
        C_projected = np.linalg.inv(S) @ S_AB @ C

        return S, C_projected


    def spade_partition(self, sigma):
        """(Deprecated) Defines the orbital partition for the occupied space.
        
        Args:
            sigma (numpy.array): singular values.

        Returns:
            partition_index (int): index of the last MO before the largest gap in sigma.

        """

        ds = [-(sigma[i+1]-sigma[i]) for i in range(self.wfn.nalpha()-1)]
        partition_index = np.argpartition(ds,-1)[-1]+1

        return partition_index


    def open_shell_subsystem(self, Ca, Cb):
        """
        Computes the potential matrices J, K, and V and subsystem energies for open shell cases.

        Args:
            Ca (numpy.array): alpha MO coefficients.
            Cb (numpy.array): beta MO coefficients.

        Returns:
            E (float): total energy of subsystem.
            E_xc (float): (DFT) Exchange-correlation energy of subsystem .
            Ja (numpy.array): alpha Coulomb matrix of subsystem.
            Jb (numpy.array): beta Coulomb matrix of subsystem.
            Ka (numpy.array): alpha Exchange matrix of subsystem.
            Kb (numpy.array): beta Exchange matrix of subsystem.
            Va (numpy.array): alpha Kohn-Sham potential matrix of subsystem.
            Vb (numpy.array): beta Kohn-Sham potential matrix of subsystem.

        """
        
        Da = Ca @ Ca.T
        Db = Cb @ Cb.T
        orbs_a = psi4.core.Matrix.from_array(Ca)
        orbs_b = psi4.core.Matrix.from_array(Cb)

        # J and K
        jk = psi4.core.JK.build(self.wfn.basisset(), self.wfn.get_basisset("DF_BASIS_SCF"),"DF")
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(orbs_a)
        jk.C_left_add(orbs_b)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        Ja = jk.J()[0].np
        Jb = jk.J()[1].np
        Ka = jk.K()[0].np
        Kb = jk.K()[1].np
        
        if hasattr(self.wfn, 'functional'): 
            # V_ks only if SCF is not HF
            if(self.wfn.functional().name() != 'HF'):
                self.wfn.Da().copy(psi4.core.Matrix.from_array(Da))
                self.wfn.Db().copy(psi4.core.Matrix.from_array(Db))
                self.wfn.form_V()
                Va = self.wfn.Va().clone().np
                Vb = self.wfn.Vb().clone().np
                E_xc = psi4.core.VBase.quadrature_values(self.wfn.V_potential())["FUNCTIONAL"]

            else:
                E_xc = 0.0
                Va = np.zeros([self.wfn.nso(), self.wfn.nso()])
                Vb = np.zeros([self.wfn.nso(), self.wfn.nso()])

            # Energy
            E = self.dot(self.wfn.H().np, Da) + self.dot(self.wfn.H().np, Db) +\
                0.5*(self.dot(Ja, Da) + self.dot(Jb, Db) + self.dot(Ja, Db) + self.dot(Jb, Da) -\
                self.wfn.functional().x_alpha()*(self.dot(Ka, Da) + self.dot(Kb, Db))) + E_xc

            return E, E_xc, Ja, Jb, Ka, Kb, Va, Vb

        else:
            E = self.dot(self.wfn.H().np, Da) + self.dot(self.wfn.H().np, Db) +\
                0.5*(self.dot(Ja, Da) + self.dot(Jb, Db) + self.dot(Ja, Db) + self.dot(Jb, Da) -\
                self.dot(Ka, Da) + self.dot(Kb, Db))

            return E, Ja, Jb, Ka, Kb


    def closed_shell_subsystem(self, C):
        """
        Computes the potential matrices J, K, and V and subsystem energies.

        Args:
            C (numpy.array): MO coefficients of subsystem.

        Returns:
            E (float): total energy of subsystem.
            E_xc (float): (DFT) Exchange-correlation energy of subsystem.
            J (numpy.array): Coulomb matrix of subsystem.
            K (numpy.array): Exchange matrix of subsystem.
            V (numpy.array): Kohn-Sham potential matrix of subsystem.

        """

        D = C @ C.T
        orbs = psi4.core.Matrix.from_array(C)

        # J and K
        if hasattr(self.wfn, 'get_basisset'):
            jk = psi4.core.JK.build(self.wfn.basisset(),self.wfn.get_basisset("DF_BASIS_SCF"),"DF")
        else:
            jk = psi4.core.JK.build(self.wfn.basisset())
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(orbs)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        J = jk.J()[0].np
        K = jk.K()[0].np

        if hasattr(self.wfn, 'functional'): 
            # V_ks only if SCF is not HF
            if(self.wfn.functional().name() != 'HF'):
                self.wfn.Da().copy(psi4.core.Matrix.from_array(D))
                self.wfn.form_V()
                V = self.wfn.Va().clone().np
                E_xc = psi4.core.VBase.quadrature_values(self.wfn.V_potential())["FUNCTIONAL"]

            else:
                E_xc = 0.0
                V = np.zeros([self.wfn.nso(), self.wfn.nso()])

            # Energy
            E = 2.0*np.einsum('ij, ij', D, self.wfn.H().np) + \
                2.0*np.einsum('ij, ij', D, J) - \
                self.wfn.functional().x_alpha()*(np.einsum('ij, ij', D, K)) + E_xc 

            return E, E_xc, J, K, V

        else:

            return J, K


    def orthonormalize(self, S, C, n_non_zero):
        """(Deprecated) Orthonormalizes a set of orbitals (vectors).

        Args:
            S (numpy.array): overlap matrix in AO basis.
            C (numpy.array): MO coefficient matrix (vectors to be orthonormalized).
            n_non_zero (int): number of orbitals that have non-zero norm. 

        Returns:
            C_orthonormal (numpy.array): set of n_non_zero orthonormal orbitals.
            
        """

        overlap = C.T @ S @ C
        v, w = np.linalg.eigh(overlap)
        idx = v.argsort()[::-1]
        v = v[idx]
        w = w[:,idx]
        C_orthonormal = C @ w
        for i in range(n_non_zero):
            C_orthonormal[:,i] = C_orthonormal[:,i]/np.sqrt(v[i])

        return C_orthonormal[:,:n_non_zero]


    def molden(self, C_span, C_ker, shell, n_env_mos):
        """Creates molden file from orbitals at the i-th shell.

        Args: 
            C_span (numpy.array): MO coefficients of the active virtuals.
            C_ker (numpy.array): MO coefficients of the frozen virtuals.
            shell (int): shell index.
            n_env_mos (int): number of orbitals in the environment.

        """
        
        n_shift = self.wfn.nso() - n_env_mos
        orbs = np.hstack((self.wfn.Ca_subset("AO","OCC").np, C_span, C_ker, self.wfn.Ca().np[:,n_shift:]))
        self.wfn.Ca().copy(psi4.core.Matrix.from_array(orbs))

        psi4.driver.molden(self.wfn, str(shell)+'.molden')


    def heatmap(self, C_span, C_ker, shell, operator):
        """Creates heatmap file from orbitals at the i-th shell.

        Args: 
            C_span (numpy.array): MO coefficients of the active virtuals.
            C_ker (numpy.array): MO coefficients of the frozen virtuals.
            shell (int): shell index.
            operator (numpy.array): matrix representation of the operator chosen to construct the shells.

        """
        
        orbs = np.hstack((C_span, C_ker))
        moOp = orbs.T @ operator @ orbs
        np.savetxt('heatmap_'+str(shell)+'.dat', moOp)

