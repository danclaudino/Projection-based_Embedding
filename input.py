import os
import psi4
import sys
import numpy as np
import run_embedding

psi4.set_memory('1000 MB')
psi4.core.set_num_threads(8)

basis = 'cc-pvdz'
low_level = 'b3lyp'
high_level = 'mp2'
n_active_atoms = 2
partition_method = 'spade'
projection_basis = 'sto-3g'

mol = psi4.geometry ("""
O       -1.1867         -0.2472         0.0000
H       -1.9237         0.3850  0.0000
H       -0.0227         1.1812  0.8852
C       0.0000  0.5526  0.0000
H       -0.0227         1.1812  -0.8852
C       1.1879  -0.3829         0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093         0.8869
H       1.1184  -1.0093         -0.8869

symmetry c1
""")

psi4.core.be_quiet()
psi4.core.set_output_file('output.dat', True)
psi4.set_options({"save_jk": "true", 'basis': basis, 'ints_tolerance': 1.0e-10})
eng, wfn = psi4.energy(low_level, return_wfn=True)

run_embedding.run_closed_shell(wfn, mol, n_active_atoms, high_level, partition_method = partition_method)
