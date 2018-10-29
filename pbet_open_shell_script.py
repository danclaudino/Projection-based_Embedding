
"""
    Daniel Claudino - October/2018

    Script to run a projection-based embedding theory
    calculation with a closed-shell (HF/KS) reference
    in conjunction with the pbet_closed_shell.py module.

    Before using this script, make sure you have changed your 
    Psi4 installation according to the instructions provided.

"""
import os
import psi4 
import numpy as np
import pbet_open_shell as pbet

"""

    *** Start of the user input ***

    o Set memory
    o Set number of threads
    o Set molecule and job specification
    o Arrange your atoms in order to have the active ones come first 
    o Set 'nAtomEnv' as the number of atoms in the environment 
    o Set 'low_theory' as the level of theory for the environment 
    o Set 'high_theory' as the level of theory for the embedded system
    o Set 'basis' as the basis set

    Results from the embedded calculation are saved in "input.log"

"""

psi4.set_memory('1000 MB')
psi4.core.set_num_threads(6)
low_theory = ''
high_theory = ''
basis = ''
nAtomEnv = 

mol = psi4.geometry ("""

symmetry c1
""")

"""
    *** End of the user input ***

    Running a Psi4 calculation at the 'low_theory'/'basis' level
"""

psi4.set_options({"save_jk": "true", 'basis': basis, "reference": "uhf"})
eng, wfn = psi4.energy(low_theory, return_wfn=True)

"""
    Running embedding calculation
"""

pbet.embedding(wfn, mol, nAtomEnv, high_theory, basis)


