import psi4
import sys
sys.path.append('path_to_Projection-based_Embedding')
import run_embedding

# Enter embedding options in the dictionary below:

options = {}
options['basis'] = 'sto-3g' # basis set 
options['low_level'] = 'b3lyp' # level of theory of the environment 
options['high_level'] = 'mp2' # level of theory of the embedded system
options['n_active_atoms'] = 2 # number of active atoms (first n atoms in the geometry string)
options['geometry'] = """
O       -1.1867 -0.2472  0.0000
H       -1.9237  0.3850  0.0000
H       -0.0227  1.1812  0.8852
C        0.0000  0.5526  0.0000
H       -0.0227  1.1812 -0.8852
C        1.1879 -0.3829  0.0000
H        2.0985  0.2306  0.0000
H        1.1184 -1.0093  0.8869
H        1.1184 -1.0093 -0.8869

"""

# Extra options
options['charge'] = 0 # Charge
options['reference'] = 'rhf' # Change to uhf/rohf for open-shells
options['memory'] = '1000 MB' # Memory to be allocated for the Psi4 run
options['num_threads'] = 8 # Number of threads for the Psi4 run
options['partition_method'] = 'spade' # Partition method (primarily for the occupied space)
options['projection_basis'] = '' # Projection basis for the occupied space
options['virt_proj_basis'] = 'sto-3g' # Projection basis for the virtual space
options['operator_name'] = 'F' # operator to be used to grow the virtual shells
options['n_occupied_shell'] = 0 # number of occupied shells (currently not working)
options['n_virtual_shell'] = '' # number of virtuals shells
options['molden'] = True # whether or not generate molden and heatmap files

#run_embedding.psi4_open_shell(options)
run_embedding.psi4_closed_shell(options)
