# Projection-based Embedding with SPADE and concentric local orbitals

Python modules to run projection-based embedding calculations using SPADE orbitals and concentric local orbitals

SPADE ([J. Chem. Theory Comput. 2019, 15, 1053.](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01112) provides a powerful way to partition occupied orbital spaces for projecion-based embedding. 

Concentric local orbitals ([ChemRxiv](https://chemrxiv.org/articles/Simple_and_Efficient_Truncation_of_Virtual_Spaces_in_Embedded_Wave_Functions_via_Concentric_Localization/8846108)) provides great computational efficiency by truncating the virtual space in line with the embedded wave function.

## How to use 

Fork my Psi4 repository and compile it according to the provided instructions in the Psi4 manual ([Compiling and Installing from Source](http://psicode.org/psi4manual/1.1/build_planning.html)). The modules and the example input file assume Psi4 is used as a Python module, which can be accomplished by following [Using Psi4 as a Python Module](http://psicode.org/psi4manual/1.1/build_planning.html). I have been using Python3.6, but I would expect there is nothing that prevents it to work with Python2 apart from some potential minor syntactic changes. SciPy/NumPy is also required.

