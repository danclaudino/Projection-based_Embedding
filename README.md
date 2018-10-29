# Projection-based Embedding with SPADE orbitals

Python script and modules to run projection-based embedding calculations using SPADE orbitals

## Requirements

* Psi4 source code (v.1.2 or higher)
* Python (2.7 or higher)
* NumPy (1.14 or higher) 

## Necessary modifications to Psi4 source code

Psi4 does not currently have a functionality that allows the user to provide a wavefunction object to an SCF calculation (HF/KS). Since projection-based embedding works by incorporating the embedding potential to the core Hamiltonian (or Fock matrix) the source code needs to be slightly modified. 

* For restricted references (closed-shells):

1. Open `top-level-psi4-dir/psi4/src/psi4/libscf_solver/hf.cc` 
2. Find the `form_H()` function
3. Add the following  to the end of `form_H()`:  
`    if (std::ifstream("newH.dat"))  `  
`    {  `  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        outfile->Printf( " ===> Reading embedded core Hamiltonian <=== \n\n");`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        int nso = 0;  `  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        for(int h=0; h < nirrep_; h++) nso += nsopi_[h];`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        double H_elem;`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        FILE* input = fopen("newH.dat", "r");`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        for(int i=0; i < nso; i++) {`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            for (int j=0; j < nso; j++) { `  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            int statusvalue=fscanf(input, "%lf", &H_elem);`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            H_->set(i, j, H_elem); // ignore nuclear potential`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            }`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        }`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        fclose(input);`  
`    }`  
  
* For unrestricted references (open-shells):

1. Open `top-level-psi4-dir/psi4/src/psi4/libscf_solver/uhf.cc`
2. Find the `common_init()` function and add the following:
`    Va_emb  = SharedMatrix(factory_->create_matrix("Embedding V alpha"));`
`    Vb_emb  = SharedMatrix(factory_->create_matrix("Embedding V beta"));`


3. Find the `form_F()` function and make sure it begins like this:  
`    Fa_->copy(H_);`
`    if (std::ifstream("Va_emb.dat"))`
`    {`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        outfile->Printf( " ===> Reading alpha embedding potential <=== \n\n");`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        int nso = 0;`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        for(int h=0; h < nirrep_; h++) nso += nsopi_[h];`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        double V_ij;`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        FILE* input = fopen("Va_emb.dat", "r");`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        for(int i=0; i < nso; i++) {`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            for (int j=0; j < nso; j++) {`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            int statusvalue=fscanf(input, "%lf", &V_ij);`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            Va_emb->set(i, j, V_ij);`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            }`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        }`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        fclose(input);`
`    }`
`    Fa_->add(Va_emb);`
`    Fa_->add(Ga_);`
`    Fb_->copy(H_);`
`    if (std::ifstream("Vb_emb.dat"))`
`    {`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        outfile->Printf( " ===> Reading beta embedding potential <=== \n\n");`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        int nso = 0;`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        for(int h=0; h < nirrep_; h++) nso += nsopi_[h];`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        double V_ij;`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        FILE* input = fopen("Vb_emb.dat", "r");`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        for(int i=0; i < nso; i++) {`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            for (int j=0; j < nso; j++) {`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            int statusvalue=fscanf(input, "%lf", &V_ij);`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            Vb_emb->set(i, j, V_ij);`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`            }`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        }`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`        fclose(input);`
`    }`
`    Fb_->add(Vb_emb);`

4. Open `top-level-psi4-dir/psi4/src/psi4/libscf_solver/uhf.h` and add the following to the UHF class:
`
    SharedMatrix Va_emb, Vb_emb;
`
  
In both cases, make sure you have add the `fstream` library in order to allow for the files to be read.

### Minor considerations
* ROKS is not implemented in Psi4, but the embedding potential does not seem to be affected by the choice of reference.
* The closed-shell case works by incorporating the embedding potential to the core Hamiltonian, while the same potential in included in the Fock matrix when using UHF/UKS references. This is because the core Hamiltonian is independent of spin. I could have the potential added to the Fock matrix in the closed-shell case as well, but as you can see, the open-shell case requires reading the alpha and beta potential at each iteration, and I would rather avoid it.
* This is set up primarily to work with GGA and hybrid functionals (I have wrote this with PBE and B3LYP in mind). Even though it should work with any functional (maybe with further modifications), I have not seen any reason to resort to more complicated functionals, as the embedding is fairly insensitive to the choice of functional.
