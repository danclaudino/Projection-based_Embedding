import numpy as np

def matrix_dot(A, B):
    """ Computes the trace (dot product) of matrices A and B

    Args:
        A, B (numpy.array): matrices to compute tr(A*B)

    Returns:
        The trace (dot product) of A * B
    """

    return np.einsum('ij,ij', A, B) 

def print_sigma(sigma, n, outfile):
    """ Formats the printing of singular values for the PRIME shells.

    Args:
        sigma (numpy.array or list): singular values
        n (int): PRIME shell index
        outfile (file): file where sigma is to be printed

    """

    outfile.write('\n{:>10} {:>2d}\n'.format('Shell #', n))
    outfile.write('  ------------\n')
    outfile.write('{:^5} \t {:^14}\n'.format('#','Singular value'))
    for i in range(len(sigma)):
        outfile.write('{:^5d} \t {:>12.10f}\n'.format(i, sigma[i]))

    outfile.write('\n')

def banner(outfile, partition_method):

    outfile.write('\n')
    outfile.write(' \t------------------------------------------------------\n')
    outfile.write(' \t|             Projection-based Embedding             |\n')
    outfile.write(' \t|                                                    |\n')
    outfile.write(' \t|                 Daniel Claudino                    |\n')
    outfile.write(' \t|                    June 2019                       |\n')
    outfile.write(' \t------------------------------------------------------\n')
    outfile.write('\n')
    outfile.write(' Main references: \n\n')
    outfile.write('     Projection-based embedding:\n')
    outfile.write('     F.R. Manby, M. Stella, J.D. Goodpaster, T.F. Miller. III,\n')
    outfile.write('     J. Chem. Theory Comput. 2012, 8, 2564.\n\n')

    if partition_method == 'spade':
        outfile.write('     SPADE partition: D. Claudino, N.J. Mayhall,\n')
        outfile.write('     J. Chem. Theory Comput. 2019, 15, 1053.\n')


def print_rhf(wfn, n_act_mos, E_A, E_B, G, nre, embed_E_A, density_correction, embed_SCF, overlap, outfile):

    outfile.write('\n')
    outfile.write('Number of orbitals in A: %s\n' % n_act_mos)
    outfile.write('Number of orbitals in B: %s\n' % (wfn.nalpha() - n_act_mos))
    outfile.write('\n')
    outfile.write('{:<7} {:<6} \t\t\t = {:>16.10f}\n'.format('('+wfn.functional().name()+')', 'E[A]', E_A))
    outfile.write('{:<7} {:<6} \t\t\t = {:>16.10f}\n'.format('('+wfn.functional().name()+')', 'E[B]', E_B))
    outfile.write('Intersystem interaction G \t = {:>16.10f}\n'.format(G))
    outfile.write('Nuclear repulsion energy \t = {:>16.10f}\n'.format(nre))
    outfile.write('{:<7} {:<6} \t\t\t = {:>16.10f}\n'.format('('+wfn.functional().name()+')', 'E[A+B]', E_A + E_B + G + nre))
    outfile.write('\n')
    outfile.write('Embedded SCF E[A] \t\t = {:>16.10f}\n'.format(embed_E_A))
    outfile.write('Embedded density correction \t = {:>16.10f}\n'.format(density_correction))
    outfile.write('Embedded HF-in-{:<5} E[A] \t = {:>16.10f}\n'.format(wfn.functional().name(), embed_SCF))
    outfile.write('<SD_before|SD_after> \t\t = {:>16.10f}\n'.format(abs(overlap)))
    outfile.write('\n')


def print_uhf(wfn, n_act_alpha, n_act_beta, E_A, E_B, G, nre, embed_E_A, density_correction, embed_SCF, overlap, outfile):

    outfile.write('\n')
    outfile.write('Number of alpha orbitals in A: %s\n' % n_act_alpha)
    outfile.write('Number of beta orbitals in A: %s\n' % n_act_beta)
    outfile.write('Number of alpha orbitals in B: %s\n' % (wfn.nalpha() - n_act_alpha))
    outfile.write('Number of beta orbitals in B: %s\n' % (wfn.nbeta() - n_act_beta))
    outfile.write('\n')
    outfile.write('{:<7} {:<6} \t\t\t = {:>16.10f}\n'.format('('+wfn.functional().name()+')', 'E[A]', E_A))
    outfile.write('{:<7} {:<6} \t\t\t = {:>16.10f}\n'.format('('+wfn.functional().name()+')', 'E[B]', E_B))
    outfile.write('Intersystem interaction G \t = {:>16.10f}\n'.format(G))
    outfile.write('Nuclear repulsion energy \t = {:>16.10f}\n'.format(nre))
    outfile.write('{:<7} {:<6} \t\t\t = {:>16.10f}\n'.format('('+wfn.functional().name()+')', 'E[A+B]', E_A + E_B + G + nre))
    outfile.write('\n')
    outfile.write('Embedded SCF E[A] \t\t = {:>16.10f}\n'.format(embed_E_A))
    outfile.write('Embedded density correction \t = {:>16.10f}\n'.format(density_correction))
    outfile.write('Embedded HF-in-{:<5} E[A] \t = {:>16.10f}\n'.format(wfn.functional().name(), embed_SCF))
    outfile.write('<SD_before|SD_after> \t\t = {:>16.10f}\n'.format(abs(overlap/2.0)))
    outfile.write('\n')


