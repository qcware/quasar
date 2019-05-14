import numpy as np
from .aiem import AIEM
from .aiem_data import AIEMPauli
from .aiem_data import AIEMMonomer
from .aiem_data import AIEMUtil

class AIEMGradCheck(object):

    @staticmethod
    def compute_fd_gradient_pauli(
        aiem,
        h0=1.0E-7,
        h1=1.0E-7,
        h2=1.0E-7,
        print_level=0,
        ):
    
        # Print override
        aiem2 = AIEM(options=aiem.options.copy().set_values({
            'print_level' : print_level,
            }))
        
        G_fci = [AIEMPauli.zeros_like(aiem.aiem_hamiltonian_pauli) for _ in range(aiem.nstate)]
        G_vqe = [AIEMPauli.zeros_like(aiem.aiem_hamiltonian_pauli) for _ in range(aiem.nstate)]
        G_cis = [AIEMPauli.zeros_like(aiem.aiem_hamiltonian_pauli) for _ in range(aiem.nstate)]
    
        # E
        
        pauli_p = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
        pauli_p.E += h0
        aiem_p = AIEM(options=aiem2.options.copy().set_values({
            'aiem_hamiltonian_pauli' : pauli_p,
            }))
        aiem_p.compute_energy(
            param_values_ref=aiem.vqe_circuit.param_values,
            cis_C_ref=aiem.cis_C,
            vqe_C_ref=aiem.vqe_C,
            )
    
        pauli_m = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
        pauli_m.E -= h0
        aiem_m = AIEM(options=aiem2.options.copy().set_values({
            'aiem_hamiltonian_pauli' : pauli_m,
            }))
        aiem_m.compute_energy(
            param_values_ref=aiem.vqe_circuit.param_values,
            cis_C_ref=aiem.cis_C,
            vqe_C_ref=aiem.vqe_C,
            )
        
        for I in range(aiem.nstate):
            G_fci[I].E = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * h0)
            G_vqe[I].E = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * h0)
            G_cis[I].E = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * h0)
    
        # Z
        
        for A in range(aiem.N):
    
            pauli_p = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_p.Z[A] += h1
            aiem_p = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            pauli_m = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_m.Z[A] -= h1
            aiem_m = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
        
            for I in range(aiem.nstate):
                G_fci[I].Z[A] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h1)
                G_vqe[I].Z[A] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h1)
                G_cis[I].Z[A] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h1)
    
        # X
        
        for A in range(aiem.N):
    
            pauli_p = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_p.X[A] += h1
            aiem_p = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            pauli_m = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_m.X[A] -= h1
            aiem_m = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
        
            for I in range(aiem.nstate):
                G_fci[I].X[A] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h1)
                G_vqe[I].X[A] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h1)
                G_cis[I].X[A] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h1)

        # XX
        
        for A, B in aiem.aiem_hamiltonian_pauli.ABs:
            
            if A > B: continue
    
            pauli_p = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_p.XX[A,B] += h2
            pauli_p.XX[B,A] += h2
            aiem_p = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            pauli_m = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_m.XX[A,B] -= h2
            pauli_m.XX[B,A] -= h2
            aiem_m = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
        
            for I in range(aiem.nstate):
                G_fci[I].XX[A,B] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h2) 
                G_vqe[I].XX[A,B] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h2) 
                G_cis[I].XX[A,B] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h2) 
                G_fci[I].XX[B,A] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h2) 
                G_vqe[I].XX[B,A] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h2) 
                G_cis[I].XX[B,A] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h2) 
    
        # ZZ
        
        for A, B in aiem.aiem_hamiltonian_pauli.ABs:
            
            if A > B: continue
    
            pauli_p = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_p.ZZ[A,B] += h2
            pauli_p.ZZ[B,A] += h2
            aiem_p = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            pauli_m = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_m.ZZ[A,B] -= h2
            pauli_m.ZZ[B,A] -= h2
            aiem_m = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
        
            for I in range(aiem.nstate):
                G_fci[I].ZZ[A,B] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h2) 
                G_vqe[I].ZZ[A,B] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h2) 
                G_cis[I].ZZ[A,B] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h2) 
                G_fci[I].ZZ[B,A] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h2) 
                G_vqe[I].ZZ[B,A] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h2) 
                G_cis[I].ZZ[B,A] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h2) 
    
        # XZ/ZX
        
        for A, B in aiem.aiem_hamiltonian_pauli.ABs:
            
            pauli_p = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_p.XZ[A,B] += h2
            pauli_p.ZX[B,A] += h2
            aiem_p = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            pauli_m = AIEMPauli.copy(aiem.aiem_hamiltonian_pauli)
            pauli_m.XZ[A,B] -= h2
            pauli_m.ZX[B,A] -= h2
            aiem_m = AIEM(options=aiem2.options.copy().set_values({
                'aiem_hamiltonian_pauli' : pauli_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )

            for I in range(aiem.nstate):
                G_fci[I].XZ[A,B] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h2) 
                G_vqe[I].XZ[A,B] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h2) 
                G_cis[I].XZ[A,B] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h2) 
                G_fci[I].ZX[B,A] = (aiem_p.fci_E[I] - aiem_m.fci_E[I]) / (2.0 * h2) 
                G_vqe[I].ZX[B,A] = (aiem_p.vqe_E[I] - aiem_m.vqe_E[I]) / (2.0 * h2) 
                G_cis[I].ZX[B,A] = (aiem_p.cis_E[I] - aiem_m.cis_E[I]) / (2.0 * h2) 

        return G_fci, G_vqe, G_cis
        
    @staticmethod
    def test_fd_gradient_pauli(
        aiem,
        h0=1.0E-7,
        h1=1.0E-7,
        h2=1.0E-7,
        print_level=0,
        **kwargs
        ):

        G_fci = [aiem.compute_fci_dm(I=I, relaxed=True) for I in range(aiem.nstate)]
        G_vqe = [aiem.compute_vqe_dm(I=I, relaxed=True, **kwargs) for I in range(aiem.nstate)]
        G_cis = [aiem.compute_cis_dm(I=I, relaxed=True) for I in range(aiem.nstate)]

        G_fci_fd, G_vqe_fd, G_cis_fd = AIEMGradCheck.compute_fd_gradient_pauli(
            aiem=aiem,
            h0=h0,
            h1=h1,
            h2=h2,
            print_level=print_level,
            )

        print('==> Pauli Gradient Check <==\n')

        print('Analytical vs. Finite Difference:\n')
        print('h0 = %r' % h0)
        print('h1 = %r' % h1)
        print('h2 = %r' % h2)
        print('')
        for label, G, G_fd in zip(['FCI', 'VQE', 'CIS'], [G_fci, G_vqe, G_cis], [G_fci_fd, G_vqe_fd, G_cis_fd]):
        
            print('Method: %s\n' % label)
            print('%2s: %11s %11s %11s %11s %11s %11s %11s' % (
                'I', 'E', 'X', 'Z', 'XX', 'XZ', 'ZX', 'ZZ'))
            for I in range(len(G)):
                G1 = G[I]
                G2 = G_fd[I]
                print('%2d: %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E' % (
                    I,
                    np.max(np.abs(G1.E - G2.E)),
                    np.max(np.abs(G1.X - G2.X)),
                    np.max(np.abs(G1.Z - G2.Z)),
                    np.max(np.abs(G1.XX - G2.XX)),
                    np.max(np.abs(G1.XZ - G2.XZ)),
                    np.max(np.abs(G1.ZX - G2.ZX)),
                    np.max(np.abs(G1.ZZ - G2.ZZ)),
                    ))

            print('')

        print('Method to Method (Analytical):\n')
        for label, Gs1, Gs2 in zip(['FCI-VQE', 'FCI-CIS', 'VQE-CIS'], [G_fci, G_fci, G_vqe], [G_vqe, G_cis, G_cis]):
        
            print('Method: %s\n' % label)
            print('%2s: %11s %11s %11s %11s %11s %11s %11s' % (
                'I', 'E', 'X', 'Z', 'XX', 'XZ', 'ZX', 'ZZ'))
            for I in range(len(G)):
                G1 = Gs1[I]
                G2 = Gs2[I]
                print('%2d: %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E' % (
                    I,
                    np.max(np.abs(G1.E - G2.E)),
                    np.max(np.abs(G1.X - G2.X)),
                    np.max(np.abs(G1.Z - G2.Z)),
                    np.max(np.abs(G1.XX - G2.XX)),
                    np.max(np.abs(G1.XZ - G2.XZ)),
                    np.max(np.abs(G1.ZX - G2.ZX)),
                    np.max(np.abs(G1.ZZ - G2.ZZ)),
                    ))
            print('')

        print('==> End Pauli Gradient Check <==\n')

        return G_fci, G_vqe, G_cis, G_fci_fd, G_vqe_fd, G_cis_fd

    @staticmethod
    def compute_fd_gradient_monomer(
        aiem,
        hE=1.0E-7,
        hM=1.0E-6,
        hR=1.0E-6,
        print_level=0,
        ):

        # Print override
        aiem2 = AIEM(options=aiem.options.copy().set_values({
            'print_level' : print_level,
            }))
        
        G_fci = [AIEMMonomer.zeros_like(aiem.aiem_monomer) for _ in range(aiem.nstate)]
        G_vqe = [AIEMMonomer.zeros_like(aiem.aiem_monomer) for _ in range(aiem.nstate)]
        G_cis = [AIEMMonomer.zeros_like(aiem.aiem_monomer) for _ in range(aiem.nstate)]
    
        # EH
        
        for A in range(aiem.N):
    
            aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
            aiem_monomer_p.EH[A] += hE
            aiem_p = AIEM(aiem2.options.copy().set_values({
                'aiem_monomer' : aiem_monomer_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
            aiem_monomer_m.EH[A] -= hE
            aiem_m = AIEM(aiem2.options.copy().set_values({
                'aiem_monomer' : aiem_monomer_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )

            for I in range(aiem.nstate):
                G_fci[I].EH[A] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hE)
                G_vqe[I].EH[A] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hE)
                G_cis[I].EH[A] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hE)
    
        # ET
        
        for A in range(aiem.N):
    
            aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
            aiem_monomer_p.ET[A] += hE
            aiem_p = AIEM(aiem2.options.copy().set_values({
                'aiem_monomer' : aiem_monomer_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
            aiem_monomer_m.ET[A] -= hE
            aiem_m = AIEM(aiem2.options.copy().set_values({
                'aiem_monomer' : aiem_monomer_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )

            for I in range(aiem.nstate):
                G_fci[I].ET[A] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hE)
                G_vqe[I].ET[A] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hE)
                G_cis[I].ET[A] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hE)
    
        # EP
        
        for A in range(aiem.N):
    
            aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
            aiem_monomer_p.EP[A] += hE
            aiem_p = AIEM(aiem2.options.copy().set_values({
                'aiem_monomer' : aiem_monomer_p,
                }))
            aiem_p.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )
    
            aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
            aiem_monomer_m.EP[A] -= hE
            aiem_m = AIEM(aiem2.options.copy().set_values({
                'aiem_monomer' : aiem_monomer_m,
                }))
            aiem_m.compute_energy(
                param_values_ref=aiem.vqe_circuit.param_values,
                cis_C_ref=aiem.cis_C,
                vqe_C_ref=aiem.vqe_C,
                )

            for I in range(aiem.nstate):
                G_fci[I].EP[A] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hE)
                G_vqe[I].EP[A] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hE)
                G_cis[I].EP[A] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hE)
    
        # MH
        
        for A in range(aiem.N):
    
            for T in range(3):

                aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_p.MH[A,T] += hM
                aiem_p = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_p,
                    }))
                aiem_p.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )
    
                aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_m.MH[A,T] -= hM
                aiem_m = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_m,
                    }))
                aiem_m.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )

                for I in range(aiem.nstate):
                    G_fci[I].MH[A,T] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hM)
                    G_vqe[I].MH[A,T] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hM)
                    G_cis[I].MH[A,T] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hM)
    
        # MT
        
        for A in range(aiem.N):
    
            for T in range(3):

                aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_p.MT[A,T] += hM
                aiem_p = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_p,
                    }))
                aiem_p.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )
    
                aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_m.MT[A,T] -= hM
                aiem_m = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_m,
                    }))
                aiem_m.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )

                for I in range(aiem.nstate):
                    G_fci[I].MT[A,T] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hM)
                    G_vqe[I].MT[A,T] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hM)
                    G_cis[I].MT[A,T] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hM)
    
        # MP
        
        for A in range(aiem.N):
    
            for T in range(3):

                aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_p.MP[A,T] += hM
                aiem_p = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_p,
                    }))
                aiem_p.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )
    
                aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_m.MP[A,T] -= hM
                aiem_m = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_m,
                    }))
                aiem_m.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )

                for I in range(aiem.nstate):
                    G_fci[I].MP[A,T] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hM)
                    G_vqe[I].MP[A,T] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hM)
                    G_cis[I].MP[A,T] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hM)
    
        # R0
        
        for A in range(aiem.N):
    
            for T in range(3):

                aiem_monomer_p = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_p.R0[A,T] += hR
                aiem_p = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_p,
                    }))
                aiem_p.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )
    
                aiem_monomer_m = AIEMMonomer.copy(aiem.aiem_monomer)
                aiem_monomer_m.R0[A,T] -= hR
                aiem_m = AIEM(aiem2.options.copy().set_values({
                    'aiem_monomer' : aiem_monomer_m,
                    }))
                aiem_m.compute_energy(
                    param_values_ref=aiem.vqe_circuit.param_values,
                    cis_C_ref=aiem.cis_C,
                    vqe_C_ref=aiem.vqe_C,
                    )

                for I in range(aiem.nstate):
                    G_fci[I].R0[A,T] = (aiem_p.fci_tot_E[I] - aiem_m.fci_tot_E[I]) / (2.0 * hR)
                    G_vqe[I].R0[A,T] = (aiem_p.vqe_tot_E[I] - aiem_m.vqe_tot_E[I]) / (2.0 * hR)
                    G_cis[I].R0[A,T] = (aiem_p.cis_tot_E[I] - aiem_m.cis_tot_E[I]) / (2.0 * hR)
    
        return G_fci, G_vqe, G_cis
        
    @staticmethod
    def test_fd_gradient_monomer(
        aiem,
        hE=1.0E-7,
        hM=1.0E-6,
        hR=1.0E-6,
        print_level=0,
        **kwargs
        ):

        G_fci = [AIEMUtil.pauli_to_monomer_grad(monomer=aiem.aiem_monomer, pauli=aiem.compute_fci_dm(I=I, relaxed=True)) for I in range(aiem.nstate)]
        G_vqe = [AIEMUtil.pauli_to_monomer_grad(monomer=aiem.aiem_monomer, pauli=aiem.compute_vqe_dm(I=I, relaxed=True, **kwargs)) for I in range(aiem.nstate)]
        G_cis = [AIEMUtil.pauli_to_monomer_grad(monomer=aiem.aiem_monomer, pauli=aiem.compute_cis_dm(I=I, relaxed=True)) for I in range(aiem.nstate)]
        
        G_fci_fd, G_vqe_fd, G_cis_fd = AIEMGradCheck.compute_fd_gradient_monomer(
            aiem=aiem,
            hE=hE,
            hM=hM,
            hR=hR,
            print_level=print_level,
            )
    
        print('==> Prop Gradient Check <==\n')

        print('Analytical vs. Finite Difference:\n')
        print('hE = %r' % hE)
        print('hM = %r' % hM)
        print('hR = %r' % hR)
        print('')
        for label, G, G_fd in zip(['FCI', 'VQE', 'CIS'], [G_fci, G_vqe, G_cis], [G_fci_fd, G_vqe_fd, G_cis_fd]):
        
            print('Method: %s\n' % label)
            print('%2s: %11s %11s %11s %11s %11s %11s %11s' % (
                'I', 'EH', 'ET', 'EP', 'MH', 'MT', 'MP', 'R0'))
            for I in range(len(G)):
                G1 = G[I]
                G2 = G_fd[I]
                print('%2d: %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E' % (
                    I,
                    np.max(np.abs(G1.EH - G2.EH)),
                    np.max(np.abs(G1.ET - G2.ET)),
                    np.max(np.abs(G1.EP - G2.EP)),
                    np.max(np.abs(G1.MH - G2.MH)),
                    np.max(np.abs(G1.MT - G2.MT)),
                    np.max(np.abs(G1.MP - G2.MP)),
                    np.max(np.abs(G1.R0 - G2.R0)),
                    ))
            print('')

        print('Method to Method (Analytical):\n')
        for label, Gs1, Gs2 in zip(['FCI-VQE', 'FCI-CIS', 'VQE-CIS'], [G_fci, G_fci, G_vqe], [G_vqe, G_cis, G_cis]):
        
            print('Method: %s\n' % label)
            print('%2s: %11s %11s %11s %11s %11s %11s %11s' % (
                'I', 'EH', 'ET', 'EP', 'MH', 'MT', 'MP', 'R0'))
            for I in range(len(G)):
                G1 = Gs1[I]
                G2 = Gs2[I]
                print('%2d: %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E %11.3E' % (
                    I,
                    np.max(np.abs(G1.EH - G2.EH)),
                    np.max(np.abs(G1.ET - G2.ET)),
                    np.max(np.abs(G1.EP - G2.EP)),
                    np.max(np.abs(G1.MH - G2.MH)),
                    np.max(np.abs(G1.MT - G2.MT)),
                    np.max(np.abs(G1.MP - G2.MP)),
                    np.max(np.abs(G1.R0 - G2.R0)),
                    ))
            print('')

        print('==> End Prop Gradient Check <==\n')

        return G_fci, G_vqe, G_cis, G_fci_fd, G_vqe_fd, G_cis_fd
