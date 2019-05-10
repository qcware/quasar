import numpy as np
from .circuit import Circuit
from .parameters import FixedParameterGroup
from .parameters import CompositeParameterGroup
from . import pauli

class Collocation(object):

    # => Energies <= #

    @staticmethod
    def compute_energy_and_pauli_dm(
        backend,
        nmeasurement,
        hamiltonian,
        circuit,
        ):

        pauli_dm = backend.compute_pauli_dm(
            circuit=circuit,
            pauli=hamiltonian,
            nmeasurement=nmeasurement,
            )
        E = pauli_dm.dot(hamiltonian)
        return E, pauli_dm

    @staticmethod
    def compute_sa_energy_and_pauli_dm(
        backend,
        nmeasurement,
        hamiltonian,
        circuit,
        reference_circuits,
        reference_weights,
        ):

        pauli_dm = pauli.Pauli.zeros_like(hamiltonian)
        E = 0.0
        for reference, weight in zip(reference_circuits, reference_weights):
            circuit2 = Circuit.concatenate([reference, circuit])
            E2, pauli_dm2 = Collocation.compute_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=circuit2,
                )
            E += weight * E2
            pauli_dm += weight * pauli_dm2
        return E, pauli_dm

    # => Gradients <= #

    @staticmethod
    def compute_gradient(
        backend,
        nmeasurement,
        hamiltonian,
        circuit,
        parameter_group,
        ):

        circuit2 = circuit.copy()
        Z = circuit2.param_values
        G = np.zeros_like(Z)
        for A in range(len(Z)):
            if not parameter_group.active_raw[A]: continue
            Zp = Z.copy()
            Zp[A] += np.pi / 4.0
            circuit2.set_param_values(Zp)
            Ep = Collocation.compute_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=circuit2,
                )[0]
            Zm = Z.copy()
            Zm[A] -= np.pi / 4.0
            circuit2.set_param_values(Zm)
            Em = Collocation.compute_energy_and_pauli_dm(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=circuit2,
                )[0]
            G[A] = (Ep - Em)

        G2 = parameter_group.compute_chain_rule1(Z, G)
        return G2
                
    @staticmethod
    def compute_sa_gradient(
        backend,
        nmeasurement,
        hamiltonian,
        circuit,
        parameter_group,
        reference_circuits,
        reference_weights,
        ):

        G = np.zeros((parameter_group.nparam,))
        for reference, weight in zip(reference_circuits, reference_weights):
            circuit2 = Circuit.concatenate([reference, circuit]) 
            fixed_group = FixedParameterGroup(reference.param_values) 
            composite_group = CompositeParameterGroup([fixed_group, parameter_group])
            G2 = Collocation.compute_gradient(
                backend=backend,
                nmeasurement=nmeasurement,
                hamiltonian=hamiltonian,
                circuit=circuit2,
                parameter_group=composite_group,
                )
            G += weight * G2
        return G

            
            
        
            

