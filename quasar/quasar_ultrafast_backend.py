import numpy as np
from .quasar_backend import QuasarSimulatorBackend
from .circuit import Gate
from .circuit import Circuit

class QuasarUltrafastBackend(QuasarSimulatorBackend):

    def __init__(
        self,
        ):
        pass

    def __str__(self):
        return 'Quasar Ultrafast Simulator Backend (Statevector)'

    def run_pauli_expectation_value_gradient(
        self,
        circuit,
        pauli,
        nmeasurement=None,
        statevector=None,
        min_qubit=None,
        nqubit=None,
        dtype=np.complex128,
        parameter_indices=None,
        **kwargs):

        ipauli_gates = {
            'Rx' : Gate.U1(np.array([[0.0+0.0j, 0.0+1.0j], [0.0+1.0j, 0.0+0.0j]], dtype=np.complex128)),
            'Ry' : Gate.U1(np.array([[0.0+0.0j, 1.0+0.0j], [-1.0+0.0j, 0.0+0.0j]], dtype=np.complex128)),
            'Rz' : Gate.U1(np.array([[0.0+1.0j, 0.0+0.0j], [0.0+0.0j, 0.0-1.0j]], dtype=np.complex128)),
        }

        # Default to parameter shift rule
        if nmeasurement is not None:
            return super().run_pauli_expectation_value_gradient(
                circuit=circuit,
                pauli=pauli,
                nmeasurement=nmeasurement,
                statevector=statevector,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                parameter_indices=parameter_indices,
                **kwargs)

        # Offsets
        min_qubit = circuit.min_qubit if min_qubit is None else min_qubit
        nqubit = circuit.nqubit if nqubit is None else nqubit
        
        # Default to taking the gradient with respect to all parameters
        if parameter_indices is None:
            parameter_indices = list(range(circuit.nparameter))

        # Check that the gradient formula is known for these parameters (i.e., Rx, Ry, Rz gates)
        parameter_keys = circuit.parameter_keys
        for parameter_index in parameter_indices:
            key = parameter_keys[parameter_index]
            times, qubits, key2 = key
            gate = circuit.gates[(times, qubits)]
            if not gate.name in ('Rx', 'Ry', 'Rz'): 
                raise RuntimeError('Unknown gradient rule: presently can only differentiate Rx, Ry, Rz gates: %s' % gate)

        # Reduced parameter keys (always Rx/Ry/Rz gates)
        parameter_gate_keys = circuit.parameter_gate_keys
        parameter_gate_keys = [parameter_gate_keys[_] for _ in parameter_indices]
         
        # Split circuit into pivot gates (target Rx/Ry/Rz) and edge gates (U, V, W, ...)
        Ws = []
        Rs = []
        W = Circuit()
        for key, gate in circuit.gates.items():
            times, qubits = key
            if key in parameter_gate_keys:
                # Pivot Gate
                R = Circuit().add_gate(gate=gate, qubits=qubits, times=times)
                Rs.append(R)
                Ws.append(W)
                W = Circuit()
            else:
                # Edge Gate
                W.add_gate(gate=gate, qubits=qubits, times=times)
        Ws.append(W)
        Ws = Ws[1:]

        # Initial statevectors
        gamma = self.run_statevector(
            circuit=circuit,
            statevector=statevector,
            dtype=dtype,
            min_qubit=min_qubit,
            nqubit=nqubit,
            **kwargs)
        lamda = pauli.matrix_vector_product(
            statevector=gamma,
            dtype=dtype,
            min_qubit=min_qubit,
            nqubit=nqubit,
            )

        # Evaluate the gradient by the ultrafast rule
        gradient = np.zeros((len(parameter_indices),), dtype=dtype)
        for index in reversed(range(len(parameter_indices))):
            W = Ws[index]
            R = Rs[index]
            gate_key = parameter_gate_keys[index]
            gate_name = circuit.gates[gate_key].name
            gate_qubit = gate_key[1][0]
            RT = R.adjoint()
            WT = W.adjoint()
            RTWT = Circuit.join_in_time([WT, RT])
            G = Circuit().add_gate(ipauli_gates[gate_name], gate_qubit)

            # Timestep
            gamma = self.run_statevector(
                circuit=RTWT,
                statevector=gamma,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            lamda = self.run_statevector(
                circuit=RTWT,
                statevector=lamda,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)

            # Gradient Evaluation
            lamda2 = self.run_statevector(
                circuit=G,
                statevector=lamda,
                min_qubit=min_qubit,
                nqubit=nqubit,
                dtype=dtype,
                **kwargs)
            V = np.sum(lamda2.conj() * gamma)
            gradient[index] = + 2.0 * np.real(V) 
             
        return gradient

    
