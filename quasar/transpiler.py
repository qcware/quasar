import sortedcontainers
from .circuit import ControlledGate


class Transpiler(object):

    """ class Transpiler is a collection of static utility functions that
        transform Circuit -> Circuit to exchange certain gate libraries. There
        are many potential use cases for these functions:

            - Flip control polarity on ControlledGates so that only positive
              controls are used.
            - Exchange ControlledGates for known Gates.
            - Remove redundant pairs of gates such as X * X or H * H.

        Many of these operations have preconditions on the required form of the
        input Circuit, such as:
            - Non-composite Circuit: the Circuit must have no CompositeGates
              (these drastically increase the complexity of transpilation
              operations). One should explode the Circuit before performing
              transpiler operations with this form.

        Many of these operations have known sequential application properties,
        such as:
            - Idempotence: op(op(circuit)) = op(circuit)
            - Nilpotence: op**N(circuit) = op**(N-1)(circuit) for some N

        These operations all return modified copies of the underlying circuits.
    """

    @staticmethod
    def polarize_controls(
        circuit,
        ):

        """ Add necessary X corrector gates to polarize all ControlledGate
            controls to True.

        Preconditions: Non-composite Circuit
        Transformation class: Idempotent
        """

        if circuit.is_composite: raise RuntimeError('circuit is composite')
        
        # Which starting times need X gates?
        expanders = sortedcontainers.SortedSet()
        for key, gate in circuit.gates.items():
            times, qubits = key
            if isinstance(gate, ControlledGate) and not all(gate.controls):
                expanders.add(times[0])
    
        # Map of old time -> new time with room for X corrector gates
        time_map = []
        counter = circuit.min_time
        for time in circuit.times:
            if time in expanders:
                time_map.append(counter+1)
                counter += 3
            else:
                time_map.append(counter)
                counter += 1

        # Circuit with room for X corrector gates
        circuit2 = circuit.slice(
            times=circuit.times, 
            times_to=time_map, 
            )

        # Perform the control polarization and mark X correctors
        X_gate_positions = []
        for key, gate in circuit2.gates.items():
            times, qubits = key
            if not isinstance(gate, ControlledGate) or all(gate.controls):
                continue
            for index, control in enumerate(gate.controls):
                if control: continue 
                # Mark X correctors
                X_gate_positions.append((times[0]-1, qubits[index]))
                X_gate_positions.append((times[0]+1, qubits[index]))
            # Polarize the controls
            gate.controls = [True]*gate.ncontrol

        # Add the X correctors
        for time, qubit in X_gate_positions:
            circuit2.X(qubit=qubit, times=time)

        return circuit2

