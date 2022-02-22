from hypothesis.strategies import (lists, integers, composite, sampled_from,
                                   floats, permutations)
from quasar.circuit import Gate, Circuit
import inspect
import math

# This doesn't work to get all the gates because some entries in Gate
# are functions that return a gate
# allowed_gates = [x for x in Gate.__dict__.values() if isinstance(x, Gate)]
# so we do this for now, which must be refreshed if Gate is extended
allowed_gate_names = [
        'I', 'X', 'Y', 'Z', 'H', 'S', 'ST', 'T', 'TT', 'Rx2', 'Rx2T', 'CX',
        'CY', 'CZ', 'CS', 'CST', 'SWAP', 'CCX', 'CSWAP', 'Rx', 'Ry', 'Rz',
        'RBS', 'iRBS', 'u1', 'u2', 'u3', 'SO4', 'SO42', 'CF', 'R_ion',
        'Rx_ion', 'Ry_ion', 'Rz_ion', 'XX_ion'] # , 'U1', 'U2'

allowed_gates = [
    getattr(Gate, x) for x in allowed_gate_names
]


@composite
def gates(draw, allowed_gates=allowed_gates):
    result = draw(sampled_from(allowed_gates))
    if callable(result):
        s = inspect.signature(result)
        kwargs = {}
        angles = floats(min_value=0, max_value=2 * math.pi)
        for p in s.parameters:
            # right now we're assuming that any parameter is an angle
            # from 0 to 2*pi,
            # although in the future we could filter by parameter name
            value = draw(angles)
            kwargs[p] = value
        result = result(**kwargs)
    return result


@composite
def circuits(draw,
             min_qubits,
             max_qubits,
             min_length,
             max_length,
             allowed_gates=allowed_gates,
             use_all_allowed_gates=False):
    length = draw(integers(min_value=min_length, max_value=max_length))
    num_qubits = draw(integers(min_value=min_qubits, max_value=max_qubits))
    if use_all_allowed_gates:
        circuit_gates = draw(permutations(allowed_gates))
        assert min_length >= len(allowed_gates)
        max_gate_qubits = max((x.nqubit for x in circuit_gates))
        assert min_qubits >= max_gate_qubits
    circuit_gates = draw(
        lists(gates(allowed_gates), min_size=length, max_size=length).filter(
            lambda x: all([y.nqubit <= num_qubits for y in x])))
    result = Circuit()
    for gate in circuit_gates:
        qubits = draw(
            lists(integers(min_value=0, max_value=num_qubits - 1),
                  min_size=gate.nqubit,
                  max_size=gate.nqubit,
                  unique=True))
        result.add_gate(gate, tuple(qubits))
    return result
