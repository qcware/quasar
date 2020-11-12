import quasar
import numpy as np
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, floats, complex_numbers, composite
from hypothesis.extra.numpy import arrays
import numpy as np
import qiskit


@composite
def statevector(draw, length=integers):
    result = draw(
        arrays(dtype=np.complex128,
               shape=(length, ),
               elements=complex_numbers(allow_infinity=False,
                                        allow_nan=False)))
    norm = np.linalg.norm(result)
    assume(norm > 0)
    result = result / norm
    norm = np.linalg.norm(result)
    assume(norm == 1.0)
    return result


# in the following tests, we set deadline to None because
# it takes so long to import qiskit on demand, even with
# doing much of the work above, that pytest flags the tests
# as "flaky" since the first test will take around 800ms and the
# second around 5ms on my workstation.
@given(theta=floats(0, 2 * np.pi))
@settings(deadline=None)
def test_rbs(theta: float):

    circuit = quasar.Circuit().X(0)
    circuit.add_gate(quasar.Gate.RBS(theta=theta), (0, 1))

    backend = quasar.QuasarSimulatorBackend()
    results = backend.run_statevector(circuit)

    assert np.isclose(results, [0, np.sin(theta), np.cos(theta), 0]).all()

    backend = quasar.QiskitSimulatorBackend()
    results = backend.run_statevector(circuit)

    assert np.isclose(results, [0, np.sin(theta), np.cos(theta), 0]).all()


@given(theta=floats(0, 2 * np.pi))
@settings(deadline=None)
def test_irbs(theta: float):
    circuit = quasar.Circuit().X(0)
    circuit.add_gate(quasar.Gate.iRBS(theta=theta), (0, 1))

    backend = quasar.QuasarSimulatorBackend()
    results = backend.run_statevector(circuit)

    assert np.isclose(
        results,
        [0, -1.j * np.sin(theta), np.cos(theta), 0]).all()

    # backend = quasar.QiskitSimulatorBackend()
    # results = backend.run_statevector(circuit)

    # assert np.isclose(
    #     results,
    #     [0, -1.j * np.sin(theta), np.cos(theta), 0]).all()

