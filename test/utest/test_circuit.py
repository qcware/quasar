import quasar
import unittest
import sortedcontainers

class TestCircuit(unittest.TestCase):

    def test_empty_circuit(self):

        circuit = quasar.Circuit()
        
        self.assertEqual(circuit.ngate, 0)
        self.assertEqual(circuit.ngate1, 0)
        self.assertEqual(circuit.ngate2, 0)
        self.assertEqual(circuit.ngate3, 0)
        self.assertEqual(circuit.ngate4, 0)
        self.assertEqual(circuit.ngate_nqubit(0), 0)
        self.assertEqual(circuit.ngate_nqubit(1), 0)
        self.assertEqual(circuit.ngate_nqubit(2), 0)
        self.assertEqual(circuit.ngate_nqubit(3), 0)
        self.assertEqual(circuit.ngate_nqubit(4), 0)
        self.assertEqual(circuit.max_gate_nqubit, 0)
        self.assertEqual(circuit.max_gate_ntime, 0)
        self.assertEqual(circuit.min_time, 0)
        self.assertEqual(circuit.max_time, -1)
        self.assertEqual(circuit.ntime, 0)
        self.assertEqual(circuit.ntime_sparse, 0)
        self.assertEqual(circuit.min_qubit, 0)
        self.assertEqual(circuit.max_qubit, -1)
        self.assertEqual(circuit.nqubit, 0)
        self.assertEqual(circuit.nqubit_sparse, 0)
        self.assertFalse(circuit.is_controlled)
        self.assertFalse(circuit.is_composite)

        self.assertIsInstance(circuit, quasar.Circuit)
        self.assertIsInstance(circuit.gates, sortedcontainers.SortedDict)
        self.assertIsInstance(circuit.times, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.qubits, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.times_and_qubits, sortedcontainers.SortedSet)

        self.assertEqual(circuit.gates, sortedcontainers.SortedDict([]))
        self.assertEqual(circuit.times, sortedcontainers.SortedSet([]))
        self.assertEqual(circuit.qubits, sortedcontainers.SortedSet([]))
        self.assertEqual(circuit.times_and_qubits, sortedcontainers.SortedSet([]))

        self.assertIsInstance(str(circuit), str)

    def test_h_circuit(self): 
        
        circuit = quasar.Circuit().add_gate(quasar.Gate.H, 0, copy=False)
        
        self.assertEqual(circuit.ngate, 1)
        self.assertEqual(circuit.ngate1, 1)
        self.assertEqual(circuit.ngate2, 0)
        self.assertEqual(circuit.ngate3, 0)
        self.assertEqual(circuit.ngate4, 0)
        self.assertEqual(circuit.ngate_nqubit(0), 0)
        self.assertEqual(circuit.ngate_nqubit(1), 1)
        self.assertEqual(circuit.ngate_nqubit(2), 0)
        self.assertEqual(circuit.ngate_nqubit(3), 0)
        self.assertEqual(circuit.ngate_nqubit(4), 0)
        self.assertEqual(circuit.max_gate_nqubit, 1)
        self.assertEqual(circuit.max_gate_ntime, 1)
        self.assertEqual(circuit.min_time, 0)
        self.assertEqual(circuit.max_time, 0)
        self.assertEqual(circuit.ntime, 1)
        self.assertEqual(circuit.ntime_sparse, 1)
        self.assertEqual(circuit.min_qubit, 0)
        self.assertEqual(circuit.max_qubit, 0)
        self.assertEqual(circuit.nqubit, 1)
        self.assertEqual(circuit.nqubit_sparse, 1)
        self.assertFalse(circuit.is_controlled)
        self.assertFalse(circuit.is_composite)

        self.assertIsInstance(circuit, quasar.Circuit)
        self.assertIsInstance(circuit.gates, sortedcontainers.SortedDict)
        self.assertIsInstance(circuit.times, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.qubits, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.times_and_qubits, sortedcontainers.SortedSet)

        self.assertEqual(circuit.gates, sortedcontainers.SortedDict([(((0,), (0,)), quasar.Gate.H)]))
        self.assertEqual(circuit.times, sortedcontainers.SortedSet([0]))
        self.assertEqual(circuit.qubits, sortedcontainers.SortedSet([0]))
        self.assertEqual(circuit.times_and_qubits, sortedcontainers.SortedSet([(0, 0)]))

        self.assertIsInstance(str(circuit), str)

    def test_ghz_circuit(self): 
        
        circuit = quasar.Circuit()
        circuit.add_gate(quasar.Gate.H, 0, copy=False)
        circuit.add_gate(quasar.Gate.CX, (0,1), copy=False)
        circuit.add_gate(quasar.Gate.CX, qubits=(1,2), copy=False)
        
        self.assertEqual(circuit.ngate, 3)
        self.assertEqual(circuit.ngate1, 1)
        self.assertEqual(circuit.ngate2, 2)
        self.assertEqual(circuit.ngate3, 0)
        self.assertEqual(circuit.ngate4, 0)
        self.assertEqual(circuit.ngate_nqubit(0), 0)
        self.assertEqual(circuit.ngate_nqubit(1), 1)
        self.assertEqual(circuit.ngate_nqubit(2), 2)
        self.assertEqual(circuit.ngate_nqubit(3), 0)
        self.assertEqual(circuit.ngate_nqubit(4), 0)
        self.assertEqual(circuit.max_gate_nqubit, 2)
        self.assertEqual(circuit.max_gate_ntime, 1)
        self.assertEqual(circuit.min_time, 0)
        self.assertEqual(circuit.max_time, 2)
        self.assertEqual(circuit.ntime, 3)
        self.assertEqual(circuit.ntime_sparse, 3)
        self.assertEqual(circuit.min_qubit, 0)
        self.assertEqual(circuit.max_qubit, 2)
        self.assertEqual(circuit.nqubit, 3)
        self.assertEqual(circuit.nqubit_sparse, 3)
        self.assertFalse(circuit.is_controlled)
        self.assertFalse(circuit.is_composite)

        self.assertIsInstance(circuit, quasar.Circuit)
        self.assertIsInstance(circuit.gates, sortedcontainers.SortedDict)
        self.assertIsInstance(circuit.times, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.qubits, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.times_and_qubits, sortedcontainers.SortedSet)

        self.assertEqual(circuit.gates, sortedcontainers.SortedDict([
            (((0,), (0,)), quasar.Gate.H),
            (((1,), (0, 1)), quasar.Gate.CX),
            (((2,), (1, 2)), quasar.Gate.CX),
            ]))
        self.assertEqual(circuit.times, sortedcontainers.SortedSet([0, 1, 2]))
        self.assertEqual(circuit.qubits, sortedcontainers.SortedSet([0, 1, 2]))
        self.assertEqual(circuit.times_and_qubits, sortedcontainers.SortedSet([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]))

        self.assertIsInstance(str(circuit), str)

    def test_sparse_circuit(self): 
        
        circuit = quasar.Circuit()
        circuit.add_gate(quasar.Gate.H, 0, copy=False)
        circuit.add_gate(quasar.Gate.CX, (0,1), times=(2,), copy=False)
        circuit.add_gate(quasar.Gate.CX, qubits=(3,4), times=(4,), copy=False)
        
        self.assertEqual(circuit.ngate, 3)
        self.assertEqual(circuit.ngate1, 1)
        self.assertEqual(circuit.ngate2, 2)
        self.assertEqual(circuit.ngate3, 0)
        self.assertEqual(circuit.ngate4, 0)
        self.assertEqual(circuit.ngate_nqubit(0), 0)
        self.assertEqual(circuit.ngate_nqubit(1), 1)
        self.assertEqual(circuit.ngate_nqubit(2), 2)
        self.assertEqual(circuit.ngate_nqubit(3), 0)
        self.assertEqual(circuit.ngate_nqubit(4), 0)
        self.assertEqual(circuit.max_gate_nqubit, 2)
        self.assertEqual(circuit.max_gate_ntime, 1)
        self.assertEqual(circuit.min_time, 0)
        self.assertEqual(circuit.max_time, 4)
        self.assertEqual(circuit.ntime, 5)
        self.assertEqual(circuit.ntime_sparse, 3)
        self.assertEqual(circuit.min_qubit, 0)
        self.assertEqual(circuit.max_qubit, 4)
        self.assertEqual(circuit.nqubit, 5)
        self.assertEqual(circuit.nqubit_sparse, 4)
        self.assertFalse(circuit.is_controlled)
        self.assertFalse(circuit.is_composite)

        self.assertIsInstance(circuit, quasar.Circuit)
        self.assertIsInstance(circuit.gates, sortedcontainers.SortedDict)
        self.assertIsInstance(circuit.times, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.qubits, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.times_and_qubits, sortedcontainers.SortedSet)

        self.assertEqual(circuit.gates, sortedcontainers.SortedDict([
            (((0,), (0,)), quasar.Gate.H),
            (((2,), (0, 1)), quasar.Gate.CX),
            (((4,), (3, 4)), quasar.Gate.CX),
            ]))
        self.assertEqual(circuit.times, sortedcontainers.SortedSet([0, 2, 4]))
        self.assertEqual(circuit.qubits, sortedcontainers.SortedSet([0, 1, 3, 4]))
        self.assertEqual(circuit.times_and_qubits, sortedcontainers.SortedSet([(0, 0), (2, 0), (2, 1), (4, 3), (4, 4)]))

        self.assertIsInstance(str(circuit), str)

    def test_shifted_sparse_circuit(self): 
        
        circuit = quasar.Circuit()
        circuit.add_gate(quasar.Gate.H, -2, times=-2, copy=False)
        circuit.add_gate(quasar.Gate.CX, (0,1), times=(2,), copy=False)
        circuit.add_gate(quasar.Gate.CX, qubits=(3,4), times=(4,), copy=False)
        
        self.assertEqual(circuit.ngate, 3)
        self.assertEqual(circuit.ngate1, 1)
        self.assertEqual(circuit.ngate2, 2)
        self.assertEqual(circuit.ngate3, 0)
        self.assertEqual(circuit.ngate4, 0)
        self.assertEqual(circuit.ngate_nqubit(0), 0)
        self.assertEqual(circuit.ngate_nqubit(1), 1)
        self.assertEqual(circuit.ngate_nqubit(2), 2)
        self.assertEqual(circuit.ngate_nqubit(3), 0)
        self.assertEqual(circuit.ngate_nqubit(4), 0)
        self.assertEqual(circuit.max_gate_nqubit, 2)
        self.assertEqual(circuit.max_gate_ntime, 1)
        self.assertEqual(circuit.min_time, -2)
        self.assertEqual(circuit.max_time, 4)
        self.assertEqual(circuit.ntime, 7)
        self.assertEqual(circuit.ntime_sparse, 3)
        self.assertEqual(circuit.min_qubit, -2)
        self.assertEqual(circuit.max_qubit, 4)
        self.assertEqual(circuit.nqubit, 7)
        self.assertEqual(circuit.nqubit_sparse, 5)
        self.assertFalse(circuit.is_controlled)
        self.assertFalse(circuit.is_composite)

        self.assertIsInstance(circuit, quasar.Circuit)
        self.assertIsInstance(circuit.gates, sortedcontainers.SortedDict)
        self.assertIsInstance(circuit.times, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.qubits, sortedcontainers.SortedSet)
        self.assertIsInstance(circuit.times_and_qubits, sortedcontainers.SortedSet)

        self.assertEqual(circuit.gates, sortedcontainers.SortedDict([
            (((-2,), (-2,)), quasar.Gate.H),
            (((2,), (0, 1)), quasar.Gate.CX),
            (((4,), (3, 4)), quasar.Gate.CX),
            ]))
        self.assertEqual(circuit.times, sortedcontainers.SortedSet([-2, 2, 4]))
        self.assertEqual(circuit.qubits, sortedcontainers.SortedSet([-2, 0, 1, 3, 4]))
        self.assertEqual(circuit.times_and_qubits, sortedcontainers.SortedSet([(-2, -2), (2, 0), (2, 1), (4, 3), (4, 4)]))

        self.assertIsInstance(str(circuit), str)

    def test_circuit_equivalence(self):
        
        # Reflexivity
        circuit1 = quasar.Circuit().H(0).CX(0, 1)
        self.assertTrue(quasar.Circuit.test_equivalence(circuit1, circuit1))

        # Reflexivity (different objects)
        circuit1 = quasar.Circuit().H(0).CX(0, 1)
        circuit2 = quasar.Circuit().H(0).CX(0, 1)
        self.assertTrue(quasar.Circuit.test_equivalence(circuit1, circuit2))

        # Reflexivity (different build order)
        circuit1 = quasar.Circuit().CX(0, 1, times=1).H(0, times=0)
        circuit2 = quasar.Circuit().H(0).CX(0, 1)
        self.assertTrue(quasar.Circuit.test_equivalence(circuit1, circuit2))

        # Same layout, different gates
        circuit1 = quasar.Circuit().Z(0).CX(0, 1)
        circuit2 = quasar.Circuit().H(0).CX(0, 1)
        self.assertFalse(quasar.Circuit.test_equivalence(circuit1, circuit2))

        # Same layout, different times
        circuit1 = quasar.Circuit().H(0).CX(0, 1)
        circuit2 = quasar.Circuit().H(0, time_start=-1).CX(1, 2)
        self.assertFalse(quasar.Circuit.test_equivalence(circuit1, circuit2))

        # Same layout, different qubits
        circuit1 = quasar.Circuit().H(0).CX(0, 1)
        circuit2 = quasar.Circuit().H(1).CX(1, 2)
        self.assertFalse(quasar.Circuit.test_equivalence(circuit1, circuit2))

if __name__ == '__main__':

    unittest.main()
